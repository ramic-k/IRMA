"""Emitter tests: ENDF decks (iel=10 + disordered classic), spectra YAML,
ncrystal YAML — every artifact through its real parser/loader gate."""
import json
import os

import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("phonopy")
yaml = pytest.importorskip("yaml")

from ase.build import bulk                          # noqa: E402
from ase.calculators.emt import EMT                 # noqa: E402

from irma.mlip.bundle import write_bundle           # noqa: E402
from irma.mlip.calculators import CalculatorSpec    # noqa: E402
from irma.mlip.emit import (                        # noqa: E402
    emit_endf_decks, emit_ncrystal_yaml, emit_spectra_yaml, resolve_species)
from irma.mlip.phonons import compute_force_constants  # noqa: E402
from irma.mlip.relax import relax                   # noqa: E402

SPEC = CalculatorSpec("emt")
QUIET = lambda *a, **k: None                        # noqa: E731


def _bundle(tmpdir, model, mesh=(4, 4, 4), **kw):
    rr, pr = model
    return write_bundle(str(tmpdir), phonon_result=pr,
                        relax_result=rr, calc_meta={"potential": "emt"},
                        args_used={}, mesh=mesh, progress=QUIET, **kw)


@pytest.fixture(scope="module")
def al_bundle(tmp_path_factory, al_model):
    return _bundle(tmp_path_factory.mktemp("al"), al_model)


@pytest.fixture(scope="module")
def cuau_bundle(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("cuau")
    rr = relax(bulk("CuAu", "rocksalt", a=4.1), EMT(), fmax=0.01, nmax=200)
    pr = compute_force_constants(
        rr.atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=1,
        scratch_dir=os.path.join(str(tmpdir), "scratch"), progress=QUIET)
    return _bundle(tmpdir, (rr, pr))


@pytest.fixture(scope="module")
def dis_bundle(tmp_path_factory, al_model):
    return _bundle(tmp_path_factory.mktemp("dis"), al_model, disordered=True)


def _parse(path):
    from irma.core.deck import TokenReader, parse_leapr_input
    from irma.gui.deck_text import parse_deck_to_staging
    tokens, lines, _, tl = parse_leapr_input(path)
    return parse_deck_to_staging(
        TokenReader(tokens, token_lines=tl, filename=str(path),
                    raw_lines=lines), str(path))


def test_species_resolution_defaults_and_overrides(al_bundle):
    species = resolve_species(al_bundle, progress=QUIET)
    (al,) = species
    # A phonopy model names an element, so the default identity is the
    # natural element (ENDF A = 0) and the constants come from that same
    # entry: identity and physics can never disagree.
    assert (al.symbol, al.Z, al.A) == ("Al", 13, 0)
    assert al.za == 13000
    assert al.identity_source == "natural"
    assert al.sigma_free_b < al.sigma_bound_b

    over = resolve_species(
        al_bundle, overrides={"Al": {"sigma_inc_b": 0.5}}, progress=QUIET)
    assert over[0].sigma_inc_b == 0.5
    assert over[0].constants_source == "override"

    with pytest.raises(ValueError, match="same element"):
        resolve_species(al_bundle, nuclides={"Al": "13-C"}, progress=QUIET)


def test_energy_dependent_prefill_is_refused(al_bundle, tmp_path):
    # natural B is flagged energy-dependent in the nuclear table
    from ase import Atoms
    from ase.io import write as ase_write
    poscar = tmp_path / "structure_relaxed.vasp"
    ase_write(str(poscar), Atoms("B", cell=[4, 4, 4], pbc=True,
                                 scaled_positions=[[0, 0, 0]]),
              direct=True, format="vasp")
    fake = type(al_bundle)(path=str(tmp_path), phonopy_yaml="",
                           structure=str(poscar), manifest={})
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        resolve_species(fake, progress=QUIET)
    # an awr-only override does not unlock the flagged constants...
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        resolve_species(fake, overrides={"B": {"awr": 100.0}}, progress=QUIET)
    # ...explicit scattering constants do
    out = resolve_species(fake, overrides={"B": {"b_coh_fm": 5.0,
                                                 "sigma_inc_b": 1.0}},
                          progress=QUIET)
    assert out[0].constants_source == "override"


def test_iel10_deck_monatomic(al_bundle, tmp_path):
    paths = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                            out_dir=str(tmp_path), progress=QUIET)
    (path,) = paths
    st = _parse(path)
    assert st["mat"] == 45 and int(st["za"]) == 13000
    assert st["iel"] == 10 and st["inelastic_mode"] == 2 and st["iint"] == 1
    assert st["elastic_mode"] == 2                 # MEF is the default
    assert st["temperatures"] == [296.0]
    assert len(st["atoms"]) == 1 and st["atoms"][0]["npos"] == 4
    assert st["nalpha"] == len(st["alpha"])
    assert st["noncubic"]["yaml"] == al_bundle.phonopy_yaml
    assert st["noncubic"]["ndir"] == 10000
    man = json.load(open(os.path.join(str(tmp_path), "emit_manifest.json")))
    assert man["elastic_format"] == "mef" and man["inelastic_mode"] == 2


def test_iel10_polyatomic_all_groups_on_every_deck(cuau_bundle, tmp_path):
    paths = emit_endf_decks(cuau_bundle, temperature_k=296.0,
                            mats={"Cu": 100, "Au": 200},
                            out_dir=str(tmp_path), progress=QUIET)
    assert len(paths) == 2
    zas = set()
    for path in paths:
        st = _parse(path)
        assert len(st["atoms"]) == 2          # ALL 6d groups on every deck
        pairs = {(a["Z"], a["A"]) for a in st["atoms"]}
        assert pairs == {(29, 0), (79, 0)}     # natural Cu, natural Au
        zas.add(int(st["za"]))
    assert zas == {29000, 79000}


def test_mat_is_required_and_validated(al_bundle):
    with pytest.raises(ValueError, match="MAT numbers are required"):
        emit_endf_decks(al_bundle, temperature_k=296.0, mats={},
                        progress=QUIET)
    with pytest.raises(ValueError, match="integer >= 1"):
        emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 0},
                        progress=QUIET)


def test_disordered_classic_deck(dis_bundle, tmp_path):
    paths = emit_endf_decks(dis_bundle, temperature_k=296.0,
                            mats={"Al": 45}, out_dir=str(tmp_path),
                            progress=QUIET)
    (path,) = paths
    st = _parse(path)
    assert st["iel"] == 0 and st["iint"] == 0
    assert st["ncold"] == 0 and st["nsk"] == 0
    assert st["mat"] == 45 and int(st["za"]) == 13000
    # continuous spectrum present, no translational, no discrete
    text = open(path).read()
    assert "0.0 0.0 1.0 /" in text            # twt=0 -> incoherent elastic
    # Card 5 spr carries the free-atom equivalent of the TOTAL bound XS
    from irma.core.nuclear_data import lookup
    al = lookup("Al")
    expected_free = al.sigma_bound_b * (al.awr / (al.awr + 1)) ** 2
    assert st["spr"] == pytest.approx(expected_free, rel=1e-6)


def test_spectra_yaml_crystal_loads(al_bundle, tmp_path):
    out = emit_spectra_yaml(al_bundle, temperature_k=296.0,
                            out_path=str(tmp_path / "spectra.yaml"),
                            progress=QUIET)
    from irma.spectra.config import SpectraConfig
    cfg = SpectraConfig.from_dict(yaml.safe_load(open(out)))
    assert cfg.material.scatterers[0].symbol == "Al"
    assert cfg.physics.inelastic_mode == 2


def test_spectra_yaml_polyatomic_keeps_resolved_species_order(cuau_bundle,
                                                              tmp_path):
    """The emitted scatterer list is the bundle's resolved species order.

    The spectra path has no principal scatterer (unlike the ENDF and
    NCrystal targets, which write one deck/pack per principal): the forward
    model evaluates every scatterer in one pass, so the emitter must not
    reorder the list.
    """
    species = resolve_species(cuau_bundle, progress=QUIET)
    out = emit_spectra_yaml(cuau_bundle, temperature_k=296.0,
                            out_path=str(tmp_path / "spectra.yaml"),
                            progress=QUIET)
    from irma.spectra.config import SpectraConfig
    cfg = SpectraConfig.from_dict(yaml.safe_load(open(out)))
    assert [s.symbol for s in cfg.material.scatterers] == \
        [s.symbol for s in species]


def test_spectra_yaml_disordered_mode0(dis_bundle, tmp_path):
    out = emit_spectra_yaml(dis_bundle, temperature_k=296.0,
                            out_path=str(tmp_path / "spectra.yaml"),
                            progress=QUIET)
    from irma.spectra.config import SpectraConfig
    cfg = SpectraConfig.from_dict(yaml.safe_load(open(out)))
    assert cfg.physics.inelastic_mode == 0
    sc = cfg.material.scatterers[0]
    # incoherent-total elastic convention
    assert sc.sigma_inc_b == pytest.approx(sc.sigma_bound_b)
    assert os.path.isfile(str(tmp_path / "dos_Al.dat"))


def test_ncrystal_yaml_crystal_loads_and_disordered_refused(al_bundle,
                                                            dis_bundle,
                                                            tmp_path):
    out = emit_ncrystal_yaml(al_bundle, temperature_k=296.0,
                             material_id="al_test",
                             out_path=str(tmp_path / "nc.yaml"),
                             progress=QUIET)
    from irma.ncrystal.config import NCrystalExportConfig
    cfg = NCrystalExportConfig.from_yaml(out)
    assert cfg.material_id == "al_test"
    assert cfg.num_directions == 10000
    with pytest.raises(ValueError, match="disordered"):
        emit_ncrystal_yaml(dis_bundle, temperature_k=296.0,
                           out_path=str(tmp_path / "nc2.yaml"),
                           progress=QUIET)


def test_emit_manifest_records_provenance(al_bundle, tmp_path):
    emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                    out_dir=str(tmp_path), progress=QUIET)
    rec = json.load(open(os.path.join(str(tmp_path), "emit_manifest.json")))
    (sp,) = rec["species"]
    assert sp["za"] == 13000 and sp["mat"] == 45
    assert sp["identity_source"] == "natural"
    assert rec["fingerprint"] == al_bundle.fingerprint


def test_adversarial_overrides_are_rejected(al_bundle):
    with pytest.raises(ValueError, match="unknown field"):
        resolve_species(al_bundle, overrides={"Al": {"sigma_lnc_b": 1.0}},
                        progress=QUIET)
    with pytest.raises(ValueError, match="not present in the structure"):
        resolve_species(al_bundle, overrides={"Cu": {"awr": 63.0}},
                        progress=QUIET)
    with pytest.raises(ValueError, match="not present in the structure"):
        resolve_species(al_bundle, nuclides={"Cu": "65-Cu"}, progress=QUIET)
    with pytest.raises(ValueError, match="inconsistent"):
        resolve_species(al_bundle,
                        overrides={"Al": {"sigma_bound_b": 99.0}},
                        progress=QUIET)
    # a CONSISTENT bound value is accepted as a cross-check
    from irma.core.nuclear_data import lookup
    al = lookup("Al")
    ok = resolve_species(
        al_bundle, overrides={"Al": {"sigma_bound_b": al.sigma_bound_b}},
        progress=QUIET)
    assert ok[0].sigma_bound_b == pytest.approx(al.sigma_bound_b, rel=1e-6)


def test_explicit_isotope_supplies_its_own_constants(cuau_bundle):
    from irma.core.nuclear_data import lookup
    species = resolve_species(cuau_bundle, nuclides={"Cu": "65-Cu"},
                              progress=QUIET)
    cu = next(s for s in species if s.symbol == "Cu")
    iso = lookup("65-Cu")
    assert (cu.Z, cu.A) == (29, 65)
    assert cu.b_coh_fm == pytest.approx(iso.b_coh_fm)
    assert cu.sigma_inc_b == pytest.approx(iso.sigma_inc_b)
    assert cu.awr == pytest.approx(iso.awr)


def test_unstable_model_is_refused_for_dos_paths(dis_bundle, tmp_path):
    import copy
    from irma.mlip.bundle import Bundle
    doctored = Bundle(path=dis_bundle.path,
                      phonopy_yaml=dis_bundle.phonopy_yaml,
                      structure=dis_bundle.structure,
                      manifest=copy.deepcopy(dis_bundle.manifest))
    doctored.manifest["phonons"]["n_imaginary"] = 5
    with pytest.raises(ValueError, match="imaginary"):
        emit_endf_decks(doctored, temperature_k=296.0, mats={"Al": 45},
                        out_dir=str(tmp_path), progress=QUIET)
    paths = emit_endf_decks(doctored, temperature_k=296.0, mats={"Al": 45},
                            out_dir=str(tmp_path), allow_unstable=True,
                            progress=QUIET)
    assert len(paths) == 1


def test_emit_overwrite_guard(al_bundle, tmp_path):
    kw = dict(temperature_k=296.0, mats={"Al": 45}, out_dir=str(tmp_path),
              progress=QUIET)
    (path,) = emit_endf_decks(al_bundle, **kw)
    original = open(path).read()

    with pytest.raises(FileExistsError, match="overwrite"):
        emit_endf_decks(al_bundle, **kw)
    assert open(path).read() == original


def test_born_bundle_emits_use_born_zero_with_embedded_nac(tmp_path, al_model,
                                                            born_file):
    born = born_file(tmp_path, al_model[1].phonon)
    b = _bundle(tmp_path / "bundle", al_model, born_path=born)
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac
    assert phonopy_yaml_embeds_nac(b.phonopy_yaml)
    (path,) = emit_endf_decks(b, temperature_k=296.0, mats={"Al": 45},
                              out_dir=str(tmp_path / "out"), progress=QUIET)
    st = _parse(path)
    assert st["noncubic"]["use_born"] == 0     # NAC rides embedded, not BORN


def test_emitted_iel10_deck_runs_through_the_engine(al_bundle, tmp_path):
    (deck,) = emit_endf_decks(al_bundle, temperature_k=296.0,
                              mats={"Al": 45}, out_dir=str(tmp_path),
                              _preview=True, progress=QUIET)
    from irma.core.driver import run_leapr
    out = tmp_path / "al_mlip.endf"
    result = run_leapr(deck, str(out))
    assert result.iel == 10
    assert out.is_file() and out.stat().st_size > 10000


def test_emitted_classic_deck_runs_and_carries_bound_total_sb(dis_bundle,
                                                              tmp_path):
    (deck,) = emit_endf_decks(dis_bundle, temperature_k=296.0,
                              mats={"Al": 45}, out_dir=str(tmp_path),
                              progress=QUIET)
    from irma.core.driver import run_leapr
    out = tmp_path / "al_classic.endf"
    result = run_leapr(deck, str(out))
    assert result.iel == -1                    # incoherent-elastic route
    assert out.is_file()
    from endf_parserpy import EndfParserPy
    mt2 = EndfParserPy(ignore_number_mismatch=True, ignore_zero_mismatch=True,
                       ignore_varspec_mismatch=True).parsefile(
        str(out), include=[(7, 2)])[7][2]
    from irma.core.nuclear_data import lookup
    assert mt2["LTHR"] == 2
    assert mt2["SB"] == pytest.approx(lookup("Al").sigma_bound_b, rel=1e-4)


def test_species_dos_survives_gamma_only_mesh(tmp_path, al_model):
    """The disordered path defaults to a Gamma-only mesh, where every band
    is flat; the histogram DOS must still count every mode."""
    from irma.mlip.bundle import load_bundle
    from irma.mlip.emit import _species_dos

    _bundle(tmp_path, al_model, disordered=True, mesh=(1, 1, 1))
    bundle = load_bundle(str(tmp_path))
    species = resolve_species(bundle, progress=QUIET)
    dos = _species_dos(bundle, species, QUIET)
    for sym, (e, rho) in dos.items():
        assert rho.sum() > 0.0, f"empty Gamma-only species DOS for {sym}"


def test_sef_option_and_mode1_deck(al_bundle, tmp_path):
    (deck,) = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                              out_dir=str(tmp_path), inelastic_mode=1,
                              elastic_format="sef", _preview=True,
                              progress=QUIET)
    st = _parse(deck)
    assert st["elastic_mode"] == 1
    assert st["inelastic_mode"] == 1
    assert st["iint"] == 0                 # incoherent-approx: log-lin
    assert st["noncubic"]["yaml"] == al_bundle.phonopy_yaml
    man = json.load(open(tmp_path / "emit_manifest.json"))
    assert man["elastic_format"] == "sef" and man["inelastic_mode"] == 1


def test_mode0_polyatomic_deck_carries_partial_spectra_and_runs(
        cuau_bundle, tmp_path, capsys):
    decks = emit_endf_decks(cuau_bundle, temperature_k=296.0,
                            mats={"Cu": 100, "Au": 200},
                            out_dir=str(tmp_path), inelastic_mode=0,
                            allow_unstable=True,   # EMT CuAu is unstable
                            _preview=True, progress=QUIET)
    (cu,) = [d for d in decks if d.endswith("endf_Cu.input")]
    st = _parse(cu)
    assert st["inelastic_mode"] == 0
    assert st["elastic_mode"] == 2
    assert st["iint"] == 0
    from irma.core.driver import run_leapr
    out = tmp_path / "cu_mode0.endf"
    result = run_leapr(cu, str(out))
    captured = capsys.readouterr()
    # the Au species carries its own Card 6e spectrum: no inherited-lambda
    # fallback, hence no warning
    assert "Partial spectrum" in captured.out
    assert "INHERITED" not in captured.out
    assert result.iel == 10
    assert out.is_file() and out.stat().st_size > 10000


def test_disordered_rejects_crystal_options(dis_bundle, tmp_path):
    for kw in ({"inelastic_mode": 0}, {"elastic_format": "sef"}):
        with pytest.raises(ValueError, match="not applicable"):
            emit_endf_decks(dis_bundle, temperature_k=296.0, mats={"Al": 45},
                            out_dir=str(tmp_path), progress=QUIET, **kw)
    assert os.listdir(tmp_path) == []     # refused before anything landed


# ------------------------------------ provenance, paths, emitted mesh ----


def _check_comment_cards(bundle, deck_path, extra=""):
    """The writer maps comment card 1 onto the 66-column structured
    MF1/MT451 header and cards 6+ onto free-text DESCRIPTION records; the
    provenance must survive that mapping untruncated."""
    st = _parse(deck_path)
    comments = st["comments"]
    assert len(comments) >= 6
    c1 = comments[0]
    assert len(c1.rstrip()) <= 66              # nothing past the mapped cols
    assert c1.startswith(" ")                  # column-1 blank -> ZSYMAM
    assert c1[22:32].startswith("EVAL-")       # EDATE columns carry a date
    assert bundle.path not in c1               # provenance is NOT on card 1
    assert bundle.fingerprint[:16] not in c1
    assert all(not c.strip() for c in comments[1:5])   # REF/HSUB slots
    desc = comments[5:]
    assert all(len(c.rstrip()) <= 66 for c in desc)    # no [:66] loss
    from irma.mlip.emit import GENERATED_NOTE
    prov = (f"{GENERATED_NOTE}; bundle {bundle.path}; "
            f"fingerprint {bundle.fingerprint[:16]}{extra}")
    squash = lambda s: "".join(s.split())              # noqa: E731
    assert squash(" ".join(desc)) == squash(prov)


def test_deck_provenance_survives_mf1_column_mapping(al_bundle, dis_bundle,
                                                     tmp_path):
    (deck,) = emit_endf_decks(al_bundle, temperature_k=296.0,
                              mats={"Al": 45},
                              out_dir=str(tmp_path / "xt"),
                              _preview=True, progress=QUIET)
    _check_comment_cards(al_bundle, deck)
    (classic,) = emit_endf_decks(dis_bundle, temperature_k=296.0,
                                 mats={"Al": 45},
                                 out_dir=str(tmp_path / "dis"),
                                 progress=QUIET)
    _check_comment_cards(dis_bundle, classic,
                         extra="; disordered classic path")


def test_disordered_spectra_dos_paths_are_absolute(dis_bundle, tmp_path,
                                                   monkeypatch):
    """A relative --out-dir must still yield a YAML that resolves
    from any working directory (matching the crystalline branch's
    abspathed phonopy_yaml)."""
    monkeypatch.chdir(tmp_path)
    out = emit_spectra_yaml(dis_bundle, temperature_k=296.0,
                            out_path=os.path.join("rel_out", "spectra.yaml"),
                            progress=QUIET)
    (sc,) = yaml.safe_load(open(out))["material"]["scatterers"]
    assert os.path.isabs(sc["dos_file"])
    assert os.path.isfile(sc["dos_file"])
    monkeypatch.chdir(tmp_path / "rel_out")    # a different cwd still works
    assert os.path.isfile(sc["dos_file"])


def test_emitted_mesh_is_production_density(al_bundle):
    """The mesh prefilled into emitted inputs is the campaign-density
    rule (int(98/a)+1 per axis, floor 8), not the bundle's quick-look
    mesh: prefill quality is the emitted files' contract."""
    from irma.mlip.emit import _emit_mesh
    mesh = _emit_mesh(al_bundle)
    assert mesh == [25, 25, 25]              # a = 4.05 A: int(98/4.05)+1
    assert mesh != list(al_bundle.manifest["phonons"]["mesh"])


def test_emitted_mesh_disordered_keeps_bundle_mesh(dis_bundle):
    """Disordered bundles keep their own (Gamma-quality) mesh: the
    classic DOS path never sums over q, so the production rule must not
    apply."""
    from irma.mlip.emit import _emit_mesh
    assert _emit_mesh(dis_bundle) == [4, 4, 4]      # the fixture's bundle mesh


def test_min_phonon_energy_reaches_every_emitted_file(al_bundle, tmp_path):
    """One option, three files: the deck carries the optional one-value card
    before Card 6g, and both YAML configurations carry the same key."""
    import yaml
    from pathlib import Path
    (path,) = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                              out_dir=str(tmp_path), min_phonon_energy_mev=0.5,
                              progress=QUIET)
    st = _parse(path)
    assert st["noncubic"]["min_phonon_energy_mev"] == 0.5
    lines = Path(path).read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip() == "0.5 /")
    assert lines[i + 1].split()[:2] == ["10000", "1000"]        # Card 6g follows
    # the default writes no card at all
    (plain,) = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                               out_dir=str(tmp_path / "plain"), progress=QUIET)
    assert _parse(plain)["noncubic"]["min_phonon_energy_mev"] == 0.0
    assert not any(line.strip() == "0.5 /" for line in Path(plain).read_text().splitlines())
    spectra = emit_spectra_yaml(al_bundle, temperature_k=296.0,
                                out_path=str(tmp_path / "spectra.yaml"),
                                min_phonon_energy_mev=0.5, progress=QUIET)
    assert yaml.safe_load(Path(spectra).read_text())["physics"]["min_phonon_energy_meV"] == 0.5
    ncrystal = emit_ncrystal_yaml(al_bundle, temperature_k=296.0,
                                  out_path=str(tmp_path / "ncrystal.yaml"),
                                  min_phonon_energy_mev=0.5, progress=QUIET)
    assert yaml.safe_load(Path(ncrystal).read_text())["export"]["min_phonon_energy_meV"] == 0.5
