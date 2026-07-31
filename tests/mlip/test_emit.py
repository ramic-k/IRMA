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


def _make_bundle(tmpdir, atoms, disordered=False, mesh=(4, 4, 4)):
    rr = relax(atoms, EMT(), fmax=0.01, nmax=200)
    pr = compute_force_constants(
        rr.atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=1,
        scratch_dir=os.path.join(str(tmpdir), "scratch"), progress=QUIET)
    return write_bundle(str(tmpdir), phonon_result=pr,
                        relax_result=rr, calc_meta={"potential": "emt"},
                        args_used={}, mesh=mesh, disordered=disordered,
                        progress=QUIET)


@pytest.fixture(scope="module")
def al_bundle(tmp_path_factory):
    return _make_bundle(tmp_path_factory.mktemp("al"),
                        bulk("Al", "fcc", a=4.05, cubic=True))


@pytest.fixture(scope="module")
def cuau_bundle(tmp_path_factory):
    atoms = bulk("CuAu", "rocksalt", a=4.1)
    return _make_bundle(tmp_path_factory.mktemp("cuau"), atoms)


@pytest.fixture(scope="module")
def dis_bundle(tmp_path_factory):
    return _make_bundle(tmp_path_factory.mktemp("dis"),
                        bulk("Al", "fcc", a=4.05, cubic=True),
                        disordered=True)


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
    assert (al.symbol, al.Z, al.A) == ("Al", 13, 27)   # most-abundant default
    assert al.za == 13027
    assert al.identity_source == "most-abundant default"
    assert al.sigma_free_b < al.sigma_bound_b

    over = resolve_species(
        al_bundle, overrides={"Al": {"sigma_inc_b": 0.5}}, progress=QUIET)
    assert over[0].sigma_inc_b == 0.5
    assert over[0].constants_source == "override"

    with pytest.raises(ValueError, match="same element"):
        resolve_species(al_bundle, nuclides={"Al": "13-C"}, progress=QUIET)


def test_energy_dependent_prefill_is_refused(al_bundle, tmp_path):
    from irma.core.nuclear_data import NUCLIDES
    flagged = [n for (z, a), n in NUCLIDES.items()
               if a == 0 and n.energy_dependent]
    if not flagged:
        pytest.skip("no natural energy-dependent entries in the table")
    sym = flagged[0].symbol
    from ase import Atoms
    from ase.io import write as ase_write
    poscar = tmp_path / "structure_relaxed.vasp"
    ase_write(str(poscar), Atoms(sym, cell=[4, 4, 4], pbc=True,
                                 scaled_positions=[[0, 0, 0]]),
              direct=True, format="vasp")
    fake = type(al_bundle)(path=str(tmp_path), phonopy_yaml="",
                           structure=str(poscar), manifest={})
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        resolve_species(fake, progress=QUIET)
    # explicit overrides unlock it
    out = resolve_species(fake, overrides={sym: {"b_coh_fm": 5.0,
                                                 "sigma_inc_b": 1.0}},
                          progress=QUIET)
    assert out[0].constants_source == "override"


def test_iel10_deck_monatomic(al_bundle, tmp_path):
    paths = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                            out_dir=str(tmp_path), progress=QUIET)
    (path,) = paths
    st = _parse(path)
    assert st["mat"] == 45 and int(st["za"]) == 13027
    assert st["iel"] == 10 and st["inelastic_mode"] == 2 and st["iint"] == 1
    assert st["temperatures"] == [296.0]
    assert len(st["atoms"]) == 1 and st["atoms"][0]["npos"] == 4
    assert st["nalpha"] == len(st["alpha"])
    assert st["noncubic"]["yaml"] == al_bundle.phonopy_yaml
    assert st["noncubic"]["ndir"] == 10000
    assert os.path.isfile(os.path.join(str(tmp_path), "emit_manifest.json"))


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
        assert pairs == {(29, 63), (79, 197)}  # Cu-63, Au-197 defaults
        zas.add(int(st["za"]))
    assert zas == {29063, 79197}


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
    assert st["mat"] == 45 and int(st["za"]) == 13027
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


def test_spectra_yaml_polyatomic_principal_first(cuau_bundle, tmp_path):
    out = emit_spectra_yaml(cuau_bundle, temperature_k=296.0,
                            principal="Au",
                            out_path=str(tmp_path / "spectra.yaml"),
                            progress=QUIET)
    from irma.spectra.config import SpectraConfig
    cfg = SpectraConfig.from_dict(yaml.safe_load(open(out)))
    assert cfg.material.scatterers[0].symbol == "Au"
    with pytest.raises(ValueError, match="principal"):
        emit_spectra_yaml(cuau_bundle, temperature_k=296.0,
                          principal="Xx",
                          out_path=str(tmp_path / "s2.yaml"), progress=QUIET)


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
    assert sp["za"] == 13027 and sp["mat"] == 45
    assert sp["identity_source"] == "most-abundant default"
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


def test_energy_dependent_needs_scattering_constants_not_any_override(
        al_bundle, tmp_path):
    from irma.core.nuclear_data import NUCLIDES
    flagged = [n for (z, a), n in NUCLIDES.items()
               if a == 0 and n.energy_dependent]
    if not flagged:
        pytest.skip("no natural energy-dependent entries in the table")
    sym = flagged[0].symbol
    from ase import Atoms
    from ase.io import write as ase_write
    poscar = tmp_path / "structure_relaxed.vasp"
    ase_write(str(poscar), Atoms(sym, cell=[4, 4, 4], pbc=True,
                                 scaled_positions=[[0, 0, 0]]),
              direct=True, format="vasp")
    fake = type(al_bundle)(path=str(tmp_path), phonopy_yaml="",
                           structure=str(poscar), manifest={})
    # an awr-only override must NOT unlock flagged constants
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        resolve_species(fake, overrides={sym: {"awr": 100.0}},
                        progress=QUIET)


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


def test_emit_overwrite_guard_and_failure_preservation(al_bundle, tmp_path,
                                                       monkeypatch):
    kw = dict(temperature_k=296.0, mats={"Al": 45}, out_dir=str(tmp_path),
              progress=QUIET)
    (path,) = emit_endf_decks(al_bundle, **kw)
    original = open(path).read()

    with pytest.raises(FileExistsError, match="overwrite"):
        emit_endf_decks(al_bundle, **kw)
    assert open(path).read() == original

    # a failed gate must preserve the existing file and leave no .tmp
    import irma.mlip.emit as emit_mod
    monkeypatch.setattr(emit_mod, "validate_deck_semantics",
                        lambda staged: ["injected failure"])
    with pytest.raises(RuntimeError, match="injected failure"):
        emit_endf_decks(al_bundle, overwrite=True, **kw)
    assert open(path).read() == original
    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".tmp")]


def test_born_bundle_emits_use_born_zero_with_embedded_nac(tmp_path):
    from phonopy.structure.symmetry import Symmetry
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    rr = relax(atoms, EMT(), fmax=0.01, nmax=100)
    pr = compute_force_constants(
        rr.atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=1,
        scratch_dir=str(tmp_path / "scratch"), progress=QUIET)
    n_indep = len(Symmetry(pr.phonon.primitive).get_independent_atoms())
    born = tmp_path / "BORN"
    born.write_text("14.4\n2.0 0 0  0 2.0 0  0 0 2.0\n"
                    + "\n".join(["1.5 0 0  0 1.5 0  0 0 1.5"] * n_indep)
                    + "\n")
    b = write_bundle(str(tmp_path / "bundle"), phonon_result=pr,
                     relax_result=rr, calc_meta={"potential": "emt"},
                     args_used={}, mesh=(4, 4, 4), born_path=str(born),
                     progress=QUIET)
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac
    assert phonopy_yaml_embeds_nac(b.phonopy_yaml)
    (path,) = emit_endf_decks(b, temperature_k=296.0, mats={"Al": 45},
                              out_dir=str(tmp_path / "out"), progress=QUIET)
    st = _parse(path)
    assert st["noncubic"]["use_born"] == 0     # NAC rides embedded, not BORN


def test_emitted_iel10_deck_RUNS_through_the_engine(al_bundle, tmp_path):
    (deck,) = emit_endf_decks(al_bundle, temperature_k=296.0,
                              mats={"Al": 45}, out_dir=str(tmp_path),
                              _preview=True, progress=QUIET)
    from irma.core.driver import run_leapr
    out = tmp_path / "al_mlip.endf"
    result = run_leapr(deck, str(out))
    assert result.iel == 10
    assert out.is_file() and out.stat().st_size > 10000


def test_emitted_classic_deck_RUNS_and_carries_bound_total_sb(dis_bundle,
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


def test_species_dos_survives_gamma_only_mesh(tmp_path):
    """The disordered path defaults to a Gamma-only mesh,
    where EVERY band has zero tetrahedron width -- _species_dos must
    fall back to smearing instead of silently emitting an empty DOS."""
    from irma.mlip.bundle import load_bundle
    from irma.mlip.emit import _species_dos

    _make_bundle(tmp_path, bulk("Al", "fcc", a=4.05, cubic=True),
                 disordered=True, mesh=(1, 1, 1))
    bundle = load_bundle(str(tmp_path))
    species = resolve_species(bundle, progress=QUIET)
    dos = _species_dos(bundle, species, QUIET)
    for sym, (e, rho) in dos.items():
        assert rho.sum() > 0.0, f"empty Gamma-only species DOS for {sym}"


def test_emitted_deck_defaults_to_mef(al_bundle, tmp_path):
    (deck,) = emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                              out_dir=str(tmp_path), _preview=True,
                              progress=QUIET)
    st = _parse(deck)
    assert st["elastic_mode"] == 2 and st["inelastic_mode"] == 2
    manifest = json.load(open(os.path.join(str(tmp_path),
                                           "emit_manifest.json")))
    assert manifest["elastic_format"] == "mef"
    assert manifest["inelastic_mode"] == 2


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


def test_mode0_polyatomic_deck_carries_partial_spectra_and_RUNS(
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


def test_mode2_polyatomic_run_suppresses_inherited_warning(
        cuau_bundle, tmp_path, capsys):
    decks = emit_endf_decks(cuau_bundle, temperature_k=296.0,
                            mats={"Cu": 100, "Au": 200},
                            out_dir=str(tmp_path), _preview=True,
                            progress=QUIET)
    (cu,) = [d for d in decks if d.endswith("endf_Cu.input")]
    from irma.core.driver import run_leapr
    run_leapr(cu, str(tmp_path / "cu_mode2.endf"))
    captured = capsys.readouterr()
    # modes 1/2 take the directional Debye-Waller branch: the classic
    # per-species lambdas are unused and the fallback warning is suppressed
    assert "INHERITED" not in captured.out
    assert "displacement tensors" in captured.out


def test_disordered_rejects_inelastic_mode(dis_bundle, tmp_path):
    with pytest.raises(ValueError, match="not applicable"):
        emit_endf_decks(dis_bundle, temperature_k=296.0, mats={"Al": 45},
                        out_dir=str(tmp_path), inelastic_mode=0,
                        progress=QUIET)


def test_bad_elastic_format_rejected(al_bundle, tmp_path):
    with pytest.raises(ValueError, match="elastic_format"):
        emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                        out_dir=str(tmp_path), elastic_format="cef",
                        progress=QUIET)


# ------------------------------------------------- QA remediation tests ----


def _check_comment_cards(bundle, deck_path, extra=""):
    """MLP-1: the writer maps comment card 1 onto the 66-column structured
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


def test_species_dos_honours_recorded_dos_smearing(dis_bundle, monkeypatch):
    """MLP-3: the --dos-smearing width the bundle records must reach the
    projected-DOS call feeding Card 11/12, Card 6e, and dos_*.dat."""
    import copy
    from irma.mlip import bundle as bundle_mod
    from irma.mlip.bundle import Bundle
    from irma.mlip.emit import _species_dos

    recorded = []
    real = bundle_mod.dos_grid_and_fallback

    def wrapper(phonon, method, dos_sigma_mev=None, **kw):
        recorded.append(dos_sigma_mev)
        return real(phonon, method, dos_sigma_mev=dos_sigma_mev, **kw)

    monkeypatch.setattr(bundle_mod, "dos_grid_and_fallback", wrapper)

    doctored = Bundle(path=dis_bundle.path,
                      phonopy_yaml=dis_bundle.phonopy_yaml,
                      structure=dis_bundle.structure,
                      manifest=copy.deepcopy(dis_bundle.manifest))
    doctored.manifest["input"]["args"]["dos_smearing"] = 4.0
    species = resolve_species(doctored, progress=QUIET)
    msgs = []
    dos = _species_dos(doctored, species, msgs.append)
    assert recorded[-1] == 4.0
    assert any("4" in m and "smearing" in m for m in msgs)
    for sym, (e, rho) in dos.items():
        assert rho.sum() > 0.0

    # near-miss: without a recorded width the default path is unchanged
    recorded.clear()
    _species_dos(dis_bundle, species, QUIET)
    assert recorded[-1] is None


def test_default_isotope_za_warns_constants_are_natural(al_bundle,
                                                        cuau_bundle):
    """MLP-4: a defaulted isotope ZA on natural-element constants must be
    flagged, with the override spelled out."""
    msgs = []
    resolve_species(al_bundle, progress=msgs.append)
    joined = "\n".join(msgs)
    assert "WARNING" in joined
    assert "NATURAL-abundance" in joined
    assert "--nuclide Al=27-Al" in joined
    assert "--species" in joined

    # near-miss: an explicit nuclide identity carries its own constants --
    # no mismatch, no warning for that species (Au still defaults + warns)
    msgs = []
    resolve_species(cuau_bundle, nuclides={"Cu": "65-Cu"},
                    progress=msgs.append)
    warnings_ = [m for m in msgs if "WARNING" in m]
    assert not any("Cu" in w for w in warnings_)
    assert any("Au" in w for w in warnings_)

    # near-miss: fully overridden constants leave nothing natural to flag
    msgs = []
    resolve_species(al_bundle,
                    overrides={"Al": {"awr": 26.98, "b_coh_fm": 3.449,
                                      "sigma_inc_b": 0.0082}},
                    progress=msgs.append)
    assert "WARNING" not in "\n".join(msgs)


def test_disordered_rejects_sef_and_manifest_records_choices(
        dis_bundle, al_bundle, tmp_path):
    """MLP-5/DOC-9: --elastic-format is rejected on the classic path the
    same way --inelastic-mode is, and the emit manifest records the
    effective choices where they apply."""
    out = tmp_path / "sef_dis"
    with pytest.raises(ValueError, match="not applicable"):
        emit_endf_decks(dis_bundle, temperature_k=296.0, mats={"Al": 45},
                        out_dir=str(out), elastic_format="sef",
                        progress=QUIET)
    assert os.listdir(out) == []          # rejected before anything landed

    # near-miss: the crystal path accepts sef and records the choice
    out2 = tmp_path / "sef_xt"
    emit_endf_decks(al_bundle, temperature_k=296.0, mats={"Al": 45},
                    out_dir=str(out2), inelastic_mode=1,
                    elastic_format="sef", _preview=True, progress=QUIET)
    man = json.load(open(out2 / "emit_manifest.json"))
    assert man["elastic_format"] == "sef"
    assert man["inelastic_mode"] == 1

    # near-miss: the disordered default (mef, the CLI argparse default)
    # still emits
    out3 = tmp_path / "mef_dis"
    paths = emit_endf_decks(dis_bundle, temperature_k=296.0,
                            mats={"Al": 45}, out_dir=str(out3),
                            progress=QUIET)
    assert len(paths) == 1


def test_disordered_spectra_dos_paths_are_absolute(dis_bundle, tmp_path,
                                                   monkeypatch):
    """MLP-8: a relative --out-dir must still yield a YAML that resolves
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


def test_late_deck_conflict_publishes_nothing(cuau_bundle, tmp_path):
    """CDX-1: a FileExistsError on the LAST target (second deck, or the
    manifest) must surface before the FIRST artifact is published."""
    species = resolve_species(cuau_bundle, progress=QUIET)
    first, last = species[0].symbol, species[-1].symbol
    kw = dict(temperature_k=296.0, mats={"Cu": 100, "Au": 200},
              _preview=True, progress=QUIET)

    out = tmp_path / "late_deck"
    out.mkdir()
    sentinel = out / f"endf_{last}.input"
    sentinel.write_text("sentinel\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        emit_endf_decks(cuau_bundle, out_dir=str(out), **kw)
    assert not (out / f"endf_{first}.input").exists()
    assert not (out / "emit_manifest.json").exists()
    assert sentinel.read_text() == "sentinel\n"

    out2 = tmp_path / "late_manifest"
    out2.mkdir()
    (out2 / "emit_manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        emit_endf_decks(cuau_bundle, out_dir=str(out2), **kw)
    assert not [f for f in os.listdir(out2) if f.startswith("endf_")]

    # near-miss: a legitimate --overwrite still replaces the full set
    paths = emit_endf_decks(cuau_bundle, out_dir=str(out2), overwrite=True,
                            **kw)
    assert len(paths) == 2
    assert (out2 / "emit_manifest.json").read_text() != "{}\n"


def test_disordered_spectra_conflict_publishes_no_dos(cuau_bundle, tmp_path):
    """CDX-1 (spectra): a conflict on the last dos_*.dat must publish
    neither the earlier dos files nor the YAML."""
    import copy
    from irma.mlip.bundle import Bundle
    doctored = Bundle(path=cuau_bundle.path,
                      phonopy_yaml=cuau_bundle.phonopy_yaml,
                      structure=cuau_bundle.structure,
                      manifest=copy.deepcopy(cuau_bundle.manifest))
    doctored.manifest["disordered"] = True
    species = resolve_species(doctored, progress=QUIET)
    first, last = species[0].symbol, species[-1].symbol
    out = tmp_path / "spectra_conflict"
    out.mkdir()
    sentinel = out / f"dos_{last}.dat"
    sentinel.write_text("sentinel\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        emit_spectra_yaml(doctored, temperature_k=296.0,
                          out_path=str(out / "spectra.yaml"),
                          allow_unstable=True, progress=QUIET)
    assert not (out / f"dos_{first}.dat").exists()
    assert not (out / "spectra.yaml").exists()
    assert sentinel.read_text() == "sentinel\n"


def test_preflight_emit_targets_covers_every_target(al_bundle, dis_bundle,
                                                    tmp_path):
    """CDX-1 (cross-target helper for the CLI): every path of a --to
    invocation is guarded up front, and nothing is written."""
    from irma.mlip.emit import preflight_emit_targets
    out = tmp_path / "pf"
    out.mkdir()
    paths = preflight_emit_targets(al_bundle,
                                   ["endf", "spectra", "ncrystal"],
                                   out_dir=str(out))
    assert {os.path.basename(p) for p in paths} == {
        "endf_Al.input", "emit_manifest.json", "spectra.yaml",
        "ncrystal.yaml"}
    assert os.listdir(out) == []               # preflight writes nothing

    (out / "ncrystal.yaml").write_text("stale\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        preflight_emit_targets(al_bundle, ["endf", "ncrystal"],
                               out_dir=str(out))
    assert preflight_emit_targets(al_bundle, ["endf", "ncrystal"],
                                  out_dir=str(out), overwrite=True)

    # disordered spectra plans the dos sidecars too
    dpaths = preflight_emit_targets(dis_bundle, ["spectra"],
                                    out_dir=str(out))
    assert "dos_Al.dat" in {os.path.basename(p) for p in dpaths}

    with pytest.raises(ValueError, match="unknown emit target"):
        preflight_emit_targets(al_bundle, ["endf", "bogus"],
                               out_dir=str(out))
