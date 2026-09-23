"""Bundle write/load/validate tests on the EMT dev backend.

Each test gets its own output directory; the shared module fixture only
provides the (immutable-by-contract) relax/FC results, and the no-leak test
pins that write_bundle cannot contaminate a later bundle through the shared
phonon object."""
import json
import os

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("phonopy")

from ase.build import bulk                          # noqa: E402
from ase.calculators.emt import EMT                 # noqa: E402

from irma.mlip.bundle import (                      # noqa: E402
    Bundle, load_bundle, validate_bundle, write_bundle)
from irma.mlip.calculators import CalculatorSpec    # noqa: E402
from irma.mlip.phonons import compute_force_constants  # noqa: E402
from irma.mlip.relax import relax                   # noqa: E402

SPEC = CalculatorSpec("emt")
QUIET = lambda *a, **k: None                        # noqa: E731


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    """One relaxed fcc-Al model + FC shared by all tests (read-only)."""
    scratch = tmp_path_factory.mktemp("model-scratch")
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    rr = relax(atoms, EMT(), fmax=0.01, nmax=100)
    pr = compute_force_constants(
        rr.atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=1,
        scratch_dir=str(scratch / "scratch"), progress=QUIET)
    return rr, pr


def _write(outdir, model, **kw):
    rr, pr = model
    kw.setdefault("mesh", (4, 4, 4))
    kw.setdefault("args_used", {"structure": "Al.test", "potential": "emt"})
    kw.setdefault("calc_meta", {"potential": "emt", "package": "ase"})
    kw.setdefault("progress", QUIET)
    return write_bundle(str(outdir), phonon_result=pr, relax_result=rr, **kw)


def _born_file(tmp_path, with_factor, phonon):
    """Synthetic BORN with exactly the symmetry-independent atom rows the
    parser demands for this model."""
    from phonopy.structure.symmetry import Symmetry
    n_indep = len(Symmetry(phonon.primitive).get_independent_atoms())
    # line 1 is ALWAYS the factor line; a non-numeric token means "use the
    # code default" (phonopy file_IO.get_born_parameters)
    lines = ["14.4" if with_factor else "default"]
    lines.append("2.0 0 0  0 2.0 0  0 0 2.0")       # dielectric
    for _ in range(n_indep):
        lines.append("1.5 0 0  0 1.5 0  0 0 1.5")
    path = tmp_path / "BORN"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_bundle_roundtrip_and_validation(model, tmp_path):
    b = _write(tmp_path, model)
    assert isinstance(b, Bundle)
    rr, pr = model

    for name in ("phonopy.yaml", "structure_relaxed.vasp", "manifest.json",
                 "dos.dat"):
        assert os.path.isfile(os.path.join(str(tmp_path), name)), name

    from irma.core.phonopy_io import resolve_force_constants_source
    assert resolve_force_constants_source(b.phonopy_yaml) == {}

    m = json.load(open(os.path.join(str(tmp_path), "manifest.json")))
    assert m["schema"] == 1 and m["kind"] == "irma-mlip-phonon-bundle"
    assert m["fingerprint"] == pr.fingerprint
    assert m["created_utc"]
    assert m["relaxation"]["converged"] is True
    assert m["relaxation"]["nmax"] == 100
    assert m["displacements"]["count"] == pr.n_displacements
    assert m["displacements"]["delta"] == 0.03
    assert m["displacements"]["supercell"] == [2, 2, 2]
    assert m["nac_embedded"] is False and m["born"]["path"] is None
    assert m["phonons"]["n_modes"] == 4 * 4 * 4 * 12   # full-mesh count
    assert m["phonons"]["n_modes_irreducible"] <= m["phonons"]["n_modes"]
    assert len(m["sha256"]["phonopy_yaml"]) == 64

    dos = np.loadtxt(os.path.join(str(tmp_path), "dos.dat"))
    assert dos.ndim == 2 and dos.shape[1] == 2
    assert (dos[:, 1] >= -1e-12).all()
    assert 20.0 < dos[:, 0].max() < 80.0

    b2 = load_bundle(str(tmp_path))
    assert b2.fingerprint == pr.fingerprint
    assert validate_bundle(str(tmp_path)) == []


def test_input_structure_hash_recorded(model, tmp_path):
    src = tmp_path / "input.cif"
    src.write_text("fake structure input\n")
    b = _write(tmp_path / "b", model, input_structure_path=str(src))
    assert b.manifest["input"]["structure_sha256"] is not None
    assert b.manifest["input"]["structure_path"].endswith("input.cif")


def test_overwrite_guard_covers_everything(model, tmp_path):
    _write(tmp_path, model)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write(tmp_path, model)

    # any non-scratch content triggers the guard, not just owned files
    other = tmp_path / "fresh"
    other.mkdir()
    (other / "users_notes.txt").write_text("keep me?")
    with pytest.raises(FileExistsError):
        _write(other, model)

    # overwrite replaces owned files, incl. a stale dos.png
    stale_png = tmp_path / "dos.png"
    stale_png.write_bytes(b"stale")
    b = _write(tmp_path, model, overwrite=True)
    assert validate_bundle(b.path) == []


def test_born_roundtrip_explicit_factor(model, tmp_path):
    rr, pr = model
    born = _born_file(tmp_path, with_factor=True, phonon=pr.phonon)
    b = _write(tmp_path / "b", model, born_path=born)
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac
    assert phonopy_yaml_embeds_nac(b.phonopy_yaml)
    assert b.manifest["nac_embedded"] is True
    assert b.manifest["born"]["sha256"] is not None
    assert validate_bundle(b.path) == []
    # the shared phonon object is NOT left carrying NAC (no-leak contract)
    assert pr.phonon.nac_params is None


def test_born_roundtrip_without_factor_uses_phonopy_default(model, tmp_path):
    rr, pr = model
    born = _born_file(tmp_path, with_factor=False, phonon=pr.phonon)
    b = _write(tmp_path / "b", model, born_path=born)
    assert b.manifest["nac_embedded"] is True
    import phonopy
    from irma.core.phonopy_io import pinned_primitive_matrix_kwargs
    ph = phonopy.load(b.phonopy_yaml, log_level=0,
                      **pinned_primitive_matrix_kwargs(b.phonopy_yaml))
    assert ph.nac_params is not None
    assert ph.nac_params["factor"] == pytest.approx(14.3996517, rel=1e-6)


def test_second_bundle_after_born_bundle_is_nac_free(model, tmp_path):
    rr, pr = model
    born = _born_file(tmp_path, with_factor=True, phonon=pr.phonon)
    _write(tmp_path / "with_nac", model, born_path=born)
    b2 = _write(tmp_path / "without", model)      # would raise before the fix
    assert b2.manifest["nac_embedded"] is False
    assert validate_bundle(b2.path) == []


def test_check_born_rows_fail_fast(model, tmp_path):
    from irma.mlip.bundle import check_born_rows
    rr, pr = model
    good = _born_file(tmp_path, with_factor=True, phonon=pr.phonon)
    assert check_born_rows(good, rr.atoms) is None

    # one row too few: the message names both counts and the way out
    lines = open(good).read().strip().splitlines()
    bad = tmp_path / "BORN_short"
    bad.write_text("\n".join(lines[:-1]) + "\n")
    problem = check_born_rows(bad, rr.atoms)
    assert problem is not None
    assert "symmetry-independent" in problem
    assert "--snap-symmetry" in problem

    assert check_born_rows(tmp_path / "missing", rr.atoms) is not None


def test_validate_ignores_a_stray_born_in_the_cwd(model, tmp_path,
                                                  monkeypatch):
    # found live in the ZrO2 campaign: an unrelated ./BORN file next to
    # where validate runs must not leak NAC into the reload check
    b = _write(tmp_path / "clean", model)
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    _born_file(workdir, with_factor=True, phonon=model[1].phonon)
    assert (workdir / "BORN").exists()
    monkeypatch.chdir(workdir)
    assert validate_bundle(b.path) == []


def test_validate_catches_tampering(model, tmp_path):
    b = _write(tmp_path, model)

    # content change -> sha mismatch
    with open(b.structure, "a") as fh:
        fh.write("# tampered\n")
    problems = validate_bundle(b.path)
    assert any("sha256 mismatch" in p for p in problems)

    # manifest claiming NAC the yaml lacks
    b2 = _write(tmp_path / "b2", model)
    mpath = os.path.join(b2.path, "manifest.json")
    m = json.load(open(mpath))
    m["nac_embedded"] = True
    json.dump(m, open(mpath, "w"))
    problems = validate_bundle(b2.path)
    assert any("nac_embedded" in p for p in problems)

    # missing file
    b3 = _write(tmp_path / "b3", model)
    os.remove(b3.structure)
    assert any("missing file" in p for p in validate_bundle(b3.path))


def test_manifest_path_traversal_is_rejected(model, tmp_path):
    b = _write(tmp_path, model)
    mpath = os.path.join(b.path, "manifest.json")
    m = json.load(open(mpath))
    m["files"]["structure"] = "../../../etc/passwd"
    json.dump(m, open(mpath, "w"))
    with pytest.raises(ValueError, match="unsafe"):
        load_bundle(b.path)
    assert validate_bundle(b.path)                # problem list, no crash


def test_load_rejects_non_bundles(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_bundle(str(tmp_path))
    (tmp_path / "manifest.json").write_text('{"kind": "something-else"}')
    with pytest.raises(ValueError, match="not an irma-mlip"):
        load_bundle(str(tmp_path))
    (tmp_path / "manifest.json").write_text('[1, 2, 3]')
    with pytest.raises(ValueError):
        load_bundle(str(tmp_path))
    (tmp_path / "manifest.json").write_text('{broken json')
    assert validate_bundle(str(tmp_path))         # problem list, no crash


def _flat_crystal(k_a=0.0, k_b=5.0):
    """2-atom cell on a 2x1x1 supercell: sublattice A is an x-chain with
    spring k_a (dispersive when k_a > 0, from 0 up to its band top);
    sublattice B is a per-atom diagonal spring k_b with NO coupling
    (every B band exactly flat). Lets tests dial pure-flat, mixed, and
    crossing spectra from one fixture."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    cell = PhonopyAtoms(symbols=["Na", "Cl"],
                        cell=np.eye(3) * 4.0,
                        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])
    ph = Phonopy(cell, supercell_matrix=np.diag([2, 1, 1]), log_level=0)
    # supercell atom order: A@l0, A@l1, B@l0, B@l1
    fc = np.zeros((4, 4, 3, 3))
    fc[0, 1] = fc[1, 0] = -k_a * np.eye(3)
    fc[0, 0] = fc[1, 1] = 2.0 * k_a * np.eye(3)
    fc[2, 2] = fc[3, 3] = k_b * np.eye(3)
    ph.force_constants = fc
    return ph


def _integral(e, rho):
    return (np.trapezoid(rho, e) if hasattr(np, "trapezoid")
            else np.trapz(rho, e))


def test_dos_flat_band_smearing_fallback():
    """Numerically dispersionless bands have zero linear-tetrahedron width
    and drop out of the DOS entirely (found live: the scawtite O-H stretch
    on a 1x1x1 supercell). All-flat spectrum -> spread guard fires."""
    from irma.mlip.bundle import _dos_and_census

    ph = _flat_crystal(k_a=0.0, k_b=5.0)        # every band flat
    e, rho, census = _dos_and_census(ph, [4, 4, 4])
    assert census.get("dos_smearing_fallback_mev") == 1.0
    assert _integral(e, rho) == pytest.approx(6, rel=0.05)
    assert np.all(np.diff(e) < 0.51)            # pitch never coarser than 0.5


def test_dos_mixed_flat_and_dispersive_triggers_fallback():
    """One flat sublattice inside an otherwise dispersive spectrum must
    still trigger the fallback (min over bands, not max/all)."""
    from irma.mlip.bundle import _dos_and_census

    # k_b large: flat B bands sit ABOVE the dispersive A range (no crossing)
    ph = _flat_crystal(k_a=2.0, k_b=40.0)
    e, rho, census = _dos_and_census(ph, [8, 2, 2])
    assert census.get("dos_smearing_fallback_mev") == 1.0
    assert _integral(e, rho) == pytest.approx(6, rel=0.05)


def test_dos_explicit_sigma_gets_a_resolving_grid():
    """--dos-smearing far below the pitch must refine the grid: a 0.05 meV
    sigma sampled every 0.5 meV integrates to almost anything."""
    from irma.mlip.bundle import _dos_and_census

    ph = _flat_crystal(k_a=0.0, k_b=5.0)
    e, rho, census = _dos_and_census(ph, [4, 4, 4], dos_sigma_mev=0.05)
    assert "dos_smearing_fallback_mev" not in census    # explicit sigma path
    assert np.all(np.diff(e) < 0.05 / 2 + 1e-9)
    assert _integral(e, rho) == pytest.approx(6, rel=0.05)


def test_dos_dispersive_bands_keep_tetrahedron(model):
    """A normal dispersive crystal must NOT trigger the smearing fallback
    (the tetrahedron DOS already integrates to the band count)."""
    from irma.mlip.bundle import _dos_and_census

    _rr, pr = model
    e, rho, census = _dos_and_census(pr.phonon, [6, 6, 6])
    integral = np.trapezoid(rho, e) if hasattr(np, "trapezoid") \
        else np.trapz(rho, e)
    assert "dos_smearing_fallback_mev" not in census
    # tetrahedron integral approximates the band count (coarse-mesh
    # tolerance; the fallback decision itself is by band spread, not this)
    n_bands = 3 * len(pr.phonon.primitive)
    assert integral == pytest.approx(n_bands, rel=0.15)


# ---- SEC-1: validate must not execute code from the bundle -------------------

def _rehash_manifest(bundle_path):
    """Make the manifest self-consistent again after editing an artifact
    (an attacker-built bundle passes the sha256 gate by construction)."""
    from irma.mlip.bundle import _sha256
    mpath = os.path.join(bundle_path, "manifest.json")
    m = json.load(open(mpath))
    for label, name in m["files"].items():
        if name:
            m["sha256"][label] = _sha256(os.path.join(bundle_path, name))
    json.dump(m, open(mpath, "w"))


def test_validate_rejects_a_python_tagged_phonopy_yaml(model, tmp_path):
    """A received bundle whose phonopy.yaml carries a `!!python/` tag must
    be REJECTED before phonopy parses it -- `validate` is the documented
    first action on an untrusted artifact, and phonopy's YAML loader
    executes those tags at parse time. The manifest hash gate does not
    help: this bundle is internally consistent."""
    b = _write(tmp_path / "hostile", model)
    canary = tmp_path / "pwned"
    # PREPENDED, deliberately: this exact file was verified to load
    # successfully under phonopy.load AND execute the payload (phonopy's
    # loader does not stop at the tag), so the guard is the only thing
    # standing between `validate` and code execution
    body = open(b.phonopy_yaml).read()
    with open(b.phonopy_yaml, "w") as fh:
        fh.write(f'exploit: !!python/object/apply:os.system '
                 f'["touch {canary}"]\n' + body)
    _rehash_manifest(b.path)

    problems = validate_bundle(b.path)
    assert problems, "a python-tagged phonopy.yaml must not validate"
    assert any("!!python/" in p for p in problems), problems
    assert not canary.exists()              # nothing from the bundle ran
    # and the rejection is the FIRST phonopy-facing problem: no
    # reload/parse diagnostics leaked out alongside it
    assert not any("reload" in p for p in problems), problems


def test_validate_accepts_a_clean_bundle_unchanged(model, tmp_path):
    """Near-miss: the guard must not reject legitimate bundles."""
    b = _write(tmp_path / "clean", model)
    assert validate_bundle(b.path) == []


def test_born_default_factor_survives_without_phonopy_units(model, tmp_path,
                                                            monkeypatch):
    """REL-12: phonopy.units is deprecated and slated for removal, and the
    extras declare no upper bound; the BORN default factor must come from
    the modern physical_units API when it exists, so a phonopy without the
    legacy module still works."""
    import sys

    pytest.importorskip("phonopy.physical_units")
    monkeypatch.setitem(sys.modules, "phonopy.units", None)   # ImportError
    born = _born_file(tmp_path, with_factor=False, phonon=model[1].phonon)
    b = _write(tmp_path / "b", model, born_path=born)
    assert b.manifest["nac_embedded"] is True

    import phonopy
    from irma.core.phonopy_io import pinned_primitive_matrix_kwargs
    ph = phonopy.load(b.phonopy_yaml, log_level=0,
                      **pinned_primitive_matrix_kwargs(b.phonopy_yaml))
    assert ph.nac_params["factor"] == pytest.approx(14.3996517, rel=1e-6)
