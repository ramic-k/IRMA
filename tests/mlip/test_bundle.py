"""Bundle write/load/validate tests on the EMT dev backend.

Each test gets its own output directory; the shared al_model fixture
(conftest.py) only provides the (immutable-by-contract) relax/FC results,
and the no-leak assert pins that write_bundle cannot contaminate a later
bundle through the shared phonon object."""
import json
import os
import sys

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("phonopy")

from irma.mlip.bundle import (                      # noqa: E402
    Bundle, load_bundle, validate_bundle, write_bundle)

QUIET = lambda *a, **k: None                        # noqa: E731


def _write(outdir, model, **kw):
    rr, pr = model
    kw.setdefault("mesh", (4, 4, 4))
    kw.setdefault("args_used", {"structure": "Al.test", "potential": "emt"})
    kw.setdefault("calc_meta", {"potential": "emt", "package": "ase"})
    kw.setdefault("progress", QUIET)
    return write_bundle(str(outdir), phonon_result=pr, relax_result=rr, **kw)


def test_bundle_roundtrip_and_validation(al_model, tmp_path):
    b = _write(tmp_path, al_model)
    assert isinstance(b, Bundle)
    rr, pr = al_model

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


def test_input_structure_hash_recorded(al_model, tmp_path):
    src = tmp_path / "input.cif"
    src.write_text("fake structure input\n")
    b = _write(tmp_path / "b", al_model, input_structure_path=str(src))
    assert b.manifest["input"]["structure_sha256"] is not None
    assert b.manifest["input"]["structure_path"].endswith("input.cif")


def test_overwrite_guard_covers_everything(al_model, tmp_path):
    _write(tmp_path, al_model)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write(tmp_path, al_model)

    # any non-scratch content triggers the guard, not just owned files
    other = tmp_path / "fresh"
    other.mkdir()
    (other / "users_notes.txt").write_text("keep me?")
    with pytest.raises(FileExistsError):
        _write(other, al_model)

    # overwrite replaces owned files, incl. a stale dos.png
    stale_png = tmp_path / "dos.png"
    stale_png.write_bytes(b"stale")
    b = _write(tmp_path, al_model, overwrite=True)
    assert validate_bundle(b.path) == []


def test_born_roundtrip_explicit_factor(al_model, tmp_path, born_file):
    rr, pr = al_model
    born = born_file(tmp_path, pr.phonon)
    b = _write(tmp_path / "b", al_model, born_path=born)
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac
    assert phonopy_yaml_embeds_nac(b.phonopy_yaml)
    assert b.manifest["nac_embedded"] is True
    assert b.manifest["born"]["sha256"] is not None
    assert validate_bundle(b.path) == []
    # the shared phonon object is NOT left carrying NAC (no-leak contract)
    assert pr.phonon.nac_params is None


def test_born_roundtrip_without_factor_uses_phonopy_default(al_model, tmp_path,
                                                            born_file,
                                                            monkeypatch):
    # the default factor comes from phonopy.physical_units; the deprecated
    # phonopy.units module must not be needed
    monkeypatch.setitem(sys.modules, "phonopy.units", None)
    rr, pr = al_model
    born = born_file(tmp_path, pr.phonon, with_factor=False)
    b = _write(tmp_path / "b", al_model, born_path=born)
    assert b.manifest["nac_embedded"] is True
    import phonopy
    from irma.core.phonopy_io import pinned_primitive_matrix_kwargs
    ph = phonopy.load(b.phonopy_yaml, log_level=0,
                      **pinned_primitive_matrix_kwargs(b.phonopy_yaml))
    assert ph.nac_params is not None
    assert ph.nac_params["factor"] == pytest.approx(14.3996517, rel=1e-6)


def test_check_born_rows_fail_fast(al_model, tmp_path, born_file):
    from irma.mlip.bundle import check_born_rows
    rr, pr = al_model
    good = born_file(tmp_path, pr.phonon)
    assert check_born_rows(good, rr.atoms) is None

    # phonopy's own writers put a '# ...' comment on line 1
    lines = open(good).read().strip().splitlines()
    header = tmp_path / "BORN_header"
    header.write_text("\n".join(["# epsilon and Z* of atoms 1"] + lines[1:]) + "\n")
    assert check_born_rows(header, rr.atoms) is None

    # one row too few: the message names the way out
    bad = tmp_path / "BORN_short"
    bad.write_text("\n".join(lines[:-1]) + "\n")
    problem = check_born_rows(bad, rr.atoms)
    assert problem is not None
    assert "--snap-symmetry" in problem

    assert check_born_rows(tmp_path / "missing", rr.atoms) is not None


def test_validate_ignores_a_stray_born_in_the_cwd(al_model, tmp_path,
                                                  born_file, monkeypatch):
    # found live in the ZrO2 campaign: an unrelated ./BORN file next to
    # where validate runs must not leak NAC into the reload check
    b = _write(tmp_path / "clean", al_model)
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    born_file(workdir, al_model[1].phonon)
    assert (workdir / "BORN").exists()
    monkeypatch.chdir(workdir)
    assert validate_bundle(b.path) == []


def test_validate_catches_tampering(al_model, tmp_path):
    b = _write(tmp_path, al_model)

    # content change -> sha mismatch
    with open(b.structure, "a") as fh:
        fh.write("# tampered\n")
    problems = validate_bundle(b.path)
    assert any("sha256 mismatch" in p for p in problems)

    # manifest claiming NAC the yaml lacks
    b2 = _write(tmp_path / "b2", al_model)
    mpath = os.path.join(b2.path, "manifest.json")
    m = json.load(open(mpath))
    m["nac_embedded"] = True
    json.dump(m, open(mpath, "w"))
    problems = validate_bundle(b2.path)
    assert any("nac_embedded" in p for p in problems)

    # missing file
    b3 = _write(tmp_path / "b3", al_model)
    os.remove(b3.structure)
    assert any("missing file" in p for p in validate_bundle(b3.path))


def test_load_rejects_non_bundles(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_bundle(str(tmp_path))
    (tmp_path / "manifest.json").write_text('{"kind": "something-else"}')
    with pytest.raises(ValueError, match="not an irma-mlip"):
        load_bundle(str(tmp_path))


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
    return np.trapezoid(rho, e)


def test_dos_mixed_flat_and_dispersive_triggers_fallback():
    """One flat sublattice inside an otherwise dispersive spectrum must
    still trigger the fallback (min over bands, not max/all)."""
    from irma.mlip.bundle import _dos_and_census

    # k_b large: flat B bands sit ABOVE the dispersive A range (no crossing)
    ph = _flat_crystal(k_a=2.0, k_b=40.0)
    e, rho, census = _dos_and_census(ph, [8, 2, 2])
    assert census.get("dos_smearing_fallback_mev") == 1.0
    assert _integral(e, rho) == pytest.approx(6, rel=0.05)
    assert np.all(np.diff(e) < 0.51)            # pitch never coarser than 0.5


def test_dos_dispersive_bands_keep_tetrahedron(al_model):
    """A normal dispersive crystal must NOT trigger the smearing fallback
    (the tetrahedron DOS already integrates to the band count)."""
    from irma.mlip.bundle import _dos_and_census

    _rr, pr = al_model
    e, rho, census = _dos_and_census(pr.phonon, [6, 6, 6])
    assert "dos_smearing_fallback_mev" not in census
    # tetrahedron integral approximates the band count (coarse-mesh
    # tolerance; the fallback decision itself is by band spread, not this)
    n_bands = 3 * len(pr.phonon.primitive)
    assert _integral(e, rho) == pytest.approx(n_bands, rel=0.15)


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


def test_validate_rejects_a_python_tagged_phonopy_yaml(al_model, tmp_path):
    """A received bundle whose phonopy.yaml carries a `!!python/` tag must
    be REJECTED before phonopy parses it -- `validate` is the documented
    first action on an untrusted artifact, and phonopy's YAML loader
    executes those tags at parse time. The manifest hash gate does not
    help: this bundle is internally consistent."""
    b = _write(tmp_path / "hostile", al_model)
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
