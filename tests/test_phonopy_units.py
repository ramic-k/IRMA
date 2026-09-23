"""phonopy calculator units: one crystal, two calculators, one answer.

phonopy stores cells in the CALCULATOR's native length unit and never
converts them on load, and it converts dynamical-matrix eigenvalues to THz
with a calculator-specific factor. IRMA has to undo both, or a model built
with quantum-espresso (bohr, 108.97 THz factor) computes different physics
from the identical crystal built with vasp (Angstrom, 15.63 THz factor).

Everything here is driven off a TWIN PAIR built from the vendored graphite
model: the same crystal and the same force constants written twice, once
through phonopy's vasp interface and once through its qe interface, with the
lattice divided by Bohr and the force constants scaled by Bohr^2/Rydberg (the
exact unit change between eV/A^2 and Ry/au^2). The pair is therefore the same
physics by construction, and every quantity IRMA derives from it must agree.
"""
import argparse
import pathlib

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_GRAPHITE = _HERE / "mode2_euphonic_n1_validation" / "graphite"

# A small mesh: these tests check units, not convergence.
_MESH = [4, 4, 2]


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """``(angstrom_yaml, bohr_yaml)`` for one crystal, built with phonopy."""
    pytest.importorskip("phonopy")
    import phonopy
    from phonopy import Phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS
    from phonopy.physical_units import get_physical_units
    from phonopy.structure.atoms import PhonopyAtoms

    units = get_physical_units()
    bohr = units.Bohr             # Angstrom per bohr
    rydberg = units.Rydberg       # eV per Rydberg

    src = phonopy.load(
        str(_GRAPHITE / "phonopy.yaml"),
        force_constants_filename=str(_GRAPHITE / "FORCE_CONSTANTS"),
        log_level=0)
    uc = src.unitcell
    fc = np.array(src.force_constants)
    root = tmp_path_factory.mktemp("phonopy_unit_twins")

    def write(name, calculator, length_divisor, fc_factor):
        out = root / name
        out.mkdir()
        cell = PhonopyAtoms(symbols=[str(s) for s in uc.symbols],
                            cell=np.array(uc.cell) / length_divisor,
                            scaled_positions=np.array(uc.scaled_positions),
                            masses=np.array(uc.masses))
        ph = Phonopy(cell, supercell_matrix=src.supercell_matrix,
                     primitive_matrix=src.primitive_matrix,
                     calculator=calculator, log_level=0)
        ph.force_constants = fc * fc_factor
        ph.save(str(out / "phonopy.yaml"),
                settings={"force_constants": False, "displacements": False})
        write_FORCE_CONSTANTS(ph.force_constants,
                              filename=str(out / "FORCE_CONSTANTS"),
                              p2s_map=ph.primitive.p2s_map)
        return str(out / "phonopy.yaml")

    return (write("angstrom_vasp", None, 1.0, 1.0),
            write("bohr_qe", "qe", bohr, bohr ** 2 / rydberg))


def _bohr():
    from phonopy.physical_units import get_physical_units
    return get_physical_units().Bohr


# ------------------------------------------------------------------ units ---


# ------------------------------------------------------- geometry readers ---

def test_prefill_reader_accepts_and_converts_a_bohr_model(twins):
    """The prefill used to REFUSE a bohr model. Now it converts, and the two
    twins fill Card 6c/6d with the same numbers."""
    from irma.core.phonopy_io import load_phonopy_primitive_structure
    ang, bohr_yaml = twins
    a = load_phonopy_primitive_structure(ang)
    b = load_phonopy_primitive_structure(bohr_yaml)
    assert a.symbols == b.symbols
    assert np.allclose(a.lattice_ang, b.lattice_ang, rtol=0, atol=1e-12)
    assert np.allclose(a.scaled_positions, b.scaled_positions, rtol=0, atol=1e-12)
    assert np.allclose(a.cellpar, b.cellpar, rtol=1e-12, atol=1e-9)
    # ... and the numbers are the physical graphite cell, not the bohr one.
    assert a.cellpar[0] == pytest.approx(2.4606, abs=1e-6)
    assert b.cellpar[0] == pytest.approx(2.4606, abs=1e-6)


def test_ncrystal_primitive_loader_agrees_across_calculators(twins):
    """irma.ncrystal writes the base NCMAT cell from this loader."""
    pytest.importorskip("phonopy")
    from irma.ncrystal.build import load_primitive_info
    ang, bohr_yaml = twins
    sa, ma, pa, la = load_primitive_info(ang)
    sb, mb, pb, lb = load_primitive_info(bohr_yaml)
    assert sa == sb
    assert np.allclose(ma, mb, rtol=0, atol=0)          # amu, never converted
    assert np.allclose(pa, pb, rtol=0, atol=0)          # fractional, ditto
    assert np.allclose(la, lb, rtol=0, atol=1e-12)
    assert la[0, 0] == pytest.approx(2.4606, abs=1e-6)


# ---------------------------------------------------- mode-1/2 engine ------

def _model_context(path):
    from irma.core.noncubic_inelastic_context import build_model_context
    return build_model_context(argparse.Namespace(
        phonopy_yaml=path, mesh=list(_MESH), born=None,
        force_constants=None, force_sets=None))


@pytest.fixture(scope="module")
def twin_contexts(twins):
    pytest.importorskip("phonopy")
    return _model_context(twins[0]), _model_context(twins[1])


def test_model_context_geometry_agrees_across_calculators(twin_contexts):
    ca, cb = twin_contexts
    la = np.asarray(ca["primitive"].cell, float)
    lb = np.asarray(cb["primitive"].cell, float)
    assert np.allclose(la, lb, rtol=0, atol=1e-12)
    assert la[0, 0] == pytest.approx(2.4606, abs=1e-6)     # Angstrom, not bohr
    assert np.allclose(ca["rec_lat_no_2pi"], cb["rec_lat_no_2pi"],
                       rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(ca["primitive"].masses, float),
                       np.asarray(cb["primitive"].masses, float))
    assert list(ca["primitive"].symbols) == list(cb["primitive"].symbols)


def test_model_context_frequencies_agree_across_calculators(twin_contexts):
    """The twins are the same physics: same THz frequencies from different
    native units. This is what makes the geometry comparison meaningful --
    a converted cell paired with an unconverted force constant would show up
    here."""
    ca, cb = twin_contexts
    fa = np.asarray(ca["mesh"].frequencies, float)
    fb = np.asarray(cb["mesh"].frequencies, float)
    assert fa.shape == fb.shape
    assert np.abs(fa - fb).max() < 1e-9 * np.abs(fa).max()
    assert ca["max_mode_energy_mev"] == pytest.approx(
        cb["max_mode_energy_mev"], rel=1e-9)


def test_model_context_carries_the_models_own_frequency_factor(twin_contexts):
    """The coherent one-phonon path solves the dynamical matrix itself, so it
    needs the model's factor -- which is NOT the same number for the two
    twins even though their frequencies are."""
    ca, cb = twin_contexts
    assert ca["frequency_factor_to_thz"] == pytest.approx(15.633302, rel=1e-6)
    assert cb["frequency_factor_to_thz"] == pytest.approx(108.970772, rel=1e-6)


def test_reduced_q_at_a_physical_Q_agrees_across_calculators(twin_contexts):
    """The load-bearing derived quantity. q_red is where the dynamical matrix
    and the exp(2 pi i q.r) structure-factor phases are evaluated for a given
    |Q| in 1/Angstrom; with an unconverted bohr cell it came out 1.89x off, so
    the coherent one-phonon term sampled the wrong point in the zone."""
    ca, cb = twin_contexts
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(7, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    # The coherent worker forms q_red = |Q| * solve(rec_lat_no_2pi, direction) / 2 pi.
    qa = np.linalg.solve(ca["rec_lat_no_2pi"], dirs.T)
    qb = np.linalg.solve(cb["rec_lat_no_2pi"], dirs.T)
    assert np.abs(qa - qb).max() < 1e-12


def test_coherent_one_phonon_frequencies_agree_across_calculators(twins):
    """End of the chain: the frequencies the coherent worker actually solves
    for. These come from _batched_qpoints_eigh, which used to multiply every
    model's eigenvalues by the VASP factor."""
    pytest.importorskip("phonopy")
    import phonopy
    from irma.core.noncubic_workers import _batched_qpoints_eigh
    from irma.core.phonopy_io import (isolated_phonopy_cwd,
                                      resolve_force_constants_source)

    q = np.array([[0.1, 0.2, 0.3], [0.25, 0.0, 0.0], [0.4, -0.3, 0.1]])
    out = []
    for path in twins:
        with isolated_phonopy_cwd():
            ph = phonopy.load(phonopy_yaml=path, is_nac=False, log_level=0,
                              **resolve_force_constants_source(path))
        freqs, _ = _batched_qpoints_eigh(ph.dynamical_matrix, q,
                                         float(ph.unit_conversion_factor))
        # phonopy's own answer for the same model, as the reference
        ph.run_qpoints(q)
        assert np.allclose(freqs, np.asarray(ph.qpoints.frequencies),
                           rtol=1e-10, atol=1e-10)
        out.append(freqs)
    assert np.abs(out[0] - out[1]).max() < 1e-8 * np.abs(out[0]).max()


# --------------------------------------------------- Angstrom regression ---

def test_angstrom_models_are_untouched():
    """The committed Angstrom model comes through the conversion bit for bit."""
    pytest.importorskip("phonopy")
    import phonopy
    from irma.core.phonopy_io import (angstrom_primitive, isolated_phonopy_cwd,
                                      pinned_primitive_matrix_kwargs)
    path = str(_GRAPHITE / "phonopy.yaml")
    with isolated_phonopy_cwd():
        ph = phonopy.load(phonopy_yaml=path, is_nac=False, log_level=0,
                          produce_fc=False,
                          **pinned_primitive_matrix_kwargs(path))
    prim = angstrom_primitive(ph)
    raw = np.array(ph.primitive.cell, dtype=float)
    assert np.array_equal(prim.cell, raw)      # identical bytes, no rescale
