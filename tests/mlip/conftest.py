"""Hermeticity for the mlip test package.

Every test gets a private irma-mlip cache: the developer's real
~/.cache/irma-mlip (env registry, downloaded checkpoints) must never
leak into stub-based tests -- a registered `mace` environment would
silently turn make_calculator stubs into live subprocess dispatch.
"""
import pytest


@pytest.fixture(autouse=True)
def _private_mlip_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("IRMA_MLIP_CACHE",
                       str(tmp_path_factory.mktemp("mlip-cache")))
    for var in list(__import__("os").environ):
        if var.startswith("IRMA_MLIP_PYTHON_"):
            monkeypatch.delenv(var)
    yield


@pytest.fixture(scope="session")
def al_model(tmp_path_factory):
    """One relaxed fcc-Al model and its EMT force constants, shared by the
    bundle and emit tests (read-only: write_bundle restores the phonon
    object's NAC state)."""
    from ase.build import bulk
    from ase.calculators.emt import EMT

    from irma.mlip.calculators import CalculatorSpec
    from irma.mlip.phonons import compute_force_constants
    from irma.mlip.relax import relax

    rr = relax(bulk("Al", "fcc", a=4.05, cubic=True), EMT(), fmax=0.01,
               nmax=100)
    pr = compute_force_constants(
        rr.atoms, CalculatorSpec("emt"), supercell=(2, 2, 2), delta=0.03,
        jobs=1, scratch_dir=str(tmp_path_factory.mktemp("al-model")),
        progress=lambda *a, **k: None)
    return rr, pr


@pytest.fixture
def born_file():
    """Factory: write a synthetic BORN in a directory, with exactly the
    symmetry-independent atom rows phonopy demands for the phonon model."""
    def make(directory, phonon, with_factor=True):
        from phonopy.structure.symmetry import Symmetry
        n_indep = len(Symmetry(phonon.primitive).get_independent_atoms())
        # line 1 is ALWAYS the factor line; a non-numeric token means "use
        # the code default" (phonopy file_IO.get_born_parameters)
        lines = ["14.4" if with_factor else "default",
                 "2.0 0 0  0 2.0 0  0 0 2.0"]                # dielectric
        lines += ["1.5 0 0  0 1.5 0  0 0 1.5"] * n_indep
        path = directory / "BORN"
        path.write_text("\n".join(lines) + "\n")
        return str(path)
    return make
