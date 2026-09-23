"""phonopy_io.py — Load phonopy mesh data and compute anisotropic DOS tensors.

Provides tools for reading phonon eigenvectors from phonopy and computing
the anisotropic phonon DOS tensor ρ_{d,ij}(ε) needed for the phonopy-backed
directional Debye-Waller / standalone MT4 treatments (inelastic_mode=1/2, iel=10).

The DOS tensor ρ_{d,ij}(ε) for atom d and Cartesian directions i, j:

    ρ_{d,ij}(ε_k) = (1/N_total) Σ_{q,ν} Re[e_{d,i,ν}(q) · conj(e_{d,j,ν}(q))]
                     × w_{q,ν} × G(ε_k − ε_{q,ν}, σ)

where e_{d,i,ν}(q) is the phonopy unit-norm eigenvector component,
w_{q,ν} is the BZ weight, G is a Gaussian kernel, and N_total = Σ w_{q,ν}.

Normalization: the scalar partial DOS g_d(ε) = (1/3)Σ_i ρ_{d,ii}(ε)
satisfies ∫g_d(ε)dε = 1 (consistent with LEAPR's tbeta=1 convention),
because phonopy eigenvectors satisfy Σ_{d,i}|e_{d,i,ν}|² = 1 (unit norm).

Phonopy eigenvector convention (np.linalg.eigh):
    mesh.eigenvectors shape = (N_q, 3*N_atoms, N_branches)
    axis 1 = component index 3*d+i  (ATOM-FIRST: atom d, direction i=0,1,2)
    axis 2 = mode index ν
    i.e., eigenvectors[q, 3*d+i, ν] = component of atom d, direction i, mode ν.
    Correct reshape: .transpose(0,2,1).reshape(N_q, N_branches, N_atoms, 3).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


from irma.core.constants import (
    HBAR, AMU, EV, BK,
    HBAR2_OVER_2MN_MEV_A2, THZ_TO_EV,
    MODE_ENERGY_FLOOR_MEV, GAMMA_ACOUSTIC_FLOOR_MEV,
)


from contextlib import contextmanager


@contextmanager
def isolated_phonopy_cwd():
    """Pin the process cwd to a fresh empty directory while phonopy loads.

    phonopy.load auto-reads FORCE_CONSTANTS / force_constants.hdf5 /
    FORCE_SETS / BORN from the process working directory as a last resort —
    even when explicit paths are given (its force-constants selection
    probes ./FORCE_CONSTANTS before building FC from an explicit
    force_sets_filename). IRMA passes every wanted source as an absolute
    path (or relies on yaml-embedded data, which outranks the fallbacks),
    so the correct behavior for ALL residual fallbacks is to find nothing:
    an empty scratch cwd guarantees that, making the loaded model depend
    only on the deck and the named phonopy.yaml — never on where IRMA
    happens to run.
    """
    import os
    import shutil
    import tempfile

    prev = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="irma_phonopy_load_")
    os.chdir(scratch)
    try:
        yield
    finally:
        os.chdir(prev)
        # rmtree, not rmdir: if phonopy wrote anything into the scratch cwd the
        # dir is non-empty and rmdir would leave it (and its contents) in /tmp.
        shutil.rmtree(scratch, ignore_errors=True)


@contextmanager
def openmp_unpinned_serial_setup():
    """Temporarily lift the OpenMP single-thread pin for a SERIAL phonopy run.

    The process-wide pin (OMP_NUM_THREADS=1 before native libraries load)
    exists for the spawn worker pools (get_context("spawn")), where
    free-threaded workers oversubscribe the machine. The big SERIAL phonopy
    mesh runs (the full and symmetry-reduced eigensolves) happen long before
    any pool is spawned, on otherwise-idle cores: phonopy's C
    dynamical-matrix build parallelizes over q-points with no cross-q
    reductions, so per-q results are unchanged (gated at ENDF tape
    precision). BLAS pools stay at one thread — a threaded 12x12 eigh is
    pathologically slower. The previous limit is restored on exit, before any
    worker pool starts; the worker initializer re-pins each spawned worker as
    a second line of defense. No-op when threadpoolctl is not installed.
    """
    try:
        import threadpoolctl
    except ImportError:
        yield
        return
    import os as _os
    n = max(1, _os.cpu_count() or 1)
    with threadpoolctl.threadpool_limits(limits=n, user_api="openmp"):
        yield


def _open_phonopy_yaml(phonopy_yaml_path, **kwargs):
    """Open a (possibly compressed) phonopy.yaml as text."""
    import bz2
    import gzip
    import lzma
    import os

    path = str(phonopy_yaml_path)
    ext = os.path.splitext(path)[1].lower()
    opener = {".gz": gzip.open, ".xz": lzma.open, ".lzma": lzma.open,
              ".bz2": bz2.open}.get(ext, open)
    return opener(path, "rt", **kwargs)


def _phonopy_yaml_has_top_level_key(phonopy_yaml_path, keys) -> bool:
    """Cheap streaming scan of a (possibly compressed) phonopy.yaml for a
    top-level key — the file can be large when force constants are embedded.
    """
    with _open_phonopy_yaml(phonopy_yaml_path) as f:
        for line in f:
            if line.startswith(keys):
                return True
    return False


def reject_unsafe_phonopy_yaml(phonopy_yaml_path) -> None:
    """Refuse to hand an UNTRUSTED phonopy.yaml to phonopy's YAML parser.

    TRUST BOUNDARY. phonopy parses phonopy.yaml with PyYAML's unsafe
    loader (``yaml.load(fp, Loader=CLoader)`` in
    phonopy/interface/phonopy_yaml.py), so a ``!!python/object/apply:``
    tag in the file executes arbitrary code AT PARSE TIME. Any
    phonopy.yaml that did not originate on this machine — a received
    bundle, a downloaded example, a deck pointing at someone else's
    model — must therefore be treated as untrusted input, and every call
    site that feeds such a file into ``phonopy.load`` (or any other
    phonopy parse) must call this guard FIRST.

    The check is the same cheap streaming scan the embeds-detection
    helpers use (no YAML parse, compressed files supported). Rejected,
    with ValueError, is any line carrying:

    - ``!!python/`` — the shorthand form of the unsafe tag namespace;
    - ``tag:yaml.org,2002:python`` — the same namespace via a verbatim
      ``!<...>`` tag;
    - a ``%TAG`` directive — never emitted by phonopy, and the only way
      to alias the python tag namespace past a textual scan.

    A legitimate phonopy.yaml contains none of these in any scalar, so
    false positives are not a practical concern. Scanning is not a
    sandbox: it makes the known code-execution vector fail closed, it
    does not make phonopy's parser safe.
    """
    path = str(phonopy_yaml_path)
    with _open_phonopy_yaml(path, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if "!!python/" in line or "tag:yaml.org,2002:python" in line:
                raise ValueError(
                    f"{path}:{lineno}: refusing to parse: the file carries "
                    f"a non-standard '!!python/' YAML tag, which phonopy's "
                    f"unsafe YAML loader would EXECUTE as code; this is not "
                    f"a legitimate phonopy.yaml — do not open it with "
                    f"phonopy tooling")
            if line.startswith("%TAG"):
                raise ValueError(
                    f"{path}:{lineno}: refusing to parse: the file carries "
                    f"a %TAG directive, which phonopy never emits and which "
                    f"can alias the code-executing '!!python/' YAML tag "
                    f"namespace; this is not a legitimate phonopy.yaml")


def phonopy_yaml_embeds_nac(phonopy_yaml_path) -> bool:
    """True if the phonopy.yaml carries embedded NAC parameters.

    Matches the current top-level ``nac:`` block as well as the older
    flat ``born_effective_charge:`` / ``dielectric_constant:`` keys that
    phonopy's parser still accepts. Used to decide phonopy.load's
    ``is_nac``: NAC saved inside the deck-named model file is part of that
    model and is honored, but phonopy's fallback of auto-reading a file
    named ``BORN`` from the process working directory is never allowed —
    whether NAC is applied must depend only on the deck (Card 6f) and the
    named phonopy.yaml, not on where IRMA runs.
    """
    return _phonopy_yaml_has_top_level_key(
        phonopy_yaml_path,
        ("nac:", "born_effective_charge:", "dielectric_constant:"),
    )


def phonopy_yaml_embeds_force_constants(phonopy_yaml_path) -> bool:
    """True if the phonopy.yaml carries an embedded force-constants block."""
    return _phonopy_yaml_has_top_level_key(
        phonopy_yaml_path, ("force_constants:",)
    )


def pinned_primitive_matrix_kwargs(phonopy_yaml_path) -> dict:
    """kwargs pinning phonopy.load's primitive-cell semantics across versions.

    phonopy 4 changed the meaning of an OMITTED ``primitive_matrix``: it now
    defaults to ``"auto"`` (symmetry-guessed primitive), where phonopy 2/3
    used the yaml's stored matrix, or the identity when the yaml stores none.
    An auto-guessed non-identity matrix re-bases the primitive cell, which
    re-interprets mesh dimensions and shifts frequencies enough to break
    byte-pinned tapes and frozen validation references.

    Contract (identical physics on every supported phonopy):
    - yaml stores a top-level ``primitive_matrix:``: pass nothing — the
      stored value wins under every version, and passing an explicit value
      would OVERRIDE a stored non-identity matrix.
    - yaml stores none: pass ``primitive_matrix="P"`` (identity), pinning
      the phonopy 2/3 behavior under phonopy 4's ``"auto"`` default.
    """
    if _phonopy_yaml_has_top_level_key(phonopy_yaml_path,
                                       ("primitive_matrix:",)):
        return {}
    return {"primitive_matrix": "P"}


def resolve_force_constants_source(phonopy_yaml_path) -> dict:
    """Locate the force constants belonging to the named phonopy model.

    Force constants embedded in the yaml itself outrank every file in
    phonopy.load (they win even over an explicit filename), so they are
    reported first ({} — phonopy reads them from the yaml). Otherwise the
    directory of phonopy.yaml is searched in the order the directional-DW
    loader has always used: ``force_constants.hdf5``, ``FORCE_CONSTANTS``
    (text), ``FORCE_SETS``; the matching phonopy.load keyword argument
    (``force_constants_filename`` or ``force_sets_filename``) is returned.

    Raises FileNotFoundError when no source exists: phonopy.load's fallback
    of searching the process working directory is never allowed — which
    model gets computed must depend only on the deck-named phonopy.yaml,
    not on where IRMA runs.
    """
    import os

    if phonopy_yaml_embeds_force_constants(phonopy_yaml_path):
        print(f"  Using force constants embedded in {phonopy_yaml_path}", flush=True)
        return {}
    yaml_dir = os.path.dirname(os.path.abspath(str(phonopy_yaml_path)))
    fc_hdf5 = os.path.join(yaml_dir, 'force_constants.hdf5')
    fc_text = os.path.join(yaml_dir, 'FORCE_CONSTANTS')
    fc_sets = os.path.join(yaml_dir, 'FORCE_SETS')
    if os.path.exists(fc_hdf5):
        print(f"  Using force constants: {fc_hdf5}", flush=True)
        return {'force_constants_filename': fc_hdf5}
    if os.path.exists(fc_text):
        print(f"  Using force constants: {fc_text}", flush=True)
        return {'force_constants_filename': fc_text}
    if os.path.exists(fc_sets):
        print(f"  Using force sets: {fc_sets}", flush=True)
        return {'force_sets_filename': fc_sets}
    raise FileNotFoundError(
        f"No force constants found for {phonopy_yaml_path}: checked the "
        f"yaml itself for an embedded force_constants block, then "
        f"{fc_hdf5}, {fc_text}, {fc_sets}."
    )


# -------------------------------------------------- primitive structure ----

@dataclass
class AngstromPrimitiveCell:
    """A phonopy primitive cell with its lattice converted to Angstrom.

    phonopy keeps cells in the calculator's length unit (bohr for qe, abinit,
    siesta, ...); IRMA works in Angstrom and converts once, here. The live
    phonopy object keeps its native cell, which its force constants and
    frequency factor are paired with.
    """

    cell: np.ndarray                # (3, 3) row vectors, ANGSTROM
    scaled_positions: np.ndarray    # (N_sites, 3) fractional
    symbols: List[str]
    masses: np.ndarray              # (N_sites,) amu


def angstrom_primitive(phonon) -> AngstromPrimitiveCell:
    """Geometry of a loaded phonopy object's primitive cell, lattice in Angstrom."""
    from phonopy.physical_units import get_calculator_physical_units
    units = get_calculator_physical_units(phonon.calculator)
    prim = phonon.primitive
    cell = np.array(prim.cell, dtype=float)
    if units.distance_to_A != 1.0:
        # Printed only when it fires, so Angstrom models keep their logs.
        print(f"  Converting the phonopy cell from {units.length_unit} to Angstrom "
              f"(1 {units.length_unit} = {units.distance_to_A:.9f} A)", flush=True)
        cell = cell * units.distance_to_A
    return AngstromPrimitiveCell(
        cell=cell,
        scaled_positions=np.array(prim.scaled_positions, dtype=float),
        symbols=[str(x) for x in prim.symbols],
        masses=np.array(prim.masses, dtype=float),
    )


def load_phonopy(phonopy_yaml_path, born_path=None, force_constants_filename=None,
                 force_sets_filename=None, geometry_only=False):
    """Load a phonopy model the way every IRMA path does.

    The yaml is checked for unsafe tags first, the primitive matrix is pinned,
    and phonopy's C backend is used (the Rust backend's thread pool ignores
    the worker thread pinning). phonopy runs in an empty scratch directory,
    because it probes ./FORCE_CONSTANTS, ./FORCE_SETS and ./BORN even when
    explicit paths are given. Force constants: the explicit paths, else
    ``resolve_force_constants_source``. NAC: the explicit BORN file, else NAC
    embedded in the yaml, never ./BORN. ``geometry_only`` reads neither.
    """
    import os

    import phonopy

    path = os.path.abspath(str(phonopy_yaml_path))
    reject_unsafe_phonopy_yaml(path)
    kwargs = dict(pinned_primitive_matrix_kwargs(path), lang="C")
    embeds_nac = False
    if geometry_only:
        kwargs.update(produce_fc=False, is_nac=False, log_level=0)
    else:
        if force_constants_filename is None and force_sets_filename is None:
            kwargs.update(resolve_force_constants_source(path))
        else:
            if force_constants_filename is not None:
                kwargs["force_constants_filename"] = os.path.abspath(
                    str(force_constants_filename))
            if force_sets_filename is not None:
                kwargs["force_sets_filename"] = os.path.abspath(str(force_sets_filename))
        born = None
        if born_path is not None:
            born = os.path.abspath(str(born_path))
            if not os.path.isfile(born):
                raise FileNotFoundError(
                    f"BORN corrections were requested but the BORN file does "
                    f"not exist: {born}")
            print(f"  NAC from {born}", flush=True)
        else:
            embeds_nac = phonopy_yaml_embeds_nac(path)
            if embeds_nac:
                print(f"  NAC embedded in {path}", flush=True)
        kwargs.update(born_filename=born, is_nac=born is not None or embeds_nac)
    with isolated_phonopy_cwd():
        ph = phonopy.load(phonopy_yaml=path, **kwargs)
    if embeds_nac and ph.nac_params is None:
        raise RuntimeError(
            f"{path} embeds NAC keys but phonopy could not parse NAC parameters "
            f"from them; fix the file or supply an explicit BORN file (Card 6f "
            f"use_born=1).")
    return ph


def _cell_parameters(lattice):
    """``(a, b, c, alpha, beta, gamma)`` from a 3x3 row-vector lattice."""
    cell = np.asarray(lattice, dtype=float).reshape(3, 3)
    lengths = np.linalg.norm(cell, axis=1)
    angles = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        denom = lengths[j] * lengths[k]
        cosang = 0.0 if denom == 0.0 else float(
            np.dot(cell[j], cell[k]) / denom)
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))))
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]),
            angles[0], angles[1], angles[2])


@dataclass
class PhonopyPrimitiveStructure:
    """The primitive cell of a phonopy model, in IRMA's Card 6c/6d terms."""

    cellpar: tuple              # (a, b, c [Angstrom], alpha, beta, gamma [deg])
    lattice_ang: np.ndarray     # (3, 3) row vectors, Angstrom
    symbols: List[str]          # per site, e.g. ['C', 'C', 'C', 'C']
    scaled_positions: np.ndarray  # (N_sites, 3) fractional


def load_phonopy_primitive_structure(phonopy_yaml_path
                                     ) -> PhonopyPrimitiveStructure:
    """Read a phonopy model's PRIMITIVE cell through phonopy itself.

    The primitive cell, not the yaml's ``unit_cell``/``primitive_cell``
    block, because that is the cell IRMA's crystal cards must describe:
    ``irma.core.crystal`` matches every Card 6d position against the
    phonopy PRIMITIVE-cell mesh for inelastic_mode 1/2, and rejects the
    deck when they do not correspond. Reading it through ``phonopy.load``
    (with the cross-version primitive-matrix pin) is the only way to get
    the same cell the engine will build: the yaml's ``primitive_cell``
    block is whatever the model's author last wrote there, and the stored
    ``primitive_matrix`` is applied to ``unit_cell`` at load time.

    A geometry-only load (no force constants, no NAC); the cell is
    converted to Angstrom as the mode-1/2 engine does.
    Raises ``ImportError`` without phonopy, ``ValueError`` for an unsafe
    yaml, and whatever phonopy raises for an unreadable file.
    """
    try:
        ph = load_phonopy(phonopy_yaml_path, geometry_only=True)
    except ImportError:
        raise ImportError(
            "phonopy is required to read a phonopy model's crystal "
            "structure. Install with: pip install phonopy")
    prim = angstrom_primitive(ph)
    return PhonopyPrimitiveStructure(
        cellpar=_cell_parameters(prim.cell),
        lattice_ang=prim.cell,
        symbols=prim.symbols,
        scaled_positions=prim.scaled_positions,
    )


@dataclass
class PhonopyMeshData:
    """Phonon mesh data loaded from phonopy.

    All eigenvectors are the raw eigenvectors of the dynamical matrix
    (unit-norm, NOT mass-weighted individually). They satisfy:
        Σ_{d,i} |e_{d,i,ν}(q)|² = 1   for each (q, ν).
    """
    qpoints: np.ndarray         # (N_q, 3) fractional BZ coordinates
    frequencies_ev: np.ndarray  # (N_q, N_branches) in eV; negative = imaginary
    eigenvectors: np.ndarray    # (N_q, N_branches, N_atoms, 3) complex128
    weights: np.ndarray         # (N_q,) integer BZ weights (1 for full mesh)
    masses_amu: np.ndarray      # (N_atoms,) atomic masses in amu
    atom_symbols: List[str]     # e.g. ['C', 'C', 'C', 'C']
    atom_positions: np.ndarray  # (N_atoms, 3) fractional coords in primitive cell
    min_phonon_energy_mev: float = 0.0  # user cutoff; 0 preserves legacy floors
    phonopy_mesh_object: object = field(default=None, repr=False, compare=False)

    @property
    def n_qpoints(self):
        """Number of mesh q-points."""
        return self.qpoints.shape[0]

    @property
    def n_branches(self):
        """Number of phonon branches (3 x atoms in the primitive cell)."""
        return self.frequencies_ev.shape[1]

    @property
    def n_atoms(self):
        """Number of atoms in the primitive cell."""
        return len(self.masses_amu)


def warn_dynamic_instability(frequencies_ev, qpoints, *, emit=print) -> None:
    """Warn if the phonon mesh has imaginary modes away from Gamma.

    Mirrors the ENDF driver's stability diagnostic so the exporter and other mesh
    consumers surface the same signal: an imaginary mode AT Gamma is acoustic-sum-
    rule / NAC numerical noise (routine and expected), but one at a NON-Gamma
    q-point is a genuine finite-wavevector dynamical instability worth flagging
    before a model is trusted (or a pack is baked from it). Warnings go to ``emit``
    (default ``print``); nothing is raised.
    """
    freqs = np.asarray(frequencies_ev, dtype=float)
    qpts = np.asarray(qpoints, dtype=float)
    is_gamma_q = np.all(np.abs(qpts) < 1.0e-9, axis=1)
    nongamma_imag = (freqs < -1.0e-6) & ~is_gamma_q[:, None]
    if np.any(nongamma_imag):
        emit(f"WARNING: {int(np.count_nonzero(nongamma_imag))} imaginary phonon "
             f"mode(s) at non-Gamma q-points (min "
             f"{np.min(freqs[nongamma_imag]) * 1000:.3f} meV) -- the phonon model "
             "is DYNAMICALLY UNSTABLE (a finite-wavevector soft mode); review the "
             "structure and force constants before trusting this evaluation.")
    elif np.any(freqs < -1.0e-4):
        emit("WARNING: imaginary modes near Gamma (min "
             f"{np.min(freqs) * 1000:.2f} meV) beyond the acoustic-sum-rule noise "
             "band; they are excluded from the grid, but review the phonon model.")


# A user phonon-energy cutoff that moves the mean-square displacement trace
# by more than this fraction is reported as a warning (author decision
# 2026-09-16): on graphite a 5 meV cutoff removes 0.13% of the modes but 29%
# of the displacement.
PHONON_CUTOFF_WARN_FRACTION = 0.01


def validate_min_phonon_energy_mev(value) -> float:
    """Return a finite, nonnegative user phonon-energy cutoff in meV."""
    cutoff = float(value)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("minimum phonon energy must be finite and nonnegative")
    return cutoff


def mode_floor_mask(energies_mev, qpoints, n_branches,
                    min_phonon_energy_mev=0.0):
    """Two-tier low-energy mode mask for flattened (q, branch) mode arrays.

    Keeps a mode when its energy exceeds MODE_ENERGY_FLOOR_MEV (1 ueV,
    pure overflow guard) — except at Gamma q-points, where the threshold
    is GAMMA_ACOUSTIC_FLOOR_MEV so the three Goldstone modes (whose
    numerical frequency is acoustic-sum-rule noise) can never leak a
    kT/omega^2 divergence into a mesh sum. See constants.py.
    """
    energies_mev = np.asarray(energies_mev, dtype=float)
    qpoints = np.asarray(qpoints, dtype=float)
    is_gamma_q = np.all(np.abs(qpoints) < 1.0e-9, axis=1)
    is_gamma_mode = np.repeat(is_gamma_q, n_branches)
    floor = np.where(is_gamma_mode, GAMMA_ACOUSTIC_FLOOR_MEV,
                     MODE_ENERGY_FLOOR_MEV)
    user_floor = validate_min_phonon_energy_mev(min_phonon_energy_mev)
    if user_floor > 0.0:
        floor = np.maximum(floor, user_floor)
    return energies_mev > floor


def coherent_mode_mask(energies_mev, q_folded, min_phonon_energy_mev=0.0):
    """Which modes the coherent one-phonon term keeps at its folded q points.

    The coherent term evaluates the dynamical matrix at the folded momentum
    transfers, so ``energies_mev`` is (points, branches) and ``q_folded`` the
    (points, 3) reduced q of each row. Without a user cutoff it keeps every
    positive mode, which is the established coherent behaviour (a Gamma
    Goldstone mode has zero or negative energy there and drops out). With a
    cutoff it applies the same rule as every mesh consumer,
    ``mode_floor_mask`` with the cutoff, so all terms are built from one
    population.
    """
    energies = np.asarray(energies_mev, dtype=float)
    cutoff = validate_min_phonon_energy_mev(min_phonon_energy_mev)
    if cutoff <= 0.0:
        return energies > 0.0
    n_points, n_branches = energies.shape
    mask = mode_floor_mask(energies.reshape(-1), np.asarray(q_folded, dtype=float),
                           n_branches, cutoff)
    return mask.reshape(n_points, n_branches)


def phonon_cutoff_summary(mesh_data, temperature_K):
    """What a user phonon-energy cutoff removed from ``mesh_data``, at one temperature.

    Compares the cutoff mask with the baseline mask (the automatic floors
    alone) on the same mesh and reports, as a dictionary: the imaginary
    mode count, the modes the baseline floors already drop, the ADDITIONAL
    positive modes the cutoff removes with their Brillouin-zone-weighted
    fraction, the per-atom DOS trace deficit (the weighted eigenvector
    weight those modes carried, out of 3 per atom), and the mean-square
    displacement trace per atom before and after the cutoff at
    ``temperature_K``. ``warn`` is True when the displacement trace moved by
    more than ``PHONON_CUTOFF_WARN_FRACTION``. Displacements weight modes as
    1/E^2, so a cutoff that removes a negligible share of the modes can
    still remove a large share of the Debye-Waller exponent; the report
    exists so that is never silent.
    """
    from dataclasses import replace
    cutoff = validate_min_phonon_energy_mev(mesh_data.min_phonon_energy_mev)
    freq = np.asarray(mesh_data.frequencies_ev, dtype=float)
    n_q, n_branches = freq.shape
    energies = freq.reshape(-1) * 1.0e3
    weights = np.repeat(np.asarray(mesh_data.weights, dtype=float), n_branches)
    total_weight = float(np.sum(weights))
    baseline = mode_floor_mask(energies, mesh_data.qpoints, n_branches, 0.0)
    with_cutoff = mode_floor_mask(energies, mesh_data.qpoints, n_branches, cutoff)
    removed = baseline & ~with_cutoff
    eig = np.asarray(mesh_data.eigenvectors).reshape(energies.size, -1, 3)
    per_atom_weight = np.sum(np.abs(eig) ** 2, axis=2)               # (modes, atoms)
    # per-atom DOS trace integrates to 3 over the q-point weight, so the
    # deficit is normalised by the q weight, not by the mode weight
    q_weight = total_weight / n_branches
    trace_deficit = (weights[removed, None] * per_atom_weight[removed]).sum(axis=0) / q_weight
    u_after = thermal_displacement_matrices_perq(mesh_data, temperature_K)
    u_before = thermal_displacement_matrices_perq(replace(mesh_data, min_phonon_energy_mev=0.0),
                                                  temperature_K)
    trace_before = np.einsum("aii->a", u_before)
    trace_after = np.einsum("aii->a", u_after)
    mean_before = float(np.mean(trace_before))
    mean_after = float(np.mean(trace_after))
    change = (mean_after - mean_before) / mean_before if mean_before > 0.0 else 0.0
    return {
        "min_phonon_energy_meV": cutoff,
        "temperature_K": float(temperature_K),
        "mode_count": int(energies.size),
        "imaginary_modes": int(np.count_nonzero(energies < 0.0)),
        "baseline_floor_excluded_modes": int(np.count_nonzero(~baseline)),
        "cutoff_removed_modes": int(np.count_nonzero(removed)),
        "cutoff_removed_weight_fraction": float(weights[removed].sum() / total_weight),
        "dos_trace_deficit_per_atom": [float(x) for x in trace_deficit],
        "trace_u_before_A2": [float(x) for x in trace_before],
        "trace_u_after_A2": [float(x) for x in trace_after],
        "mean_trace_u_change": float(change),
        "warn": bool(abs(change) > PHONON_CUTOFF_WARN_FRACTION),
    }


def format_phonon_cutoff_summary(summary):
    """The log lines for ``phonon_cutoff_summary``."""
    s = summary
    lines = [
        f"Phonon-energy cutoff {s['min_phonon_energy_meV']:g} meV at "
        f"{s['temperature_K']:g} K: {s['imaginary_modes']} imaginary mode(s) and "
        f"{s['baseline_floor_excluded_modes']} mode(s) under the automatic floors "
        f"were already excluded; the cutoff removes {s['cutoff_removed_modes']} more "
        f"of {s['mode_count']} modes ({100.0 * s['cutoff_removed_weight_fraction']:.4f}% "
        f"of the mode weight). Per-atom DOS trace 3 -> "
        + ", ".join(f"{3.0 - d:.6f}" for d in s["dos_trace_deficit_per_atom"])
        + f". Mean-square displacement trace per atom "
        f"{np.mean(s['trace_u_before_A2']):.6g} -> {np.mean(s['trace_u_after_A2']):.6g} A^2 "
        f"({100.0 * s['mean_trace_u_change']:+.2f}%). No replacement spectrum; the "
        f"coherent one-phonon term is truncated the same way.",
    ]
    if s["warn"]:
        lines.append(
            f"WARNING: the cutoff changed the mean-square displacement by "
            f"{100.0 * abs(s['mean_trace_u_change']):.1f}%, above "
            f"{100.0 * PHONON_CUTOFF_WARN_FRACTION:g}%: the Debye-Waller factors "
            f"and every term built from them now describe a truncated vibrational "
            f"model, not the phonon model as computed. Displacements weight modes "
            f"as 1/E^2, so low-energy modes matter far more than their count.")
    return lines


def tdm_freq_min_thz(qpoints):
    """Two-tier mode floor as a global phonopy ``freq_min`` (in THz).

    Phonopy's ``ThermalDisplacementMatrices`` API only accepts a GLOBAL
    frequency cutoff, so the two-tier floor collapses to: the Gamma-tier
    guard when the mesh contains a Gamma point (tiny-positive
    Goldstone/ASR noise would otherwise poison U ~ coth/omega) and the
    1-ueV overflow guard on Gamma-free (shifted MP) meshes. EVERY
    ThermalDisplacementMatrices construction must take its freq_min from
    here so the MT2 and MT4 Debye-Waller tensors stay mutually
    consistent.
    """
    qpoints = np.asarray(qpoints, dtype=float)
    has_gamma = bool(np.any(np.all(np.abs(qpoints) < 1.0e-9, axis=1)))
    floor_mev = GAMMA_ACOUSTIC_FLOOR_MEV if has_gamma else MODE_ENERGY_FLOOR_MEV
    return floor_mev * 1.0e-3 / THZ_TO_EV


def load_phonopy_mesh(phonopy_yaml_path, mesh_dim, born_path=None,
                      force_constants_filename=None, force_sets_filename=None,
                      min_phonon_energy_mev=0.0):
    """Load phonon eigenvectors from phonopy and return a PhonopyMeshData.

    The model is loaded by :func:`load_phonopy` (force constants and NAC as
    described there).

    Parameters
    ----------
    phonopy_yaml_path : str
        Path to phonopy.yaml.
    mesh_dim : list of 3 ints
        Monkhorst-Pack mesh dimensions, e.g. [40, 40, 40].
    born_path : str or None
        Path to BORN file for non-analytical correction (NAC).
        If None, no NAC is applied.
    force_constants_filename, force_sets_filename : str or None
        Explicit force-constants / force-sets file; overrides the
        yaml-adjacent discovery.

    Returns
    -------
    PhonopyMeshData
    """
    min_phonon_energy_mev = validate_min_phonon_energy_mev(
        min_phonon_energy_mev)
    try:
        ph = load_phonopy(phonopy_yaml_path, born_path=born_path,
                          force_constants_filename=force_constants_filename,
                          force_sets_filename=force_sets_filename)
    except ImportError:
        raise ImportError(
            "phonopy is required for non-cubic inelastic calculations. "
            "Install with: pip install phonopy")

    # Use is_mesh_symmetry=False (full Monkhorst-Pack mesh, all weights=1).
    # The symmetrized (irreducible BZ) mesh gives WRONG per-atom DOS tensors
    # for non-symmorphic space groups: the irreducible q-points represent only
    # one star member, so atom permutations from other star members are missing.
    # With the full mesh (q and all its symmetry images explicitly included),
    # the BZ average correctly gives equivalent values for symmetry-equivalent
    # atoms.  This is also why phonopy requires is_mesh_symmetry=False for its
    # own thermal_displacement_matrices calculation.
    print(f"  Running phonopy mesh {mesh_dim[0]}×{mesh_dim[1]}×{mesh_dim[2]} "
          f"with eigenvectors (full mesh)...", flush=True)
    with openmp_unpinned_serial_setup():
        ph.run_mesh(mesh_dim, with_eigenvectors=True, is_mesh_symmetry=False)
    mesh_obj = ph.mesh
    prim = ph.primitive
    freq_ev = mesh_obj.frequencies * THZ_TO_EV  # (N_q, N_branches), negative = imaginary
    n_q, n_branches = freq_ev.shape
    n_atoms = len(prim.masses)
    # phonopy eigenvectors are [q, 3*atom+xyz, mode]; reorder to [q, mode, atom, xyz].
    eigs = mesh_obj.eigenvectors.transpose(0, 2, 1).reshape(n_q, n_branches, n_atoms, 3)
    weights = mesh_obj.weights
    masses_amu = np.array(prim.masses, dtype=float)
    symbols = list(prim.symbols)
    atom_positions = np.array(prim.scaled_positions, dtype=float)
    print(f"  Loaded {n_q} q-points, {n_branches} branches, {n_atoms} atoms "
          f"(total BZ weight = {int(np.sum(weights))})", flush=True)

    return PhonopyMeshData(
        qpoints=mesh_obj.qpoints,
        frequencies_ev=freq_ev,
        eigenvectors=eigs,
        weights=weights,
        masses_amu=masses_amu,
        atom_symbols=symbols,
        atom_positions=atom_positions,
        min_phonon_energy_mev=min_phonon_energy_mev,
        phonopy_mesh_object=ph.mesh,
    )


def thermal_displacement_matrices_perq(mesh_data, temperature_k):
    """Thermal-displacement tensor U_ij (Angstrom^2) with the PER-Q mode floor.

    Direct mode sum, mirroring phonopy's ``ThermalDisplacementMatrices``:

        U_ij(d) = (1/N) Σ_{q,nu} (hbar^2 / (2 M_d E_qnu)) coth(E/2kT)
                                  · Re[e_{d,i,nu}(q) e*_{d,j,nu}(q)]

    with the same two-tier per-q ``mode_floor_mask`` used by the DOS-tensor and
    one-phonon paths (Goldstone guard AT Gamma, 1-ueV overflow guard elsewhere),
    rather than phonopy's single GLOBAL ``freq_min``. The eigenvectors are the
    unit-norm dynamical-matrix eigenvectors (Σ_{d,i}|e|²=1), so the displacement
    of atom d carries the 1/M_d mass factor explicitly.

    On a Gamma-free mesh the per-q floor equals the global floor, so this
    reproduces phonopy exactly (pinned by tests). On a Gamma-containing mesh it
    keeps the off-Gamma low-omega modes in [1 ueV, GAMMA_ACOUSTIC_FLOOR] that
    phonopy's global Gamma-tier cutoff would wrongly drop -- the modes that carry
    real 1/omega-weighted Debye-Waller weight -- keeping the MT2/MT4 DW tensor
    consistent with the one-phonon mode set.
    """
    n_atoms = mesh_data.n_atoms
    n_branches = mesh_data.n_branches
    n_total = float(np.sum(mesh_data.weights))
    T = float(temperature_k)

    freq_flat = mesh_data.frequencies_ev.reshape(-1)                  # (N_modes,) eV
    weights_flat = np.repeat(mesh_data.weights, n_branches).astype(float)
    eigs_flat = mesh_data.eigenvectors.reshape(-1, n_atoms, 3)        # (N_modes,Na,3)

    valid = mode_floor_mask(
        freq_flat * 1.0e3, mesh_data.qpoints, n_branches,
        mesh_data.min_phonon_energy_mev)
    E = freq_flat[valid]                                  # (M,) eV, all > floor > 0
    wt = weights_flat[valid]
    eig = eigs_flat[valid]                                # (M, N_atoms, 3) complex

    two_n_plus_1 = 1.0 / np.tanh(E / (2.0 * BK * T))      # coth(E/2kT) = (2n+1)
    # hbar^2/(2 M_d E) in Angstrom^2 == PREFACTOR / (M_d[amu] * E[eV])
    prefactor = HBAR**2 * 1.0e16 / (2.0 * AMU * EV)       # Angstrom^2 * amu * eV
    masses = np.asarray(mesh_data.masses_amu, dtype=float)            # (N_atoms,)

    scal = (wt * two_n_plus_1 / E)[:, None] * (prefactor / masses)[None, :]  # (M,Na)
    # Chunked accumulation bounds the (chunk, Na, 3, 3) complex intermediate
    # (~0.3 GB per 2^18 modes for a 9-atom cell), so dense meshes of large
    # cells cannot spike memory; the fixed chunk size keeps the summation
    # order deterministic run to run.
    U = np.zeros((n_atoms, 3, 3), dtype=float)
    chunk = 1 << 18
    for start in range(0, eig.shape[0], chunk):
        eig_c = eig[start:start + chunk]
        R = np.real(eig_c[:, :, :, None] * np.conj(eig_c[:, :, None, :]))
        U += np.einsum('md,mdij->dij', scal[start:start + chunk], R,
                       optimize=True)
    U /= n_total
    return 0.5 * (U + U.transpose(0, 2, 1))               # symmetrize i<->j


def compute_thermal_displacement_matrices(mesh_data, temperature_k):
    """Compute thermal displacement matrices U_ij in Angstrom^2.

    Always the explicit per-q vectorized mode sum
    (thermal_displacement_matrices_perq), for three reasons. Consistency:
    on a Gamma-CONTAINING mesh a single global frequency floor (the Gamma
    tier, as in phonopy's ``ThermalDisplacementMatrices`` API) would drop
    the off-Gamma modes in [1 ueV, GAMMA_ACOUSTIC_FLOOR] that the
    one-phonon sum keeps, whereas the per-q two-tier floor keeps the DW
    tensor consistent with the one-phonon mode set. Speed: the einsum sum
    is vectorized over all modes (~1 s on a 40^3 mesh of a 9-atom cell,
    where a per-q Python loop takes ~a minute). Independence: it needs
    only the mesh arrays — no phonopy import — so reconstructed
    PhonopyMeshData objects work too. On a Gamma-free mesh the per-q floor
    coincides with phonopy's global ``freq_min`` and the result matches
    phonopy's ``ThermalDisplacementMatrices`` to ~1e-7 A^2 (equivalence
    pinned by tests/test_tdm_perq_floor.py).
    """
    return thermal_displacement_matrices_perq(mesh_data, temperature_k)


def thermal_displacements_to_f_matrix(thermal_mats_ang2, awr_by_atom, tev):
    """Convert phonopy U_ij tensors to IRMA's dimensionless F-matrix.

    IRMA's internal non-cubic Debye-Waller tensor is defined by

        2W(kappa_hat) = alpha * (kappa_hat . F . kappa_hat),

    with ``alpha = (ħ² kappa²) / (2 A m_n kT)``.

    Phonopy's ``thermal_displacement_matrices`` return

        U_ij = <u_i u_j>

    in Angstrom^2. Matching the two conventions gives

        F = A * kT * U / (ħ² / 2m_n),

    where ``A`` is the atom mass ratio to the neutron and ``kT`` must use the
    same energy units as ``ħ² / 2m_n``. Here we use meV and Angstrom.
    """
    thermal_mats = np.asarray(thermal_mats_ang2, dtype=float)
    awr = np.asarray(awr_by_atom, dtype=float)
    if thermal_mats.shape[:1] != awr.shape[:1]:
        raise ValueError(
            "thermal_mats_ang2 and awr_by_atom must have matching atom counts"
        )

    kT_meV = float(tev) * 1000.0
    scale = (awr * kT_meV / HBAR2_OVER_2MN_MEV_A2)[:, np.newaxis, np.newaxis]
    return thermal_mats * scale


def compute_dos_tensor(mesh_data, freq_max_ev, n_freq, sigma_ev=None,
                       chunk_size=5000):
    """Compute anisotropic DOS tensor from phonopy eigenvectors.

    Returns the 3×3 partial DOS tensor per atom on a uniform energy grid:

        ρ_{d,ij}(ε_k) = (1/N_total) Σ_{q,ν} Re[e_{d,i,ν}·conj(e_{d,j,ν})]
                         × w_{q,ν} × G(ε_k − ε_{q,ν}, σ)

    where G(x, σ) = exp(−x²/2σ²) / (σ√2π) is the Gaussian kernel and
    N_total = Σ_{q,ν} w_{q,ν} is the total BZ weight.

    The scalar partial DOS g_d(ε) = (1/3)Σ_i ρ_{d,ii}(ε) satisfies
    ∫g_d(ε)dε ≈ 1 (one per atom, consistent with tbeta=1 in LEAPR).

    Parameters
    ----------
    mesh_data : PhonopyMeshData
    freq_max_ev : float
        Maximum energy [eV] for the DOS grid.
    n_freq : int
        Number of points on the uniform energy grid [0, freq_max_ev].
    sigma_ev : float or None
        Gaussian smearing width [eV]. Default: 2 × grid spacing.
    chunk_size : int
        Modes processed per batch (memory control). Default 5000 gives
        ~50 MB peak usage for a 40×40×40 mesh with 4 atoms.

    Returns
    -------
    dos_tensor : ndarray, shape (N_atoms, 3, 3, n_freq), float64
        DOS tensor in eV⁻¹.
    energy_grid : ndarray, shape (n_freq,), float64
        Uniform energy grid in eV from 0 to freq_max_ev.
    """
    n_atoms = mesh_data.n_atoms
    n_q = mesh_data.n_qpoints
    n_branches = mesh_data.n_branches
    n_total = float(np.sum(mesh_data.weights))

    delta_ev = freq_max_ev / max(n_freq - 1, 1)
    if sigma_ev is None or sigma_ev <= 0.0:
        sigma_ev = 2.0 * delta_ev

    energy_grid = np.linspace(0.0, freq_max_ev, n_freq)
    dos_tensor = np.zeros((n_atoms, 3, 3, n_freq), dtype=np.float64)

    # Flatten (q, ν) → mode index
    freq_flat = mesh_data.frequencies_ev.flatten()               # (N_modes,)
    weights_flat = np.repeat(mesh_data.weights, n_branches)      # (N_modes,)
    eigs_flat = mesh_data.eigenvectors.reshape(
        n_q * n_branches, n_atoms, 3)                            # (N_modes, N_atoms, 3)

    # Filter: keep only positive-frequency modes above the two-tier mode
    # floor (Goldstone guard at Gamma, 1 ueV overflow guard elsewhere).
    # Imaginary/negative modes are unphysical for the DOS tensor and should
    # not be broadened onto the positive energy grid.
    valid_mask = mode_floor_mask(
        freq_flat * 1.0e3, mesh_data.qpoints, n_branches,
        mesh_data.min_phonon_energy_mev)
    freq_valid = freq_flat[valid_mask]
    wt_valid = weights_flat[valid_mask].astype(np.float64)
    eigs_valid = eigs_flat[valid_mask]                           # (N_valid, N_atoms, 3)

    n_valid = int(np.sum(valid_mask))
    n_skipped = len(freq_flat) - n_valid
    print(f"  DOS tensor: {n_valid} valid modes, {n_skipped} skipped "
          f"(below the mode floor or imaginary); "
          f"σ = {sigma_ev*1000:.2f} meV, Δε = {delta_ev*1000:.2f} meV", flush=True)

    inv_gauss_norm = 1.0 / (sigma_ev * np.sqrt(2.0 * np.pi))
    two_sig2 = 2.0 * sigma_ev**2

    for start_idx in range(0, n_valid, chunk_size):
        end_idx = min(start_idx + chunk_size, n_valid)

        freq_c = freq_valid[start_idx:end_idx]   # (C,)
        wt_c = wt_valid[start_idx:end_idx]        # (C,)
        eig_c = eigs_valid[start_idx:end_idx]     # (C, N_atoms, 3)

        # Outer products: R[m,d,i,j] = Re[eig[m,d,i] * conj(eig[m,d,j])]
        # eig_c shape: (C, N_atoms, 3)
        R = np.real(
            eig_c[:, :, :, np.newaxis] *
            np.conj(eig_c[:, :, np.newaxis, :])
        )  # (C, N_atoms, 3, 3)

        # Gaussian kernel: G[m, k] = exp(-(ε_k - ε_m)² / 2σ²) × norm × w_m
        diff = energy_grid[np.newaxis, :] - freq_c[:, np.newaxis]  # (C, n_freq)
        G = np.exp(-diff**2 / two_sig2) * inv_gauss_norm             # (C, n_freq)
        G *= wt_c[:, np.newaxis]                                      # weight each mode

        # Accumulate: dos_tensor[d,i,j,k] += Σ_m R[m,d,i,j] * G[m,k]
        dos_tensor += np.einsum('mdij,mk->dijk', R, G, optimize=True)

    # Normalise by total BZ weight
    dos_tensor /= n_total

    # Symmetrise: ρ_{d,ij} should be real and symmetric (i↔j) by construction,
    # but floating-point asymmetry from complex eigenvectors may introduce tiny
    # off-diagonal imaginary residuals that were already taken as Re[…] above.
    dos_tensor = 0.5 * (dos_tensor + dos_tensor.transpose(0, 2, 1, 3))

    return dos_tensor, energy_grid
