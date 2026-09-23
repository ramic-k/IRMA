"""phonopy_io.py — Load phonopy mesh data and compute anisotropic DOS tensors.

The DOS tensor of atom d and Cartesian directions i, j, used by the
inelastic_mode=1/2 paths:

    ρ_{d,ij}(ε_k) = (1/N_total) Σ_{q,ν} Re[e_{d,i,ν}(q) · conj(e_{d,j,ν}(q))]
                     × w_{q,ν} × G(ε_k − ε_{q,ν}, σ)

with e the unit-norm phonopy eigenvector, w the Brillouin-zone weight,
N_total = Σ w and G(x, σ) = exp(−x²/2σ²)/(σ√2π). Because Σ_{d,i}|e|² = 1, the
scalar partial DOS g_d = (1/3)Σ_i ρ_{d,ii} integrates to 1 (LEAPR's tbeta=1).
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
    """Run phonopy in a fresh empty directory: phonopy.load probes
    ./FORCE_CONSTANTS, ./FORCE_SETS and ./BORN even when explicit paths are
    given, and what gets loaded must not depend on where IRMA runs."""
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
        shutil.rmtree(scratch, ignore_errors=True)


@contextmanager
def openmp_unpinned_serial_setup():
    """Let phonopy's OpenMP q-point loop use all cores for a serial run_mesh.

    Per-q results are unchanged; BLAS stays pinned to one thread.
    """
    import os

    import threadpoolctl
    with threadpoolctl.threadpool_limits(limits=os.cpu_count() or 1, user_api="openmp"):
        yield


def _open_phonopy_yaml(phonopy_yaml_path, mode="rt"):
    """Open a (possibly compressed) phonopy.yaml, as text by default."""
    import bz2
    import gzip
    import lzma
    import os

    path = str(phonopy_yaml_path)
    ext = os.path.splitext(path)[1].lower()
    opener = {".gz": gzip.open, ".xz": lzma.open, ".lzma": lzma.open,
              ".bz2": bz2.open}.get(ext, open)
    return opener(path, mode)


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
    """Reject a phonopy.yaml that could execute code in phonopy's loader.

    phonopy parses with PyYAML's unsafe loader, where a python tag executes
    code. phonopy writes UTF-8 with no tags and no directives, so this
    refuses any '!!' or '!<' tag (every global tag starts with one, whatever
    %-escapes follow), any %TAG directive, and a UTF-16 byte-order mark,
    which would hide the text from this scan. Call it before any phonopy.load.
    """
    path = str(phonopy_yaml_path)
    with _open_phonopy_yaml(path, "rb") as f:
        for lineno, line in enumerate(f, 1):
            if lineno == 1 and line.startswith((b"\xff\xfe", b"\xfe\xff")):
                raise ValueError(
                    f"{path}: refusing to parse: a UTF-16 file (phonopy "
                    f"writes UTF-8)")
            if (b"!!" in line or b"!<" in line
                    or line.lstrip(b"\xef\xbb\xbf").startswith(b"%TAG")):
                text = line.decode("utf-8", "replace").strip()[:80]
                raise ValueError(
                    f"{path}:{lineno}: refusing to parse: a YAML tag or %TAG "
                    f"directive, which phonopy's loader can execute (phonopy "
                    f"writes neither): {text!r}")


def phonopy_yaml_embeds_nac(phonopy_yaml_path) -> bool:
    """True if the phonopy.yaml carries embedded NAC parameters: the current
    ``nac:`` block or the older flat keys phonopy still accepts."""
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
    """kwargs pinning phonopy.load's primitive cell.

    phonopy 4 defaults an omitted ``primitive_matrix`` to "auto" (a guessed
    primitive cell). When the yaml stores a matrix pass nothing, since an
    explicit value would override it; otherwise pass "P" (the identity).
    """
    if _phonopy_yaml_has_top_level_key(phonopy_yaml_path,
                                       ("primitive_matrix:",)):
        return {}
    return {"primitive_matrix": "P"}


def resolve_force_constants_source(phonopy_yaml_path) -> dict:
    """phonopy.load kwargs for the named model's force constants.

    Embedded in the yaml ({}), else force_constants.hdf5, FORCE_CONSTANTS or
    FORCE_SETS next to it. Raises when there is none, since phonopy's fallback
    of searching the working directory is never used.
    """
    import os

    if phonopy_yaml_embeds_force_constants(phonopy_yaml_path):
        print(f"  Using force constants embedded in {phonopy_yaml_path}", flush=True)
        return {}
    yaml_dir = os.path.dirname(os.path.abspath(str(phonopy_yaml_path)))
    for name, key in (("force_constants.hdf5", "force_constants_filename"),
                      ("FORCE_CONSTANTS", "force_constants_filename"),
                      ("FORCE_SETS", "force_sets_filename")):
        path = os.path.join(yaml_dir, name)
        if os.path.exists(path):
            print(f"  Using {name}: {path}", flush=True)
            return {key: path}
    raise FileNotFoundError(
        f"No force constants found for {phonopy_yaml_path} (embedded, "
        f"force_constants.hdf5, FORCE_CONSTANTS, FORCE_SETS)")


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
    explicit paths are given. Force constants: embedded in the yaml, else the
    explicit paths, else ``resolve_force_constants_source``. Embedded force
    constants win (phonopy's rule); an explicit path given with them is not
    used, and a warning says so. NAC: the explicit BORN file, else NAC
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
            if phonopy_yaml_embeds_force_constants(path):
                unused = ", ".join(str(p) for p in (force_constants_filename,
                                                    force_sets_filename)
                                   if p is not None)
                print(f"WARNING: {path} embeds force constants, which phonopy "
                      f"uses; {unused} is not used", flush=True)
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


def warn_dynamic_instability(frequencies_ev, qpoints) -> None:
    """Warn about imaginary modes: any away from Gamma (a real instability),
    or ones at Gamma beyond the acoustic-sum-rule noise."""
    freqs = np.asarray(frequencies_ev, dtype=float)
    qpts = np.asarray(qpoints, dtype=float)
    is_gamma_q = np.all(np.abs(qpts) < 1.0e-9, axis=1)
    nongamma_imag = (freqs < -1.0e-6) & ~is_gamma_q[:, None]
    if np.any(nongamma_imag):
        print(f"WARNING: {int(np.count_nonzero(nongamma_imag))} imaginary phonon "
             f"mode(s) at non-Gamma q-points (min "
             f"{np.min(freqs[nongamma_imag]) * 1000:.3f} meV) -- the phonon model "
             "is DYNAMICALLY UNSTABLE (a finite-wavevector soft mode); review the "
             "structure and force constants before trusting this evaluation.")
    elif np.any(freqs < -1.0e-4):
        print("WARNING: imaginary modes near Gamma (min "
             f"{np.min(freqs) * 1000:.2f} meV) beyond the acoustic-sum-rule noise "
             "band; they are excluded from the grid, but review the phonon model.")


# A phonon-energy cutoff that moves the mean-square displacement trace by more
# than this fraction is a warning (a 5 meV cutoff on graphite removes 0.13% of
# the modes but 29% of the displacement).
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
    if min_phonon_energy_mev > 0.0:
        floor = np.maximum(floor, min_phonon_energy_mev)
    return energies_mev > floor


def coherent_mode_mask(energies_mev, q_folded, min_phonon_energy_mev=0.0):
    """Modes the coherent one-phonon term keeps; ``energies_mev`` is
    (points, branches) at the (points, 3) folded reduced q.

    Without a cutoff every positive mode is kept (the established coherent
    behaviour); with one, ``mode_floor_mask`` applies as for the mesh sums.
    """
    energies = np.asarray(energies_mev, dtype=float)
    if min_phonon_energy_mev <= 0.0:
        return energies > 0.0
    n_points, n_branches = energies.shape
    mask = mode_floor_mask(energies.reshape(-1), np.asarray(q_folded, dtype=float),
                           n_branches, min_phonon_energy_mev)
    return mask.reshape(n_points, n_branches)


def phonon_cutoff_summary(mesh_data, temperature_K):
    """What a user phonon-energy cutoff removed from ``mesh_data`` at one temperature.

    A dictionary with the modes the automatic floors already drop, the extra
    modes the cutoff removes and their weight, the per-atom DOS trace deficit,
    and the mean-square displacement trace per atom with and without the
    cutoff. Displacements weight modes as 1/E^2, so removing few modes can
    remove much of the Debye-Waller exponent; ``warn`` is True when the trace
    moved by more than ``PHONON_CUTOFF_WARN_FRACTION``.
    """
    from dataclasses import replace
    cutoff = float(mesh_data.min_phonon_energy_mev)
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
    u_after = compute_thermal_displacement_matrices(mesh_data, temperature_K)
    u_before = compute_thermal_displacement_matrices(
        replace(mesh_data, min_phonon_energy_mev=0.0), temperature_K)
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
        f"({100.0 * s['mean_trace_u_change']:+.2f}%).",
    ]
    if s["warn"]:
        lines.append(
            f"WARNING: the phonon-energy cutoff changed the mean-square "
            f"displacement by {100.0 * abs(s['mean_trace_u_change']):.1f}% (above "
            f"{100.0 * PHONON_CUTOFF_WARN_FRACTION:g}%); the Debye-Waller factors "
            f"describe a truncated phonon model.")
    return lines


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
        yaml-adjacent discovery, but not force constants embedded in the
        yaml (those win, with a warning).

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

    # The full mesh: an irreducible q-point stands for one star member only,
    # so per-atom tensors would miss the atom permutations of the others.
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


def compute_thermal_displacement_matrices(mesh_data, temperature_k):
    """Thermal-displacement tensors U_ij [Angstrom^2], one per atom:

        U_ij(d) = (1/N) Σ_{q,nu} (hbar^2 / (2 M_d E_qnu)) coth(E/2kT)
                                  · Re[e_{d,i,nu}(q) e*_{d,j,nu}(q)]

    The eigenvectors are unit-norm, so the 1/M_d factor is explicit. The modes
    are those of the two-tier ``mode_floor_mask`` used by the DOS tensor and
    the one-phonon sums; on a Gamma-free mesh this matches phonopy's
    thermal-displacement matrices.
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

    kT_meV = float(tev) * 1000.0
    scale = (awr * kT_meV / HBAR2_OVER_2MN_MEV_A2)[:, np.newaxis, np.newaxis]
    return thermal_mats * scale


def compute_dos_tensor(mesh_data, freq_max_ev, n_freq, sigma_ev=None,
                       chunk_size=5000):
    """The 3×3 partial DOS tensor per atom on a uniform energy grid (see the
    module docstring).

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
        Modes processed per batch.

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

    freq_flat = mesh_data.frequencies_ev.flatten()               # (N_modes,)
    weights_flat = np.repeat(mesh_data.weights, n_branches)      # (N_modes,)
    eigs_flat = mesh_data.eigenvectors.reshape(
        n_q * n_branches, n_atoms, 3)                            # (N_modes, N_atoms, 3)

    # Modes above the two-tier floor; imaginary modes are dropped.
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

        R = np.real(
            eig_c[:, :, :, np.newaxis] *
            np.conj(eig_c[:, :, np.newaxis, :])
        )  # (C, N_atoms, 3, 3)

        diff = energy_grid[np.newaxis, :] - freq_c[:, np.newaxis]  # (C, n_freq)
        G = np.exp(-diff**2 / two_sig2) * inv_gauss_norm             # (C, n_freq)
        G *= wt_c[:, np.newaxis]                                      # weight each mode

        dos_tensor += np.einsum('mdij,mk->dijk', R, G, optimize=True)

    dos_tensor /= n_total
    # R is exactly symmetric; this is a no-op safeguard.
    dos_tensor = 0.5 * (dos_tensor + dos_tensor.transpose(0, 2, 1, 3))

    return dos_tensor, energy_grid
