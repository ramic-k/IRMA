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

import inspect

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
#
# UNITS. phonopy does NOT keep cells in Angstrom. Each calculator interface
# has a native length unit (phonopy.physical_units.get_calculator_physical_units:
# `distance_to_A`), and the readers keep the cell in THAT unit -- the Quantum
# ESPRESSO reader even converts an Angstrom CELL_PARAMETERS block *to* bohr
# (phonopy/interface/qe.py, `factor = 1.0 / Bohr`). phonopy.load builds its
# Phonopy object straight from the yaml's `unit_cell` with no conversion
# (phonopy/cui/load.py reads the units table only for `factor` and
# `nac_factor`); `distance_to_A` is applied only by the structure-file
# CONVERTER (phonopy/interface/calculator.py) and the random-displacement
# writer (phonopy/api_phonopy.py). So a phonopy.yaml written from a qe /
# abinit / elk / siesta / wien2k / DFTB+ / TURBOMOLE / fleur / abacus / qlm run
# carries `physical_unit: length: "au"` and a cell in BOHR, and
# `phonopy.load(...).primitive.cell` hands that bohr cell straight back.
#
# IRMA is Angstrom throughout, so every phonopy load path converts the cell
# ONCE, here, through `angstrom_primitive()`. Downstream modules
# (noncubic_inelastic_context, noncubic_engine, irma.spectra, irma.ncrystal)
# then genuinely receive Angstrom and never need to know about calculator
# units -- the `_ang` suffixes they use are true.
#
# WHAT IS NOT CONVERTED, because phonopy already reports it in fixed units for
# every calculator:
#   - frequencies: phonopy applies the calculator's own `factor` (VaspToTHz,
#     PwscfToTHz, ...) inside the dynamical-matrix solve, so `mesh.frequencies`
#     is THz everywhere. This is also why the LIVE phonopy object must keep its
#     native-unit cell: `angstrom_primitive` returns a converted COPY of the
#     geometry and never mutates `phonon.primitive`.
#   - masses: amu everywhere (there is no mass factor in the units table, and
#     phonopy skips the yaml's `atomic_mass` key when cross-checking units).
#   - scaled (fractional) positions and mesh q-points: dimensionless.
#   - eigenvectors: unit-norm, dimensionless; their Cartesian directions are
#     invariant under a uniform cell rescale.
#   - thermal-displacement tensors U_ij: built here from eigenvectors, masses
#     and mode energies with an explicit hbar^2/(2 M E) constant -- the cell
#     never enters, so they are Angstrom^2 for every calculator already.

_ANGSTROM_UNIT_NAMES = ("angstrom", "angstroms", "ang", "a")
_BOHR_UNIT_NAMES = ("au", "a.u.", "bohr", "bohrs")


def phonopy_calculator_length_units(calculator):
    """``(factor_to_angstrom, unit_name)`` for a phonopy calculator name.

    ``calculator`` is phonopy's interface mode (``None`` == vasp). Reads
    phonopy's own table, so a new calculator interface is picked up
    automatically. Supports the phonopy 4 dataclass and the phonopy 2/3
    dict form of the same table.
    """
    try:
        from phonopy.physical_units import get_calculator_physical_units as _get
    except ImportError:                                  # phonopy 2/3
        from phonopy.interface.calculator import (
            get_default_physical_units as _get)
    units = _get(calculator)
    if isinstance(units, dict):
        return (float(units["distance_to_A"]),
                str(units.get("length_unit", "") or ""))
    return (float(units.distance_to_A),
            str(getattr(units, "length_unit", "") or ""))


def _length_unit_factor_from_name(unit_name):
    """Angstrom per unit for a phonopy ``length_unit`` string, or None.

    phonopy only ever writes ``"angstrom"`` or ``"au"``; the spellings below
    are the tolerant superset. The bohr number is read out of phonopy's own
    calculator table (``qe`` is ``au`` in every phonopy version) rather than
    hardcoded, so there is exactly one numeric authority for the conversion.
    """
    name = (unit_name or "").strip().lower()
    if name in _ANGSTROM_UNIT_NAMES:
        return 1.0
    if name in _BOHR_UNIT_NAMES:
        return phonopy_calculator_length_units("qe")[0]
    return None


def phonopy_yaml_length_unit(phonopy_yaml_path):
    """The ``physical_unit: length:`` value recorded in the yaml, or None.

    Cheap streaming scan (no YAML parse, compressed files supported), so a
    yaml with embedded force constants is not materialized to answer a
    one-key question. Returns ``None`` when the file records no length unit
    (hand-written yamls and very old phonopy versions); the calculator table
    is then the authority.
    """
    in_block = False
    with _open_phonopy_yaml(phonopy_yaml_path, errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if not line[0].isspace():
                in_block = line.startswith("physical_unit:")
                continue
            if in_block:
                stripped = line.strip()
                if stripped.startswith("length:"):
                    return stripped.split(":", 1)[1].strip().strip("\"'")
    return None


def phonopy_model_length_to_angstrom(phonopy_yaml_path, calculator):
    """``(factor, unit_name)``: multiply this model's lengths to get Angstrom.

    The yaml's own ``physical_unit: length:`` wins when it names a unit we
    recognize -- it is what the writer declared the numbers to be, it needs no
    phonopy import, and it stays right even if a future phonopy retunes its
    table. Otherwise the calculator table is the authority (hand-written and
    pre-``physical_unit`` yamls record nothing).

    Raises ``ValueError`` for the two genuinely unusable cases: a recorded unit
    name we cannot map to a factor, and a calculator interface phonopy's own
    table does not know (phonopy raises that one; it is re-raised with the file
    named).
    """
    recorded = phonopy_yaml_length_unit(phonopy_yaml_path)
    if recorded is not None:
        factor = _length_unit_factor_from_name(recorded)
        if factor is None:
            raise ValueError(
                f"{phonopy_yaml_path}: the file records 'physical_unit: "
                f"length: {recorded}', which is not a length unit IRMA "
                f"recognizes (expected 'angstrom' or 'au'). Refusing to guess "
                f"a conversion factor -- fix the file, or rewrite the model "
                f"through phonopy's structure converter.")
        return factor, recorded.strip()
    try:
        return phonopy_calculator_length_units(calculator)
    except ValueError as exc:
        raise ValueError(
            f"{phonopy_yaml_path}: phonopy has no physical-units entry for "
            f"the '{calculator}' calculator interface recorded in this file, "
            f"so the length unit of its cell is unknown ({exc}).") from exc


@dataclass
class AngstromPrimitiveCell:
    """A phonopy primitive cell with its lattice converted to Angstrom.

    Stands in for phonopy's ``Primitive`` wherever IRMA reads GEOMETRY only
    (``cell``, ``scaled_positions``, ``symbols``, ``masses``) -- the mode-1/2
    model context stores one of these so every consumer downstream of the load
    gets Angstrom without knowing that calculator units exist. It deliberately
    is NOT a phonopy object: the live ``phonon.primitive`` must keep its
    native-unit cell, because phonopy's dynamical matrix pairs that cell with
    force constants in the same native units and with the calculator's
    frequency ``factor``.
    """

    cell: np.ndarray                # (3, 3) row vectors, ANGSTROM
    scaled_positions: np.ndarray    # (N_sites, 3) fractional
    symbols: List[str]
    masses: np.ndarray              # (N_sites,) amu
    length_to_angstrom: float = 1.0
    length_unit: str = "angstrom"


def angstrom_primitive(phonon, phonopy_yaml_path) -> AngstromPrimitiveCell:
    """Geometry of a loaded phonopy object's primitive cell, lattice in Angstrom.

    THE single unit-conversion point for the whole package: every IRMA path
    that needs a phonopy cell goes through here, so a bohr-native model
    (qe/abinit/siesta/wien2k/...) and its Angstrom twin produce identical
    downstream physics. ``phonon`` is left untouched.
    """
    factor, unit_name = phonopy_model_length_to_angstrom(
        phonopy_yaml_path, getattr(phonon, "calculator", None))
    prim = phonon.primitive
    if hasattr(prim, "get_scaled_positions"):            # older phonopy API
        scaled = np.array(prim.get_scaled_positions(), dtype=float)
        symbols = [str(s) for s in prim.get_chemical_symbols()]
        masses = np.array(prim.get_masses(), dtype=float)
        cell = np.array(prim.get_cell(), dtype=float)
    else:
        scaled = np.array(prim.scaled_positions, dtype=float)
        symbols = [str(s) for s in prim.symbols]
        masses = np.array(prim.masses, dtype=float)
        cell = np.array(prim.cell, dtype=float)
    if factor != 1.0:
        # Printed only when it fires, so Angstrom models keep byte-identical logs.
        print(f"  Converting the phonopy cell from {unit_name} to Angstrom "
              f"(1 {unit_name} = {factor:.9f} A)", flush=True)
        cell = cell * factor
    return AngstromPrimitiveCell(
        cell=cell,
        scaled_positions=scaled,
        symbols=symbols,
        masses=masses,
        length_to_angstrom=float(factor),
        length_unit=str(unit_name or "angstrom"),
    )


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

    No force constants are read (``produce_fc=False``, no explicit FC
    path, isolated cwd), so this is a cheap geometry-only load; NAC is off
    for the same reason. phonopy is imported lazily, so callers that never
    click this path (classic decks) need not have it installed.

    A model whose cell is in the calculator's native unit (bohr, for
    qe/abinit/siesta/wien2k/...) is CONVERTED to Angstrom, not refused: the
    mode-1/2 engine converts at the same boundary, so the filled Card 6c/6d
    describe the same cell the engine will run.

    Raises ``ImportError`` without phonopy, ``ValueError`` for an unsafe
    yaml or an unresolvable length unit, and whatever phonopy raises for an
    unreadable file.
    """
    import os

    try:
        import phonopy
    except ImportError:
        raise ImportError(
            "phonopy is required to read a phonopy model's crystal "
            "structure. Install with: pip install phonopy")

    path = os.path.abspath(str(phonopy_yaml_path))

    # TRUST BOUNDARY (SEC-1): the path can name any file the user picked;
    # scan for code-executing YAML tags BEFORE phonopy's unsafe loader runs.
    reject_unsafe_phonopy_yaml(path)

    load_params = inspect.signature(phonopy.load).parameters
    kwargs = dict(pinned_primitive_matrix_kwargs(path))
    if "produce_fc" in load_params:
        # Geometry only: never build force constants for a structure read.
        kwargs["produce_fc"] = False
    if "log_level" in load_params:
        kwargs["log_level"] = 0
    # isolated_phonopy_cwd: phonopy probes the process cwd for
    # FORCE_CONSTANTS / FORCE_SETS / BORN; the structure it reports must
    # depend only on the named yaml, never on where IRMA runs.
    with isolated_phonopy_cwd():
        ph = phonopy.load(phonopy_yaml=path, is_nac=False, **kwargs)

    prim = angstrom_primitive(ph, path)
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


def mode_floor_mask(energies_mev, qpoints, n_branches):
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
    return energies_mev > floor


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
                      force_constants_filename=None, force_sets_filename=None):
    """Load phonon eigenvectors from phonopy and return a PhonopyMeshData.

    Force constants: explicit caller paths win; otherwise they are read from
    the yaml itself if embedded, else from the same directory as
    phonopy_yaml_path (force_constants.hdf5, FORCE_CONSTANTS, then
    FORCE_SETS, in that order) -- the same priority the mode-1/2 engine
    context applies.

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
    import os

    try:
        import phonopy
    except ImportError:
        raise ImportError(
            "phonopy is required for non-cubic inelastic calculations. "
            "Install with: pip install phonopy")


    # Absolutize before the isolated_phonopy_cwd pin below.
    phonopy_yaml_path = os.path.abspath(str(phonopy_yaml_path))
    if born_path is not None:
        born_path = os.path.abspath(str(born_path))

    # TRUST BOUNDARY (SEC-1): refuse a phonopy.yaml carrying code-executing
    # YAML tags BEFORE phonopy's unsafe loader parses it. This is the widest
    # entry point -- Card 6f / iel=10 decks and irma.spectra.dos_from_phonopy
    # all funnel through here.
    reject_unsafe_phonopy_yaml(phonopy_yaml_path)

    # Locate the force constants belonging to the named model: an explicit
    # caller path wins; else discovery (embedded in the yaml, else hdf5 >
    # text FORCE_CONSTANTS > FORCE_SETS next to it). phonopy.load() would
    # otherwise search the CURRENT WORKING DIRECTORY, which must never
    # decide the model.
    if force_constants_filename is None and force_sets_filename is None:
        fc_kwargs = resolve_force_constants_source(phonopy_yaml_path)
    else:
        fc_kwargs = {}
        if force_constants_filename is not None:
            fc_kwargs["force_constants_filename"] = os.path.abspath(
                str(force_constants_filename))
        if force_sets_filename is not None:
            fc_kwargs["force_sets_filename"] = os.path.abspath(
                str(force_sets_filename))

    # NAC policy: an explicit BORN path (Card 6f use_born=1) wins; otherwise
    # NAC embedded in the named phonopy.yaml is honored; phonopy's fallback
    # of auto-reading ./BORN from the process working directory is disabled
    # (is_nac=False) so the applied physics never depends on the run cwd.
    embeds_nac = phonopy_yaml_embeds_nac(phonopy_yaml_path)
    # Force phonopy's C/OpenMP backend, NOT the Rust `phonors` backend. phonopy>=4
    # defaults to lang="Rust" when `phonors` is installed; phonors uses a rayon
    # global thread pool that limit_native_threads_to_one() cannot pin. The
    # mode-1/2 engine runs a spawn ProcessPool (Card 6f ncpu>1), so an
    # uncontrolled rayon pool in every worker oversubscribes the cores (and
    # deadlocked workers outright under the fork start method used previously;
    # observed on phonopy 4.2.1 + Python 3.14 macOS). The C backend is pinned to
    # one OMP thread by limit_native_threads_to_one() and is safe in the worker
    # pool. Only pass `lang` when this phonopy accepts it: older phonopy predates
    # phonors and has no such parameter (would raise TypeError otherwise).
    backend_kwargs = (
        {"lang": "C"}
        if "lang" in inspect.signature(phonopy.load).parameters
        else {}
    )
    try:
        # isolated_phonopy_cwd: phonopy probes the cwd for FORCE_CONSTANTS/
        # FORCE_SETS/BORN even when explicit paths are given; with every
        # wanted source passed as an absolute path, the fallbacks must find
        # nothing.
        with isolated_phonopy_cwd():
            ph = phonopy.load(
                phonopy_yaml=phonopy_yaml_path,
                born_filename=born_path,
                is_nac=(born_path is not None) or embeds_nac,
                **pinned_primitive_matrix_kwargs(phonopy_yaml_path),
                **backend_kwargs,
                **fc_kwargs,
            )
    except Exception as exc:
        if born_path is not None:
            # BORN was explicitly requested: degrading silently to a NAC-less
            # calculation would corrupt the physics unnoticed.
            raise RuntimeError(
                f"BORN corrections were requested but the BORN file could "
                f"not be loaded from {born_path}: {exc}"
            ) from exc
        raise
    if born_path is not None:
        if ph.nac_params is None:
            raise RuntimeError(
                f"BORN corrections were requested but phonopy.load returned "
                f"no NAC parameters from {born_path}."
            )
        print(f"  Loaded NAC parameters from {born_path}", flush=True)
    elif embeds_nac:
        if ph.nac_params is None:
            # The yaml declares NAC keys that phonopy could not parse into
            # parameters. Refusing to guess: continuing NAC-less (or letting
            # phonopy fall back to a ./BORN in the cwd) would silently change
            # the physics.
            raise RuntimeError(
                f"{phonopy_yaml_path} embeds NAC keys but phonopy could not "
                f"parse NAC parameters from them; fix the file or supply an "
                f"explicit BORN file (Card 6f use_born=1)."
            )
        print(f"  Using NAC parameters embedded in {phonopy_yaml_path}", flush=True)

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
    # ph.mesh (Mesh object) replaces get_mesh_dict(): the accessor is
    # deprecated in phonopy 4 and the property exists on every supported
    # version with identical array contents.
    mesh_obj = ph.mesh

    # Frequencies: (N_q, N_branches) in THz → eV (negative = imaginary)
    freq_ev = mesh_obj.frequencies * THZ_TO_EV  # (N_q, N_branches)

    n_q = freq_ev.shape[0]
    n_branches = freq_ev.shape[1]
    # n_atoms: support both old API (get_number_of_atoms) and new (len/masses)
    prim = ph.primitive
    if hasattr(prim, 'get_number_of_atoms'):
        n_atoms = prim.get_number_of_atoms()
    else:
        n_atoms = len(prim.masses)

    # Eigenvectors: phonopy uses np.linalg.eigh convention throughout.
    # Mesh.eigenvectors, shape (N_q, 3*N_atoms, N_branches):
    #   axis 0 = q-points
    #   axis 1 = component index 3*d+i  (ATOM-FIRST: d=atom index, i=0,1,2 for x,y,z)
    #   axis 2 = mode/branch index ν
    # i.e., eigs_raw[q, 3*d+i, ν] = component of atom d, direction i, mode ν.
    # To get our target shape (N_q, N_branches, N_atoms, 3) — eigs[q, ν, d, i]:
    #   transpose(0,2,1) swaps axes 1↔2 → (N_q, N_branches, 3*N_atoms)
    #   reshape to (N_q, N_branches, N_atoms, 3) using atom-first split.
    eigs_raw = mesh_obj.eigenvectors
    expected_last = n_atoms * 3

    if eigs_raw.ndim == 3 and eigs_raw.shape == (n_q, n_branches, expected_last):
        # SANITY GUARD, not a layout discriminator: phonopy's Mesh
        # eigenvectors are ALWAYS the np.linalg.eigh 3D array
        # (N_q, 3*N_atoms, N_branches), atom-first. Because
        # n_branches == expected_last == 3*n_atoms for a phonon problem, this
        # shape test cannot distinguish atom-first from a hypothetical
        # mode-first layout; it just confirms the expected dimensions before the
        # (correct, for the sole atom-first convention) transpose below.
        eigs = eigs_raw.transpose(0, 2, 1).reshape(n_q, n_branches, n_atoms, 3)
    elif eigs_raw.ndim == 3 and eigs_raw.shape == (expected_last, n_branches, n_q):
        # Older phonopy: (3*N_atoms, N_branches, N_q), atom-first.
        # .T → (N_q, N_branches, 3*N_atoms) with [q, ν, 3*d+i] indexing.
        eigs = eigs_raw.T.reshape(n_q, n_branches, n_atoms, 3)
    else:
        raise ValueError(
            f"Unexpected eigenvector shape from phonopy: {eigs_raw.shape}. "
            f"Expected (N_q={n_q}, N_branches={n_branches}, N_atoms*3={expected_last}).")

    weights = mesh_obj.weights
    if weights is None:
        weights = np.ones(n_q, dtype=int)
    # Support both old API (get_masses/get_chemical_symbols) and new (masses/symbols)
    if hasattr(prim, 'get_masses'):
        masses_amu = np.array(prim.get_masses(), dtype=float)
        symbols = list(prim.get_chemical_symbols())
    else:
        masses_amu = np.array(prim.masses, dtype=float)
        symbols = list(prim.symbols)

    n_total = int(np.sum(weights))
    print(f"  Loaded {n_q} q-points, {n_branches} branches, {n_atoms} atoms "
          f"(total BZ weight = {n_total})", flush=True)

    # Primitive cell geometry.
    # atom_positions: fractional coordinates in primitive cell (N_atoms, 3).
    # UNITS: nothing PhonopyMeshData carries has a length dimension -- q-points
    # and positions are fractional, frequencies are THz->eV, masses are amu,
    # eigenvectors are unit-norm -- so this loader needs no cell conversion.
    # The paths that DO need the lattice call angstrom_primitive() instead.
    # NOTE: the reciprocal lattice is intentionally NOT stored here. The live
    # noncubic path derives its own b-matrix WITHOUT the 2π factor (see
    # noncubic_inelastic_context: rec_lat_no_2pi = inv(primitive.cell), and the
    # /2π applied in noncubic_engine when going to reduced coords). A stored
    # 2π·(A⁻¹)ᵀ field would carry the opposite convention — a latent trap.
    if hasattr(prim, 'get_scaled_positions'):
        atom_positions = np.array(prim.get_scaled_positions(), dtype=float)
    else:
        atom_positions = np.array(prim.scaled_positions, dtype=float)

    return PhonopyMeshData(
        qpoints=mesh_obj.qpoints,
        frequencies_ev=freq_ev,
        eigenvectors=eigs,
        weights=weights,
        masses_amu=masses_amu,
        atom_symbols=symbols,
        atom_positions=atom_positions,
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

    valid = mode_floor_mask(freq_flat * 1.0e3, mesh_data.qpoints, n_branches)
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
        freq_flat * 1.0e3, mesh_data.qpoints, n_branches)
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
