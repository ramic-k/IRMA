"""``SpectraConfig`` -- the neutron-scattering input contract.

A *separate* structured file (YAML primary; TOML and JSON also accepted), not
the positional LEAPR deck, because instrument geometry, the physical meV energy
axis, ``dQ``, and the elastic-line switch are not first-class deck fields. One
config maps one-to-one onto a ``compute_spectrum`` call, so a GUI-saved
config and a CLI run are byte-for-byte the same calculation.

Public surface: the dataclass tree (``SpectraConfig`` + the four section
dataclasses + ``Scatterer``), ``load(path)`` / ``dump(cfg, path)``,
``validate(cfg)``, and ``run_spectra(cfg, ...)`` (the thin config ->
``compute_spectrum`` adapter). ``PyYAML`` is imported lazily so
``import irma.spectra.config`` works without it (JSON/TOML still load).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

# VISION preset defaults (kept in lockstep with irma.spectra.sqe)
from irma.spectra.sqe import VISION_EF_MEV, VISION_SIGMA_COEFFS, sigma_of_E

_GEOMETRIES = ("vision", "indirect", "direct")
_COMBINES = ("mean", "sum")
_RESOLUTION_SHAPES = ("gaussian", "lorentzian")
_RESOLUTION_MODELS = ("poly", "chopper")
_CHOPPER_KEYS = ("instrument", "package", "frequency")
_OUTPUT_MODES = ("cuts", "map")
_CUT_BYS = ("angles", "q")


class SpectraConfigError(ValueError):
    """Raised on a malformed or physically inconsistent ``SpectraConfig``."""


def _require_exact_int(value, name, minimum):
    """Exact-integer config field: reject bool, non-finite, and FRACTIONAL
    values instead of truncating them.

    ``int(3.9)`` silently becomes 3 (and a fractional site index silently
    selects a different site), so integer-coded fields demand exact integral
    values; numeric strings ("6") and integral floats (6.0) still pass."""
    if isinstance(value, bool):
        raise SpectraConfigError(
            f"{name} must be an integer >= {minimum}, not a boolean ({value!r})")
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        raise SpectraConfigError(
            f"{name} must be an integer >= {minimum}, got {value!r}")
    if not math.isfinite(fvalue) or fvalue != int(fvalue):
        raise SpectraConfigError(
            f"{name} must be an EXACT integer >= {minimum} (fractional values "
            f"are not truncated), got {value!r}")
    ivalue = int(fvalue)
    if ivalue < minimum:
        raise SpectraConfigError(
            f"{name} must be an integer >= {minimum}, got {value!r}")
    return ivalue


def _strict_bool(name, val):
    """Strictly-typed boolean config field: bool (or 0/1) only.

    ``bool('off')`` is True, so silently accepting strings would flip physics
    switches on a ``--set physics.elastic=off`` or YAML quoting slip. numpy
    bool/integer scalars count as bools: programmatic configs legitimately
    carry ``np.True_`` or ``np.int64(1)`` from array-derived values."""
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)) and int(val) in (0, 1):
        return bool(val)
    raise SpectraConfigError(f"{name} must be true/false, got {val!r}")


# -----------------------------------------------------------------------------
# dataclass tree
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class Scatterer:
    """Optional per-site scattering override (else read from the phonopy yaml).

    ``dos_file`` + ``dos_unit`` + ``multiplicity`` are used only by the DOS-based
    path (inelastic_mode=0): the partial phonon-DOS file for this species, the
    frequency unit of its first column ('meV', 'eV', 'cm-1', 'THz'), and the
    number of its atoms in the represented cell. ``positions`` (fractional
    coordinates, one [x,y,z] per atom of this species) feeds the mode-0 coherent
    Bragg peaks together with ``material.lattice`` and ``b_coh_fm``.
    """
    symbol: str
    sigma_bound_b: Optional[float] = None
    awr: Optional[float] = None
    b_coh_fm: Optional[float] = None
    sigma_inc_b: Optional[float] = None
    dos_file: Optional[str] = None
    dos_unit: str = "meV"
    multiplicity: int = 1
    positions: Optional[list] = None       # fractional [x,y,z] sites (mode-0 elastic)


@dataclasses.dataclass
class MaterialConfig:
    """The phonon model -- references the SAME artifacts the ENDF side names.

    ``phonopy_yaml`` drives the eigenvector path (inelastic_mode 1/2); the
    DOS-based path (inelastic_mode 0) leaves it unset and instead reads each
    scatterer's ``dos_file``. ``lattice`` = [a, b, c, alpha, beta, gamma]
    (Angstrom, degrees) supplies the unit cell for the mode-0 coherent-elastic
    Bragg peaks (the iel=10 analogue); it is unused by modes 1/2.
    """
    phonopy_yaml: Optional[str] = None
    born: Optional[str] = None
    force_constants: Optional[str] = None
    force_sets: Optional[str] = None
    mesh: list = dataclasses.field(default_factory=lambda: [40, 40, 40])
    temperature_K: float = 296.0
    scatterers: list = dataclasses.field(default_factory=list)
    lattice: Optional[list] = None         # [a,b,c,alpha,beta,gamma] (mode-0 elastic)


# Readable aliases for the integer inelastic_mode. The integers stay
# canonical; the aliases exist because the deck-side and spectra-side mode 0
# mean different calculations (isotropic-DW cubic vs the DOS forward model)
# and a name is harder to misread across the two surfaces than a bare 0.
INELASTIC_MODE_ALIASES = {"dos": 0, "incoherent": 1, "coherent": 2}


@dataclasses.dataclass
class PhysicsConfig:
    """Physics selections: inelastic mode, DOS source, phonon order, sampling."""

    inelastic_mode: Union[int, str] = 2     # 0/'dos' | 1/'incoherent' | 2/'coherent' (exact 1ph; the validated default)
    dos_source: str = "file"                # mode-0 DOS origin: file (per-scatterer) | phonopy
    max_phonon_order: Union[int, str] = "auto"   # int >= 1 or "auto"
    min_phonon_energy_meV: float = 0.0      # modes 1/2: remove modes <= this (0 = automatic floors only)
    n_directions: int = 10000
    multiphonon_directions: int = 1000
    jobs: Optional[int] = None          # worker processes; null = auto (all CPU cores)
    elastic: bool = True
    elastic_kind: str = "both"              # both | coherent (Bragg) | incoherent (DW)
    # isotropic (trace/3 scalar W' per species) | directional (orientation-
    # averaged <exp(-Q^2 uhat.U.uhat)> per atom; modes 1/2 only -- needs the
    # engine's displacement tensors). Same option name as the NCrystal export.
    incoherent_elastic_mode: str = "isotropic"
    elastic_from_tape: Optional[str] = None
    include_energy_gain: bool = True
    gain_side: str = "direct"               # direct (explicit Bose factors) | detailed_balance (mirror)
    kinematic_kf_ki: bool = False


@dataclasses.dataclass
class GridConfig:
    """Energy/Q grid of the computed spectrum or map."""

    e_min_meV: float = 0.0
    e_max_meV: float = 250.0
    de_meV: float = 0.5                     # one default everywhere: CLI + GUI agree
    dq_max_invA: float = 0.05               # S(Q,E) Q-support spacing (ONE meaning)
    # 2-D map Q-axis maximum (GUI map launch + `irma spectra map`). 1-D spectrum
    # runs derive their Q support from the instrument locus and do not read it.
    q_max_invA: Optional[float] = None
    q_pad_invA: float = 0.5


@dataclasses.dataclass
class InstrumentConfig:
    """Instrument geometry, fixed energy, angle coverage, and resolution."""

    geometry: str = "vision"                # vision | indirect | direct
    e_fixed_meV: float = VISION_EF_MEV      # Ef (vision/indirect) OR Ei (direct)
    angles_deg: Optional[list] = None       # None -> preset fills it (vision)
    q_cuts: Optional[list] = None           # optional constant-|Q| cuts [1/A]
    bank_halfwidth_deg: float = 5.0
    sigma_coeffs: Optional[list] = None     # None -> vision preset poly
    resolution_shape: str = "gaussian"      # gaussian (sigma poly) | lorentzian (HWHM poly)
    resolution_model: str = "poly"          # poly (sigma poly) | chopper (auto, any PyChop instrument)
    chopper_spec: Optional[dict] = None     # instrument,package,frequency for resolution_model=chopper
    combine: str = "mean"
    # Output selection (the Run button's product). 'cuts' = 1-D spectra;
    # 'map' = the dense 2-D S(Q,E) heatmap (any geometry -- the map path
    # dispatches direct/indirect kinematics itself). Map-only fields below are
    # ignored for 'cuts'.
    output_mode: str = "cuts"               # cuts | map
    cut_by: str = "angles"                  # angles | q   (fixed-cuts sub-mode)
    cut_dq_invA: Optional[float] = None     # constant-Q cut band width [1/A]; None -> thin slice
    map_coverage_deg: Optional[list] = None  # [2th_min, 2th_max] mask band; None -> instrument default
    map_mask: bool = True                   # mask the 2-D map to the accessible (q,E) band
    export_components: bool = False         # save/plot inel+elastic breakdown (else total only)


@dataclasses.dataclass
class SpectraConfig:
    """Complete forward-model configuration (material, physics, grid,
    instrument), loadable from YAML/TOML/JSON."""

    material: MaterialConfig
    physics: PhysicsConfig = dataclasses.field(default_factory=PhysicsConfig)
    grid: GridConfig = dataclasses.field(default_factory=GridConfig)
    instrument: InstrumentConfig = dataclasses.field(default_factory=InstrumentConfig)

    # -- (de)serialization ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "SpectraConfig":
        """Build (and type-normalize) a config from a parsed mapping.

        Unknown keys raise -- a typo'd field is a config error, not a silent
        no-op. Sequence fields are coerced to lists of the right scalar so the
        round-trip ``load(dump(cfg)) == cfg`` is exact.
        """
        if not isinstance(d, dict):
            raise SpectraConfigError(f"config root must be a mapping, got {type(d).__name__}")
        # Unknown SECTIONS must fail as loudly as unknown keys inside one: a
        # typo'd 'instrumnet:' would otherwise be dropped wholesale and the run
        # would complete (exit 0) with the default instrument instead.
        unknown = set(d) - {"material", "physics", "grid", "instrument"}
        if unknown:
            raise SpectraConfigError(
                f"unknown config section(s) {sorted(unknown)}; allowed "
                "sections: ['material', 'physics', 'grid', 'instrument'] -- "
                "check for a typo (a misspelled section would otherwise be "
                "silently ignored)")
        if "material" not in d:
            raise SpectraConfigError("config is missing the required 'material' section")
        # Every top-level section must BE a mapping before dict() touches it:
        # a scalar (e.g. --set material=5, or a fat-fingered YAML section)
        # used to reach dict(5) and escape as a raw TypeError instead of the
        # clean schema error (review CLI-2).
        for _sec in ("material", "physics", "grid", "instrument"):
            if _sec in d and not isinstance(d[_sec], dict):
                raise SpectraConfigError(
                    f"config section {_sec!r} must be a mapping of keys, got "
                    f"{type(d[_sec]).__name__} ({d[_sec]!r})")
        _scat_raw = d["material"].get("scatterers", [])
        if not isinstance(_scat_raw, (list, tuple)):
            raise SpectraConfigError(
                "material.scatterers must be a list of mappings, got "
                f"{type(_scat_raw).__name__}")
        for _s in _scat_raw:
            if not isinstance(_s, dict):
                raise SpectraConfigError(
                    "each material.scatterers entry must be a mapping, got "
                    f"{type(_s).__name__} ({_s!r})")
        mat_d = dict(d["material"])
        scat = [_build(Scatterer, dict(s)) for s in mat_d.pop("scatterers", [])]
        material = _build(MaterialConfig, mat_d)
        material.scatterers = scat
        material.mesh = [_require_exact_int(x, "material.mesh entries", 1)
                         for x in material.mesh]
        if material.lattice is not None:
            material.lattice = [float(x) for x in material.lattice]
        for s in material.scatterers:
            if s.positions is not None:
                s.positions = [[float(x) for x in p] for p in s.positions]
        physics = _build(PhysicsConfig, dict(d.get("physics", {})))
        if isinstance(physics.max_phonon_order, str) and physics.max_phonon_order != "auto":
            raise SpectraConfigError(
                f"physics.max_phonon_order must be an int or 'auto', got "
                f"{physics.max_phonon_order!r}")
        try:
            from irma.core.phonopy_io import validate_min_phonon_energy_mev
            physics.min_phonon_energy_meV = validate_min_phonon_energy_mev(
                physics.min_phonon_energy_meV)
        except (TypeError, ValueError) as exc:
            raise SpectraConfigError(
                f"physics.min_phonon_energy_meV must be a finite number >= 0 "
                f"(meV), got {physics.min_phonon_energy_meV!r}: {exc}") from None
        physics.elastic = _strict_bool("physics.elastic", physics.elastic)
        physics.include_energy_gain = _strict_bool(
            "physics.include_energy_gain", physics.include_energy_gain)
        physics.kinematic_kf_ki = _strict_bool(
            "physics.kinematic_kf_ki", physics.kinematic_kf_ki)
        grid = _build(GridConfig, dict(d.get("grid", {})))
        instrument = _build(InstrumentConfig, dict(d.get("instrument", {})))
        if instrument.angles_deg is not None:
            instrument.angles_deg = [float(a) for a in instrument.angles_deg]
        if instrument.q_cuts is not None:
            instrument.q_cuts = [float(q) for q in instrument.q_cuts]
        if instrument.sigma_coeffs is not None:
            instrument.sigma_coeffs = [float(c) for c in instrument.sigma_coeffs]
        if instrument.map_coverage_deg is not None:
            instrument.map_coverage_deg = [float(a) for a in instrument.map_coverage_deg]
        if instrument.cut_dq_invA is not None:
            instrument.cut_dq_invA = float(instrument.cut_dq_invA)
        instrument.map_mask = _strict_bool("instrument.map_mask", instrument.map_mask)
        instrument.export_components = _strict_bool(
            "instrument.export_components", instrument.export_components)
        if instrument.chopper_spec is not None:
            cs = dict(instrument.chopper_spec)
            # Accept PyChop's [resolution, frame] frequency list (the resolution
            # disk -- first element -- sets the burst) as well as a bare scalar,
            # matching chopper_resolution._norm_frequency; positivity is enforced
            # by validate(), so a missing/zero value is tolerated here.
            freq = cs.get("frequency", 0.0)
            if isinstance(freq, (list, tuple)):
                freq = freq[0] if len(freq) else 0.0
            # None (a YAML `null` / an omitted CLI flag) normalizes to the empty
            # value, so validate() reports the missing key BY NAME instead of a
            # raw float(None) TypeError or the baffling "unknown chopper
            # instrument 'None'".
            instrument.chopper_spec = {
                "instrument": str(cs.get("instrument") or ""),
                "package": str(cs.get("package") or ""),
                "frequency": float(freq if freq is not None else 0.0),
            }
        return cls(material=material, physics=physics, grid=grid, instrument=instrument)

    def to_dict(self) -> dict:
        """Plain nested dict (canonical section order) for serialization."""
        return {
            "material": dataclasses.asdict(self.material),
            "physics": dataclasses.asdict(self.physics),
            "grid": dataclasses.asdict(self.grid),
            "instrument": dataclasses.asdict(self.instrument),
        }


def _build(dc_type, d: dict):
    """Construct a dataclass from a dict, rejecting unknown keys."""
    fields = {f.name for f in dataclasses.fields(dc_type)}
    unknown = set(d) - fields
    if unknown:
        raise SpectraConfigError(
            f"{dc_type.__name__}: unknown field(s) {sorted(unknown)}; "
            f"allowed: {sorted(fields)}")
    return dc_type(**d)


# -----------------------------------------------------------------------------
# load / dump  (YAML primary; TOML + JSON accepted; canonical dump is YAML)
# -----------------------------------------------------------------------------
def _parse(text: str, suffix: str) -> dict:
    """Parse config text by file suffix (.yaml/.yml, .toml, .json)."""
    s = suffix.lower()
    if s == ".json":
        return json.loads(text)
    if s == ".toml":
        import tomllib                      # stdlib (the >= 3.11 floor guarantees it)
        return tomllib.loads(text)
    # default + .yaml/.yml
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SpectraConfigError(
            "PyYAML is required to read YAML configs; install pyyaml or use "
            "a .json / .toml config") from e
    return yaml.safe_load(text)


def load(path, validate_cfg: bool = True) -> SpectraConfig:
    """Load a SpectraConfig from YAML/TOML/JSON (by extension) and validate it."""
    path = Path(path)
    cfg = SpectraConfig.from_dict(_parse(path.read_text(), path.suffix))
    if validate_cfg:
        validate(cfg)
    return cfg


def dump(cfg: SpectraConfig, path) -> Path:
    """Serialize a SpectraConfig (canonical YAML; .json by extension)."""
    path = Path(path)
    data = cfg.to_dict()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(data, indent=2))
        return path
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SpectraConfigError(
            "PyYAML is required to write YAML configs; dump to a .json path "
            "instead") from e
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return path


# -----------------------------------------------------------------------------
# validation
# -----------------------------------------------------------------------------
def validate(cfg: SpectraConfig) -> SpectraConfig:
    """Raise SpectraConfigError on a malformed/inconsistent config; else return it."""
    m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument

    # Normalize a string inelastic_mode alias to its canonical integer FIRST,
    # so every later check (and every consumer of the validated config) sees
    # an int. Unknown strings fail here with the alias table in the message.
    if isinstance(p.inelastic_mode, str):
        _alias = p.inelastic_mode.strip().lower()
        if _alias not in INELASTIC_MODE_ALIASES:
            raise SpectraConfigError(
                f"physics.inelastic_mode must be 0, 1, 2 or one of "
                f"{sorted(INELASTIC_MODE_ALIASES)}, got {p.inelastic_mode!r}")
        p.inelastic_mode = INELASTIC_MODE_ALIASES[_alias]

    # Phonon-model source: inelastic_mode 0 reads per-species DOS files; modes
    # 1/2 read the phonopy yaml (+ mesh).
    if p.inelastic_mode == 0:
        if p.dos_source not in ("file", "phonopy"):
            raise SpectraConfigError(
                f"physics.dos_source must be 'file' or 'phonopy', got {p.dos_source!r}")
        if not m.scatterers:
            raise SpectraConfigError(
                "inelastic_mode=0 (DOS) requires material.scatterers, each with "
                "awr and sigma_bound_b (+ a dos_file when dos_source='file')")
        # dos_source='phonopy' reads the partial DOS from the phonopy.yaml + mesh;
        # 'file' reads each scatterer's 2-column dos_file.
        if p.dos_source == "phonopy":
            if not m.phonopy_yaml:
                raise SpectraConfigError(
                    "inelastic_mode=0 with dos_source='phonopy' needs material.phonopy_yaml")
            if len(m.mesh) != 3 or any(int(x) <= 0 for x in m.mesh):
                raise SpectraConfigError(
                    f"material.mesh must be three positive ints, got {m.mesh}")
        for s in m.scatterers:
            if p.dos_source == "file" and not s.dos_file:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r}: inelastic_mode=0 (dos_source='file') "
                    "needs a dos_file")
            if s.awr is None or s.sigma_bound_b is None:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r}: inelastic_mode=0 needs awr and "
                    "sigma_bound_b")
            s.multiplicity = _require_exact_int(
                s.multiplicity, f"scatterer {s.symbol!r} multiplicity", 1)
            if s.positions is not None and int(s.multiplicity) != len(s.positions):
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r}: multiplicity ({s.multiplicity}) "
                    f"must equal the number of positions ({len(s.positions)})")
            # The incoherent elastic line sums every species' channel; a missing
            # sigma_inc_b would silently zero that species, so demand an explicit
            # value (0.0 declares "no incoherent channel" deliberately).
            if (p.elastic and p.elastic_kind in ("both", "incoherent")
                    and s.sigma_inc_b is None):
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r}: the incoherent elastic line "
                    f"(elastic_kind={p.elastic_kind!r}) needs an explicit "
                    "sigma_inc_b for every scatterer -- set 0.0 to declare no "
                    "incoherent channel, or use elastic_kind='coherent'")
        # Optional mode-0 coherent elastic: a crystal (lattice + per-species
        # fractional sites + b_coh_fm) enables the Bragg peaks. Validate it only
        # when a lattice is supplied; without it the coherent channel is skipped
        # (elastic_kind='incoherent' never needs the crystal).
        if m.lattice is not None:
            if len(m.lattice) != 6 or not all(math.isfinite(x) for x in m.lattice):
                raise SpectraConfigError(
                    "material.lattice must be six finite numbers "
                    "[a,b,c,alpha,beta,gamma], got " f"{m.lattice}")
            a_, b_, c_, al_, be_, ga_ = (float(x) for x in m.lattice)
            if not (a_ > 0 and b_ > 0 and c_ > 0):
                raise SpectraConfigError(
                    f"material.lattice a,b,c must be > 0, got {(a_, b_, c_)}")
            if not all(0.0 < ang < 180.0 for ang in (al_, be_, ga_)):
                raise SpectraConfigError(
                    f"material.lattice angles must be in (0,180) deg, got "
                    f"{(al_, be_, ga_)}")
            # positions + b_coh_fm are required only when the coherent peaks are
            # actually wanted; an incoherent-only line runs lattice-free.
            want_coh = p.elastic and p.elastic_kind in ("both", "coherent")
            for s in m.scatterers:
                if want_coh and (not s.positions or s.b_coh_fm is None):
                    raise SpectraConfigError(
                        f"scatterer {s.symbol!r}: the mode-0 coherent elastic "
                        "line (material.lattice set) needs positions + b_coh_fm "
                        "for every scatterer")
                for pos in (s.positions or ()):
                    if len(pos) != 3 or not all(math.isfinite(x) for x in pos):
                        raise SpectraConfigError(
                            f"scatterer {s.symbol!r}: each position must be three "
                            f"finite fractional coords, got {pos}")
    else:
        if not m.phonopy_yaml:
            raise SpectraConfigError(
                "material.phonopy_yaml is required for inelastic_mode 1/2")
        if len(m.mesh) != 3 or any(int(x) <= 0 for x in m.mesh):
            raise SpectraConfigError(
                f"material.mesh must be three positive ints, got {m.mesh}")
    if m.temperature_K <= 0:
        raise SpectraConfigError(f"material.temperature_K must be > 0, got {m.temperature_K}")

    # Per-scatterer numeric overrides are only presence-checked above (mode-0
    # requires awr/sigma_bound_b to EXIST; tape-free elastic requires b_coh_fm),
    # so a NaN/Inf -- or a negative mass or cross section -- would flow straight
    # into the engine and every relational guard downstream would pass silently
    # (NaN compares False). Validate the VALUES here, wherever present.
    # b_coh_fm is a signed scattering length and may legitimately be negative
    # (H, Ti, Mn, ...): finite is its only constraint.
    for s in m.scatterers:
        if s.awr is not None and not (math.isfinite(s.awr) and s.awr > 0):
            raise SpectraConfigError(
                f"scatterer {s.symbol!r}: awr must be finite and > 0, got {s.awr}")
        if s.sigma_bound_b is not None and not (
                math.isfinite(s.sigma_bound_b) and s.sigma_bound_b > 0):
            raise SpectraConfigError(
                f"scatterer {s.symbol!r}: sigma_bound_b must be finite and > 0, "
                f"got {s.sigma_bound_b}")
        if s.b_coh_fm is not None and not math.isfinite(s.b_coh_fm):
            raise SpectraConfigError(
                f"scatterer {s.symbol!r}: b_coh_fm must be finite, got {s.b_coh_fm}")
        if s.sigma_inc_b is not None and not (
                math.isfinite(s.sigma_inc_b) and s.sigma_inc_b >= 0):
            raise SpectraConfigError(
                f"scatterer {s.symbol!r}: sigma_inc_b must be finite and >= 0, "
                f"got {s.sigma_inc_b}")

    if p.inelastic_mode not in (0, 1, 2):
        raise SpectraConfigError(
            f"physics.inelastic_mode must be 0 (DOS), 1 or 2, got {p.inelastic_mode}")
    if p.max_phonon_order != "auto":
        # bool is an int subclass: max_phonon_order=True used to pass the
        # isinstance check and become order 1 (review SP-1).
        p.max_phonon_order = _require_exact_int(
            p.max_phonon_order, "physics.max_phonon_order", 1)
    # Count fields demand EXACT integers: n_directions=3.9 used to truncate
    # to 3 at use time (review SP-1).
    p.n_directions = _require_exact_int(
        p.n_directions, "physics.n_directions", 1)
    p.multiphonon_directions = _require_exact_int(
        p.multiphonon_directions, "physics.multiphonon_directions", 1)
    if p.elastic_kind not in ("both", "coherent", "incoherent"):
        raise SpectraConfigError(
            "physics.elastic_kind must be 'both', 'coherent', or 'incoherent', "
            f"got {p.elastic_kind!r}")
    if p.incoherent_elastic_mode not in ("isotropic", "directional"):
        raise SpectraConfigError(
            "physics.incoherent_elastic_mode must be 'isotropic' or "
            f"'directional', got {p.incoherent_elastic_mode!r}")
    if p.incoherent_elastic_mode == "directional" and p.inelastic_mode == 0:
        raise SpectraConfigError(
            "physics.incoherent_elastic_mode = 'directional' requires "
            "inelastic_mode 1 or 2 (the DOS path has no displacement tensors)")
    if p.gain_side not in ("direct", "detailed_balance"):
        raise SpectraConfigError(
            "physics.gain_side must be 'direct' (explicit Bose factors) or "
            f"'detailed_balance' (mirror), got {p.gain_side!r}")
    if p.jobs is not None:
        # jobs=0.9 used to pass this check, then int() to 0 at use time and
        # silently fall back to ALL cores -- the opposite of the user's
        # throttle (review SP-1).
        p.jobs = _require_exact_int(p.jobs, "physics.jobs", 1)

    # NaN/Inf bypass every relational guard below (all comparisons against NaN are
    # False, and +Inf <= 0 is False), so reject non-finite scalars up front.
    _finite = {
        "material.temperature_K": m.temperature_K,
        "grid.e_min_meV": g.e_min_meV, "grid.e_max_meV": g.e_max_meV,
        "grid.de_meV": g.de_meV, "grid.dq_max_invA": g.dq_max_invA,
        "grid.q_pad_invA": g.q_pad_invA, "instrument.e_fixed_meV": ins.e_fixed_meV,
        "instrument.bank_halfwidth_deg": ins.bank_halfwidth_deg,
    }
    if g.q_max_invA is not None:
        _finite["grid.q_max_invA"] = g.q_max_invA
    for _name, _val in _finite.items():
        if not math.isfinite(_val):
            raise SpectraConfigError(f"{_name} must be finite, got {_val}")

    if g.de_meV <= 0 or g.dq_max_invA <= 0:
        raise SpectraConfigError("grid.de_meV and grid.dq_max_invA must be > 0")
    if g.e_max_meV <= g.e_min_meV:
        raise SpectraConfigError(
            f"grid.e_max_meV ({g.e_max_meV}) must exceed e_min_meV ({g.e_min_meV})")
    if g.q_max_invA is not None and g.q_max_invA <= 0:
        raise SpectraConfigError(f"grid.q_max_invA must be > 0 or null, got {g.q_max_invA}")
    if g.q_pad_invA < 0:
        raise SpectraConfigError(
            f"grid.q_pad_invA (Q-support padding) must be >= 0, got {g.q_pad_invA}")

    if ins.geometry not in _GEOMETRIES:
        raise SpectraConfigError(
            f"instrument.geometry must be one of {_GEOMETRIES}, got {ins.geometry!r}")
    if ins.combine not in _COMBINES:
        raise SpectraConfigError(
            f"instrument.combine must be one of {_COMBINES}, got {ins.combine!r}")
    # output selection: the 2-D map runs for every geometry (the map path
    # dispatches direct vs indirect kinematics itself), so no geometry guard here
    if ins.output_mode not in _OUTPUT_MODES:
        raise SpectraConfigError(
            f"instrument.output_mode must be one of {_OUTPUT_MODES}, got {ins.output_mode!r}")
    if ins.cut_by not in _CUT_BYS:
        raise SpectraConfigError(
            f"instrument.cut_by must be one of {_CUT_BYS}, got {ins.cut_by!r}")
    if ins.cut_dq_invA is not None and not (
            math.isfinite(ins.cut_dq_invA) and ins.cut_dq_invA > 0):
        raise SpectraConfigError(
            f"instrument.cut_dq_invA (constant-Q cut band) must be finite and "
            f"> 0, or null, got {ins.cut_dq_invA}")
    if ins.map_coverage_deg is not None:
        cov = ins.map_coverage_deg
        if (len(cov) != 2 or not all(math.isfinite(a) for a in cov)
                or not (0.0 < cov[0] < cov[1] < 180.0)):
            raise SpectraConfigError(
                "instrument.map_coverage_deg must be [2th_min, 2th_max] with "
                f"0 < min < max < 180 deg, got {cov}")
    if ins.resolution_shape not in _RESOLUTION_SHAPES:
        raise SpectraConfigError(
            f"instrument.resolution_shape must be one of {_RESOLUTION_SHAPES}, "
            f"got {ins.resolution_shape!r}")
    if ins.resolution_model not in _RESOLUTION_MODELS:
        raise SpectraConfigError(
            f"instrument.resolution_model must be one of {_RESOLUTION_MODELS}, "
            f"got {ins.resolution_model!r}")
    if ins.resolution_model == "chopper" and ins.resolution_shape == "lorentzian":
        # chopper_sigma_of_E returns a Gaussian sigma; feeding it to a Lorentzian
        # kernel would mis-scale the width (HWHM_Lorentz != sigma_Gauss) and use
        # the wrong line shape. Chopper resolution is Gaussian by construction.
        raise SpectraConfigError(
            "instrument.resolution_model='chopper' produces a Gaussian width and "
            "is incompatible with resolution_shape='lorentzian'; use "
            "resolution_shape='gaussian' (or resolution_model='poly').")
    if ins.resolution_model == "chopper" and ins.geometry != "direct":
        raise SpectraConfigError(
            "instrument.resolution_model='chopper' is a direct-geometry chopper "
            f"model; it requires geometry='direct' (got '{ins.geometry}'). "
            "Use resolution_model='poly' otherwise.")
    if ins.resolution_model == "chopper":
        if not ins.chopper_spec:
            raise SpectraConfigError(
                "resolution_model='chopper' requires instrument.chopper_spec "
                f"with {list(_CHOPPER_KEYS)}")
        missing = [k for k in _CHOPPER_KEYS if not ins.chopper_spec.get(k)]
        if missing:
            raise SpectraConfigError(
                f"instrument.chopper_spec is missing {missing}; "
                "resolution_model='chopper' needs all of instrument, package "
                "and frequency (CLI flags: --chopper-instrument, "
                "--chopper-package, --chopper-frequency)")
        from irma.spectra.chopper_resolution import instrument_geometry
        try:                            # validates instrument + package names
            instrument_geometry(ins.chopper_spec["instrument"],
                                ins.chopper_spec["package"])
        except ValueError as exc:
            raise SpectraConfigError(str(exc))
        _freq = float(ins.chopper_spec["frequency"])
        if not (math.isfinite(_freq) and _freq > 0):
            raise SpectraConfigError(
                f"chopper_spec.frequency must be finite and > 0 Hz, got {_freq}")
    if ins.e_fixed_meV <= 0:
        raise SpectraConfigError(f"instrument.e_fixed_meV must be > 0, got {ins.e_fixed_meV}")
    # NaN/Inf are already rejected by the finite whitelist above; this is the
    # sign/zero guard.
    if not (ins.bank_halfwidth_deg > 0):
        raise SpectraConfigError(
            f"instrument.bank_halfwidth_deg must be > 0, got {ins.bank_halfwidth_deg}")
    if ins.geometry != "vision" and ins.output_mode != "map" and not ins.angles_deg:
        raise SpectraConfigError(
            f"instrument.angles_deg is required for geometry '{ins.geometry}' "
            "fixed cuts (only the vision preset auto-fills its banks; the 2-D map "
            "uses the detector coverage instead)")
    if ins.angles_deg is not None and any(not (0.0 < a < 180.0) for a in ins.angles_deg):
        raise SpectraConfigError("instrument.angles_deg must lie strictly in (0, 180) deg")
    # q_cuts were only float-cast in from_dict; a NaN or non-positive |Q| would
    # otherwise reach the cut extractor and produce an empty/garbage spectrum.
    if ins.q_cuts is not None and any(
            not (math.isfinite(q) and q > 0) for q in ins.q_cuts):
        raise SpectraConfigError(
            f"instrument.q_cuts must be finite and > 0 (|Q| values [1/A]), "
            f"got {ins.q_cuts}")
    if ins.geometry == "direct":
        if ins.cut_by not in ("angles", "q"):
            raise SpectraConfigError(
                f"instrument.cut_by must be 'angles' or 'q', got {ins.cut_by!r}")
        # cut_by='q' takes constant-|Q| cuts only; without q_cuts the run would
        # silently produce an empty spectrum, so require them explicitly. Only
        # relevant when producing cuts -- the 2-D map ignores cut_by.
        if ins.output_mode != "map" and ins.cut_by == "q" and not ins.q_cuts:
            raise SpectraConfigError(
                "instrument.cut_by='q' requires instrument.q_cuts (constant-|Q| "
                "values [1/A]); none were given")
    # direct geometry: the energy-LOSS limit is the incident energy. In
    # down-scatter the neutron's final energy Ef = Ei - E >= 0, so the maximum
    # energy LOSS is Ei -- modes above Ei are kinematically inaccessible in
    # energy loss (you must raise Ei to reach them). Energy GAIN (up-scatter)
    # is the E < 0 side (e_min) and is NOT capped, so this only bounds e_max.
    if ins.geometry == "direct" and g.e_max_meV >= ins.e_fixed_meV:
        raise SpectraConfigError(
            f"direct geometry: grid.e_max_meV ({g.e_max_meV}) must be < the "
            f"incident energy Ei ({ins.e_fixed_meV} meV). In down-scatter the "
            f"neutron loses at most Ei (final energy >= 0), so an energy-loss "
            f"mode above {ins.e_fixed_meV} meV cannot be reached -- raise Ei to "
            f"cover it, or lower e_max. (Up-scatter / energy gain is the "
            f"e_min < 0 side and is not limited by Ei.)")
    # resolution sigma-poly must be finite and stay positive across the evaluated
    # loss range. Evaluate through the SAME sigma_of_E the forward model uses, so
    # a short coefficient list (e.g. [c0] or [c0, c1] -- the GUI emits these from
    # its trailing-blank trim, and the CLI/YAML accept any length) is zero-padded
    # identically here instead of raising a raw IndexError.
    coeffs = ins.sigma_coeffs if ins.sigma_coeffs is not None else list(VISION_SIGMA_COEFFS)
    if not all(math.isfinite(c) for c in coeffs):
        raise SpectraConfigError(
            f"instrument.sigma_coeffs must be finite (no NaN/Inf), got {coeffs}")
    if len(coeffs) > 3:
        raise SpectraConfigError(
            f"instrument.sigma_coeffs is a quadratic c0,c1,c2 (at most 3 terms); "
            f"got {len(coeffs)} -- extra terms would be silently ignored")
    # The runtime evaluates the poly at |E| (sigma_of_E), so the effective
    # domain is [0, max(e_max, |e_min|)] -- a wide energy-gain window (e_min
    # well below -e_max) probes the poly BEYOND e_max and must be validated
    # there too, or a negative width would be silently clipped downstream.
    # Sample the endpoints AND the parabola vertex: an upward poly (c2>0) has
    # its minimum at E=-c1/(2 c2), so a negative interior dip would otherwise
    # slip past an endpoint-only check.
    c = list(coeffs) + [0.0, 0.0, 0.0]
    e_hi = max(g.e_max_meV, abs(g.e_min_meV))
    samples = [0.0, 0.5 * e_hi, e_hi]
    if c[2] > 0.0:
        vertex = -c[1] / (2.0 * c[2])
        if 0.0 <= vertex <= e_hi:
            samples.append(vertex)
    for E in samples:
        sig = float(sigma_of_E(E, coeffs))
        if sig <= 0:
            raise SpectraConfigError(
                f"instrument.sigma_coeffs gives non-positive resolution sigma "
                f"({sig:.4g} meV) at E={E:.3g} meV; widen/replace the poly")
    return cfg


def check_input_files(cfg: SpectraConfig) -> SpectraConfig:
    """Preflight every path-valued config field that is set; return ``cfg``.

    Raises :class:`SpectraConfigError` naming the responsible FIELD when a
    referenced file does not exist, so a typo'd path fails up front instead of
    surfacing mid-run as a bare "[Errno 2] No such file or directory" with no
    hint which field caused it. Kept OUT of :func:`validate` deliberately: a
    config is schema-valid independent of the local filesystem (configs are
    edited, dumped and round-tripped on machines that do not hold the data);
    the CLI calls this right before a run.
    """
    m, p = cfg.material, cfg.physics
    named = [
        ("material.phonopy_yaml", m.phonopy_yaml),
        ("material.born", m.born),
        ("material.force_constants", m.force_constants),
        ("material.force_sets", m.force_sets),
        ("physics.elastic_from_tape", p.elastic_from_tape),
    ]
    named += [(f"material.scatterers[{i}] ({s.symbol}) dos_file", s.dos_file)
              for i, s in enumerate(m.scatterers)]
    for field, path in named:
        if path and not Path(path).is_file():
            raise SpectraConfigError(
                f"{field}: file not found: {path} (relative paths are "
                "resolved from the current working directory, not from the "
                "config file's location)")
    return cfg


# -----------------------------------------------------------------------------
# config -> compute_spectrum adapter
# -----------------------------------------------------------------------------
def _dos_entry(s, omega_ev, rho, multiplicity):
    """One ``compute_mode0_sqe`` species dict from a scatterer + its (omega, rho)
    partial DOS and atom multiplicity; attaches the elastic-only extras."""
    entry = {
        "symbol": s.symbol,
        "omega_ev": omega_ev,
        "rho": rho,
        "awr": float(s.awr),
        "sigma_bound_b": float(s.sigma_bound_b),
        "multiplicity": int(multiplicity),
    }
    # elastic-only extras (mode-0 coherent Bragg + incoherent DW)
    if s.b_coh_fm is not None:
        entry["b_coh_fm"] = float(s.b_coh_fm)
    if s.sigma_inc_b is not None:
        entry["sigma_inc_b"] = float(s.sigma_inc_b)
    if s.positions is not None:
        entry["positions"] = [[float(x) for x in p] for p in s.positions]
    return entry


def _assemble_dos_species(material, *, dos_source="file", born_path=None):
    """Per-species mode-0 payloads. ``dos_source='file'`` reads each scatterer's
    2-column ``dos_file``; ``'phonopy'`` derives the partial DOS (and per-species
    atom multiplicity) from ``material.phonopy_yaml`` + ``mesh`` and matches it to
    the scatterers by symbol. Assumes ``validate`` has run."""
    if dos_source == "phonopy":
        from irma.spectra.dos_from_phonopy import partial_dos_from_phonopy
        pdos = partial_dos_from_phonopy(material.phonopy_yaml, material.mesh,
                                        born_path=born_path,
                                        force_constants=material.force_constants,
                                        force_sets=material.force_sets)
        by_sym = {d["symbol"]: d for d in pdos}
        scat_syms = [s.symbol for s in material.scatterers]
        # a symbol listed twice would map both scatterers onto the SAME phonopy
        # DOS and double-count it; reject (file mode allows duplicates, phonopy not).
        dupes = sorted({x for x in scat_syms if scat_syms.count(x) > 1})
        if dupes:
            raise SpectraConfigError(
                f"dos_source='phonopy': duplicate scatterer symbol(s) {dupes} map to "
                "one phonopy partial DOS each (ambiguous); give one scatterer per "
                "element, or use dos_source='file' for inequivalent same-element sites")
        # every phonopy species must be represented, else its inelastic (and
        # elastic) contribution is silently lost from the sample total.
        missing = sorted(set(by_sym) - set(scat_syms))
        if missing:
            raise SpectraConfigError(
                f"dos_source='phonopy': the phonopy cell has species {missing} with no "
                "matching scatterer; add a scatterer (awr + sigma_bound_b) for each so "
                "they are not dropped from the sample total")
        species = []
        for s in material.scatterers:
            if s.symbol not in by_sym:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r} is not among the phonopy DOS species "
                    f"{sorted(by_sym)}; check the symbol against the phonopy cell")
            pd = by_sym[s.symbol]
            if s.positions is not None and len(s.positions) != pd["multiplicity"]:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r}: {len(s.positions)} positions but the "
                    f"phonopy cell has {pd['multiplicity']} atoms of this species")
            species.append(_dos_entry(s, pd["omega_ev"], pd["rho"], pd["multiplicity"]))
        return species

    from irma.spectra.dos_io import read_dos_2col
    species = []
    for s in material.scatterers:
        omega_ev, rho = read_dos_2col(s.dos_file, unit=s.dos_unit)
        species.append(_dos_entry(s, omega_ev, rho, s.multiplicity))
    return species


def phonopy_species(path):
    """Distinct atom symbols (first-appearance order) from a phonopy.yaml.

    Best-effort plain-YAML walk (no phonopy import); returns [] when the
    file cannot be parsed, so callers treat the species list as advisory.
    Shared by the modes-1/2 symbol check below and the GUI's
    auto-fill-elements button.
    """
    try:
        import yaml
        with open(path) as fh:
            doc = yaml.safe_load(fh)
    except Exception:
        return []
    if not isinstance(doc, dict):
        return []
    for key in ("primitive_cell", "unit_cell", "supercell"):
        cell = doc.get(key)
        if isinstance(cell, dict) and isinstance(cell.get("points"), list):
            seen = []
            for pt in cell["points"]:
                s = pt.get("symbol") if isinstance(pt, dict) else None
                if s and s not in seen:
                    seen.append(s)
            if seen:
                return seen
    return []


def _check_species_symbols(m, p):
    """Modes 1/2: every scatterer symbol must be a phonopy structure species.

    The engine matches per-site constants by exact chemical symbol and
    silently falls back to built-in defaults for unmatched symbols, so a
    C-13 row on a C structure would hybridize the row's AWR/sigma_bound
    with natural-C site constants. Refuse loudly instead. Advisory: an
    unparseable yaml gives [] and skips the check (the engine reports its
    own load error).
    """
    if p.inelastic_mode not in (1, 2) or not m.phonopy_yaml:
        return
    species = phonopy_species(m.phonopy_yaml)
    if not species:
        return
    bad = sorted({s.symbol for s in m.scatterers} - set(species))
    if bad:
        raise SpectraConfigError(
            f"scatterer symbol(s) {bad} do not match the phonopy structure's "
            f"species {species} (inelastic_mode {p.inelastic_mode} matches "
            "per-site constants by exact chemical symbol). For isotope-"
            "specific constants, use the element symbol with explicit "
            "sigma_bound_b/awr/b_coh_fm/sigma_inc_b values.")


def _elastic_inputs(m, p):
    """Resolve the config's elastic route -> (elastic_model, elastic, scatterers).

    Three routes, shared by the 1-D spectrum (``run_spectra``) and the 2-D map
    (``run_map``) so ``physics.elastic`` means the same thing in both:
      * ``elastic_from_tape``  -> a ready ElasticModel parsed from ENDF MF7/MT2;
      * ``inelastic_mode == 0``-> the DOS-derived tape-free line, built inside
        the forward call from the just-computed isotropic f0 (the coherent
        Bragg peaks when material.lattice + positions + b_coh_fm are given, the
        lattice-free incoherent Debye-Waller line otherwise / in addition --
        the forward model routes by elastic_kind and the crystal's presence);
      * modes 1/2              -> the engine's surfaced elastic_state, which
        needs b_coh_fm + awr per scatterer symbol.
    """
    if not p.elastic:
        return None, False, None
    if p.elastic_from_tape:
        from irma.spectra.elastic import from_endf_mf7mt2
        return (from_endf_mf7mt2(p.elastic_from_tape, T_K=float(m.temperature_K)),
                False, None)
    if p.inelastic_mode == 0:
        return None, True, None
    if not m.scatterers:
        raise SpectraConfigError(
            "physics.elastic=true (tape-free) needs material.scatterers with "
            "b_coh_fm + awr per symbol, or set physics.elastic_from_tape")
    missing = [s.symbol for s in m.scatterers
               if s.b_coh_fm is None or s.awr is None]
    if missing:
        raise SpectraConfigError(
            f"tape-free elastic needs b_coh_fm + awr for scatterers {missing}")
    elastic_scatterers = {
        s.symbol: {"b_coh_fm": s.b_coh_fm,
                   "sigma_inc_b": (s.sigma_inc_b or 0.0), "awr": s.awr}
        for s in m.scatterers}
    return None, True, elastic_scatterers


def run_spectra(cfg: SpectraConfig, *, workdir=None, label=None, progress=print):
    """Run the forward model described by ``cfg`` and return a ``SpectrumResult``.

    Thin, deterministic mapping from the validated config onto
    ``compute_spectrum`` keyword arguments -- the single code path the
    GUI and CLI both funnel through.
    """
    import json

    from irma.spectra.forward import compute_spectrum

    validate(cfg)
    m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument

    # Direct 'fixed cuts' sub-mode: cut_by='q' -> constant-Q cuts only (no
    # angle spectra); otherwise the bank/angle spectra are produced AND any
    # supplied q_cuts are honored alongside them (compute_spectrum supports
    # both simultaneously). Gating cuts on cut_by=='q' would leave the
    # documented --q-cuts flag silently inert: the flag-form CLI has no
    # --cut-by, so cut_by always defaulted to 'angles' there.
    cut_by = ins.cut_by if ins.geometry == "direct" else "angles"
    by_q = (cut_by == "q")
    q_cuts_arg = ins.q_cuts or None
    produce_angles = not by_q

    auto_order = (p.max_phonon_order == "auto")
    max_order = 100 if auto_order else int(p.max_phonon_order)

    # Per-SYMBOL scattering overrides: the engine broadcasts these to every
    # primitive-cell site of that symbol, so a single {symbol: value} entry
    # covers a multi-atom mono-species cell (graphite = 4 C). (Per-SITE arrays
    # would have to match the primitive-atom count, which the config -- keyed by
    # symbol -- does not carry.)  b_coh: fm -> Angstrom (x1e-5).
    _check_species_symbols(m, p)
    b_map = {s.symbol: s.b_coh_fm * 1.0e-5 for s in m.scatterers if s.b_coh_fm is not None}
    inc_map = {s.symbol: s.sigma_inc_b for s in m.scatterers if s.sigma_inc_b is not None}
    sl_json = json.dumps(b_map) if b_map else None
    inc_json = json.dumps(inc_map) if inc_map else None
    sigma_b = (m.scatterers[0].sigma_bound_b if m.scatterers else None)
    awr = (m.scatterers[0].awr if m.scatterers else None)
    if sigma_b is None or awr is None:
        raise SpectraConfigError(
            "material.scatterers must give at least the principal site's "
            "sigma_bound_b and awr (run_spectra needs the bound XS + mass ratio)")

    # inelastic_mode=0 (DOS): assemble the per-species DOS payload (from files or
    # the phonopy.yaml). The eigenvector engine is skipped by compute_spectrum
    # when dos_species is set. A lattice (+ positions + b_coh_fm) enables elastic.
    dos_species = (_assemble_dos_species(m, dos_source=p.dos_source, born_path=m.born)
                   if p.inelastic_mode == 0 else None)
    dos_crystal = (tuple(float(x) for x in m.lattice)
                   if (p.inelastic_mode == 0 and m.lattice is not None) else None)

    elastic_model, elastic_flag, elastic_scatterers = _elastic_inputs(m, p)

    return compute_spectrum(
        geometry=ins.geometry, phonopy_yaml=m.phonopy_yaml,
        temperature_k=float(m.temperature_K), mesh=tuple(m.mesh),
        sab_mass_ratio=float(awr), sab_sigma_barn=float(sigma_b),
        angles_deg=ins.angles_deg, e_fixed_meV=ins.e_fixed_meV, q_cuts=q_cuts_arg,
        cut_dq=ins.cut_dq_invA, produce_angle_spectra=produce_angles,
        dE=g.de_meV, dQ=g.dq_max_invA, e_min=g.e_min_meV, e_max=g.e_max_meV,
        bank_halfwidth_deg=ins.bank_halfwidth_deg, sigma_coeffs=ins.sigma_coeffs,
        resolution_shape=ins.resolution_shape, resolution_model=ins.resolution_model,
        chopper_spec=ins.chopper_spec,
        combine=ins.combine, inelastic_mode=p.inelastic_mode,
        dos_species=dos_species, dos_crystal=dos_crystal,
        num_directions=p.n_directions,
        multiphonon_num_directions=p.multiphonon_directions,
        multiphonon_max_order=max_order, auto_multiphonon_order=auto_order,
        min_phonon_energy_mev=float(p.min_phonon_energy_meV),
        jobs=p.jobs,  # None -> compute_spectrum auto-detects all cores
        force_constants=m.force_constants, force_sets=m.force_sets, born_path=m.born,
        site_scattering_lengths_angstrom=None,
        site_incoherent_cross_sections_barn=None,
        scattering_lengths_json=sl_json, incoherent_cross_sections_json=inc_json,
        elastic_model=elastic_model, elastic=elastic_flag,
        elastic_kind=p.elastic_kind, elastic_scatterers=elastic_scatterers,
        incoherent_elastic_mode=p.incoherent_elastic_mode,
        include_gain=p.include_energy_gain, gain_side=p.gain_side,
        kinematic_factor=p.kinematic_kf_ki, q_pad=g.q_pad_invA,
        workdir=workdir, label=label, progress=progress)


def run_map(cfg, *, q_min=0.0, q_max=None, dQ_map=None, angle_range=None,
            broaden=True, workdir=None, label=None, progress=print):
    """Run a dense 2-D S(Q,E) powder map for ``cfg`` (config -> compute_sqe_map).

    Config-driven defaults (a None argument defers to the config):
      * ``q_max``      -> ``grid.q_max_invA``, else auto-covers the
        kinematic envelope (capped at 40 1/A);
      * ``dQ_map``     -> ``grid.dq_max_invA``;
      * ``angle_range``-> ``instrument.map_coverage_deg``, else the cut
        ``instrument.angles_deg``, else (45, 135).
    The grid uses the config's e_min/e_max/de for energy and q_min/q_max/dQ_map
    for Q. ``angle_range=(2th_min, 2th_max)`` sets the kinematic-envelope overlay.
    """
    import json

    from irma.spectra.forward import compute_sqe_map

    validate(cfg)
    m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument

    # run_map's OWN kwargs are call-site overrides that bypass the config
    # schema (validate() never sees them), so a NaN/Inf/negative value would
    # reach the engine unchecked. Validate them here, naming the kwarg, BEFORE
    # any compute is dispatched.
    for _name, _val in (("q_min", q_min), ("q_max", q_max), ("dQ_map", dQ_map)):
        if _val is not None and not math.isfinite(_val):
            raise SpectraConfigError(f"run_map kwarg {_name} must be finite, got {_val}")
    if q_min < 0:
        raise SpectraConfigError(f"run_map kwarg q_min must be >= 0, got {q_min}")
    if dQ_map is None:
        dQ_map = float(cfg.grid.dq_max_invA)
    if dQ_map <= 0:
        raise SpectraConfigError(f"run_map kwarg dQ_map must be > 0, got {dQ_map}")
    if angle_range is not None:
        # Same physical two-theta contract as instrument.map_coverage_deg:
        # the kwarg route used to accept (-10, 200) and forward it to the
        # engine (review SP-3).
        _ar = list(angle_range)
        if (len(_ar) != 2 or not all(math.isfinite(a) for a in _ar)
                or not (0.0 < _ar[0] < _ar[1] < 180.0)):
            raise SpectraConfigError(
                "run_map kwarg angle_range must be (2theta_min, 2theta_max) "
                f"with 0 < min < max < 180 deg, got {angle_range}")

    auto_order = (p.max_phonon_order == "auto")
    max_order = 100 if auto_order else int(p.max_phonon_order)
    _check_species_symbols(m, p)
    b_map = {s.symbol: s.b_coh_fm * 1.0e-5 for s in m.scatterers if s.b_coh_fm is not None}
    inc_map = {s.symbol: s.sigma_inc_b for s in m.scatterers if s.sigma_inc_b is not None}
    sigma_b = (m.scatterers[0].sigma_bound_b if m.scatterers else None)
    awr = (m.scatterers[0].awr if m.scatterers else None)
    if sigma_b is None or awr is None:
        raise SpectraConfigError(
            "material.scatterers must give the principal sigma_bound_b and awr")
    if angle_range is None:
        cov = ins.map_coverage_deg
        # Mirror the GUI precedence: for a chopper instrument with no explicit
        # map_coverage_deg, use its tabulated 2-theta span (ARCS/SEQUOIA/HYSPEC)
        # rather than the generic [45, 135] fallback.
        if not cov and ins.resolution_model == "chopper" and ins.chopper_spec:
            from irma.spectra.chopper_resolution import default_coverage
            cov = default_coverage(ins.chopper_spec.get("instrument", ""))
        if not cov:
            cov = ins.angles_deg or [45.0, 135.0]
        angle_range = (min(cov), max(cov))

    if q_max is None:
        if g.q_max_invA:
            q_max = float(g.q_max_invA)
        else:
            # Auto-cover the kinematic envelope (the GUI rule): a hardcoded
            # cap leaves an empty band wherever the arch reaches past it.
            from irma.spectra.forward import kinematic_envelope
            Eg = np.arange(g.e_min_meV, g.e_max_meV + 0.5 * g.de_meV, g.de_meV)
            _, q_hi = kinematic_envelope(ins.geometry, ins.e_fixed_meV,
                                         angle_range[0], angle_range[1], Eg)
            finite_q = q_hi[np.isfinite(q_hi)]
            q_max = float(finite_q.max()) + 0.5 if finite_q.size else 13.0
            q_max = min(q_max, 40.0)
    if q_max <= q_min:
        raise SpectraConfigError(
            f"run_map q_max ({q_max}) must exceed q_min ({q_min}) -- q_max "
            "defaults to grid.q_max_invA, else the kinematic envelope")

    dos_species = (_assemble_dos_species(m, dos_source=p.dos_source, born_path=m.born)
                   if p.inelastic_mode == 0 else None)
    dos_crystal = (tuple(float(x) for x in m.lattice)
                   if (p.inelastic_mode == 0 and m.lattice is not None) else None)
    elastic_model, elastic_flag, elastic_scatterers = _elastic_inputs(m, p)

    return compute_sqe_map(
        geometry=ins.geometry, phonopy_yaml=m.phonopy_yaml,
        temperature_k=float(m.temperature_K), mesh=tuple(m.mesh),
        sab_mass_ratio=float(awr), sab_sigma_barn=float(sigma_b),
        e_fixed_meV=ins.e_fixed_meV, angle_range_deg=tuple(angle_range),
        q_min=q_min, q_max=q_max, dQ_map=dQ_map,
        e_min=g.e_min_meV, e_max=g.e_max_meV, dE=g.de_meV,
        inelastic_mode=p.inelastic_mode, dos_species=dos_species,
        num_directions=p.n_directions,
        multiphonon_num_directions=p.multiphonon_directions,
        multiphonon_max_order=max_order, auto_multiphonon_order=auto_order,
        min_phonon_energy_mev=float(p.min_phonon_energy_meV),
        jobs=p.jobs,
        force_constants=m.force_constants, force_sets=m.force_sets, born_path=m.born,
        sigma_coeffs=ins.sigma_coeffs, resolution_shape=ins.resolution_shape,
        resolution_model=ins.resolution_model,
        chopper_spec=ins.chopper_spec,
        broaden=broaden, include_gain=p.include_energy_gain,
        gain_side=p.gain_side,
        elastic_model=elastic_model, elastic=elastic_flag,
        elastic_kind=p.elastic_kind, elastic_scatterers=elastic_scatterers,
        incoherent_elastic_mode=p.incoherent_elastic_mode,
        dos_crystal=dos_crystal,
        scattering_lengths_json=(json.dumps(b_map) if b_map else None),
        incoherent_cross_sections_json=(json.dumps(inc_map) if inc_map else None),
        workdir=workdir, label=label, progress=progress)
