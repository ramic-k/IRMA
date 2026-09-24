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
from pathlib import Path
from typing import Optional, Union

import numpy as np

# VISION preset defaults (from irma.spectra.sqe)
from irma.spectra.sqe import (
    VISION_BANKS, VISION_EF_MEV, VISION_SIGMA_COEFFS, sigma_of_E)

_CHOPPER_KEYS = ("instrument", "package", "frequency")
# Allowed values of the string-valued config fields.
_CHOICES = {
    "physics.dos_source": ("file", "phonopy"),
    "physics.elastic_kind": ("both", "coherent", "incoherent"),
    "physics.incoherent_elastic_mode": ("isotropic", "directional"),
    "physics.gain_side": ("direct", "detailed_balance"),
    "instrument.geometry": ("vision", "indirect", "direct"),
    "instrument.combine": ("mean", "sum"),
    "instrument.output_mode": ("cuts", "map"),
    "instrument.cut_by": ("angles", "q"),
    "instrument.resolution_shape": ("gaussian", "lorentzian"),
    "instrument.resolution_model": ("poly", "chopper"),
}


class SpectraConfigError(ValueError):
    """Raised on a malformed or physically inconsistent ``SpectraConfig``."""


def _require_exact_int(value, name, minimum):
    """Exact-integer config field: numeric strings ("6") and integral floats
    (6.0) pass; bools, fractions and non-finite values are rejected rather
    than truncated."""
    try:
        f = float(value)
        ok = not isinstance(value, bool) and f == int(f) and f >= minimum
    except (TypeError, ValueError, OverflowError):
        ok = False
    if not ok:
        raise SpectraConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(f)


def _number(value, name):
    """A numeric field as parsed, with number strings converted: YAML 1.1
    loads an exponent literal without a point or a signed exponent ('2e2',
    '1.5e3') as a string."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise SpectraConfigError(f"{name} must be a number, got {value!r}") from None
    return value


def _numbers(value, name):
    """A list-of-numbers field as floats."""
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        raise SpectraConfigError(
            f"{name} must be a list of numbers, got {value!r}") from None


# numeric scalar fields of each section, converted by _number in from_dict
_NUMBER_FIELDS = {
    "scatterer": ("sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b"),
    "material": ("temperature_K",),
    "grid": ("e_min_meV", "e_max_meV", "de_meV", "dq_max_invA", "q_max_invA",
             "q_pad_invA"),
    "instrument": ("e_fixed_meV", "bank_halfwidth_deg"),
}


def _convert_numbers(obj, section, prefix):
    for name in _NUMBER_FIELDS[section]:
        setattr(obj, name, _number(getattr(obj, name), f"{prefix}.{name}"))


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
    """One scattering species. Modes 1/2 need ``b_coh_fm`` and ``sigma_inc_b``
    for every species except C, which has built-in values.

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
    dq_max_invA: float = 0.05               # S(Q,E) Q-support spacing
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
    sigma_coeffs: Optional[list] = None     # None -> VISION poly (vision/indirect), constant 0.02*Ei sigma (direct)
    resolution_shape: str = "gaussian"      # gaussian (sigma poly) | lorentzian (HWHM poly)
    resolution_model: str = "poly"          # poly (sigma poly) | chopper (auto; the eight instruments in chopper_resolution)
    chopper_spec: Optional[dict] = None     # instrument,package,frequency for resolution_model=chopper
    combine: str = "mean"
    # Output selection (the Run button's product). 'cuts' = 1-D spectra;
    # 'map' = the dense 2-D S(Q,E) heatmap (any geometry -- the map path
    # dispatches direct/indirect kinematics itself). Map-only fields below are
    # ignored for 'cuts'.
    output_mode: str = "cuts"               # cuts | map
    cut_by: str = "angles"                  # angles | q   (fixed-cuts sub-mode)
    cut_dq_invA: Optional[float] = None     # half-width of the constant-Q cut band [1/A]; None -> thin slice
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
        # Unknown sections must fail like unknown keys inside one: a
        # typo'd 'instrumnet:' would otherwise be dropped wholesale and the run
        # would complete (exit 0) with the default instrument instead.
        unknown = set(d) - {"material", "physics", "grid", "instrument"}
        if unknown:
            raise SpectraConfigError(
                f"unknown config section(s) {sorted(unknown)}; allowed: "
                "['material', 'physics', 'grid', 'instrument']")
        if "material" not in d:
            raise SpectraConfigError("config is missing the required 'material' section")
        mat_d = dict(d["material"])
        scat = [_build(Scatterer, dict(s)) for s in mat_d.pop("scatterers", [])]
        material = _build(MaterialConfig, mat_d)
        material.scatterers = scat
        _convert_numbers(material, "material", "material")
        material.mesh = [_require_exact_int(x, "material.mesh entries", 1)
                         for x in material.mesh]
        if material.lattice is not None:
            material.lattice = _numbers(material.lattice, "material.lattice")
        for s in material.scatterers:
            _convert_numbers(s, "scatterer", f"scatterer {s.symbol!r}")
            if s.positions is not None:
                s.positions = [_numbers(p, f"scatterer {s.symbol!r} positions")
                               for p in s.positions]
        physics = _build(PhysicsConfig, dict(d.get("physics", {})))
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
        _convert_numbers(grid, "grid", "grid")
        instrument = _build(InstrumentConfig, dict(d.get("instrument", {})))
        _convert_numbers(instrument, "instrument", "instrument")
        for f in ("angles_deg", "q_cuts", "sigma_coeffs", "map_coverage_deg"):
            if getattr(instrument, f) is not None:
                setattr(instrument, f, _numbers(getattr(instrument, f),
                                                f"instrument.{f}"))
        if instrument.cut_dq_invA is not None:
            instrument.cut_dq_invA = float(instrument.cut_dq_invA)
        instrument.map_mask = _strict_bool("instrument.map_mask", instrument.map_mask)
        instrument.export_components = _strict_bool(
            "instrument.export_components", instrument.export_components)
        if instrument.chopper_spec is not None:
            cs = dict(instrument.chopper_spec)
            unknown = set(cs) - set(_CHOPPER_KEYS)
            if unknown:
                raise SpectraConfigError(
                    f"unknown instrument.chopper_spec key(s) {sorted(unknown)}; "
                    f"allowed: {list(_CHOPPER_KEYS)}")
            instrument.chopper_spec = {k: cs.get(k) for k in _CHOPPER_KEYS}
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
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpectraConfigError(f"config could not be parsed: {exc}") from None


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
    """Raise SpectraConfigError on a malformed/inconsistent config; else return it.

    Range checks are written as ``not x > 0`` so a NaN fails them too.
    """
    from irma.core.noncubic_helpers import _FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM
    m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument

    # A string inelastic_mode alias becomes its integer first, so every later
    # check and every consumer of the validated config sees an int.
    if isinstance(p.inelastic_mode, str):
        alias = p.inelastic_mode.strip().lower()
        if alias not in INELASTIC_MODE_ALIASES:
            raise SpectraConfigError(
                f"physics.inelastic_mode must be 0, 1, 2 or one of "
                f"{sorted(INELASTIC_MODE_ALIASES)}, got {p.inelastic_mode!r}")
        p.inelastic_mode = INELASTIC_MODE_ALIASES[alias]
    if p.inelastic_mode not in (0, 1, 2):
        raise SpectraConfigError(
            f"physics.inelastic_mode must be 0 (DOS), 1 or 2, got {p.inelastic_mode}")
    for key, allowed in _CHOICES.items():
        section, name = key.split(".")
        value = getattr(getattr(cfg, section), name)
        if value not in allowed:
            raise SpectraConfigError(f"{key} must be one of {allowed}, got {value!r}")

    # Phonon model: modes 1/2 and the mode-0 phonopy DOS read the phonopy yaml.
    if p.inelastic_mode != 0 or p.dos_source == "phonopy":
        if not m.phonopy_yaml:
            raise SpectraConfigError(
                "material.phonopy_yaml is required for inelastic_mode 1/2 and "
                "for inelastic_mode 0 with dos_source='phonopy'")
        if len(m.mesh) != 3:
            raise SpectraConfigError(f"material.mesh must be three positive ints, got {m.mesh}")
    if not m.temperature_K > 0:
        raise SpectraConfigError(f"material.temperature_K must be > 0, got {m.temperature_K}")
    for s in m.scatterers:
        name = f"scatterer {s.symbol!r}"
        if s.awr is not None and not s.awr > 0:
            raise SpectraConfigError(f"{name}: awr must be > 0, got {s.awr}")
        if s.sigma_bound_b is not None and not s.sigma_bound_b > 0:
            raise SpectraConfigError(f"{name}: sigma_bound_b must be > 0, got {s.sigma_bound_b}")
        if s.sigma_inc_b is not None and not s.sigma_inc_b >= 0:
            raise SpectraConfigError(f"{name}: sigma_inc_b must be >= 0, got {s.sigma_inc_b}")
        # the engine has built-in values for carbon only
        if (p.inelastic_mode in (1, 2)
                and s.symbol not in _FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM
                and (s.b_coh_fm is None or s.sigma_inc_b is None)):
            raise SpectraConfigError(
                f"{name}: inelastic_mode {p.inelastic_mode} needs b_coh_fm and "
                "sigma_inc_b (built-in values exist only for C)")

    if p.inelastic_mode == 0:
        if not m.scatterers:
            raise SpectraConfigError(
                "inelastic_mode=0 (DOS) requires material.scatterers, each with "
                "awr and sigma_bound_b")
        lat = m.lattice
        if lat is not None and not (len(lat) == 6 and all(x > 0 for x in lat[:3])
                                    and all(0 < x < 180 for x in lat[3:])):
            raise SpectraConfigError(
                "material.lattice must be [a,b,c,alpha,beta,gamma] with a,b,c > 0 "
                f"and angles in (0,180) deg, got {lat}")
        # The coherent Bragg peaks need a lattice; the incoherent line does not.
        want_coh = p.elastic and p.elastic_kind in ("both", "coherent") and lat is not None
        want_inc = p.elastic and p.elastic_kind in ("both", "incoherent")
        for s in m.scatterers:
            name = f"scatterer {s.symbol!r}"
            if p.dos_source == "file" and not s.dos_file:
                raise SpectraConfigError(f"{name}: dos_source='file' needs a dos_file")
            if s.awr is None or s.sigma_bound_b is None:
                raise SpectraConfigError(f"{name}: inelastic_mode=0 needs awr and sigma_bound_b")
            s.multiplicity = _require_exact_int(s.multiplicity, f"{name} multiplicity", 1)
            if s.positions is not None and s.multiplicity != len(s.positions):
                raise SpectraConfigError(
                    f"{name}: multiplicity ({s.multiplicity}) must equal the number "
                    f"of positions ({len(s.positions)})")
            if any(len(pos) != 3 for pos in s.positions or ()):
                raise SpectraConfigError(
                    f"{name}: each position must be three fractional coordinates, "
                    f"got {s.positions}")
            if want_inc and s.sigma_inc_b is None:
                raise SpectraConfigError(
                    f"{name}: the incoherent elastic line needs sigma_inc_b "
                    "(0.0 for no incoherent channel)")
            if want_coh and (not s.positions or s.b_coh_fm is None):
                raise SpectraConfigError(
                    f"{name}: the coherent elastic line (material.lattice set) "
                    "needs positions and b_coh_fm")

    if p.max_phonon_order != "auto":
        p.max_phonon_order = _require_exact_int(
            p.max_phonon_order, "physics.max_phonon_order", 1)
    p.n_directions = _require_exact_int(p.n_directions, "physics.n_directions", 1)
    p.multiphonon_directions = _require_exact_int(
        p.multiphonon_directions, "physics.multiphonon_directions", 1)
    if p.jobs is not None:
        p.jobs = _require_exact_int(p.jobs, "physics.jobs", 1)
    if p.incoherent_elastic_mode == "directional" and p.inelastic_mode == 0:
        raise SpectraConfigError(
            "physics.incoherent_elastic_mode = 'directional' requires "
            "inelastic_mode 1 or 2 (the DOS path has no displacement tensors)")

    if not (g.de_meV > 0 and g.dq_max_invA > 0):
        raise SpectraConfigError("grid.de_meV and grid.dq_max_invA must be > 0")
    if not g.e_max_meV > g.e_min_meV:
        raise SpectraConfigError(
            f"grid.e_max_meV ({g.e_max_meV}) must exceed e_min_meV ({g.e_min_meV})")
    if g.q_max_invA is not None and not g.q_max_invA > 0:
        raise SpectraConfigError(f"grid.q_max_invA must be > 0 or null, got {g.q_max_invA}")
    if not g.q_pad_invA >= 0:
        raise SpectraConfigError(f"grid.q_pad_invA must be >= 0, got {g.q_pad_invA}")

    if ins.cut_dq_invA is not None and not ins.cut_dq_invA > 0:
        raise SpectraConfigError(
            f"instrument.cut_dq_invA must be > 0 or null, got {ins.cut_dq_invA}")
    cov = ins.map_coverage_deg
    if cov is not None and not (len(cov) == 2 and 0.0 < cov[0] < cov[1] < 180.0):
        raise SpectraConfigError(
            "instrument.map_coverage_deg must be [2th_min, 2th_max] with "
            f"0 < min < max < 180 deg, got {cov}")
    if not ins.e_fixed_meV > 0:
        raise SpectraConfigError(f"instrument.e_fixed_meV must be > 0, got {ins.e_fixed_meV}")
    if ins.resolution_model == "chopper":
        if ins.geometry != "direct" or ins.resolution_shape != "gaussian":
            raise SpectraConfigError(
                "instrument.resolution_model='chopper' needs geometry='direct' "
                "and resolution_shape='gaussian'")
        spec = ins.chopper_spec or {}
        missing = [k for k in _CHOPPER_KEYS if not spec.get(k)]
        if missing:
            raise SpectraConfigError(
                f"instrument.chopper_spec is missing {missing} (CLI flags: "
                "--chopper-instrument, --chopper-package, --chopper-frequency)")
        from irma.spectra.chopper_resolution import (
            _norm_frequency, chopper_sigma_of_E, instrument_geometry)
        try:
            geom = instrument_geometry(spec["instrument"], spec["package"])
            freq = _norm_frequency(spec["frequency"])
            if freq > geom["max_frequency"]:
                raise ValueError(
                    f"frequency {freq:g} Hz is above the {spec['instrument']} "
                    f"maximum of {geom['max_frequency']} Hz")
            # raises when this package and frequency do not transmit Ei
            chopper_sigma_of_E(np.array([0.0]), Ei=ins.e_fixed_meV, **spec)
        except (TypeError, ValueError) as exc:
            raise SpectraConfigError(f"instrument.chopper_spec: {exc}") from None
    if not ins.bank_halfwidth_deg > 0:
        raise SpectraConfigError(
            f"instrument.bank_halfwidth_deg must be > 0, got {ins.bank_halfwidth_deg}")
    if ins.geometry != "vision" and ins.output_mode != "map" and not ins.angles_deg:
        raise SpectraConfigError(
            f"instrument.angles_deg is required for geometry '{ins.geometry}' "
            "(only the vision preset fills its banks)")
    banks = sorted(VISION_BANKS.values())
    if (ins.geometry == "vision" and ins.angles_deg is not None
            and sorted(ins.angles_deg) != banks):
        raise SpectraConfigError(
            f"instrument.angles_deg {ins.angles_deg} does not apply to the vision "
            f"preset (fixed banks at {banks} deg); use geometry 'indirect' for "
            "other angles")
    if ins.angles_deg is not None and not all(0.0 < a < 180.0 for a in ins.angles_deg):
        raise SpectraConfigError("instrument.angles_deg must lie strictly in (0, 180) deg")
    if ins.q_cuts is not None and not all(q > 0 for q in ins.q_cuts):
        raise SpectraConfigError(f"instrument.q_cuts must be > 0 [1/A], got {ins.q_cuts}")
    if ins.geometry == "direct":
        if ins.output_mode != "map" and ins.cut_by == "q" and not ins.q_cuts:
            raise SpectraConfigError("instrument.cut_by='q' requires instrument.q_cuts")
        if not g.e_max_meV < ins.e_fixed_meV:
            raise SpectraConfigError(
                f"direct geometry: grid.e_max_meV ({g.e_max_meV}) must be below the "
                f"incident energy Ei ({ins.e_fixed_meV} meV)")
    coeffs = ins.sigma_coeffs if ins.sigma_coeffs is not None else VISION_SIGMA_COEFFS
    if len(coeffs) > 3:
        raise SpectraConfigError(
            f"instrument.sigma_coeffs is c0,c1,c2 (at most 3 terms), got {coeffs}")
    # The width is evaluated at |E| over the output grid (and at E=0).
    E = np.linspace(0.0, max(g.e_max_meV, abs(g.e_min_meV)), 1000)
    if not np.all(sigma_of_E(E, coeffs) > 0):
        raise SpectraConfigError(
            f"instrument.sigma_coeffs {list(coeffs)} give a non-positive resolution "
            f"width on |E| <= {E[-1]:g} meV")
    return cfg


def check_input_files(cfg: SpectraConfig) -> SpectraConfig:
    """Preflight every path-valued config field that is set; return ``cfg``.

    Raises :class:`SpectraConfigError` naming the responsible field when a
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
        # A duplicated symbol would count one phonopy DOS twice; a missing one
        # would drop that species from the sample total.
        dupes = sorted({x for x in scat_syms if scat_syms.count(x) > 1})
        if dupes:
            raise SpectraConfigError(
                f"dos_source='phonopy': duplicate scatterer symbol(s) {dupes}; give "
                "one scatterer per element (or use dos_source='file')")
        missing = sorted(set(by_sym) - set(scat_syms))
        if missing:
            raise SpectraConfigError(
                f"dos_source='phonopy': phonopy species {missing} have no scatterer")
        species = []
        for s in material.scatterers:
            if s.symbol not in by_sym:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r} is not among the phonopy DOS species "
                    f"{sorted(by_sym)}")
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
    """Modes 1/2: every scatterer symbol must be a phonopy structure species
    (the engine would silently use built-in constants for an unmatched
    symbol). An unparseable yaml skips the check."""
    if p.inelastic_mode not in (1, 2) or not m.phonopy_yaml:
        return
    species = phonopy_species(m.phonopy_yaml)
    if not species:
        return
    bad = sorted({s.symbol for s in m.scatterers} - set(species))
    if bad:
        raise SpectraConfigError(
            f"scatterer symbol(s) {bad} are not phonopy structure species "
            f"{species}; scatterers are matched by exact chemical symbol (give "
            "an isotope its element symbol with explicit constants)")


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


def _forward_kwargs(cfg, progress):
    """Keyword arguments shared by ``compute_spectrum`` and ``compute_sqe_map``
    for a validated config."""
    m, p, ins = cfg.material, cfg.physics, cfg.instrument
    _check_species_symbols(m, p)
    principal = m.scatterers[0] if m.scatterers else None
    if principal is None or principal.sigma_bound_b is None or principal.awr is None:
        raise SpectraConfigError(
            "material.scatterers must give at least the principal site's "
            "sigma_bound_b and awr")
    # Per-symbol overrides; the engine applies each to every site of that
    # symbol. b_coh: fm -> Angstrom.
    b_map = {s.symbol: s.b_coh_fm * 1.0e-5 for s in m.scatterers if s.b_coh_fm is not None}
    inc_map = {s.symbol: s.sigma_inc_b for s in m.scatterers if s.sigma_inc_b is not None}
    mode0 = p.inelastic_mode == 0
    dos_species = (_assemble_dos_species(m, dos_source=p.dos_source, born_path=m.born)
                   if mode0 else None)
    elastic_model, elastic_flag, elastic_scatterers = _elastic_inputs(m, p)
    auto_order = p.max_phonon_order == "auto"
    return dict(
        geometry=ins.geometry, phonopy_yaml=m.phonopy_yaml,
        temperature_k=float(m.temperature_K), mesh=tuple(m.mesh),
        sab_mass_ratio=float(principal.awr), sab_sigma_barn=float(principal.sigma_bound_b),
        e_fixed_meV=ins.e_fixed_meV, e_min=cfg.grid.e_min_meV, e_max=cfg.grid.e_max_meV,
        dE=cfg.grid.de_meV, sigma_coeffs=ins.sigma_coeffs,
        resolution_shape=ins.resolution_shape, resolution_model=ins.resolution_model,
        chopper_spec=ins.chopper_spec, inelastic_mode=p.inelastic_mode,
        dos_species=dos_species,
        dos_crystal=(tuple(float(x) for x in m.lattice)
                     if mode0 and m.lattice is not None else None),
        num_directions=p.n_directions, multiphonon_num_directions=p.multiphonon_directions,
        multiphonon_max_order=100 if auto_order else int(p.max_phonon_order),
        auto_multiphonon_order=auto_order,
        min_phonon_energy_mev=float(p.min_phonon_energy_meV), jobs=p.jobs,
        force_constants=m.force_constants, force_sets=m.force_sets, born_path=m.born,
        scattering_lengths_json=json.dumps(b_map) if b_map else None,
        incoherent_cross_sections_json=json.dumps(inc_map) if inc_map else None,
        elastic_model=elastic_model, elastic=elastic_flag, elastic_kind=p.elastic_kind,
        elastic_scatterers=elastic_scatterers,
        incoherent_elastic_mode=p.incoherent_elastic_mode,
        include_gain=p.include_energy_gain, gain_side=p.gain_side, progress=progress)


def run_spectra(cfg: SpectraConfig, *, progress=print):
    """Run the forward model described by ``cfg`` and return a ``SpectrumResult``.

    Thin, deterministic mapping from the validated config onto
    ``compute_spectrum`` keyword arguments -- the single code path the
    GUI and CLI both funnel through.
    """
    from irma.spectra.forward import compute_spectrum

    validate(cfg)
    p, g, ins = cfg.physics, cfg.grid, cfg.instrument
    return compute_spectrum(
        **_forward_kwargs(cfg, progress),
        angles_deg=ins.angles_deg, q_cuts=ins.q_cuts or None, cut_dq=ins.cut_dq_invA,
        # direct cut_by='q' gives constant-Q cuts only; otherwise the angle
        # spectra are produced alongside any q_cuts
        produce_angle_spectra=not (ins.geometry == "direct" and ins.cut_by == "q"),
        dQ=g.dq_max_invA, bank_halfwidth_deg=ins.bank_halfwidth_deg,
        combine=ins.combine, kinematic_factor=p.kinematic_kf_ki, q_pad=g.q_pad_invA)


def run_map(cfg, *, q_min=0.0, q_max=None, dQ_map=None, angle_range=None,
            broaden=True, progress=print):
    """Run a dense 2-D S(Q,E) powder map for ``cfg`` (config -> compute_sqe_map).

    Config-driven defaults (a None argument defers to the config):
      * ``q_max``      -> ``grid.q_max_invA``, else auto-covers the
        kinematic envelope (capped at 40 1/A);
      * ``dQ_map``     -> ``grid.dq_max_invA``;
      * ``angle_range``-> ``instrument.map_coverage_deg``, else the chopper
        instrument's coverage, else the cut ``instrument.angles_deg``, else
        (45, 135).
    The grid uses the config's e_min/e_max/de for energy and q_min/q_max/dQ_map
    for Q. ``angle_range=(2th_min, 2th_max)`` sets the kinematic-envelope overlay.
    """
    from irma.spectra.forward import compute_sqe_map

    validate(cfg)
    g, ins = cfg.grid, cfg.instrument
    if angle_range is not None and not (
            len(angle_range) == 2 and 0.0 < angle_range[0] < angle_range[1] < 180.0):
        raise SpectraConfigError(
            "run_map angle_range must be (2theta_min, 2theta_max) with "
            f"0 < min < max < 180 deg, got {angle_range}")
    if dQ_map is None:
        dQ_map = float(g.dq_max_invA)
    if angle_range is None:
        cov = ins.map_coverage_deg
        if not cov and ins.resolution_model == "chopper":
            from irma.spectra.chopper_resolution import default_coverage
            cov = default_coverage(ins.chopper_spec["instrument"])
        if not cov:
            cov = ins.angles_deg or [45.0, 135.0]
        angle_range = (min(cov), max(cov))
    if q_max is None:
        if g.q_max_invA:
            q_max = float(g.q_max_invA)
        else:
            # cover the kinematic envelope, capped at 40 1/A
            from irma.spectra.forward import kinematic_envelope
            Eg = np.arange(g.e_min_meV, g.e_max_meV + 0.5 * g.de_meV, g.de_meV)
            _, q_hi = kinematic_envelope(ins.geometry, ins.e_fixed_meV,
                                         angle_range[0], angle_range[1], Eg)
            finite_q = q_hi[np.isfinite(q_hi)]
            q_max = min(float(finite_q.max()) + 0.5 if finite_q.size else 13.0, 40.0)
    if not (0 <= q_min < q_max and dQ_map > 0):
        raise SpectraConfigError(
            f"run_map needs 0 <= q_min < q_max and dQ_map > 0, got q_min={q_min}, "
            f"q_max={q_max}, dQ_map={dQ_map}")
    return compute_sqe_map(
        **_forward_kwargs(cfg, progress),
        angle_range_deg=tuple(angle_range), q_min=q_min, q_max=q_max, dQ_map=dQ_map,
        broaden=broaden, kinematic_factor=cfg.physics.kinematic_kf_ki)
