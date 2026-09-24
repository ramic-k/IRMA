"""Config for the IRMA→NCrystal exporter.

Reuses :class:`irma.spectra.config.MaterialConfig` / :class:`Scatterer` (the
phonon model + per-species neutron data — the SAME artifacts the ENDF and
spectra sides name) and adds a thin NCrystal-export section. It is a SEPARATE
dataclass, not a ``SpectraConfig`` section, because ``SpectraConfig.from_dict``
hard-rejects unknown sections; the two configs share the ``material`` block and
diverge only in their own knobs.

YAML shape::

    material:
      phonopy_yaml: graphite/phonopy.yaml
      mesh: [40, 40, 40]
      temperature_K: 296.0
      scatterers:
        - {symbol: C, sigma_bound_b: 5.551, awr: 11.898, b_coh_fm: 6.646, sigma_inc_b: 0.001}
    export:
      material_id: graphite
      inelastic_mode: 2
      num_directions: 10000
      multiphonon_num_directions: 1000
      multiphonon_max_order: auto
      gain_side: scaled_sym        # scaled_sym only (half-table; NCrystal
                                   # rebuilds the gain side by detailed balance)
      elastic: true
      coherent_partition_mode: principal-share
      incoherent_elastic_mode: isotropic   # or directional: sample the
                                   # orientation-averaged <exp(-2W_d(Q))>
                                   # per site (the ENDF path cannot
                                   # represent this)
"""
from __future__ import annotations

import dataclasses
import math
import re
from typing import Optional, Union

from irma.core.grids import AUTO_GRID_DEFAULTS
from irma.spectra.config import (MaterialConfig, Scatterer, SpectraConfigError,
                                 _build, _require_exact_int, _strict_bool)


# Coherent one-phonon partition across per-principal packs; kept in sync with
# irma.core.noncubic_engine ('auto' resolves to exact-total for a single group,
# principal-share otherwise).
VALID_PARTITION_MODES = ("auto", "exact-total", "principal-share")
# Incoherent-elastic Debye-Waller treatment in the baked pack: 'isotropic'
# collapses each site tensor to trace/3 (NCrystal's stock ElIncScatter);
# 'directional' has the plugin sample the orientation-averaged
# <exp(-Q^2 uhat.U.uhat)> per site (irma.core.incoherent_dw).
VALID_INCOHERENT_ELASTIC_MODES = ("isotropic", "directional")

# Readable aliases for the integer inelastic_mode (the integers stay canonical
# for deck compatibility; the export has no mode 0). Mirrors the spectra
# config's alias table so one vocabulary works across both YAML surfaces.
INELASTIC_MODE_ALIASES = {"incoherent": 1, "coherent": 2}


def _require_positive_int(value, name):
    """Exact positive integer for export fields (see _require_exact_int)."""
    return _require_exact_int(value, f"export.{name}", 1)


@dataclasses.dataclass
class NCrystalExportConfig:
    """Everything the exporter needs to bake one per-temperature pack set.

    One config → one temperature (``material.temperature_K``) → one
    ``.irmapack`` per principal scatterer. Re-run at each temperature for a
    multi-T deployment.
    """

    material: MaterialConfig
    material_id: str
    inelastic_mode: int = 2
    num_directions: int = 10000
    multiphonon_num_directions: int = 1000
    multiphonon_max_order: Union[int, str] = "auto"
    min_phonon_energy_meV: float = 0.0      # remove modes <= this (0 = automatic floors only)
    jobs: Optional[int] = None
    gain_side: str = "scaled_sym"
    elastic: bool = True
    coherent_partition_mode: str = "principal-share"
    incoherent_elastic_mode: str = "isotropic"
    # S(alpha,beta) grid (ENDF dimensionless convention, lat=1 -> 0.0253 eV ref).
    # Two ways to set it, as on the ENDF-evaluation side:
    #   explicit  -> give both alpha_grid and beta_grid (dimensionless, lat units).
    #   automatic -> omit both; the same converged grid the ENDF evaluator builds
    #                (irma.core.grids.generate_beta_grid / generate_alpha_grid) is
    #                generated from the phonon spectrum. freq_max_eV is auto-derived
    #                from the phonopy mesh when omitted; the other knobs default to
    #                the ENDF-grid defaults. (alpha is linear in Q; the uniform-alpha
    #                layout it replaces under-integrates the thermal cross section;
    #                see generate_alpha_grid.)
    alpha_grid: Optional[list] = None
    beta_grid: Optional[list] = None
    lat: int = 1
    # automatic-grid knobs (used only when alpha_grid/beta_grid are omitted).
    # freq_max_eV None -> estimate the max phonon frequency from the phonopy mesh.
    # The numeric defaults come from the ONE shared source
    # (irma.core.grids.AUTO_GRID_DEFAULTS), so this config, the GUI ENDF grid
    # tab, and the GUI NCrystal panel can never drift apart. n_upper 80 keeps
    # the high-E S(a,b) accurate in the baked pack (~20 suffices for
    # thermal-only work -- check grid convergence for your energy range).
    freq_max_eV: Optional[float] = None
    n_lower: int = AUTO_GRID_DEFAULTS["n_lower"]
    n_phonon: int = AUTO_GRID_DEFAULTS["n_phonon"]
    n_upper: int = AUTO_GRID_DEFAULTS["n_upper"]
    beta_max_eV: float = AUTO_GRID_DEFAULTS["beta_max_eV"]
    alpha_dq_invA: float = AUTO_GRID_DEFAULTS["alpha_dq_invA"]
    alpha_qcut_invA: float = AUTO_GRID_DEFAULTS["alpha_qcut_invA"]
    alpha_nlog: int = AUTO_GRID_DEFAULTS["alpha_nlog"]

    # -- validation / normalization ------------------------------------------
    def __post_init__(self) -> None:
        if not self.material_id:
            raise SpectraConfigError("export.material_id is required")
        # material_id becomes a file stem (<id>.ncmat, <id>__<symbol>.irmapack), so
        # it must be a safe basename: a path separator, '..', or an absolute path
        # would let a config write outside the requested output directory.
        _mid = str(self.material_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", _mid) or ".." in _mid:
            raise SpectraConfigError(
                "export.material_id must be a safe file stem matching "
                "[A-Za-z0-9][A-Za-z0-9_.-]* (letters, digits, '.', '_', '-'; no "
                f"path separators or '..'), got {self.material_id!r}")
        # strict bool first: a quoted YAML "false" is truthy
        self.elastic = _strict_bool("export.elastic", self.elastic)
        if isinstance(self.inelastic_mode, str):
            _alias = self.inelastic_mode.strip().lower()
            if _alias not in INELASTIC_MODE_ALIASES:
                raise SpectraConfigError(
                    f"export.inelastic_mode must be 1, 2, or one of "
                    f"{sorted(INELASTIC_MODE_ALIASES)}, got "
                    f"{self.inelastic_mode!r}")
            self.inelastic_mode = INELASTIC_MODE_ALIASES[_alias]
        self.inelastic_mode = _require_exact_int(
            self.inelastic_mode, "export.inelastic_mode", 1)
        if self.inelastic_mode not in (1, 2):
            raise SpectraConfigError(
                f"export.inelastic_mode must be 1 or 2 (or 'incoherent'/"
                f"'coherent'), got {self.inelastic_mode!r}")
        if self.gain_side != "scaled_sym":
            raise SpectraConfigError(
                f"export.gain_side must be 'scaled_sym', got {self.gain_side!r}")
        if isinstance(self.multiphonon_max_order, str):
            if self.multiphonon_max_order != "auto":
                raise SpectraConfigError(
                    "export.multiphonon_max_order must be an int or 'auto', got "
                    f"{self.multiphonon_max_order!r}")
        else:
            self.multiphonon_max_order = _require_positive_int(
                self.multiphonon_max_order, "multiphonon_max_order")
        try:
            from irma.core.phonopy_io import validate_min_phonon_energy_mev
            self.min_phonon_energy_meV = validate_min_phonon_energy_mev(
                self.min_phonon_energy_meV)
        except (TypeError, ValueError) as exc:
            raise SpectraConfigError(
                f"export.min_phonon_energy_meV must be a finite number >= 0 "
                f"(meV), got {self.min_phonon_energy_meV!r}: {exc}") from None
        self.num_directions = _require_positive_int(
            self.num_directions, "num_directions")
        self.multiphonon_num_directions = _require_positive_int(
            self.multiphonon_num_directions, "multiphonon_num_directions")
        if self.jobs is not None:
            self.jobs = _require_positive_int(self.jobs, "jobs")
        # alpha/beta grids are all-or-nothing: both (explicit) or neither (auto).
        if (self.alpha_grid is None) != (self.beta_grid is None):
            raise SpectraConfigError(
                "provide both export.alpha_grid and export.beta_grid for an "
                "explicit grid, or neither for the automatic Q/E grid")
        # explicit grids: >= 2 points, finite, strictly increasing
        if self.alpha_grid is not None:
            for name, grid, lower in (("alpha_grid", self.alpha_grid, "positive"),
                                      ("beta_grid", self.beta_grid, "nonnegative")):
                vals = [float(v) for v in grid]
                if len(vals) < 2:
                    raise SpectraConfigError(
                        f"export.{name} must contain at least two points, "
                        f"got {len(vals)}")
                if not all(math.isfinite(v) for v in vals):
                    raise SpectraConfigError(
                        f"export.{name} must contain only finite values")
                if any(b <= a for a, b in zip(vals, vals[1:])):
                    raise SpectraConfigError(
                        f"export.{name} must be strictly increasing")
                if (vals[0] <= 0.0) if lower == "positive" else (vals[0] < 0.0):
                    raise SpectraConfigError(
                        f"export.{name} values must be {lower}, got "
                        f"{vals[0]!r} first")
                setattr(self, name, vals)
        # only 0 and 1 exist (grids.py would treat any other value as 0)
        if isinstance(self.lat, bool) or self.lat not in (0, 1):
            raise SpectraConfigError(f"export.lat must be 0 or 1, got {self.lat!r}")
        self.lat = int(self.lat)
        if not self.material.scatterers:
            raise SpectraConfigError(
                "material.scatterers must list at least the principal species "
                "with sigma_bound_b + awr + b_coh_fm + sigma_inc_b")
        # Duplicate symbols would collide on the symbol-derived pack filename and
        # silently shadow each other in the by-symbol lookup.
        _syms = [s.symbol for s in self.material.scatterers]
        _dupes = sorted({s for s in _syms if _syms.count(s) > 1})
        if _dupes:
            raise SpectraConfigError(
                f"material.scatterers lists duplicate symbol(s) {_dupes}; one "
                "scatterer entry per species")
        # The inelastic engine derives its channel weights from b_coh_fm and
        # sigma_inc_b, so both are required even without the elastic block.
        for s in self.material.scatterers:
            v = s.sigma_bound_b
            if v is None or not math.isfinite(float(v)) or float(v) <= 0.0:
                raise SpectraConfigError(
                    f"scatterer {s.symbol!r} sigma_bound_b must be a finite "
                    f"positive number, got {v!r}")
            if s.awr is not None:
                v = float(s.awr)
                if not math.isfinite(v) or v <= 0.0:
                    raise SpectraConfigError(
                        f"scatterer {s.symbol!r} awr must be finite and "
                        f"positive, got {s.awr!r}")
            for field, allow_negative in (("b_coh_fm", True),
                                          ("sigma_inc_b", False)):
                val = getattr(s, field)
                if val is None:
                    raise SpectraConfigError(
                        f"scatterer {s.symbol!r} is missing {field}: the "
                        "mode-1/2 inelastic engine derives its channel "
                        "weights from b_coh_fm and sigma_inc_b, so both are "
                        "required even for an inelastic-only export "
                        "(export.elastic: false)")
                fval = float(val)
                if not math.isfinite(fval) or (fval < 0.0
                                               and not allow_negative):
                    raise SpectraConfigError(
                        f"scatterer {s.symbol!r} {field} must be finite"
                        f"{'' if allow_negative else ' and >= 0'}, got "
                        f"{val!r}")
        _temp_K = float(self.material.temperature_K)
        if not math.isfinite(_temp_K) or _temp_K <= 0.0:
            raise SpectraConfigError(
                f"material.temperature_K must be a finite value > 0, got "
                f"{self.material.temperature_K!r}")
        mesh = self.material.mesh
        try:
            mesh_ok = mesh is not None and len(mesh) == 3
        except TypeError:
            mesh_ok = False
        if not mesh_ok:
            raise SpectraConfigError(
                f"material.mesh must be three positive integers, got {mesh!r}")
        self.material.mesh = [
            _require_exact_int(m, "material.mesh entry", 1) for m in mesh]
        if not self.material.phonopy_yaml:
            raise SpectraConfigError(
                "material.phonopy_yaml is required for the phonon (inelastic_mode "
                "1/2) export")
        if self.coherent_partition_mode not in VALID_PARTITION_MODES:
            raise SpectraConfigError(
                f"export.coherent_partition_mode must be one of "
                f"{list(VALID_PARTITION_MODES)}, got "
                f"{self.coherent_partition_mode!r}")
        if self.incoherent_elastic_mode not in VALID_INCOHERENT_ELASTIC_MODES:
            raise SpectraConfigError(
                f"export.incoherent_elastic_mode must be one of "
                f"{list(VALID_INCOHERENT_ELASTIC_MODES)}, got "
                f"{self.incoherent_elastic_mode!r}")
        # The directional mode lives in the pack's elastic block; without the
        # elastic export it would be a silent no-op, so reject the combination.
        if self.incoherent_elastic_mode == "directional" and not self.elastic:
            raise SpectraConfigError(
                "export.incoherent_elastic_mode = directional requires "
                "export.elastic = true (the mode is carried by the pack's "
                "elastic block)")
        # automatic-grid knobs: validated and normalized even for an explicit grid
        if self.freq_max_eV is not None:
            self.freq_max_eV = float(self.freq_max_eV)
            if not math.isfinite(self.freq_max_eV) or self.freq_max_eV <= 0.0:
                raise SpectraConfigError(
                    "export.freq_max_eV must be finite and > 0 (or omitted to "
                    f"auto-estimate from the phonopy mesh), got {self.freq_max_eV!r}")
        self.n_phonon = _require_exact_int(self.n_phonon, "export.n_phonon", 2)
        for _f in ("n_lower", "n_upper"):
            setattr(self, _f, _require_exact_int(
                getattr(self, _f), f"export.{_f}", 0))
        # alpha_nlog < 2 would leave no log tail (the alpha grid would stop at q_cut)
        self.alpha_nlog = _require_exact_int(
            self.alpha_nlog, "export.alpha_nlog", 2)
        for _f in ("beta_max_eV", "alpha_dq_invA", "alpha_qcut_invA"):
            setattr(self, _f, float(getattr(self, _f)))
            if not math.isfinite(getattr(self, _f)) or getattr(self, _f) <= 0.0:
                raise SpectraConfigError(
                    f"export.{_f} must be finite and > 0, got {getattr(self, _f)!r}")

    @property
    def grid_mode(self) -> str:
        """``"explicit"`` when alpha_grid/beta_grid are given, else ``"auto"``."""
        return "explicit" if self.alpha_grid is not None else "auto"

    @property
    def auto_multiphonon_order(self) -> bool:
        """True when the multiphonon order is engine-sized (``"auto"``)."""
        return self.multiphonon_max_order == "auto"

    @property
    def effective_multiphonon_max_order(self) -> int:
        """Starting ``multiphonon_max_order`` handed to the engine (``auto`` → 100).

        With ``auto_multiphonon_order`` set, this 100 is the floor, not a cap: the
        engine sizes the order up toward high-Q convergence, bounded only by its
        internal safety cap (2000)."""
        return 100 if self.auto_multiphonon_order else int(self.multiphonon_max_order)

    # -- (de)serialization ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "NCrystalExportConfig":
        """Build a validated config from a parsed YAML/JSON mapping."""
        if not isinstance(d, dict):
            raise SpectraConfigError(
                f"config root must be a mapping, got {type(d).__name__}")
        unknown = set(d) - {"material", "export"}
        if unknown:
            raise SpectraConfigError(
                f"unknown config section(s) {sorted(unknown)}; allowed sections: "
                "['material', 'export']")
        if "material" not in d:
            raise SpectraConfigError("config is missing the required 'material' section")
        if "export" not in d:
            raise SpectraConfigError("config is missing the required 'export' section")
        material = _material_from_dict(dict(d["material"]))
        export_d = dict(d["export"])
        known = {f.name for f in dataclasses.fields(cls)} - {"material"}
        unknown_keys = set(export_d) - known
        if unknown_keys:
            raise SpectraConfigError(
                f"unknown export key(s) {sorted(unknown_keys)}; allowed: {sorted(known)}")
        return cls(material=material, **export_d)

    @classmethod
    def from_yaml(cls, path) -> "NCrystalExportConfig":
        """Load and validate an export config from a YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))


def _material_from_dict(mat_d: dict) -> MaterialConfig:
    """Build a :class:`MaterialConfig` from a parsed mapping (strict on typos),
    as ``SpectraConfig.from_dict`` does for its material section."""
    scatterers = [_build(Scatterer, dict(s)) for s in mat_d.pop("scatterers", [])]
    material = _build(MaterialConfig, mat_d)
    material.scatterers = scatterers
    return material
