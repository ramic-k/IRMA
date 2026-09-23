"""ENDFTSLPACK_TEXT_V1 text container (key=value), ported pattern from irma.ncrystal.pack."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = "ENDFTSLPACK_TEXT_V1"
UNITS = "angstrom_eV_barn_K"
SCHEMA = "1"
BACKEND = "endf_direct"


@dataclass
class ENDFTSLPack:
    material_id: str
    temperature_K: float
    bound_xs_barn: float
    element_mass_amu: float
    sab_representation: str = "scaled_sym_sab"
    alpha_grid: list = field(default_factory=list)
    beta_grid: list = field(default_factory=list)
    sab_values: list = field(default_factory=list)
    coh_edges_ev: list = field(default_factory=list)
    coh_cumS: list = field(default_factory=list)
    elastic_msd_a2: float | None = None
    elastic_incoherent_xs_barn: float | None = None
    elastic_scale: float = 1.0
    metadata: dict = field(default_factory=dict)


def _f(v):
    return f"{float(v):.17g}"


def _fl(vs):
    return " ".join(_f(v) for v in vs)


def _validate(pack: ENDFTSLPack) -> None:
    """Shared write_pack/read_pack validation (guards read_pack against
    hand-edited files): raises ValueError naming the field."""
    def bad(fieldname, msg):
        raise ValueError(f"invalid ENDFTSL pack field {fieldname}: {msg}")

    for name, v in (("temperature_K", pack.temperature_K),
                    ("element_mass_amu", pack.element_mass_amu)):
        if not (math.isfinite(v) and v > 0.0):
            bad(name, f"must be a finite positive number, got {v!r}")
    v = pack.bound_xs_barn
    if not (math.isfinite(v) and v >= 0.0):
        bad("bound_xs_barn", f"must be a finite non-negative number, got {v!r}")

    for name, grid in (("alpha_grid", pack.alpha_grid),
                       ("beta_grid", pack.beta_grid)):
        if len(grid) < 2:
            bad(name, f"must contain at least two points, got {len(grid)}")
        if not all(math.isfinite(x) for x in grid):
            bad(name, "must contain only finite values")
        if any(b <= a for a, b in zip(grid, grid[1:])):
            bad(name, "must be strictly increasing")
    nexp = len(pack.alpha_grid) * len(pack.beta_grid)
    if len(pack.sab_values) != nexp:
        bad("sab_values", "length must equal len(alpha_grid) x len(beta_grid) "
            f"= {nexp}, got {len(pack.sab_values)}")
    if not all(math.isfinite(x) and x >= 0.0 for x in pack.sab_values):
        bad("sab_values", "must be finite and non-negative")

    if len(pack.coh_edges_ev) != len(pack.coh_cumS):
        bad("coh_cumulative_s", "length must equal coh_edges_ev length "
            f"({len(pack.coh_edges_ev)}), got {len(pack.coh_cumS)}")
    if pack.coh_edges_ev:
        e = pack.coh_edges_ev
        if not all(math.isfinite(x) and x >= 0.0 for x in e):
            bad("coh_edges_ev", "must be finite and non-negative")
        if any(b <= a for a, b in zip(e, e[1:])):
            bad("coh_edges_ev", "must be strictly increasing")
        s = pack.coh_cumS
        if not all(math.isfinite(x) and x >= 0.0 for x in s):
            bad("coh_cumulative_s", "must be finite and non-negative")
        if any(b < a for a, b in zip(s, s[1:])):
            bad("coh_cumulative_s", "must be non-decreasing")


def write_pack(pack: ENDFTSLPack, path) -> None:
    _validate(pack)
    L = [MAGIC, f"schema_version = {SCHEMA}", f"units = {UNITS}",
         f"backend = {BACKEND}", f"material_id = {pack.material_id}",
         f"temperature_K = {_f(pack.temperature_K)}",
         f"bound_xs_barn = {_f(pack.bound_xs_barn)}",
         f"element_mass_amu = {_f(pack.element_mass_amu)}",
         f"sab_representation = {pack.sab_representation}",
         f"alpha_grid = {_fl(pack.alpha_grid)}",
         f"beta_grid = {_fl(pack.beta_grid)}",
         f"sab_values = {_fl(pack.sab_values)}"]
    if pack.coh_edges_ev:
        L.append(f"coh_edges_ev = {_fl(pack.coh_edges_ev)}")
        L.append(f"coh_cumulative_s = {_fl(pack.coh_cumS)}")
    if pack.elastic_msd_a2 is not None:
        L.append(f"elastic_msd_a2 = {_f(pack.elastic_msd_a2)}")
        L.append(f"elastic_incoherent_xs_barn = {_f(pack.elastic_incoherent_xs_barn)}")
        L.append(f"elastic_scale = {_f(pack.elastic_scale)}")
    for k, v in pack.metadata.items():
        L.append(f"meta.{k} = {v}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def read_pack(path) -> ENDFTSLPack:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != MAGIC:
        raise ValueError(f"bad pack magic in {path!r}; expected {MAGIC}")
    fields, meta = {}, {}
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = (s.strip() for s in ln.split("=", 1))
        if k.startswith("meta."):
            meta[k[5:]] = v
        else:
            fields[k] = v
    if fields.get("schema_version") != SCHEMA:
        raise ValueError("unsupported schema_version")
    if fields.get("units") != UNITS:
        raise ValueError("unsupported units")

    def g(key):
        return [float(x) for x in fields[key].split()] if fields.get(key) else []

    pack = ENDFTSLPack(
        material_id=fields["material_id"], temperature_K=float(fields["temperature_K"]),
        bound_xs_barn=float(fields["bound_xs_barn"]),
        element_mass_amu=float(fields["element_mass_amu"]),
        sab_representation=fields.get("sab_representation", "scaled_sym_sab"),
        alpha_grid=g("alpha_grid"), beta_grid=g("beta_grid"), sab_values=g("sab_values"),
        coh_edges_ev=g("coh_edges_ev"), coh_cumS=g("coh_cumulative_s"),
        elastic_msd_a2=float(fields["elastic_msd_a2"]) if "elastic_msd_a2" in fields else None,
        elastic_incoherent_xs_barn=(float(fields["elastic_incoherent_xs_barn"])
                                    if "elastic_incoherent_xs_barn" in fields else None),
        elastic_scale=float(fields.get("elastic_scale", "1.0")), metadata=meta)
    _validate(pack)
    return pack
