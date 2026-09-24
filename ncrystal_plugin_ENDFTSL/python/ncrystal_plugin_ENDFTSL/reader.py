"""Parse an ENDF/TSL tape's MF7/MT2 (elastic) + MF7/MT4 (inelastic) with endf_parserpy.

Self-contained: no irma import. Mirrors irma.core.endf_writer's backend auto-pick.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


def _parser():
    try:
        from endf_parserpy import EndfParserCpp
        return EndfParserCpp()
    except Exception:
        from endf_parserpy import EndfParserPy
        return EndfParserPy()


def _as_list(x):
    """endf_parserpy gives TAB1 bodies as 0-indexed lists but LIST/multi arrays as
    1-indexed dicts. Normalize either to a plain list in key order."""
    if isinstance(x, dict):
        keys = sorted(k for k in x.keys() if isinstance(k, int))
        return [x[k] for k in keys]
    return list(x)


@dataclass
class TSLEvaluation:
    za: float
    awr: float
    lthr: int
    lat: int
    lasym: int
    lln: int
    temps_mt4: list
    beta: list
    alpha: list
    sab: list
    b_array: list
    beta_int: list          # ENDF interpolation code of each beta interval
    coh_temps: list | None
    coh_edges_ev: list | None
    coh_cumS: list | None
    incoh_sb_barn: float | None
    incoh_temps: list | None
    incoh_Wp: list | None


def read_tsl(path) -> TSLEvaluation:
    d = _parser().parsefile(str(Path(path)))
    mt4 = d[7][4]
    lthr = 0
    coh_temps = coh_edges = coh_cumS = None
    incoh_sb = incoh_temps = incoh_Wp = None

    # ---- MF7/MT2 elastic (present iff iel != 0) ----
    if 2 in d[7]:
        mt2 = d[7][2]
        lthr = int(mt2["LTHR"])
        if lthr in (1, 3):
            tab = mt2["S_T0_table"]
            coh_edges = _as_list(tab["Eint"])
            s0 = _as_list(tab["S"])
            extra_T = _as_list(mt2.get("T", {}))          # additional temperatures
            extra_S = mt2.get("S", {})                    # dict: edge -> {tempidx -> S}
            coh_temps = [float(mt2["T0"])] + [float(t) for t in extra_T]
            cols = [s0]
            for ti in range(1, len(coh_temps)):
                cols.append([float(extra_S[e][ti]) for e in sorted(extra_S.keys())]
                            if extra_S else s0)
            coh_cumS = cols
        if lthr in (2, 3):
            incoh_sb = float(mt2["SB"])
            incoh_temps = _as_list(mt2["Tint"])
            incoh_Wp = _as_list(mt2["Wp"])

    # ---- MF7/MT4 inelastic ----
    lat = int(mt4["LAT"])
    lasym = int(mt4["LASYM"])
    lln = int(mt4["LLN"])
    beta = _as_list(mt4["beta"])
    # MF7/MT4 TAB2 regions: interval (j, j+1) uses the first region whose last
    # point NBT (1-based) is >= j + 2
    nbt = [int(n) for n in _as_list(mt4["beta_interp"]["NBT"])]
    codes = [int(c) for c in _as_list(mt4["beta_interp"]["INT"])]
    beta_int = [next(c for n, c in zip(nbt, codes) if j + 2 <= n)
                for j in range(len(beta) - 1)]
    # T0 is the principal temperature (the column parsed into `sab` below);
    # mt4["T"] holds only the LT extra temperatures. Prepend T0 so
    # temps_mt4[0] is the actual temperature of the parsed S(alpha,beta)
    # (e.g. ENDF/B-VIII.1 BeO: 293.6 K, then 400..1200 K) — dropping T0
    # here would mislabel the principal column with an extra temperature.
    temps_mt4 = [float(mt4["T0"])] + [float(t) for t in _as_list(mt4.get("T", {}))]
    # Each per-beta block S_table[bi+1] carries the alpha grid AND the principal-
    # temperature S column (S_table[bi+1]['S'], 0-indexed over alpha). The top-level
    # mt4['S'] exists only when ntempr>1 and holds the extra temperatures (indexed
    # [alpha][beta][temp]); the single-temperature fixture has no mt4['S'].
    alpha = [_as_list(mt4["S_table"][bi + 1]["alpha"]) for bi in range(len(beta))]
    # sab[bi][ai] = [S(T0), S(T1), ..., S(T_LT)] aligned with temps_mt4. The T0 column
    # is in S_table[bi+1]['S']; the LT extra temperatures are in mt4['S'][alpha][beta]
    # [temp] (all 1-based). physical_inelastic selects the column for the requested
    # temperature (which must match one of the stored temps -- no interpolation).
    n_extra = len(temps_mt4) - 1
    extra_S = mt4.get("S") if n_extra > 0 else None
    sab = []
    for bi in range(len(beta)):
        row = _as_list(mt4["S_table"][bi + 1]["S"])          # T0 alpha-vector at this beta
        block = []
        for ai in range(len(alpha[bi])):
            col = [float(row[ai])]
            for j in range(1, n_extra + 1):                  # extra temps, temps_mt4 order
                col.append(float(extra_S[ai + 1][bi + 1][j]))
            block.append(col)
        sab.append(block)
    b_array = [0.0] + _as_list(mt4["B"])      # pad so b_array[1] == ENDF B(1)

    return TSLEvaluation(
        za=float(mt4["ZA"]), awr=float(mt4["AWR"]),
        lthr=lthr, lat=lat, lasym=lasym, lln=lln,
        temps_mt4=temps_mt4, beta=[float(b) for b in beta], alpha=alpha, sab=sab,
        b_array=[float(b) for b in b_array], beta_int=beta_int,
        coh_temps=coh_temps, coh_edges_ev=coh_edges, coh_cumS=coh_cumS,
        incoh_sb_barn=incoh_sb, incoh_temps=incoh_temps, incoh_Wp=incoh_Wp)
