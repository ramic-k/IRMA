#!/usr/bin/env python
"""Plot the per-atom scattering cross section of an ENDFTSL plugin material.

This is the "use it in NCrystal" half of the workflow: once `python -m
ncrystal_plugin_ENDFTSL` has produced an `.ncmat` (+ its `.endftslpack`), NCrystal
loads it like any other material and the plugin samples the ENDF/TSL evaluation.
Here we just ask NCrystal for sigma_scatter(E) on an energy grid and plot it.

Examples
--------
# total per-atom cross section of one plugin material:
python plot_cross_section.py out/graphite.ncmat --temp 296 -o graphite_xs.png

# break the total into coherent / incoherent-elastic / inelastic channels:
python plot_cross_section.py pmma/pmma_endftsl.ncmat --temp 300 --channels -o pmma_xs.png

# (optional) overlay another NCrystal cfg for reference -- NB this is a DIFFERENT
# physics source (NCrystal's own VDOS model), so expect shape differences:
python plot_cross_section.py al2o3/al2o3_endftsl.ncmat --temp 296 \
    --stock "Al2O3_sg167_Corundum.ncmat;coh_elas=0" -o al2o3_xs.png

NCrystal cross sections are PER ATOM. The plugin needs to be importable/discovered
by NCrystal (pip-install it, or set NCRYSTAL_PLUGIN_LIST to the built .so).
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import NCrystal as NC

_CHANNELS = [("coh_elas", "coherent elastic", "tab:blue"),
             ("incoh_elas", "incoherent elastic", "tab:green"),
             ("inelas", "inelastic", "tab:orange")]


def _xs(cfg: str, energies_ev):
    sc = NC.createScatter(cfg)
    return np.array([sc.crossSectionIsotropic(float(e)) for e in energies_ev])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ncmat", nargs="+", help="one or more plugin-produced .ncmat files")
    ap.add_argument("--temp", type=float, default=293.6, help="temperature K (default 293.6)")
    ap.add_argument("--stock", default=None,
                    help="optional reference NCrystal cfg to overlay (e.g. a stock "
                         "ncmat name, possibly with ';coh_elas=0')")
    ap.add_argument("--channels", action="store_true",
                    help="also plot the coh/incoh-elastic/inelastic decomposition "
                         "(only for the FIRST ncmat)")
    ap.add_argument("--label", action="append", default=None,
                    help="legend label for each ncmat (repeat; default = file stem)")
    ap.add_argument("--title", default=None, help="plot title override")
    ap.add_argument("--legend-loc", default="best", help="matplotlib legend loc")
    ap.add_argument("--emin-mev", type=float, default=0.1)
    ap.add_argument("--emax-mev", type=float, default=1.0e4)
    ap.add_argument("--npts", type=int, default=400)
    ap.add_argument("-o", "--out", default="cross_section.png")
    a = ap.parse_args(argv)

    energies = np.geomspace(a.emin_mev / 1000.0, a.emax_mev / 1000.0, a.npts)
    e_mev = energies * 1000.0

    labels = a.label or [os.path.splitext(os.path.basename(nc))[0] for nc in a.ncmat]
    if len(labels) != len(a.ncmat):
        ap.error(f"got {len(labels)} --label but {len(a.ncmat)} ncmat")

    fig, ax = plt.subplots(figsize=(9, 6))
    for nc, lab in zip(a.ncmat, labels):
        cfg = f"{nc};temp={a.temp}K"
        tot = _xs(cfg, energies)
        ax.loglog(e_mev, tot, lw=1.7, color="black", label=f"{lab} (total)")
        ax.axhline(tot[-1], ls=":", lw=0.7, alpha=0.5, color="0.5")
    if a.channels:
        for comp, label, color in _CHANNELS:
            try:
                y = _xs(f"{a.ncmat[0]};temp={a.temp}K;comp={comp}", energies)
            except Exception:
                continue
            if np.any(y > 0):
                ax.loglog(e_mev, y, lw=1.0, ls="--", color=color, alpha=0.8, label=label)
    if a.stock:
        ax.loglog(e_mev, _xs(a.stock, energies), color="0.45", lw=1.6,
                  label=f"{a.stock} (stock NCrystal)")

    ax.set_xlabel("incident energy (meV)")
    ax.set_ylabel(r"$\sigma_{\rm scatter}$ (barn / atom)")
    ax.set_title(a.title or f"Per-atom scattering cross section  (T = {a.temp} K)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc=a.legend_loc, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
