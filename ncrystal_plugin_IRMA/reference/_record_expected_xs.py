"""Record the plugin's cross sections for the expected pack -> graphite_reference_xs.json.

Run in the PLUGIN env (NCrystal + ncrystal_plugin_IRMA installed). Invoked by
regenerate_expected.sh; takes the expected directory as argv[1]. The IRMA provenance
(git SHA / version) is read straight from the pack the producer just baked.
"""
import json
import os
import sys

import NCrystal as NC

EXPECTED = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "expected")
os.chdir(EXPECTED)                   # bare pack name in the NCMAT resolves via CWD

sha = ver = "unknown"
for line in open("graphite_reference__C.irmapack"):
    if line.startswith("meta.irma_git_sha"):
        sha = line.split("=", 1)[1].strip()
    if line.startswith("meta.irma_version"):
        ver = line.split("=", 1)[1].strip()

energies_mev = [1.0, 5.0, 25.0, 60.0, 150.0, 300.0]
xs = {}
for comp in ("total", "coh_elas", "incoh_elas", "inelas"):
    cfg = "graphite_reference.ncmat;temp=296.0K" + ("" if comp == "total" else f";comp={comp}")
    sc = NC.createScatter(cfg)
    xs[comp] = [float(sc.crossSectionIsotropic(e / 1000.0)) for e in energies_mev]

expected = {
    "description": "IRMA-produced reference for the NCrystal IRMA plugin "
                   "(graphite mode-2).",
    "irma_git_sha": sha, "irma_version": ver,
    "ncrystal_version": NC.__version__, "temperature_K": 296.0,
    "energies_meV": energies_mev, "cross_sections_barn": xs,
}
with open(os.path.join(EXPECTED, "graphite_reference_xs.json"), "w") as handle:
    json.dump(expected, handle, indent=2)
print("  recorded cross sections (IRMA SHA", sha + ");",
      "total@25meV =", round(xs["total"][2], 4), "barn")
