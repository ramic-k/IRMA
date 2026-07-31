"""Regenerate ``extinction_crysxt_expected.npz`` (the CrysXT regression reference).

Run MANUALLY in an environment with NCrystal + the CrysXT plugin built
(e.g. mantid_env) when adding/changing cases or after a CrysXT version bump:

    conda run -n mantid_env python tests/data/regenerate_crysxt_expected.py

It freezes the Be Bragg planes, the energy grid, and CrysXT's coherent-elastic
sigma for each case; ``tests/test_extinction_crysxt_expected.py`` then replays it
without CrysXT. Keep CASES identical to that test.
"""
import os
import tempfile

import numpy as np
import NCrystal as NC

OUT = os.path.join(os.path.dirname(__file__), "extinction_crysxt_expected.npz")

info = NC.createInfo("Be_sg194.ncmat")
si = info.getStructureInfo()
V, N = si["volume"], si["n_atoms"]
WL2EKIN = NC.wl2ekin(1.0)
d = np.array([dd for (h, k, l_, mult, dd, fsq) in info.hklList()])
fsq = np.array([fsq for (h, k, l_, mult, dd, fsq) in info.hklList()])
mult = np.array([mult for (h, k, l_, mult, dd, fsq) in info.hklList()], float)
be_text = NC.createTextData("Be_sg194.ncmat").rawData

lam = np.geomspace(0.5, 3.9, 120)
E = WL2EKIN / lam ** 2

CASES = [
    ("BC_mix Gauss std",  "Extinction BC_mix 8550 170 75750 Gauss rec=std"),
    ("BC_mix Gauss cls",  "Extinction BC_mix 8550 170 75750 Gauss rec=cls"),
    ("BC_mod Lorentz std", "Extinction BC_mod 8550 170 75750 Lorentz rec=std"),
    ("BC_pure primary",   "Extinction BC_pure 8550 0 0 Gauss"),
    ("BC_pure secII Fresnel", "Extinction BC_pure 8550 0 75750 Fresnel rec=std"),
    ("BC_pure secII Fresnel cls", "Extinction BC_pure 8550 0 75750 Fresnel rec=cls"),
    ("BC_pure secII Lorentz cls", "Extinction BC_pure 8550 0 75750 Lorentz rec=cls"),
    ("Sabine_uncorr rect", "Extinction Sabine_uncorr 8550 170 75750 rect"),
    ("Sabine_corr",       "Extinction Sabine_corr 8550 170 75750"),
]


def crysxt_sigma(xline):
    body = be_text + "\n@CUSTOM_CRYSXT\n  " + xline + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".ncmat", delete=False) as f:
        f.write(body); path = f.name
    try:
        sc = NC.createScatter(f"{path};comp=coh_elas")
        return np.array([sc.crossSectionIsotropic(e) for e in E])
    finally:
        os.unlink(path)


cx = np.array([crysxt_sigma(xl) for _, xl in CASES])
np.savez_compressed(
    OUT, d=d, fsq=fsq, mult=mult, V=V, N=N, WL2EKIN=WL2EKIN, E=E,
    labels=np.array([lbl for lbl, _ in CASES]), crysxt=cx)
print(f"wrote {OUT}: {len(d)} planes, {len(E)} energies, {len(CASES)} cases")
