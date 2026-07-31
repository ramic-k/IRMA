"""CrysXT regression gate (CI-safe, no NCrystal/CrysXT needed).

``tests/data/extinction_crysxt_expected.npz`` freezes the Be Bragg planes, an
energy grid, and the CrysXT plugin's coherent-elastic sigma for nine
model/recipe/distribution cases (regenerate with
``tests/data/regenerate_crysxt_expected.py`` in mantid_env). This test recomputes the same
sigma from :func:`irma.core.extinction.extinction_factor` on the frozen planes
and asserts it reproduces CrysXT — pinning the IRMA port to the reference so any
drift in the extinction models is caught in CI.
"""
import math
import os

import numpy as np
import pytest

from irma.core.extinction import extinction_factor

_EXPECTED = os.path.join(os.path.dirname(__file__), "data",
                         "extinction_crysxt_expected.npz")

# label -> (model, extinction_factor kwargs). MUST match the capture script.
CASES = {
    "BC_mix Gauss std": ("BC_mix", dict(l=8550.0, g=170.0, L=75750.0, dist="Gauss", recipe="std")),
    "BC_mix Gauss cls": ("BC_mix", dict(l=8550.0, g=170.0, L=75750.0, dist="Gauss", recipe="cls")),
    "BC_mod Lorentz std": ("BC_mod", dict(l=8550.0, g=170.0, L=75750.0, dist="Lorentz", recipe="std")),
    "BC_pure primary": ("BC_pure", dict(l=8550.0, g=0.0, L=0.0, dist="Gauss")),
    "BC_pure secII Fresnel": ("BC_pure", dict(l=8550.0, g=0.0, L=75750.0, dist="Fresnel", recipe="std")),
    "BC_pure secII Fresnel cls": ("BC_pure", dict(l=8550.0, g=0.0, L=75750.0, dist="Fresnel", recipe="cls")),
    "BC_pure secII Lorentz cls": ("BC_pure", dict(l=8550.0, g=0.0, L=75750.0, dist="Lorentz", recipe="cls")),
    "Sabine_uncorr rect": ("Sabine_uncorr", dict(l=8550.0, g=170.0, L=75750.0, dist="rect")),
    "Sabine_corr": ("Sabine_corr", dict(l=8550.0, g=170.0, L=75750.0)),
}


def _port_sigma(model, d, fsq, mult, F, V, N, WL2EKIN, E, kw):
    """Coherent-elastic sigma_coh(E) from the IRMA port on the frozen planes
    (the same kinematic comb CrysXT used), summing only planes with 2d >= lambda."""
    Nc = 1.0 / V
    out = np.empty(len(E))
    for i, e in enumerate(E):
        wl = math.sqrt(WL2EKIN / e)
        s = 0.0
        for j in range(len(d)):
            if 2.0 * d[j] >= wl:
                y = extinction_factor(model, Nc, wl, F[j], d[j], **kw)
                s += d[j] * fsq[j] * mult[j] * y
        out[i] = wl * wl / (2.0 * V * N) * s
    return out


@pytest.fixture(scope="module")
def expected():
    data = np.load(_EXPECTED)
    return {
        "d": data["d"], "fsq": data["fsq"], "mult": data["mult"].astype(float),
        "V": float(data["V"]), "N": float(data["N"]),
        "WL2EKIN": float(data["WL2EKIN"]), "E": data["E"],
        "labels": [str(x) for x in data["labels"]], "crysxt": data["crysxt"],
    }


@pytest.mark.parametrize("label", list(CASES))
def test_port_reproduces_crysxt(expected, label):
    model, kw = CASES[label]
    idx = expected["labels"].index(label)
    cx = expected["crysxt"][idx]
    F = np.sqrt(expected["fsq"]) * 1e-4
    mine = _port_sigma(model, expected["d"], expected["fsq"], expected["mult"], F,
                       expected["V"], expected["N"], expected["WL2EKIN"], expected["E"], kw)
    m = cx > 1e-3 * cx.max()
    rel = np.abs(mine[m] - cx[m]) / cx[m]
    # the port reproduces CrysXT essentially exactly; a 0.5% gate flags real drift
    assert rel.max() < 5e-3, (
        f"{label}: port deviates from CrysXT by {100 * rel.max():.3f}% (max)")


def test_all_cases_present():
    g = np.load(_EXPECTED)
    assert set(str(x) for x in g["labels"]) == set(CASES)
