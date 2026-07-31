"""Shared fixtures for the IRMA test suite.

These tests pin pure-function behavior (no phonopy / NJOY needed) so the
recently-changed, format-churn-prone surfaces (Bragg-edge grouping, multiphonon
order policy, deck parsing) can't silently regress.
"""
# Pin native BLAS/OMP thread pools to 1 BEFORE numpy is imported anywhere in the
# session (conftest.py is pytest's earliest import). The mode-1/2 engine forks
# its worker pool; a multithreaded BLAS pool spun up first leaves idle pthreads
# that can hold the glibc malloc lock at fork() and deadlock a worker (the
# fork-after-threads hazard). Setting these first means OpenBLAS never creates
# its pool, so every engine fork in the suite is safe regardless of import
# order. setdefault preserves a deliberate environment override.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from irma.core.engine import CrystalStructure, AtomSite


@pytest.fixture
def graphite():
    """Hexagonal graphite (P6_3/mmc), 4 C atoms — the standard validation crystal."""
    C = AtomSite(b_coh_fm=6.646,
                 positions=[(0.0, 0.0, 0.25), (0.0, 0.0, 0.75),
                            (1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)])
    return CrystalStructure(a=2.464, b=2.464, c=6.711,
                            alpha=90.0, beta=90.0, gamma=120.0, sites=[C])


@pytest.fixture
def dense_bragg():
    """A synthetic Bragg-edge set: a few real edges below 1 eV plus a very dense
    block above, ending in a flat emax endpoint. Returns (E, s_raw, emax).

    Deterministic (no RNG) so the grouping tests are reproducible.
    """
    E_low = np.array([0.002, 0.01, 0.05, 0.2, 0.5, 0.9])
    E_high = np.linspace(1.001, 4.999, 3000)            # dense block just above 1 eV .. 5 eV
    emax = 5.0
    E = np.concatenate([E_low, E_high, [emax]])
    s_low = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.4])
    s_high = 0.002 + 0.001 * np.cos(np.arange(3000))    # small, varied, >0
    s_raw = np.concatenate([s_low, s_high, [0.0]])       # emax endpoint: zero increment
    return E, s_raw, emax


# ---------------------------------------------------------------------------
# GUI tests must never open a REAL modal dialog.
#
# The suite runs on a live display (the GUI tests are display-gated, not
# headless), so a messagebox or file dialog from a test does not just slow
# things down: it appears on the developer's screen and blocks the whole
# run until someone clicks it. That happened twice while agents were
# running suites in the background. Individual tests already stub the
# dialogs they assert on; this is the backstop for every path that reaches
# one incidentally, e.g. a validation error raised by a deliberately
# incomplete form.
#
# A test that wants to assert on dialog behavior still patches it itself
# (mock.patch.object(module, "messagebox")), which takes precedence.
@pytest.fixture(autouse=True)
def _no_blocking_tk_dialogs(monkeypatch):
    """Make tkinter's modal dialogs non-blocking for the whole suite."""
    try:
        from tkinter import filedialog, messagebox
    except Exception:                      # no tkinter: nothing to guard
        return
    answers = {"showinfo": "ok", "showwarning": "ok", "showerror": "ok",
               "askyesno": False, "askokcancel": False,
               "askretrycancel": False, "askquestion": "no"}
    for name, result in answers.items():
        if hasattr(messagebox, name):
            monkeypatch.setattr(messagebox, name,
                                lambda *a, _r=result, **k: _r)
    for name in ("askopenfilename", "asksaveasfilename", "askdirectory"):
        if hasattr(filedialog, name):
            monkeypatch.setattr(filedialog, name, lambda *a, **k: "")
