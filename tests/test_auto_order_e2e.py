"""End-to-end check that Card 6g auto_multiphonon_order=1 raises the
multiphonon order above the deck's nphon and changes the MT4 law, on the
vendored graphite phonopy model with a tiny configuration (mesh 4^3,
ndir=40, mpdir=20, 6 alpha x 8 beta, nphon=2). A phonopy-free test of
derive_required_multiphonon_order keeps the order formula covered on
runners without phonopy.
"""
import io
import math
import os
import re
import tempfile
import contextlib

import numpy as np
import pytest

# Pure formula-path coverage needs no phonopy.
from irma.core.noncubic_engine import derive_required_multiphonon_order  # noqa: E402

_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mode2_euphonic_n1_validation", "graphite",
    "phonopy.yaml"))

# Same tiny deck as tests/test_noncubic_fast_ci.py. The mode-line 4th
# field is inelastic_mode (kept = 1); the Card 6g control line ({ctrl})
# carries the auto_multiphonon_order 3rd field. Card 3 nphon = {nphon}.
_DECK = """20 /
'fast noncubic CI deck'/
1 1 {nphon}/
31 6012. 0 0 1e-100/
11.898 4.7392 1 10 0 0/
0/
1 1 0 1/
2.467 2.467 6.701 90.0 90.0 120.0/
6 12 11.898 6.6484 0.001 4/
0.0 0.0 0.25  0.0 0.0 0.75  0.333333333333 0.666666666667 0.25  0.666666666667 0.333333333333 0.75/
'{yaml}'/
4 4 4 1 0/
{ctrl}
6 8 1/
0.1 0.4 1.0 2.5 6.0 15.0/
0.0 0.4 1.0 2.0 3.5 5.5 8.0 12.0/
296.0/
'fast nc ci'/
/
"""

# Deliberately low deck order so the auto-size branch must raise it.
_DECK_NPHON = 2
_AUTO_LOG_RE = re.compile(
    r"auto-sizing order\s+(\d+)\s*->\s*(\d+)")


def test_derive_required_multiphonon_order_raises_above_requested():
    """Pure (no-phonopy) coverage of the order-formula path the e2e
    auto-size branch consumes: a low requested order is raised to the
    required order on a soft (graphite-c-axis-like) anisotropic tensor."""
    U = np.array([np.diag([0.005, 0.005, 0.011])])   # u_max = 0.011 Angstrom^2
    q_max = 46.7                                      # work-grid Q_max (tiny deck)
    effective, required, two_w, u_max = derive_required_multiphonon_order(
        q_max, U, requested_order=_DECK_NPHON)
    assert math.isclose(u_max, 0.011)
    assert math.isclose(two_w, q_max * q_max * 0.011)
    assert required > _DECK_NPHON                      # formula demands more
    assert effective == required                       # raised to the requirement
    assert effective > _DECK_NPHON                      # ... above the deck order


# ---------------------------------------------------------------------------
# End-to-end (phonopy-gated, like test_noncubic_fast_ci.py).
# ---------------------------------------------------------------------------
pytest.importorskip("phonopy")

from irma.core.engine import run_leapr  # noqa: E402


def _run(nphon, ctrl, tag):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, f"{tag}.input")
    out = os.path.join(d, f"{tag}.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(nphon=nphon, ctrl=ctrl, yaml=_YAML))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        run_leapr(inp, out)
    return out, log.getvalue()


def _law_sum(path):
    from endf_parserpy import EndfParserPy
    mt4 = EndfParserPy().parsefile(path)[7][4]
    s_sum = 0.0
    for b, tab in mt4["S_table"].items():
        if not isinstance(b, int):
            continue
        vals = tab["S"].values() if isinstance(tab["S"], dict) else tab["S"]
        for v in vals:
            assert math.isfinite(v)
            s_sum += v
    return s_sum


@pytest.fixture(scope="module")
def runs():
    # auto_order OFF (Card 6g 2 fields -> defaults to 0) vs ON (3rd field = 1),
    # both at the same low deck nphon.
    off, off_log = _run(_DECK_NPHON, "40 20/", "autooff")
    on, on_log = _run(_DECK_NPHON, "40 20 1/", "autoon")
    return {"off": (off, off_log), "on": (on, on_log)}


def test_auto_order_raises_effective_order_above_deck_nphon(runs):
    _, on_log = runs["on"]
    m = _AUTO_LOG_RE.search(on_log)
    assert m is not None, "auto-size branch did not run (no auto-sizing log line)"
    from_order, to_order = int(m.group(1)), int(m.group(2))
    assert from_order == _DECK_NPHON                   # started at the deck order
    assert to_order > _DECK_NPHON                       # auto-raised above it

    # The OFF run must NOT take the auto-size branch (it honors the deck order).
    _, off_log = runs["off"]
    assert "auto-sizing order" not in off_log


def test_auto_order_changes_the_emitted_law(runs):
    off, _ = runs["off"]
    on, _ = runs["on"]
    s_off, s_on = _law_sum(off), _law_sum(on)
    # The auto-raised order adds the high-order multiphonon background the
    # truncated (order=2) run is missing, so the law must change.
    assert s_off != pytest.approx(s_on, rel=1.0e-9)
    assert s_on > s_off                                 # added background, not removed
    # ... and the tapes themselves differ byte-for-byte.
    assert open(off).read() != open(on).read()


def test_auto_sized_run_prints_no_reach_warning(runs):
    """An auto-sized order meets the Poisson rule, so it reaches the recoil
    ridge plus the margin: the energy-reach guard and the check on the
    computed array both stay silent."""
    _, on_log = runs["on"]
    assert "WARNING: multiphonon order" not in on_log
    assert "becomes identically zero" not in on_log
