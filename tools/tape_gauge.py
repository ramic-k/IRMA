#!/usr/bin/env python3
"""Tape gauge: the re-blessing instrument for modes-1/2 output changes.

Any change that may touch mode-1/2 ENDF bytes (performance work that reorders
floating-point arithmetic, physics changes, numerics) is gated here: run the
gauge BEFORE the change from a pristine worktree, run it AFTER on the working
tree, and diff. ``--diff`` declares tapes equivalent when they are
byte-identical or when every parsed MF7 value (MT2 coherent elastic AND MT4
inelastic) agrees within 1e-6 relative (the ENDF write precision — the
project's "tape-safe" equivalence criterion). Any structural mismatch (a dropped or
added key, a truncated array, MT2 on one side only) fails closed as
max_rel=inf.
Deliberate physics changes additionally need their own quantification
(see CHANGELOG 0.15.0's F16 entry for the worked example).

Profiles
--------
small (default)
    Fast iteration gate: graphite, 8^3 mesh, ndir 200, a 12x16 explicit grid,
    plus one spectra spectrum + one 2-D map (CSV byte gate). ~20 s total.
--big
    The production bless gauge: graphite, 40^3 mesh, the
    validated standard sampling (Card 6g: ndir=10000 mpdir=1000 auto-order),
    14 workers, and the FULL automatic alpha/beta grids regenerated in-process
    by irma.core.grids (399 x 370 at the campaign defaults) — a two-temperature
    (296 + 500 K) mode-2 tape and a one-temperature mode-1 tape. ~4 min.

Usage
-----
    # baseline from a pristine worktree of the last blessed commit:
    git -C <repo> worktree add /tmp/wt_blessed <blessed-ref>
    IRMA_GAUGE_ROOT=/tmp/wt_blessed python tools/tape_gauge.py before --big
    # candidate from the working tree:
    python tools/tape_gauge.py after --big
    python tools/tape_gauge.py --diff before after

Outputs land under $IRMA_GAUGE_DIR (default /tmp/irma_tape_gauge)/<label>/
with a <label>.json of sha256 + wall seconds per run.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("IRMA_GAUGE_ROOT",
                      os.path.dirname(_TOOL_DIR))      # repo root by default
GAUGE_DIR = os.environ.get("IRMA_GAUGE_DIR", "/tmp/irma_tape_gauge")
PY = sys.executable
YAML = os.path.join(ROOT, "tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml")

# small profile: fast iteration gate (mesh 8^3, ndir 200, explicit mini grids)
DECK = """20 /
'irma tape gauge deck'/
{ntemp} 1 20/
31 6012. 0 0 1e-100/
11.898 4.7392 1 10 0 0/
0/
1 1 0 {mode}/
2.467 2.467 6.701 90.0 90.0 120.0/
6 12 11.898 6.6484 0.001 4/
0.0 0.0 0.25  0.0 0.0 0.75  0.333333333333 0.666666666667 0.25  0.666666666667 0.333333333333 0.75/
'{yaml}'/
8 8 8 4 0/
200 100/
12 16 1/
0.05 0.1 0.2 0.4 0.7 1.0 1.5 2.5 4.0 6.0 10.0 15.0/
0.0 0.2 0.4 0.7 1.0 1.5 2.0 2.8 3.6 4.5 5.5 7.0 8.5 10.0 12.0 14.0/
{temps}'irma tape gauge'/
/
"""

# big profile: production bless gauge — full automatic grids spliced in
DECK_BIG = """20 /
'irma tape gauge 40^3'/
{ntemp} 1 100/
31 6012. 0 0 1e-100/
11.898 4.7392 1 10 0 0/
0/
1 1 0 {mode}/
2.467 2.467 6.701 90.0 90.0 120.0/
6 12 11.898 6.6484 0.001 4/
0.0 0.0 0.25  0.0 0.0 0.75  0.333333333333 0.666666666667 0.25  0.666666666667 0.333333333333 0.75/
'{yaml}'/
40 40 40 14 0/
10000 1000 1/
{grids}{temps}'irma tape gauge'/
/
"""

SPECTRA_CFG = """material:
  phonopy_yaml: {yaml}
  mesh: [6, 6, 6]
  temperature_K: 300.0
  scatterers:
    - {{symbol: C, sigma_bound_b: 5.551, awr: 11.898, b_coh_fm: 6.646,
        sigma_inc_b: 0.001}}
physics:
  inelastic_mode: 1
  max_phonon_order: 10
  n_directions: 100
  multiphonon_directions: 50
  jobs: 4
  elastic: true
grid: {{e_min_meV: 0.0, e_max_meV: 100.0, de_meV: 2.0, dq_max_invA: 0.25}}
instrument: {{geometry: vision, e_fixed_meV: 3.5}}
"""


def _auto_grid_block():
    """The campaign-standard automatic alpha/beta grids as deck text.

    Regenerated in-process so before/after runs of the SAME gauge version
    produce textually identical decks (the generators live in irma.core.grids
    and are themselves byte-pinned by the test suite).
    """
    sys.path.insert(0, ROOT)
    from irma.core.grids import generate_alpha_grid, generate_beta_grid
    freq_max = 0.2107 * 1.1                 # graphite cutoff + engine headroom
    beta = generate_beta_grid(freq_max, 296.0)
    alpha = generate_alpha_grid(beta, 11.898, 296.0)

    def fmt(arr):
        lines, cur = [], []
        for v in arr:
            cur.append(f"{v:.6E}")
            if len(cur) == 5:
                lines.append(" ".join(cur))
                cur = []
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)

    return (f"{len(alpha)} {len(beta)} 1/\n"
            f"{fmt(alpha)}/\n{fmt(beta)}/\n")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(label, big=False):
    outdir = os.path.join(GAUGE_DIR, label)
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    env = dict(os.environ, MPLCONFIGDIR="/tmp", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    res = {}

    deck_tpl = DECK_BIG if big else DECK
    grids = _auto_grid_block() if big else ""
    for tag, mode, temps in (("mode2_2T", 2, "296.0/\n500.0/\n"),
                             ("mode1_1T", 1, "296.0/\n")):
        deck = os.path.join(outdir, f"{tag}.input")
        tape = os.path.join(outdir, f"{tag}.endf")
        with open(deck, "w") as f:
            f.write(deck_tpl.format(mode=mode, yaml=YAML, grids=grids,
                                    ntemp=temps.count("/\n"), temps=temps))
        t0 = time.time()
        subprocess.run([PY, "-m", "irma", deck, tape], check=True, env=env,
                       cwd=ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        res[tag] = {"sha256": sha(tape), "seconds": round(time.time() - t0, 2)}
    if big:     # the big gauge gates the tapes only
        json.dump(res, open(os.path.join(GAUGE_DIR, f"{label}.json"), "w"),
                  indent=1)
        print(json.dumps(res, indent=1))
        return

    cfg = os.path.join(outdir, "spec.yaml")
    with open(cfg, "w") as f:
        f.write(SPECTRA_CFG.format(yaml=YAML))
    csv = os.path.join(outdir, "spec.csv")
    t0 = time.time()
    subprocess.run([PY, "-m", "irma", "spectra", "run", cfg, "-o", csv],
                   check=True, env=env, cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res["spectrum"] = {"sha256": sha(csv), "seconds": round(time.time() - t0, 2)}

    mapcsv = os.path.join(outdir, "map.csv")
    t0 = time.time()
    subprocess.run([PY, "-m", "irma", "spectra", "map", cfg, "--q-max", "8",
                    "--dq-map", "0.5", "-o", mapcsv], check=True, env=env,
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res["map"] = {"sha256": sha(mapcsv), "seconds": round(time.time() - t0, 2)}

    json.dump(res, open(os.path.join(GAUGE_DIR, f"{label}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def _walk_rel_diff(x, y):
    """Max relative difference over all numeric leaves of two parsed structures.

    Fails CLOSED: any structural mismatch — unequal dict key sets, unequal
    sequence lengths, or leaves whose types/values can't be compared — is
    folded into the metric as inf, so a candidate that drops, adds, or
    truncates data can never pass the gauge.
    """
    max_rel = 0.0

    def walk(x, y):
        nonlocal max_rel
        if isinstance(x, dict) or isinstance(y, dict):
            if not (isinstance(x, dict) and isinstance(y, dict)) \
                    or set(x) != set(y):
                max_rel = float("inf")
                return
            for k in x:
                walk(x[k], y[k])
        elif isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            if not (isinstance(x, (list, tuple)) and isinstance(y, (list, tuple))) \
                    or len(x) != len(y):
                max_rel = float("inf")
                return
            for xi, yi in zip(x, y):
                walk(xi, yi)
        elif isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
            # bool is an int subclass, so this must precede the numeric branch:
            # True vs 1 is a TYPE change in the parsed tape, not a zero diff
            if type(x) is not type(y) or bool(x) != bool(y):
                max_rel = float("inf")
        elif isinstance(x, (int, float, np.number)) \
                and isinstance(y, (int, float, np.number)):
            denom = max(abs(float(x)), abs(float(y)), 1e-300)
            max_rel = max(max_rel, abs(float(x) - float(y)) / denom)
        elif x != y:        # non-numeric leaves (e.g. strings) must match exactly
            max_rel = float("inf")

    walk(x, y)
    return max_rel


def _mf7_rel_diff(mf7_a, mf7_b):
    """Max relative difference over the MF7 sections of two parsed tapes.

    Both MT2 (coherent elastic, ~98% of MF7 content on iel=10 decks) and MT4
    (inelastic) are gauged. A section absent from BOTH tapes is fine (not
    every deck emits elastic); present on one side only is a structural
    failure (inf).
    """
    max_rel = 0.0
    for mt in (2, 4):
        if (mt in mf7_a) != (mt in mf7_b):
            return float("inf")
        if mt in mf7_a:
            max_rel = max(max_rel, _walk_rel_diff(mf7_a[mt], mf7_b[mt]))
    return max_rel


def _tape_rel_diff(path_a, path_b):
    """Max relative difference over all parsed MF7 (MT2 + MT4) values."""
    from endf_parserpy import EndfParserPy
    return _mf7_rel_diff(EndfParserPy().parsefile(path_a)[7],
                         EndfParserPy().parsefile(path_b)[7])


def diff(a, b, tol=1.0e-6):
    ja = json.load(open(os.path.join(GAUGE_DIR, f"{a}.json")))
    jb = json.load(open(os.path.join(GAUGE_DIR, f"{b}.json")))
    ok = True
    if set(ja) != set(jb):
        # an artifact appearing or vanishing between runs is itself a failure
        print(f"ARTIFACT SET MISMATCH: only-in-{a}={sorted(set(ja) - set(jb))} "
              f"only-in-{b}={sorted(set(jb) - set(ja))}")
        ok = False
    for k in sorted(set(ja) & set(jb)):
        same = ja[k]["sha256"] == jb[k]["sha256"]
        verdict = "IDENTICAL"
        if not same:
            fa = os.path.join(GAUGE_DIR, a, f"{k}.endf")
            fb = os.path.join(GAUGE_DIR, b, f"{k}.endf")
            if os.path.exists(fa) and os.path.exists(fb):
                rel = _tape_rel_diff(fa, fb)
                within = rel < tol
                ok &= within
                verdict = (f"DIFFER max_rel={rel:.3e} "
                           f"({'WITHIN' if within else '*** EXCEEDS ***'} {tol:g})")
            else:
                ok = False
                verdict = "*** DIFFER (non-tape output: byte gate only) ***"
        print(f"{k:10s} bytes={verdict} "
              f"{ja[k]['seconds']:8.2f}s -> {jb[k]['seconds']:8.2f}s "
              f"({ja[k]['seconds'] / max(jb[k]['seconds'], 1e-9):.2f}x)")
    print("GAUGE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], big="--big" in sys.argv[2:])
