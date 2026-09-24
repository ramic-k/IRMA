# Contributing to IRMA

## Development setup

```bash
pip install -e ".[phonopy,spectra,mlip]"   # matches what CI tests; phonopy only
                                           # needed for inelastic_mode=1/2, spectra
                                           # for the irma.spectra forward model,
                                           # mlip for the irma.mlip front end
pip install -r requirements-dev.txt
```

## Tests

```bash
python -m pytest                   # fast suite (no NJOY)
```

CI (`.github/workflows/ci.yml`) runs the fast suite on every push and
pull request, across Linux/macOS/Windows and Python 3.11-3.13, plus the
two NCrystal plugin gates.
The suite must pass in every CI job. Tests that need phonopy must call
`pytest.importorskip("phonopy")` so the core-only job passes; runs longer
than a few seconds and anything needing NJOY belong in the manual
validation harnesses, not the fast suite.

The `irma.mlip` tests (`tests/mlip/`) follow the same rule, with one difference:
they exercise the calculator factory through stubs and the displacement/dispatch
machinery through ASE's built-in EMT, so they need `ase`/`phonopy`/`PyYAML`
(the `[mlip]` extra) but no potential package and no network; without the
extra they skip cleanly. Tests that instantiate real potentials sit behind
the `mlip_real` marker and only run when you opt in with
`IRMA_MLIP_REAL_TESTS=1`.

## Validation harnesses (manual, not CI)

- `tests/native_LEAPR_NJOY_ENDF_validation/` — IRMA must reproduce the
  committed native LEAPR/NJOY reference tapes (7e-5 or better inelastic, exact
  elastic structure). Run after touching the classic kernels, the writer, or
  anything in the ENDF output chain.
- `tests/mode2_euphonic_n1_validation/` — the mode-2 exact one-phonon
  S(α,β) vs the committed Euphonic reference. Run after touching the mode-1/2
  engine. Regenerating the frozen reference requires `euphonic` and is only
  needed if the vendored phonon calculations change.
- `tests/mode0_validation/` — the mode-0 (DOS) VISION spectrum vs mode 1 for
  graphite. Run after touching the spectra mode-0 path.

## Refactoring policy

Behavior-preserving changes to the physics chain are verified by
byte-identical output: build a baseline of representative input files covering
every affected code path (classic, negative-temperature reuse, secondary
scatterer, single-channel (SEF) / mixed (MEF) / grouped elastic, modes 1/2
with multiphonon), produce tapes before and after, and `cmp` them. Single-process runs (`ncpu=1`) are deterministic and
byte-stable; use them for baselines.

For modes 1/2 specifically, `tools/tape_gauge.py` is the regression check:
a fast 8³ iteration profile and a `--big` production profile (graphite,
40³ mesh, standard sampling, the full automatic grids). Run the baseline
from a clean worktree of the last commit whose tapes were accepted as
reference, the candidate from the working tree, and `--diff` the two — it passes on
byte-identical tapes, or on parsed MF7/MT4 agreement within 1e-6 relative
(the ENDF write precision). Deliberate physics changes additionally need
their own quantification and a CHANGELOG record stating what moved and by
how much.

## Conventions

- Three hard dependencies only (numpy, endf-parserpy, threadpoolctl).
  The core import path (`irma`, `irma.core`, the input-file CLI) must not
  import scipy, PyYAML or phonopy; import them lazily in the paths that
  need them (modes 1/2, `irma.spectra`, `irma.ncrystal`, `irma.mlip`).
  Anything heavier belongs in validation tooling, not the package.
- Native BLAS/OpenMP threads are pinned to 1 by design (measured: threaded
  BLAS inside the `ncpu` worker pool oversubscribes and is >10x slower).
  Parallelism belongs to the Card 6f `ncpu` process pool, which uses the
  spawn start method (Windows-compatible; a driver script calling IRMA at
  module level therefore needs the standard `if __name__ == "__main__":`
  guard). `IRMA_WORKER_THREADS` overrides the per-worker thread pin for
  tuning experiments only — the default of 1 is the measured optimum.
- Input-format changes must keep the LEAPR-derived card structure and be
  validated with `DeckError` messages (card name, expected/found,
  input line) — see `tests/test_deck_errors.py`.
- Comments document the code as it is — no exploratory narration, no
  conversational asides. If a value or layout matches NJOY, cite the NJOY
  routine it matches.
