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
python -m pytest                   # fast suite (~25 s, no phonopy/NJOY)
```

CI (`.gitlab-ci.yml`) runs the fast suite on every push and merge request.
Keep it green; keep it fast — anything needing phonopy, NJOY, or more than a
few seconds belongs in the manual validation harnesses, not the fast suite.

The `irma.mlip` tests (`tests/mlip/`) follow the same rule with one twist:
they exercise the calculator factory through stubs and the displacement/dispatch
machinery through ASE's built-in EMT, so they need `ase`/`phonopy`/`PyYAML`
(the `[mlip]` extra) but no potential package and no network; without the
extra they skip cleanly. Tests that instantiate real potentials sit behind
the `mlip_real` marker and only run when you opt in with
`IRMA_MLIP_REAL_TESTS=1`.

## Validation harnesses (manual, not CI)

- `tests/native_LEAPR_NJOY_ENDF_validation/` — IRMA must reproduce the
  committed native LEAPR/NJOY reference tapes (~1e-4 inelastic, exact
  elastic structure). Run after touching the classic path, the writer, or
  anything in the ENDF output chain.
- `tests/mode2_euphonic_n1_validation/` — the mode-2 exact one-phonon law
  vs the committed Euphonic reference. Run after touching the noncubic
  engine. Regenerating the frozen reference requires `euphonic` and is only
  needed if the vendored phonon models change.

## Refactoring policy

Behavior-preserving changes to the physics chain are verified by
byte-identical output: build a baseline of representative decks covering
every affected code path (see the validation README for deck patterns —
classic, negative-temperature reuse, secondary scatterer, single-channel (SEF) / mixed (MEF) / grouped elastic, modes 1/2 with multiphonon and SCT), produce tapes before and
after, and `cmp` them. Single-process runs (`ncpu=1`) are deterministic and
byte-stable; use them for baselines.

For the noncubic modes 1/2 specifically, `tools/tape_gauge.py` is the
standing instrument: a fast 8³ iteration profile and a `--big` production
profile (graphite, 40³ mesh, standard sampling, the full automatic grids).
Run the baseline from a pristine worktree of the last blessed commit, the
candidate from the working tree, and `--diff` the two — it passes on
byte-identical tapes, or on parsed MF7/MT4 agreement within 1e-6 relative
(the ENDF write precision). Deliberate physics changes additionally need
their own quantification and a CHANGELOG record (see the 0.15.0 F16 entry
for the worked example).

## Conventions

- Three hard dependencies only (numpy, endf-parserpy, threadpoolctl).
  phonopy stays an optional extra; do not add imports of it outside lazy,
  mode-1/2-only paths. Anything heavier belongs in validation tooling, not
  the package.
- Native BLAS/OpenMP threads are pinned to 1 by design (measured: threaded
  BLAS inside the `ncpu` worker pool oversubscribes and is >10x slower).
  Parallelism belongs to the Card 6f `ncpu` process pool, which uses the
  spawn start method (Windows-compatible; a driver script calling IRMA at
  module level therefore needs the standard `if __name__ == "__main__":`
  guard). `IRMA_WORKER_THREADS` overrides the per-worker thread pin for
  tuning experiments only — the default of 1 is the measured optimum.
- Input-format changes must keep the LEAPR-derived card structure and be
  validated with friendly `DeckError` messages (card name, expected/found,
  input line) — see `tests/test_deck_errors.py`.
- Comments document the code as it is — no exploratory narration, no
  conversational asides. If a value or layout matches NJOY, cite the NJOY
  routine it matches.
