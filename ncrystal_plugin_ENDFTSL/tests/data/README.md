# ENDFTSL test fixtures

## graphite MEF (LTHR=3) tape — `../../examples/graphite/graphite_mef_296K.endf`

A vendored ENDF/TSL tape used by the converter test suite and the reference
gate. The single tracked copy lives with the graphite example
(`examples/graphite/graphite_mef_296K.endf`); the tests and reference scripts
read it from there. It carries **both** elastic components — MF7/MT2 **LTHR=3**
(coherent Bragg edges + incoherent Debye-Waller) — plus the inelastic S(α,β)
(MF7/MT4, `LAT=1, LASYM=0, LLN=0`), so v1 exercises the full elastic +
inelastic stack from a single fixture.

Generated with **IRMA v0.18.0 (fixture provenance; regeneration from a current checkout must be run FROM THE REPOSITORY ROOT -- the Card 6f phonopy path in the .input decks is repo-relative)** from `graphite_mef_296K.input`:

```bash
conda run -n irma_and_mcstas_environment python -m irma \
    graphite_mef_296K.input ../../examples/graphite/graphite_mef_296K.endf
```

- `graphite_mef_296K.input` is a copy of the repo's `examples/tsl/graphite_mode2.input`
  with **Card 6b** changed from `1 1 0 2 /` (SEF) to `2 1 0 2 /` (MEF) — the only
  edit. It is mode-2 (exact coherent one-phonon + anisotropic DW), 40×40×40 mesh,
  4000/200 powder directions, 296 K, with the phonopy model at
  `tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml`.

This is a **checked-in fixture** — regenerate only deliberately (the converter
tests read it; the converter just replays whatever the tape contains).

> NOTE: generating it needs phonopy's C backend. IRMA forces `lang="C"`
> (phonopy ≥ 4 defaults to the Rust `phonors` backend, whose rayon thread pool
> deadlocks the mode-2 worker ProcessPool — see `irma/core/phonopy_io.py`).
