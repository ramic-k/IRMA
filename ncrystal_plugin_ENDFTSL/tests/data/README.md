# ENDFTSL test fixtures

Two graphite ENDF/TSL tapes (the MEF tape was made with IRMA v0.18.0). Both
decks use the vendored
phonopy model `tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml`
(Card 6f, a repository-relative path), mode 2 (exact coherent one-phonon term,
anisotropic Debye-Waller), 4000/200 powder directions, 296 K, ZA 6012, and
explicit 200 x 426 alpha/beta grids with the default log-linear Card 4
interpolation.

| Deck | Tape | Card 6b | Mesh | Used by |
|---|---|---|---|---|
| `graphite_mef_296K.input` | `../../examples/graphite/graphite_mef_296K.endf` | `2 1 0 2` (MEF, MF7/MT2 LTHR=3: coherent Bragg edges plus incoherent Debye-Waller) | 40 x 40 x 40 | the converter tests and the reference gate |
| `graphite_cef_296K.input` | `graphite_cef_296K.endf` | `1 1 0 2` (coherent elastic only, LTHR=1) | 12 x 12 x 12 | `test_zero_out.py` |

The MEF tape carries both elastic components and the inelastic S(alpha,beta)
(MF7/MT4, `LAT=1, LASYM=0, LLN=0`), so one fixture exercises the full elastic
and inelastic stack.

To regenerate, run from the repository root:

```bash
python -m irma ncrystal_plugin_ENDFTSL/tests/data/graphite_mef_296K.input \
    ncrystal_plugin_ENDFTSL/examples/graphite/graphite_mef_296K.endf
python -m irma ncrystal_plugin_ENDFTSL/tests/data/graphite_cef_296K.input \
    ncrystal_plugin_ENDFTSL/tests/data/graphite_cef_296K.endf
```

These are checked-in fixtures; regenerate them only deliberately. The
converter reproduces whatever the tape contains.
