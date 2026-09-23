# ncrystal_plugin_ENDFTSL

A standalone NCrystal plugin that reads a complete **ENDF/TSL** thermal-scattering
evaluation — MF7/MT2 coherent **and** incoherent elastic **and** MF7/MT4 inelastic
S(α,β) — and reproduces that evaluation inside NCrystal, **verbatim**. It goes
beyond NCrystal's built-in `ncrystal_endf2ncmat`, which imports only the inelastic
S(α,β) and drops the ENDF elastic.

Self-contained: nothing here imports `irma`. (It lives in the IRMA repo for now;
IRMA is used only as a dev tool to generate the test tapes.)

## Pipeline

```
tape.endf ──python -m ncrystal_plugin_ENDFTSL──▶ <id>.endftslpack + <id>.ncmat
NCrystal.createScatter("<id>.ncmat") ──▶ the C++ plugin samples:
   inelastic   → NCrystal SABScatter            (from the pack's S(α,β))
   incoherent  → NCrystal ElIncScatter          (MSD = W'·ħ²/2mₙ)
   coherent    → NCEndfCohElasScatter (NEW)      (the tape's own S(E) Bragg edges, verbatim)
```

## Convert a tape

```bash
python -m ncrystal_plugin_ENDFTSL tsl.endf -o out/ \
    --material-id graphite --symbol C --mass 12.0107 --density 2.26 --temperature 296
```

Writes `out/graphite.endftslpack` (the per-temperature scattering data) + a
**structure-free** `out/graphite.ncmat` (no `@CELL` — coherent elastic comes from
the pack) carrying a `@CUSTOM_ENDFTSL` section. `--temperature` must be one of the
temperatures stored in the tape (the converter selects that column exactly — no
interpolation); re-run per temperature.

A **polyatomic** material lists one tape per principal scatterer in a YAML config
(atom `fraction`s summing to 1); the packs are per-atom normalized and summed.
Every channel of a tape, the coherent Bragg edges included, is scaled by its
species' atom fraction, as transport codes do:

```bash
python -m ncrystal_plugin_ENDFTSL --config beo.yaml -o out   # e.g. BeO -> Be + O packs
```

## Examples

See [`examples/`](examples/) for runnable, per-material walkthroughs — convert a
tape then **plot the cross section in NCrystal** — covering every elastic case:
graphite (`LTHR=3`), BeO (`LTHR=1` coherent), Al₂O₃ (`LTHR=0` inelastic-only), and
PMMA (`LTHR=2`, hydrogenous). `examples/plot_cross_section.py` is a reusable helper.

## Build the C++ plugin

Needs NCrystal ≥ 4.3.0 + cmake + ninja + scikit-build-core, built into an env with a
matching NCrystal:

```bash
pip install . --no-build-isolation --no-deps
```

NCrystal auto-discovers the installed plugin via `ncrystal-pypluginmgr`; or force a
load with `NCRYSTAL_PLUGIN_LIST=/abs/.../libNCPlugin_ENDFTSL.so` (the reliable path
for a compiled-C host such as a McStas instrument).

## Physics / conventions

- **Coherent elastic (MF7/MT2 LTHR=1/3):** the tape's cumulative `S(E)` Bragg-edge
  table is consumed verbatim; `σ_coh(E)=S(E)/E` (histogram, `INT=1`), `μ=1−2Eᵢ/E`.
  This reproduces what NJOY→ACE→MCNP/SCALE sample — the point of the plugin.
- **Incoherent elastic (MF7/MT2 LTHR=2/3):** `MSD = W'·ħ²/2mₙ`, `xs = σ_b` → NCrystal
  `ElIncScatter` reproduces the ENDF `(σ_b/2)(1−e⁻⁴ᴱᵂ')/(2EW')` exactly.
- **Inelastic (MF7/MT4):** physical S(α,β) (un-`LAT`'d, de-scaled) → scaled-symmetric
  half-table → NCrystal `SABScatter`.
- A missing elastic block (LTHR=1-only or 2-only tape) → that channel is cleanly absent.

## Tests

- `tests/` — the converter (reader/physics/pack/convert/ncmat/CLI) + zero-out.
- `reference/` — end-to-end gate: convert → load → coherent `σ=S/E` verbatim,
  incoherent vs the ENDF formula, total vs a vendored expected cross section.

Run: `pip install . && python -m pytest tests reference -q` (or, Python-only,
`PYTHONPATH=python python -m pytest tests`).
