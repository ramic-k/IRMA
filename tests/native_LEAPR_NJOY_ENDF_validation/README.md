# Validation against native LEAPR/NJOY reference ENDF

These are end-to-end validation cases: IRMA is run on a real evaluated
thermal-scattering deck and its ENDF output (MF7/MT2 elastic, MF7/MT4
inelastic) is compared against a reference tape: the published
ENDF/B-VIII.1 evaluation, or a freshly generated NJOY2016.78 tape (see
Reference provenance below).

## What is here

This directory is **fully self-contained** — the source deck, the IRMA
translation, and the reference output for each material all live together, so
the validation can be reproduced from the repo alone with no external data.

```
leapr_decks/                       one self-contained triple per material:
  tsl-crystalline-graphite.{leapr,input,endf.gz}   graphite,  iel=1, 10 T
  tsl-026_Fe_056.{leapr,input,endf.gz}             bcc iron,  iel=6,  6 T
  tsl-013_Al_027.{leapr,input,endf.gz}             fcc Al,    iel=4,  6 T
  tsl-HinCH2.{leapr,input,endf.gz}                 H in CH2,  iel=0 + free-gas C, 15 T
  tsl-l-CH4.{leapr,input,endf.gz}                  liquid CH4: trans (diffusion) +
                                                   4 discrete oscillators + free-gas C, 1 T
  tsl-ortho-H.{leapr,input,endf.gz}                liquid ortho-H2: coldh (ncold=1, nsk=2
                                                   S(kappa) table) + trans + discre, 7 T
  tsl-para-H.{leapr,input,endf.gz}                 liquid para-H2: coldh (ncold=2, nsk=2
                                                   S(kappa) table) + trans + discre, 7 T
  tsl-BeO.{leapr,input,endf.gz}                    BeO (ENDF model): TWO-PASS secondary
                                                   (nss=1 b7=0, O gets its own phonon
                                                   pass) + iel=3 builtin, 8 T
leapr_to_irma_input.py           converter: NJOY LEAPR deck  -> IRMA deck
validate_native_leapr_endf.py      comparator: IRMA ENDF vs reference ENDF
README.md                          this file
```

For each material, `leapr_decks/` holds three paired files:

* `.leapr`    — the original NJOY LEAPR deck (the source of record);
* `.input`    — its IRMA translation, produced by `leapr_to_irma_input.py`;
* `.endf.gz`  — the reference tape, gzipped (the comparator
  decompresses it on the fly).

The fresh references were produced with **unmodified NJOY2016.78**, which is
exactly what IRMA's classic path reimplements, so those reproductions are
bit-exact. The published references are the released ENDF/B-VIII.1 files,
whatever program generated them (see Reference provenance below).
`leapr_to_irma_input.py` is a general NJOY-LEAPR→IRMA converter, not
specialized to these materials.

### Reference provenance

For graphite/Fe/Al/CH2 the vendored reference is the published evaluated
tape. The generating code, per each tape's own MF1/MT451 comments: NJOY
LEAPR for graphite; **FLASSH** for H in CH2 (checked 2026-07-24); Al and Fe
name no generating code. For the three liquid materials the vendored reference
was generated locally by running **unmodified NJOY2016.78** on the vendored
`.leapr` deck, because the published tapes are not standard-LEAPR output:

* **ortho-H / para-H** (ENDF/B-VIII.1) were generated with **NJOY-H2D2**, the
  CAB-modified LEAPR; standard NJOY2016 run on the same decks differs from the
  published tapes by up to ~2.4 % (integral ~0.5 %).
* **l-CH4** is the 1994-era evaluation; the published tape differs from a
  fresh NJOY2016 run of its own deck by up to ~7 % locally (Teff by 0.17 %) —
  three decades of LEAPR evolution, not a model change.
* **BeO** is NJOY2016's own regression deck (`tests/23/input`, the extended
  ENDF model) — there is no published tape for it, so the reference is the
  local NJOY2016.78 run by construction.

IRMA reproduces the fresh NJOY2016 tapes **exactly (0.0 measured
difference on every S value and Teff)** for all four.

## How the IRMA decks were derived

The reference `.leapr` files are NJOY *job streams* (`reconr`/`broadr`/`leapr`/
`thermr`/`acer`/`plotr` …). IRMA runs only the LEAPR scattering-law step. It
also accepts the job stream directly (it locates the `leapr … stop` block);
the converter produces a standalone deck for readability, with a small,
well-defined translation of the LEAPR block:

1. Take the block following the `leapr` module line, up to the next module
   name or `stop` (so a multi-module job — e.g. aluminum — yields only its
   LEAPR part).
2. Reformat Card 1: NJOY writes `nout` on its own bare line; IRMA wants a
   terminated card (`25 /`).
3. Ensure Card 2 (title) is a single quoted string (the H-in-H₂O CAB deck
   gives an unquoted multi-word title that would otherwise truncate).
4. Copy every remaining card **verbatim** — Cards 3–9, the per-temperature
   detail blocks (continuous phonon spectrum, translational/diffusion weights,
   discrete oscillators), the temperature cards, and the MF1/MT451 comment
   block. The classic-path (`iel<10`) card grammar is identical between NJOY
   LEAPR and IRMA, so no field-level rewriting is needed.

### Negative-temperature convention

Several decks use the LEAPR shorthand where a **negative temperature** means
"reuse the previous temperature's scattering-law inputs unchanged and only
recompute the law at the new |T|" (the deck omits the detail block for those
temperatures). IRMA honors this convention
(`tests/test_negative_temperature.py`). The derived decks preserve the
negative-temperature cards exactly as the references write them (graphite
has 9, iron and aluminum 5 each, polyethylene 14, BeO 14 over its two
passes).

## Running the validation

```bash
# reproduce + compare one material (runs IRMA on the vendored .input deck,
# then diffs MF7 against the vendored .endf.gz reference — no external data)
python tests/native_LEAPR_NJOY_ENDF_validation/validate_native_leapr_endf.py \
    crystalline-graphite
```

`--rtol` bounds the inelastic max relative difference at *physically
significant* S (S > 1e-3·S_max); `--itol` bounds the per-temperature
α/β-integrated S ratio. The strict all-points "max rel d" is reported for
information but is dominated by a handful of near-zero tail points.

## Results

| material        | iel | ref code        | temps | inel. max-rel-d (sig.) | ΣS ratio | elastic |
|-----------------|-----|-----------------|-------|------------------------|----------|---------|
| graphite        | 1   | published (NJOY LEAPR) | 10 | 3.4e-5              | 1.1e-5   | MT2 coh: 221=221 edges, strength ratio 1.00000 |
| 026_Fe_056      | 6   | published (code not named) | 6 | 6.9e-5          | 1.9e-5   | MT2 coh: 602=602 edges, strength ratio 1.00000 |
| 013_Al_027      | 4   | published (code not named) | 6 | 6.2e-5          | 1.9e-5   | MT2 coh: 568=568 edges, strength ratio 1.00000 |
| HinCH2          | 0   | published (FLASSH) | 15 | 2.0e-5                  | 2.9e-7   | MT2 incoh-elastic (LTHR=2) |
| l-CH4           | 0   | NJOY2016.78 (local) | 1 | 0.0                    | 0.0      | none |
| ortho-H         | 0   | NJOY2016.78 (local) | 7 | 0.0                    | 0.0      | none |
| para-H          | 0   | NJOY2016.78 (local) | 7 | 0.0                    | 0.0      | none |
| BeO (two-pass)  | 3   | NJOY2016.78 (local) | 8 | 0.0                    | 0.0      | MT2 coh: 236=236 edges, strength ratio 1.00000 |

IRMA reproduces the four fresh NJOY2016.78 tapes exactly and the four
published tapes to at most 7e-5 at significant S (the α/β-integrated S to
~1e-5), with the coherent-elastic Bragg structure exact (same edge count
after degenerate-shell merging, same total strength to 5 figures). The set
covers seven distinct code paths: hexagonal (graphite, iel=1), bcc (iron, iel=6) and
fcc (aluminum, iel=4) built-in coherent elastic, a hydrogenous molecular
moderator (polyethylene, iel=0) with a free-gas secondary scatterer and
incoherent elastic (LTHR=2), liquid methane (`trans` with a diffusive
component + 4 discrete oscillators + free-gas secondary), and liquid
ortho-/para-hydrogen (`coldh` ortho/para statistics with a 1500-point S(κ)
table + translational + oscillator, with a fresh detail block at every
temperature). The `skold` step runs only for ncold = 0 (in NJOY and IRMA
alike), so these decks do not exercise it; the fast-CI minitape
`tests/test_coldh_skold_minitape.py` pins it against NJOY. The BeO case adds the
two-pass secondary scatterer (nss=1, b7=0: the O atom gets a complete second
phonon-spectrum pass and the laws are merged) and validates the iel=3 BeO
built-in coherent elastic against NJOY. Together they cover the classic
kernels `contin`, `trans`, `discre` and `coldh` and both secondary
conventions (free-gas and two-pass).

The strict all-points "max rel d" (~1e-4) is slightly larger than the
significant-S figure because of a handful of near-zero tail points.

### Materials deliberately excluded

* **Be-metal** — its reference File 7 was produced with FLASSH (an independent
  code), not NJOY LEAPR, so it is a cross-code *agreement* case rather than a
  bit-exact one and is kept out of this set. (Note: validating it well needs
  the generalized-elastic path `iel=10` with the published hcp lattice
  a=2.2856, c=3.5842 Å, because the built-in `coher(lat=2)` uses a
  cubic-approximation edge treatment with c=3.5832 Å.)
* **H-in-H₂O (CAB)** — the full 94-temperature run and ~90 MB reference are too
  heavy for a routine validation case.

## Fast checks in the committed suite

`validate_native_leapr_endf.py` is a slow, reference-dependent harness and is
not part of CI. The committed fast suite instead pins the pieces this validation
depends on:

* `tests/test_negative_temperature.py` — the negative-temperature (shared-DOS
  multiple-temperature) convention (reuse ≡ an explicit repeated positive-T
  block).
* `tests/test_native_leapr_decks.py` — the derived decks parse into the
  expected LEAPR control cards (Card 1 reformatting, title quoting,
  iel/grid/temperature header, no leftover NJOY wrapper lines).
* The minitape tests (`tests/test_coldd_minitape.py`,
  `tests/test_coldh_skold_minitape.py`, `tests/test_two_pass_minitape.py`,
  `tests/test_writer_flag_tapes.py`) — small decks whose MF7 must be byte
  identical to unmodified-NJOY2016.78 tapes, which guards the NJOY
  invariant in CI.
