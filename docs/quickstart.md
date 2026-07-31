# Quickstart

This page gets you from a fresh install to a written ENDF tape in about five
minutes. You will run a committed example deck through the command-line
interface, learn what each line of console output means, read an annotated
minimal classic deck, and find the pointers to the GUI and the rest of the
manual. It assumes IRMA is already installed (see [Installation](installation.md))
and that you can run `python -m irma`.

The defaults deserve a word before you start: every value IRMA fills
in on its own (the GUI's fresh form, the inputs `irma mlip emit`
writes, any configuration field you leave out) is the production
setting from the validation campaigns. Accepting them wholesale gives
results of the quality shown in the [validation record](validation/methodology.md),
not a rough first draft.

## Run a deck in one command

IRMA's command line takes an input deck and an output path:

```bash
python -m irma input.input out.endf
```

- `input.input` is a LEAPR-style deck (the `.input` or `.leapr` extension is
  conventional, not required).
- `out.endf` is the ENDF-6 File 7 tape IRMA writes. IRMA always writes the
  file you name on the command line; Card 1's `nout` field is kept only for
  LEAPR compatibility and is otherwise ignored.

A real, validated graphite deck ships with the repository, so you can try the
full path right now:

```bash
python -m irma \
  tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/tsl-crystalline-graphite.input \
  graphite.endf
```

This is a classic crystalline-graphite evaluation (a deck-supplied phonon
density of states (DOS), isotropic Debye-Waller, the same physics level as
`iel=10` with `inelastic_mode=0`); it runs in seconds and reproduces the
published ENDF/B-VIII.1 graphite tape to 3.4×10⁻⁵ (the graphite entry of the
[reference set](validation/methodology.md)).

If you installed IRMA with `pip`, the bare `irma` console script is equivalent
to `python -m irma`, so `irma input.input out.endf` runs the same evaluation.
The two-positional deck form also has explicit aliases that do exactly the
same thing (`irma evaluate input.input out.endf` and `irma run input.input
out.endf`), handy when you want the intent spelled out in a script.

## Reading the console output

IRMA narrates each stage to stdout as it works. For the graphite run the
output looks like this (the middle temperatures are omitted here):

```text
IRMA: graphite
  ntempr=10, iprint=2, nphon=100
  mat=30, za=130, isabt=0, ilog=0, iint=0
  awr=11.898, spr=4.73918, npr=1, iel=1, ncold=0, nsk=0
  nalpha=150, nbeta=400, lat=1

  Principal scatterer...
  Temperature 1: 296.00 K
    DW lambda = 0.868056, T_eff = 706.228
  Temperature 2: 400.00 K
    (negative T: reusing previous temperature's scattering-law inputs)
    DW lambda = 1.490620, T_eff = 748.880
  ...
  Temperature 10: 2000.00 K
    (negative T: reusing previous temperature's scattering-law inputs)
    DW lambda = 33.902922, T_eff = 2089.988
  Found 345 Bragg edges below 5 eV
  Read 34 comment cards

  Writing ENDF output...
ENDF output written to graphite.endf
  IRMA complete.
```

| Stage | What it means |
|-------|---------------|
| `IRMA: <title>` | The banner. IRMA echoes Card 2's quoted title so you can confirm the right deck loaded. |
| Echoed control cards (`ntempr`, `mat`, `awr`, `nalpha`, ...) | IRMA reads and prints back the values it parsed from Cards 3-7. Use these to verify the parser saw what you intended: a wrong `iel` or `nbeta` shows up here. |
| `Temperature N: <T> K` | One line per temperature in the deck. The S(α,β) detail block for that temperature is computed next. A negative temperature in the deck prints `(negative T: reusing previous temperature's ...)`, the LEAPR shared-spectrum convention. |
| `DW lambda = ..., T_eff = ...` | The Debye-Waller lambda and effective temperature computed from that temperature's phonon spectrum. Both end up on the tape: the lambda sets the elastic Debye-Waller suppression, and the effective temperature drives THERMR's short-collision-time extension downstream. |
| `Found <N> Bragg edges below 5 eV` | The coherent-elastic phase. Printed when the principal scatterer has an elastic treatment (`iel` ≥ 1 or `iel=10`). |
| `Read <N> comment cards` | The trailing quoted comment cards become the ENDF MF1/MT451 description. |
| `Writing ENDF output...` / `ENDF output written to <path>` | The S(α,β) table and the elastic data are assembled into MF7 and serialized to the output path. |
| `IRMA complete.` | Success. The tape at your output path is ready for downstream processing (e.g. NJOY THERMR/ACER). |

If the deck is malformed, IRMA stops at the failing card: the console shows
the lines above up to that point, then an `Input deck error:` message that
names the offending card, the expected-vs-found values, and the input line
number, then exits non-zero. A clean engine-level failure (for example a
missing phonopy file) prints `IRMA failed: <reason>`. No ENDF tape is written
in either case.

One hazard lives downstream of a successful run, and you should know about it
before taking a coherent tape to NJOY. It is important to note that stock
NJOY2016 THERMR has a `cliq` bug that turns a correct coherent
`inelastic_mode = 2` tape (graphite and similar) into ~`1e91`-barn garbage
above ~`0.27 eV`, or stalls for hours. The bug fires on the *shape* of the
tabulated S(α,β), not on its grid, so no beta-grid choice avoids it; apply the
one-line two-axis guard patch to `src/thermr.f90` first.
`inelastic_mode = 0/1` tapes and most materials are unaffected. See
[NJOY interoperability](njoy.md).

## A minimal classic deck, annotated

The deck format is NJOY free-format: values are separated by spaces and every
card is terminated by `/`. The simplest useful deck is a classic evaluation, the same physics
level as `inelastic_mode=0`: an isotropic Debye-Waller treatment plus a
cubic phonon expansion built from a scalar phonon DOS you supply on the
deck. No phonopy is needed.

Here is the card skeleton (graphite-shaped, one temperature). The phonon DOS
and grid arrays are shown as `... /` placeholders; in a real deck they are the
full lists of values.

```text
24 /                          Card 1: nout (LEAPR-compat; output goes to the CLI path)
'graphite' /                  Card 2: quoted title (echoed as the banner)
1 2 100 /                     Card 3: ntempr iprint nphon
30 6012.0 0 0 /               Card 4: mat za isabt ilog (smin defaulted; mat is the evaluation's MAT number)
11.898 4.73918 1 1 0 0 /      Card 5: awr spr npr iel ncold nsk
0 /                           Card 6: nss (0 = no secondary scatterer)
150 400 1 /                   Card 7: nalpha nbeta lat
3.32e-03 ... 6.64e+01 /       Card 8: alpha grid (nalpha values, increasing)
0.0 ... /                     Card 9: beta grid (nbeta values, increasing)
296.0 /                       Card 10: temperature [K]
7.8e-04 351 /                 Card 11: delta ni (DOS energy spacing, point count)
0.0 ... /                     Card 12: rho(1..ni) phonon DOS on the equidistant grid
0.0 0.0 1.0 /                 Card 13: twt c tbeta (translational/continuous weights)
0 /                           Card 14: nd (number of discrete oscillators)
'crystalline graphite' /      Optional MF1/MT451 comment card(s)
/                             A bare slash ends the comment section
```

What each control card carries, at a glance:

| Card | Fields | Role |
|------|--------|------|
| 1 | `nout` | Output unit (LEAPR compatibility only). |
| 2 | `'title'` | Quoted title; becomes the console banner. |
| 3 | `ntempr iprint nphon` | Temperature count, print level, phonon-expansion order. |
| 4 | `mat za isabt ilog smin` | ENDF MAT/ZA and storage flags. |
| 5 | `awr spr npr iel ncold nsk` | Principal scatterer: mass ratio, free-atom σ, atom count, elastic option, cold-H and pair-correlation options. |
| 6 | `nss b7 aws sps mss` | Secondary scatterer block (`nss=0` for none). |
| 7 | `nalpha nbeta lat` | Grid sizes; `lat=1` means grids are at the 0.0253 eV reference temperature. |
| 8 / 9 | α grid / β grid | Strictly increasing value lists, one `/` ends each card. |
| 10 | `T` | Temperature [K]; a negative value reuses the previous block. |
| 11-16 | `delta ni`, `rho`, `twt c tbeta`, `nd`, ... | The S(α,β) detail block (DOS, translational, oscillators). |

This table is an orientation, not the full specification. Field meanings,
defaults, valid ranges, the generalized-elastic cards (`iel=10`, Cards 6b-6g),
and the noncubic `inelastic_mode=1/2` workflows are documented card by card in
the [Input cards reference](input-reference.md); reach for it whenever a field
above is unclear.

Before you write your own deck, know Card 5's `iel`. It selects the
coherent-elastic treatment: `0` = none, `1`-`6` = the legacy built-in
materials (graphite, Be, BeO, Al, Pb, Fe), and `10` = generalized Bragg edges
from your own crystal structure. For a new material on a noncubic crystal,
`iel=10` is the recommended path; it also makes the anisotropic Debye-Waller
treatment and the phonopy-backed `inelastic_mode=1/2` workflows available.
The example above uses the built-in `iel=1` graphite tables, which keeps the
deck self-contained.

## Skipping the grid: let IRMA build it

Cards 8 and 9 are explicit α and β grids. The deck format has **no**
auto-grid flag. Automatic grids are built in the GUI (the **Grids** tab) or
from Python via the `irma.core.grids` functions; either way the result is an
explicit grid that lands on Cards 7–9 of the deck (or in the saved JSON
configuration). The [Automatic grids](grids.md) page explains the controls
(`N lower`, `N phonon`, `N upper`, and the linear-in-Q alpha parameters). The
default alpha grid is linear in momentum transfer Q (`dQ = 0.05 1/Å` up to
`Q = 12 1/Å`, then a logarithmic tail), which resolves the thermal upscatter
windows that a recoil-mirrored grid under-samples.

## First GUI launch

To point and click, or to inspect an unfamiliar deck field by field, launch
the graphical interface:

```bash
python -m irma --gui
```

The `irma-gui` console script does the same thing. From the GUI you can define
the crystal structure, set scattering parameters and grids, point at phonopy
data, run the calculation with a progress display, and save or load the whole
configuration as JSON. Existing `.input` / `.leapr` decks can be imported via
**File > Import Input File** to populate every field. The GUI is covered in
detail on the [Graphical interface](gui.md) page.

## Predict an instrument spectrum

The same install (plus `pip install -e ".[spectra]"`) also predicts what an
inelastic neutron scattering (INS) instrument measures. One command produces
a VISION-resolved spectrum from a phonopy model:

```bash
python -m irma spectra vision --phonopy-yaml phonopy.yaml --inelastic-mode 2 \
    --scatterer "C,5.551,11.898,6.646,0.001" --temperature 300 -o vision.csv
```

or, with no eigenvectors at all, from a phonon DOS via a config file. The
[Neutron-scattering spectra](spectra.md) page has the end-to-end example and the
full CLI/config reference; [DOS-based spectra (mode 0)](spectra-mode0.md)
covers the DOS-only workflow; ready-to-run configurations live under
`examples/spectra/`.

## Where to go next

- **[Input cards reference](input-reference.md)**: the authoritative
  card-by-card spec, including `iel=10` and the noncubic modes.
- **[Automatic grids](grids.md)**: let IRMA build α/β grids for you.
- **[Scattering modes](modes.md)**: the elastic `iel` options (including the
  generalized `iel=10` Bragg edges for any crystal), SEF (single-channel
  elastic format) vs MEF, and when to
  use `inelastic_mode` 0, 1, or 2.
- **[Neutron-scattering spectra](spectra.md)**: instrument-resolved INS
  spectra and S(Q,E) maps with `irma spectra`; mode 0's DOS-only workflow has
  [its own page](spectra-mode0.md).
- **[NJOY interoperability](njoy.md)**: preparing tapes for THERMR/ACER.
- **[Validation](validation/graphite.md)**: how IRMA's results compare against NJOY,
  Euphonic, OCLIMAX, and measured graphite cross sections.
