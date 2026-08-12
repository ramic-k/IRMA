# Quickstart

This page gets you from a fresh install to a written ENDF tape in about five
minutes. You will run a committed example input file through the command-line
interface, learn what each line of console output means, read an annotated
minimal classic input file, and find the pointers to the GUI and the rest of the
manual. It assumes IRMA is already installed (see [Installation](installation.md))
and that you can run `python -m irma`.

The defaults deserve a word before you start: every value IRMA fills
in on its own (the GUI's fresh form, the inputs `irma mlip emit`
writes, any configuration field you leave out) is the production
setting from the validation campaigns. Accepting them wholesale gives
results of the quality shown in the [validation record](validation/methodology.md),
not a rough first draft.

## Run an input file in one command

IRMA's command line takes an input file and an output path:

```bash
python -m irma input.input out.endf
```

- `input.input` is a LEAPR-style input file (the `.input` or `.leapr` extension
  is conventional, not required).
- `out.endf` is the ENDF-6 File 7 tape IRMA writes. IRMA always writes the
  file you name on the command line; Card 1's `nout` field is kept only for
  LEAPR compatibility and is otherwise ignored.

A real, validated graphite input file ships with the repository, so you can try
the full path right now (the path below exists in a clone of the repository; a
pip-only install can fetch the file from GitHub first):

```bash
python -m irma \
  tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/tsl-crystalline-graphite.input \
  graphite.endf
```

This is a classic crystalline-graphite evaluation (a phonon density of states
(DOS) supplied in the input file, isotropic Debye-Waller, the same physics level as
`iel=10` with `inelastic_mode=0`); it runs in seconds and reproduces the
published ENDF/B-VIII.1 graphite tape to 3.4×10⁻⁵ (the graphite entry of the
[reference set](validation/methodology.md)).

If you installed IRMA with `pip`, the bare `irma` console script is equivalent
to `python -m irma`, so `irma input.input out.endf` runs the same evaluation.
The two-positional form also has explicit aliases that do exactly the
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
| `IRMA: <title>` | The banner. IRMA echoes Card 2's quoted title so you can confirm the right input file loaded. |
| Echoed control cards (`ntempr`, `mat`, `awr`, `nalpha`, ...) | IRMA reads and prints back the values it parsed from Cards 3-7. Use these to verify the parser saw what you intended: a wrong `iel` or `nbeta` shows up here. |
| `Temperature N: <T> K` | One line per temperature in the input file. The S(α,β) detail block for that temperature is computed next. A negative temperature in the input file prints `(negative T: reusing previous temperature's ...)`, the LEAPR shared-spectrum convention. |
| `DW lambda = ..., T_eff = ...` | The Debye-Waller lambda and effective temperature computed from that temperature's phonon spectrum. Both end up on the tape: the lambda sets the elastic Debye-Waller suppression, and the effective temperature drives the short-collision-time extension of THERMR (NJOY's thermal processing module) downstream. |
| `Found <N> Bragg edges below 5 eV` | The coherent-elastic phase. Printed when the principal scatterer has an elastic treatment (`iel` ≥ 1 or `iel=10`). |
| `Read <N> comment cards` | The trailing quoted comment cards become the ENDF MF1/MT451 description. |
| `Writing ENDF output...` / `ENDF output written to <path>` | The S(α,β) table and the elastic data are assembled into MF7 and written to the output path. |
| `IRMA complete.` | Success. The tape at your output path is ready for downstream processing (e.g. NJOY THERMR/ACER). |

Three echoed values deserve a note. `S(α,β)` is the thermal scattering
law, tabulated in the dimensionless momentum transfer α and energy
transfer β (β = E/kT; the [theory page](theory.md) defines the
conventions). `za=130` is the historical TSL convention for graphite
carried by this classic input file (thermal libraries often assign such
codes to compound materials), while the generalized `iel=10` treatment
requires the physical `1000·Z+A` form. And `iint=0` is Card 4's optional
interpolation flag, defaulted here; the
[input file reference](input-reference.md) describes it.

MAT, on Card 4, is the ENDF material number: the integer that labels the
evaluation in a library, yours to assign.

If the input file is malformed, IRMA stops at the failing card: the console shows
the lines above up to that point, then an `Input deck error:` message that
names the offending card, the expected-vs-found values, and the input line
number, then exits non-zero. A clean engine-level failure (for example a
missing phonopy file) prints `IRMA failed: <reason>`. No ENDF tape is written
in either case.

One hazard lives downstream of a successful run, and you should know about it
before taking a coherent tape to NJOY. Stock
NJOY2016 THERMR has a `cliq` bug that turns a correct coherent
`inelastic_mode = 2` tape (graphite and similar) into ~`1e91`-barn garbage
above ~`0.27 eV`, or stalls for hours. The bug fires on the *shape* of the
tabulated S(α,β), not on its grid, so no beta-grid choice avoids it; apply the
one-line `thermr.f90` patch described in [NJOY interoperability](njoy.md)
before processing. `inelastic_mode = 0/1` tapes and most materials are
unaffected.

## A minimal classic input file, annotated

The input file format is NJOY free-format: values are separated by spaces and
every card is terminated by `/`. The simplest useful input file is a classic
evaluation, the same physics
level as `inelastic_mode=0`: an isotropic Debye-Waller treatment plus a
cubic phonon expansion built from a scalar phonon DOS you supply in the
input file. No phonopy is needed.

Here is the card skeleton (graphite-shaped, one temperature). The phonon DOS
and grid arrays are shown as `... /` placeholders; in a real input file they
are the full lists of values.

```text
24 /                          Card 1: nout (LEAPR-compat; output goes to the CLI path)
'graphite' /                  Card 2: quoted title (echoed as the banner)
1 2 100 /                     Card 3: ntempr iprint nphon
30 6012.0 0 0 /               Card 4: mat za isabt ilog (smin and iint defaulted; mat is the evaluation's MAT number)
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
| 4 | `mat za isabt ilog smin [iint]` | ENDF MAT/ZA, storage flags, and the optional interpolation flag echoed in the log. |
| 5 | `awr spr npr iel ncold nsk` | Principal scatterer: mass ratio, free-atom σ, atom count, elastic option, cold-H and pair-correlation options. |
| 6 | `nss b7 aws sps mss` | Secondary scatterer block (`nss=0` for none). |
| 7 | `nalpha nbeta lat` | Grid sizes; `lat=1` means grids are at the 0.0253 eV reference temperature. |
| 8 / 9 | α grid / β grid | Strictly increasing value lists, one `/` ends each card. |
| 10 | `T` | Temperature [K]; a negative value reuses the previous block. |
| 11-16 | `delta ni`, `rho`, `twt c tbeta`, `nd`, ... | The S(α,β) detail block (DOS, translational, oscillators). |

This table is an orientation, not the full specification. Field meanings,
defaults, valid ranges, the generalized-elastic cards (`iel=10`, Cards 6b-6g),
and the noncubic `inelastic_mode=1/2` workflows are documented card by card in
the [Input file reference](input-reference.md); reach for it whenever a field
above is unclear.

Before you write your own input file, know Card 5's `iel`. It selects the
coherent-elastic treatment: `0` = none, `1`-`6` = the legacy built-in
materials (graphite, Be, BeO, Al, Pb, Fe), and `10` = generalized Bragg edges
from your own crystal structure. For a new material on a noncubic crystal,
`iel=10` is recommended; it also makes the anisotropic Debye-Waller
treatment and the `inelastic_mode=1/2` workflows, computed from a phonopy
calculation, available.
The example above uses the built-in `iel=1` graphite tables, which keeps the
input file self-contained.

## Skipping the grid: let IRMA build it

Cards 8 and 9 are explicit α and β grids. The input file format has **no**
auto-grid flag. Automatic grids are built in the GUI (the **Grids** tab) or
from Python via the `irma.core.grids` functions; either way the result is an
explicit grid that lands on Cards 7–9 of the input file (or in the saved JSON
configuration). The [Automatic grids](grids.md) page explains the controls
(`N lower`, `N phonon`, `N upper`, and the linear-in-Q alpha parameters). The
default alpha grid is linear in momentum transfer Q (`dQ = 0.05 1/Å` up to
`Q = 12 1/Å`, then a logarithmic tail), which resolves the thermal upscatter
windows that a beta-mirrored (recoil-relation) alpha grid under-samples.

## First GUI launch

To point and click, or to inspect an unfamiliar input file field by field, launch
the graphical interface:

```bash
python -m irma --gui
```

The `irma-gui` console script does the same thing. From the GUI you can define
the crystal structure, set scattering parameters and grids, point at phonopy
data, run the calculation with a progress display, and save or load the whole
configuration as JSON. Existing `.input` / `.leapr` input files can be imported via
**File > Import Input File** to populate every field. The GUI is covered in
detail on the [Graphical interface](gui.md) page.

## Predict an instrument spectrum

The same install (plus `pip install -e ".[spectra]"`) also predicts what an
inelastic neutron scattering (INS) instrument measures. One command produces
a VISION-resolved spectrum from a phonopy calculation (in the command,
`--scatterer` decodes as symbol, bound cross section [b], atomic weight
ratio, b_coh [fm], sigma_inc [b]; IRMA's built-in table supplies these
values for any element):

```bash
python -m irma spectra vision --phonopy-yaml phonopy.yaml --inelastic-mode 2 \
    --scatterer "C,5.551,11.898,6.646,0.001" --temperature 300 -o vision.csv
```

or, with no eigenvectors at all, from a phonon DOS via a config file. The
[Neutron scattering spectra](spectra.md) page has the end-to-end example and the
full CLI/config reference; [DOS-based spectra (mode 0)](spectra-mode0.md)
covers the DOS-only workflow; ready-to-run configurations live under
`examples/spectra/`.

## Where to go next

- **[Input file reference](input-reference.md)**: the authoritative
  card-by-card spec, including `iel=10` and the noncubic modes.
- **[Automatic grids](grids.md)**: let IRMA build α/β grids for you.
- **[Scattering modes](modes.md)**: the elastic `iel` options (including the
  generalized `iel=10` Bragg edges for any crystal), SEF (single-channel
  elastic format) vs MEF, and when to
  use `inelastic_mode` 0, 1, or 2.
- **[Neutron scattering spectra](spectra.md)**: instrument-resolved INS
  spectra and S(Q,E) maps with `irma spectra`; mode 0's DOS-only workflow has
  [its own page](spectra-mode0.md).
- **[NJOY interoperability](njoy.md)**: preparing tapes for THERMR/ACER.
- **[Validation](validation/graphite.md)**: how IRMA's results compare against NJOY,
  Euphonic, OCLIMAX, and measured graphite cross sections.
