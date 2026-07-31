# Troubleshooting & FAQ

This page collects the failures you are most likely to hit while building an
IRMA evaluation (malformed decks, missing phonon data, incompatible mode
combinations, and downstream NJOY processing) and tells you what each one
means and how to fix it. IRMA is deliberate about reporting problems: a bad
deck is rejected with the offending card, the expected and found values, and
the input line number, rather than being allowed to cascade into cryptic
numerical errors. When something goes wrong, read the message first; it
usually names the exact card and line to edit.

## Deck errors and what they mean

IRMA parses the deck in NJOY free-format style: values are separated by
spaces or commas, each card is terminated by `/`, and any values omitted
before the `/` take their defaults. When a card cannot be read as written,
IRMA raises a deck error and the command-line tool prints it without a
traceback:

```
Input deck error:
  expected an integer for field 1, got the non-integral value 1.9 while reading Card 3 (ntempr iprint nphon) (input line 4 of mydeck.input)
Fix the input deck and rerun (see the README card-by-card input reference).
```

In this manual that card-by-card reference is the
[Input deck reference](input-reference.md) page.

The message has three parts: **what** went wrong (the field and the bad
value), **which card** was being read, and **where** in the file it lives
(line number and file name). Semantic checks that run after a card is fully
read report the line where the card *started*.

The command-line tool also exits with distinct status codes, so scripts can
branch on the kind of failure: a deck error (including a deck that ends
before a required card) exits with code `2`, and an otherwise meaningful
failure (a missing file, a phonopy model that will not load) exits with
code `3`.

### Common mistakes

| Symptom in the message | Likely cause | Fix |
|---|---|---|
| `input ended before this card` / `input ended after N of M expected values` | A card or array is short: the deck stops before all the values a count card promised | Supply the missing values, or correct the count on the preceding card |
| `the '/' terminator appeared after N of M expected values` | The array count (e.g. `nalpha`, `nbeta`, `ni`, `nd`) disagrees with how many numbers you actually wrote | Make the count match the data, on Card 7/8/9, 11/12, 14/15, etc. |
| `an empty '/' card appeared where this array was expected — remove the stray terminator` | A lone `/` sits where an array should begin | Delete the stray terminator line |
| `expected a number for field N, got 'word'` / `numeric-coded fields must use their codes, not words` | A numeric-coded selector was written as a word (e.g. `numerical` instead of `0`) | Use the numeric code |
| `expected an integer ... got the non-integral value 1.9` | A field that must be an integer received a fractional value | Use an integer (an exactly-integral float such as `200.0` is accepted) |
| `expected a quoted string, got the number ... (missing string card?)` | A path/title card is missing, so the next card's numbers landed in its slot | Add the missing quoted card (use `''` for an intentionally empty string) |
| A token contains `_` or an overflowing exponent (`1e999`) | Underscore digit grouping and infinities are not Fortran numeric syntax | Write a plain number |

It is important to note where the closing `/` belongs when an array spans
several lines (a long alpha grid, a phonon spectrum): it goes on the
**last data line**, right after the final value, exactly as NJOY itself
writes decks. If you instead put a lone `/` on the *next* line by itself,
IRMA (like NJOY's list-directed read) rolls that stray `/` into the
following scalar read, so that card silently takes its defaults and every
later card shifts by one. The downstream symptom is a confusing error on a
card far from the real mistake. When a deck error points somewhere
unexpected, check the array terminators just above it.

You can also hand IRMA a full NJOY job stream; it locates the
`leapr ... stop` block automatically and ignores the surrounding modules.
The deck title on Card 2 is exempt from module-name detection, so a title
that happens to contain a module word will not truncate the deck.

## phonopy not installed / force-constants resolution failures

The phonopy-backed inelastic paths (`iel=10` with `inelastic_mode=1` or `2`)
need the optional `phonopy` extra. The classic paths (`iel=0`–`6`, and
`iel=10` with `inelastic_mode=0`) do not.

```bash
pip install -e ".[phonopy]"   # adds the noncubic inelastic paths
```

For modes 1/2, Card 6f names a `phonopy.yaml`, and IRMA must find the force
constants. It resolves them in a fixed order and **never consults the
process working directory**, so the physics never depends on where you run:

1. force constants embedded in the named `phonopy.yaml`, otherwise
2. a file discovered *next to* that yaml, tried in order:
   `force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`.

If none of those is found, IRMA reports an error rather than guessing.

Three pitfalls are common. First, if IRMA reports that nothing was found
next to the yaml, put one of `force_constants.hdf5`, `FORCE_CONSTANTS`, or
`FORCE_SETS` in the same directory as the `phonopy.yaml` you named on
Card 6f. Second, a `BORN` file in your working directory is ignored: with
`use_born=0`, IRMA disables phonopy's fallback of auto-reading a `BORN`
file from the working directory, and a non-analytical-term (LO–TO)
correction is applied only via a top-level `nac:` block inside the named
`phonopy.yaml`, or when you set `use_born=1` and give a readable BORN path;
an unreadable BORN file is an error, not a silent skip. Third, the atom
positions must match the phonon model: for modes 1/2 the Card 6d fractional
positions must match the phonopy primitive cell.

An unstable phonon model warns; it does not silently pass. If the phonon
mesh contains imaginary modes (reported as negative frequencies), IRMA
prints a warning naming the most-negative mode in meV. Those modes are
excluded from the DOS-tensor grid and the mode sums, but the warning means
the phonon model itself is unstable and should be reviewed before you
trust the evaluation.

## "refusing to parse: the file carries a non-standard '!!python/' YAML tag"

Every `phonopy.yaml` handed to IRMA is scanned before phonopy sees it, and a
file carrying a `!!python/` tag (or the `tag:yaml.org,2002:python` verbatim
form, or a `%TAG` directive that could alias that namespace) is rejected with
this error. The reason: phonopy parses `phonopy.yaml` with PyYAML's *unsafe*
loader, so a `!!python/object/apply:` tag in the file executes arbitrary code
at parse time. A legitimate phonopy.yaml, one written by phonopy itself,
never contains any of these, so the scan does not reject real models, and
there is deliberately no override switch.

If you hit this on a file you received, treat the file as hostile: do not open
it with phonopy tooling anywhere. If you hit it on a file you generated
yourself, something rewrote it (for example, round-tripping it through another
program's YAML serializer, which can emit `!!python/` tags for Python
objects); regenerate it with phonopy.

The scan runs at every IRMA entry point that parses a phonopy.yaml: the deck
engine (`iel=10` with `inelastic_mode=1/2`, Card 6f), the `irma.spectra`
phonopy-DOS path, the NCrystal exporter, `irma mlip validate` and bundle
loading, and the GUI's phonopy.yaml file picker. It is a fail-closed scan of
the known code-execution vector, **not** a sandbox; see the trust-boundary
discussion in [MLIP phonon models](mlip.md) for what it does and does not
guarantee.

## "Modes 1/2 reject ncold / nsk / nss"

The phonopy-backed MT4 path builds the inelastic thermal scattering law
S(α,β) and the directional Debye–Waller factors directly from the phonon
mesh. It never constructs the S(κ) tables that feed the cold-hydrogen and
Sköld kernels, and there is no noncubic counterpart to the
secondary-scatterer pass. IRMA therefore rejects those combinations up
front, before the expensive phonon-mesh load, instead of crashing mid-run:

| You wrote | With | IRMA says (paraphrased) |
|---|---|---|
| `ncold > 0` or `nsk > 0` (Card 5) | `inelastic_mode=1/2` | "ncold/nsk pair-correlation options are not available with inelastic_mode=…" |
| `nss > 0` (Card 6) | `inelastic_mode=1/2` | "a secondary scatterer is not supported with inelastic_mode=…" |
| `nspec ≠ 0` (Card 6b) | `inelastic_mode=1/2` | "nspec must be 0 …; phonopy provides MT4 and the Debye-Waller factors, so Card 6e partial spectra are not used" |

To fix: set `ncold=0` and `nsk=0` on Card 5, `nss=0` on Card 6, and
`nspec=0` on Card 6b (omit the Card 6e blocks). Modes 1/2 decks supply
**only** the temperature cards after the grids; the legacy detail block
(continuous DOS, translational, oscillator, and Sköld cards) is not read at
all. If you genuinely need cold hydrogen, Sköld, or a mixed moderator, use
the classic paths (`inelastic_mode=0` or `iel=0`–`6`).

## Generalized elastic (`iel=10`) parse-time rejections

Two further combinations are rejected up front at parse time, with every
`inelastic_mode` (0, 1, 2), because the generalized-elastic builder would
otherwise silently write a wrong tape:

| You wrote | With | IRMA says (paraphrased) |
|---|---|---|
| `nss > 0` and `b7 = 0` (Card 6) | `iel=10` (Card 5) | "a bound two-pass secondary scatterer … is not supported with generalized elastic (Card 5 iel=10): … MF7/MT2 would be built from the secondary scatterer's Debye-Waller data instead of the principal's" |
| an `extinction` card | SEF (`elastic_mode=1`) routed to the incoherent-elastic builder | "extinction would be a silent no-op: … writes MF7/MT2 from the incoherent elastic builder, which does not apply extinction" |

The first: the generalized MF7/MT2 builder reads only the last-computed
Debye–Waller array, and the bound (`b7=0`) two-pass merge leaves that array
holding the *secondary* scatterer's data, so the whole elastic section would be
built from the wrong species. An analytic secondary (`b7=1` free gas, `b7=2`
diffusion) is single-pass and stays legal. To fix: use `b7=1`/`2`, drop the
secondary (`nss=0`), or use the classic elastic options (`iel=0`–`6`), whose
builder averages the two species.

The second: SEF writes only the dominant elastic component, and that
component is the incoherent one, which never applies extinction, when the
single-atom principal has `sigma_coh <= sigma_inc` (Card 6d), or when the
polyatomic principal is not the designated-coherent atom. To fix: use
`elastic_mode=2` (MEF) to keep a coherent component, or remove the
`extinction` card. See [Crystalline extinction](extinction.md).

## The `nphon` auto-size warning, and what to do

In modes 1/2 the multiphonon maximum order is the Card 3 `nphon`, honored
verbatim by default. If `nphon` is too small for the requested beta grid,
the tabulated S(α,β) runs out of support: it goes identically zero before
the top of the grid. IRMA detects this and warns:

```
WARNING: Selected standalone SAB array ... becomes identically zero above
beta=… while IRMA requested beta_max=…. This usually means
multiphonon_max_order (Card 3 nphon) is too small for the requested grid —
raise it or set Card 6g auto_order=1. THERMR's short-collision-time
extension covers transfers beyond the tabulated law using the tape's
effective temperature.
```

You have two fixes:

- **Set `auto_order=1`** in the third field of Card 6g. The auto-sizer raises
  the multiphonon order to converge the anisotropic Debye–Waller Poisson
  sum at the grid's Q<sub>max</sub>. As a scale: graphite with
  Q<sub>max</sub> ≈ 100 Å⁻¹ needs order ≈ 223, so a fixed `nphon=100`
  truncates S(α,β) at high Q, which is exactly what the warning reports.
- **Raise Card 3 `nphon`** by hand until the tabulated S(α,β) has genuine
  support over your beta grid.

```
10000 1000 1 /     <-- Card 6g: ndir mpdir auto_order (auto-size ON)
```

Either way, transfers beyond the tabulated region are not lost: THERMR's
short-collision-time extension, driven by the tape's effective temperature,
reconstructs them downstream. The warning is about giving the *tabulated*
region honest support, not about a hole in the physics.

## High-energy (optic) phonon structure missing at low temperature

If you generate a tape at a **cryogenic temperature** (a few K to a few tens of
K) and the high-energy-transfer structure (graphite's optic modes near
170–200 meV, for example) reads back as **zero**, the cause is underflow in
the stored symmetric S(α,β), not a physics error.

With the default `ilog=0` (Card 4), the symmetric S(α,β) is stored linearly
as `S·exp(−β/2)`. Because `β = E/kT` scales as `1/T`, at low temperature the
exponential becomes astronomically small: a 200 meV transfer has `β ≈ 8` at
296 K (`exp(−β/2) ≈ 0.02`, fine) but `β ≈ 460` at 5 K (`exp(−β/2) ≈ 1e−100`).
Those values fall below what the ENDF field can represent and are written as
**0**, and the structure is silently lost: read-back recovers
`S = S_sym·exp(+β/2)`, so a stored zero stays zero.

IRMA warns when this happens:

```
WARNING: ilog=0 (linear symmetric-law storage) at T=5 K: N S(alpha,beta)
points with significant scattering (up to E~250 meV transfer) underflow the
ENDF symmetric law [S*exp(-beta/2) < 1e-90] and are written as 0 -- the
high-energy phonon structure (e.g. optic modes) will be LOST on read-back.
Set ilog=1 (LLN log storage) on Card 4 to preserve it.
```

**Fix: set `ilog=1`** (ENDF `LLN=1`, log storage) on Card 4: add the `ilog`
and `smin` fields after `isabt`:

```
28 6012 0 1 0 /   <-- Card 4: mat za isabt ILOG=1 smin
```

With `LLN=1` the file stores `ln(S)` (for example `ln(1e−100) = −230`,
perfectly representable), so the full dynamic range survives. Any reader
must understand `LLN=1`; THERMR does, and so do IRMA's own readers
(`valplot.load_sym_sab`, the validation harness, the neutron-scattering
forward model). It is harmless at room temperature, where `ilog=0` stays
NJOY-byte-faithful.

As a rule of thumb, set `ilog=1` for any tape below ~50–100 K. The GUI's
`ilog` control carries the same guidance, and the run log warns if you
leave it at 0 and lose data. A self-contained numerical demonstration is in
`examples/lln_low_temperature_demo.py`.

## NJOY processing issues

IRMA writes ENDF-6 MF7 tapes that process through the standard
THERMR/ACER chain. The most common surprise is a coherent
`inelastic_mode = 2` tape (graphite and similar) coming out of stock
NJOY2016 THERMR as ~`1e91`-barn nonsense above ~`0.27 eV`, or stalling
for hours.

It is important to note that this is a bug in stock `thermr.f90`, not in
the IRMA tape: a `cliq` liquid-extrapolation guard tests decay along alpha
but not beta, and it fires on the *shape* of a coherent S(α,β), so no
beta-grid choice fixes it. Apply the one-line two-axis guard patch before
processing mode-2 tapes:

```fortran
! at both cliq sites in src/thermr.f90
if (sab(1,1).gt.sab(2,1).and.sab(1,1).gt.sab(1,2)) then
```

`inelastic_mode = 0/1` tapes and most materials are unaffected. For the
mechanism, the symptom table, the full RECONR/BROADR/THERMR/ACER recipe,
and tape-energy units, see [NJOY interoperability](njoy.md).

## GUI quirks

The GUI (`irma-gui`, or `python -m irma --gui`) mirrors the deck format, with
a few behaviors worth knowing.

- **Conditional fields appear only when their mode is active.** The Sköld /
  S(κ) group shows only when `nsk > 0` or `ncold > 0`; the secondary-scatterer
  group shows only when `nss > 0`; the second full temperature block appears
  only for the two-pass case (`nss > 0` with `b7 = 0`). If a field you expect
  is missing, check the selector that gates it.
- **"Preview Grid Sizes" reports the counts before you run.** On the Grids
  tab this button reports `nalpha`, `nbeta`, and the grid endpoints for the
  current automatic-grid parameters (or just the counts for a manual grid),
  so you can confirm the grid before launching a long calculation. `nalpha`
  is decoupled from `nbeta`.
- **The Phonon part warns when it is being ignored.** If `inelastic_mode=1/2`
  is selected in the Material part, the Phonon part shows a notice that its
  continuous-distribution, translational, and oscillator fields are **not
  read**; those matter only for `inelastic_mode=0`, because modes 1/2 take
  the inelastic S(α,β) from the force constants.
- **Importing a deck populates every field.** *File ▸ Import Input File*
  reads an existing `.input` or `.leapr` deck and fills the GUI; conditional
  groups update to match the imported selectors. Configurations can also be
  saved and loaded as JSON via the File menu.

## Performance expectations and `ncpu`

| Path | Rough cost (validation hardware, 2026-07) |
|---|---|
| Classic / mode-0 runs | seconds |
| Mode-2 graphite, 399×400 auto grid, 4000 directions, auto order, 8 cores | ~18 s wall |
| Mode-2 graphite, 399×400 auto grid, 10000 directions, auto order 217, 14 cores | ~20 s wall |
| 2000-column reference grids | around a minute |

The phonopy-backed modes parallelize over a worker pool sized by **Card 6f
`ncpu`** (14 was used in validation). The pool uses the spawn start method
with the compute context in shared memory, so it works identically on
Linux, macOS, and Windows. Native BLAS/OpenMP thread pools are
intentionally pinned to one thread per worker, which is measured to be the
optimum (jobs-vs-threads sweep on the graphite production pack bake:
14 procs x 1 thread 27 s, 7 x 2 28 s, 1 x 14 47 s); the
`IRMA_WORKER_THREADS` environment variable overrides the per-worker limit
for tuning experiments. Set `ncpu` to the number of cores you want to
dedicate; multiphonon cost grows linearly with the Card 6g `mpdir`
direction count (converged by ~50–100; the production default of 1000 is
the validation-campaign sampling). Soft-mode materials auto-size the
multiphonon Poisson order into the hundreds (beta-quartz: 1084), but the
convolution ladder that builds each phonon order and the sweep that
deposits the orders into S(α,β) are both bounded, per direction and atom,
by the exact float64 underflow horizon of the Poisson weights, so the deep
orders cost only what the (direction, atom) combinations that actually
reach them require.

Large grids make large tapes. ENDF tapes carry ~6-significant-figure
values, and a big grid makes a big file: a 2000×5001 grid is on the order
of 270 MB. Size the grid to the evaluation you need.

### The ENDF writer backend (`IRMA_ENDF_WRITER`)

Serializing the tape is a large share of classic-path wall time. IRMA
writes with `endf-parserpy`'s **compiled backend** (`EndfParserCpp`) by
default; it is byte-identical to the pure-Python writer and several times
faster (e.g. the 10-temperature graphite expected deck drops from ~12 s to
~4 s end to end). If your `endf-parserpy` install has no compiled module
(source-only build), IRMA prints a warning and falls back to the
pure-Python writer: identical output, slower. Override with the
`IRMA_ENDF_WRITER` environment variable: `py` forces the pure-Python
writer, `cpp` requires the compiled one (hard error if missing), unset or
`auto` is the default behavior.

## Where outputs land

IRMA writes the ENDF tape to the **output path you give on the command
line**; the file name is not taken from the deck:

```bash
python -m irma mydeck.input mymaterial.endf
```

The Card 1 `nout` field is kept only for LEAPR compatibility; it does not
control the file name. Both arguments are required; omitting the output
path is an error. Progress and any warnings (imaginary modes, the `nphon`
support warning, Bragg-edge cutoff notes) print to the console as the run
proceeds.

## How to report issues

When a result looks wrong or IRMA fails in a way this page does not cover,
please open an issue on the project tracker:

<https://github.com/ramic-k/IRMA/issues>

Include, where you can:

- the IRMA version (`python -m irma --version`),
- the full deck (or the smallest deck that reproduces the problem),
- the exact console output, including any deck-error message and line
  number,
- for modes 1/2, how the phonon model was produced (and whether
  `force_constants.hdf5`, `FORCE_CONSTANTS`, or `FORCE_SETS` was used).

A minimal reproducing deck is the single most useful thing you can attach.
The committed example decks under `tests/` are good starting points to
adapt; see the [input deck reference](input-reference.md) and
[scattering modes](modes.md) pages for the card-by-card format.
