# Scattering modes

IRMA describes a material's thermal response with three independent
choices: which coherent-elastic treatment to use (`iel`), which elastic
format to write (`elastic_mode`), and which inelastic engine to run
(`inelastic_mode`). They live on different cards and combine freely, so it
helps to think of them as three independent axes rather than a single
menu. This page describes each axis, shows how to read the three selectors
off a deck, and gives guidance for picking the right combination for cubic
versus noncubic and isotropic versus anisotropic materials.

## The three axes at a glance

| Axis | Card / field | What it controls | Values |
|------|--------------|------------------|--------|
| `iel` | Card 5, field 4 | Coherent-elastic (Bragg) treatment | `0` none/incoherent; `1`–`6` built-in materials; `10` generalized (any crystal) |
| `elastic_mode` | Card 6b, field 1 (`iel=10` only) | ENDF elastic format written | `1` = SEF (single-channel elastic format); `2` = MEF |
| `inelastic_mode` | Card 6b, field 4 (`iel=10` only) | Inelastic S(α,β) engine | `0` legacy cubic; `1` directional incoherent; `2` coherent one-phonon |

Two of the three axes exist only on the generalized path: `elastic_mode`
and `inelastic_mode` are read from Card 6b, which is present only when
`iel=10`. For the built-in materials (`iel=1`–`6`) and for `iel=0`, IRMA
uses the classic LEAPR paths and the two Card 6b selectors do not apply.

## Axis 1 (`iel`): coherent-elastic treatment

`iel` (Card 5, field 4) selects how Bragg (coherent-elastic) scattering is
handled.

| `iel` | Treatment | Notes |
|-------|-----------|-------|
| `0` | None / incoherent | No coherent elastic; with no translational mode IRMA flags incoherent elastic |
| `1`–`6` | Built-in coherent elastic | Graphite, Be, BeO, Al, Pb, Fe (in that order) |
| `10` | Generalized | Bragg edges computed from the crystal structure you supply on Cards 6c–6d; works for any material |

The built-in options reproduce the historical LEAPR `coher` lattices.
`iel=10` is the recommended path for new evaluations: it computes Bragg
edges directly from your unit cell and unlocks the `elastic_mode` and
`inelastic_mode` axes.

The generalized path also changes what the Card 4 `za` must contain. With
`iel=10`, `za` must encode the physical nuclide as `1000·Z + A` (with
`A = 0` for the natural element) so that
the principal scatterer can be matched to a Card 6d atom type. Some
thermal libraries use a TSL-convention ZA instead; that convention will
not match here.

## Axis 2 (`elastic_mode`): which elastic format to write

`elastic_mode` (Card 6b, field 1) is only meaningful when `iel=10`. It
selects the ENDF elastic representation and therefore the `LTHR` flag your
downstream codes will see.

| `elastic_mode` | Name | What is written | Resulting `LTHR` |
|----------------|------|-----------------|------------------|
| `1` | SEF (single-channel elastic format) | The atom with the dominant elastic component gets coherent elastic; other atoms get incoherent elastic with redistribution | `1` (coherent) or `2` (incoherent), per atom |
| `2` | MEF (Mixed Elastic Format) | Every atom gets both a coherent (per-atom Bragg edges) and an incoherent part | `3` (mixed) |

For a single-atom material under SEF, the coherent-versus-incoherent cross
sections decide which elastic component is written. For a polyatomic cell,
SEF selects one designated-coherent (DC) atom, the one minimizing the
incoherent contribution; that species' tape carries the coherent elastic,
and the others carry incoherent elastic.

It is important to note that MEF needs downstream support:
`elastic_mode=2` writes `LTHR=3`, so confirm that your transport and
processing chain understands the mixed elastic format before choosing it.
When in doubt, SEF (`elastic_mode=1`) is the conservative, widely
supported choice.

## Axis 3 (`inelastic_mode`): the S(α,β) engine

`inelastic_mode` (Card 6b, field 4, `iel=10` only) chooses how the
inelastic S(α,β) is built.

| `inelastic_mode` | Engine | Debye-Waller | Needs phonopy? | Reads legacy DOS cards? |
|------------------|--------|--------------|----------------|--------------------------|
| `0` | Legacy cubic phonon expansion from a tabulated DOS | Isotropic (scalar) | No | Yes (Cards 11–19 / Card 6e) |
| `1` | Directional incoherent approximation | Directional | Yes | No |
| `2` | Exact coherent + incoherent one-phonon, incoherent multiphonon | Directional | Yes | No |

`inelastic_mode=0` is the classic LEAPR-style path: an isotropic
Debye-Waller factor and a phonon expansion built from a scalar phonon
density of states given on the deck, with no phonopy required.
`inelastic_mode=1` computes a directional Debye-Waller factor for the
coherent elastic and an in-process noncubic S(α,β), using an
incoherent-approximation one-phonon term plus incoherent-approximation
multiphonons. `inelastic_mode=2` is the same as mode 1 except that the
one-phonon term is exact (coherent plus incoherent) on top of the
incoherent-approximation multiphonons.

Even in mode 2, only the one-phonon term is coherent. The multiphonon
orders (n ≥ 2) use the incoherent-approximation model (a
`sigma_total`-scaled per-atom self kernel), so coherent interference is
dropped in the tail. Runs record
`multiphonon_model = "incoherent_approximation"` in their metadata so the
mode-2 product is not mistaken for fully coherent multiphonon scattering.

It is important to note that modes 1 and 2 ignore the legacy DOS cards.
`inelastic_mode=1/2` build MT4 and the Debye-Waller factors entirely from
the phonopy model: the legacy continuous-DOS, translational, and
oscillator detail cards (Cards 11–19) are not read, and Card 6e partial
spectra are rejected (`nspec` must be `0`). The deck supplies only the
temperature cards; everything else comes from phonopy. The mixed-moderator
(`nss>0`), cold-hydrogen (`ncold`), and Sköld (`nsk`) options are also
unavailable with modes 1/2 and are rejected at parse time.

### What modes 1 and 2 require

When `inelastic_mode=1` or `2`, supply the phonopy mesh and sampling
controls between Cards 6d and 7:

- **Card 6f**: the `phonopy.yaml` path, then `mesh_nx mesh_ny mesh_nz
  ncpu use_born` (and a `BORN` path if `use_born=1`). Force constants are
  read from the yaml if embedded, otherwise discovered next to it
  (`force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`, in that order).
  The process working directory is never consulted.
- **Card 6g**: the powder-averaging sampling controls (next section).

These modes need the optional phonopy dependency:

```bash
pip install -e ".[phonopy]"   # adds the noncubic inelastic paths
```

## Choosing a combination

### Cubic vs noncubic materials

For a cubic material the Debye-Waller factor is isotropic by symmetry, so
`inelastic_mode=0` (with a good DOS) captures the physics and is fast. For
a noncubic (anisotropic) crystal, the directional engines (`1`/`2`) are
the point of `iel=10`: they carry the full Debye-Waller tensor rather than
collapsing it to a scalar.

### Isotropic vs directional Debye-Waller: why it matters

The Debye-Waller factor suppresses one-phonon scattering exponentially
with momentum transfer Q. In an anisotropic crystal that suppression is
direction-dependent, and the powder average of the exact directional
factor is not the same as applying a single orientation-averaged
(isotropic) factor.

Graphite is the standard cautionary example: its perpendicular and
in-plane Debye-Waller terms differ by roughly a factor of 6.6
(W_c / W_ab ≈ 6.6). An isotropic treatment exponentially over-suppresses
the one-phonon S(α,β) at high Q relative to the exact directional powder
average. IRMA's `inelastic_mode=2` performs the directional powder average
exactly; running `inelastic_mode=0` (isotropic Debye-Waller) on the same
phonon model reproduces the over-suppressed roll-off, isolating the cause.

![Directional vs isotropic Debye-Waller attenuation in the graphite one-phonon S(α,β) at E ≈ 2 meV](assets/validation/graphite/fig_graphite_dw_directional.png)

The exact one-phonon treatment has been cross-checked against an
independent code: IRMA's mode-2 one-phonon S(α,β) has been cross-validated
against Euphonic (using the same phonon model) for graphite and Be,
coherent component against coherent component, and the integrals of the
symmetric S(α,β) agree to ratios of 1.00001 (graphite) and 1.0002 (Be).
For strongly anisotropic crystals, prefer `inelastic_mode=2`; reserve
`inelastic_mode=0` for cubic materials or for reproducing legacy isotropic
evaluations.

## Card 6g: sampling controls

Card 6g (modes 1/2 only) sets the powder-averaging resolution and the
multiphonon order:

```
ndir  mpdir  [auto_order]  /
```

| Field | Name | Meaning |
|-------|------|---------|
| 1 | `ndir` | Coherent powder-average directions (golden-spiral quadrature). Must be ≥ 1 |
| 2 | `mpdir` | Multiphonon powder-average directions. The powder average converges by ~50–100; the recommended 1000 keeps headroom against residual azimuthal asymmetry from incomplete averaging over symmetry-equivalent directions (cost is linear in `mpdir`). Must be ≥ 1 |
| 3 | `auto_order` *(optional)* | `0` = honor the Card 3 `nphon` multiphonon order verbatim (default); `1` = auto-size the multiphonon order |

The incoherent powder average is always the exact numerical orientational
average; there is no method selector.

### Recommended Card 6g

The recommended production setting (the GUI and `irma mlip emit` default)
is:

```
10000 1000 1 /
```

That is 10000 coherent powder directions, 1000 multiphonon directions, and
auto-sized multiphonon order: the same sampling the validation campaign
ran. The powder averages converge well below these counts, so a faster
exploratory run at `4000 200 1` is fine; the default carries the
cross-code-comparison headroom.

Auto-sizing is what keeps the tabulated S(α,β) converged at high Q. With
`auto_order=1`, IRMA raises the multiphonon order as needed to converge
the anisotropic-Debye-Waller Poisson sum at the grid's Q_max: graphite on
a grid reaching Q_max ≈ 100 1/Å needs order ≈ 223, so a fixed `nphon=100`
truncates the table at high Q, and the engine warns when this happens.
Leaving `auto_order=0` honors the Card 3 `nphon` exactly. That is
appropriate when you want full control, but make sure `nphon` (or
`auto_order=1`) gives the tabulated S(α,β) genuine support over your
requested beta grid. Energy transfers beyond the tabulated range are
covered downstream by THERMR's short-collision-time extension.

## Bragg-edge grouping (optional)

For materials with very dense high-energy coherent-elastic edge structure,
Card 6b accepts two optional trailing fields that enable ENDF-102 §7.2.2
Bragg-edge grouping:

```
elastic_mode  nat  nspec  inelastic_mode  [bins_per_decade]  [threshold_eV]  /
```

| Field | Name | Meaning |
|-------|------|---------|
| 5 | `bins_per_decade` | Number of log-uniform bins per decade above the threshold. `0` or absent = grouping off (default) |
| 6 | `threshold_eV` | Energy above which edges are merged. Default `1` eV |

Above `threshold_eV`, the dense edge steps are merged into
`bins_per_decade` log-uniform bins per decade with structure-factor-weighted
placement. The grouping preserves the cumulative S and the total cross
section, so it shrinks the tape's edge list without changing the physics
your transport code integrates. It is off by default; existing four-field
Card 6b decks are unaffected.

Grouping merges only the edges above the threshold and keeps every
sub-threshold edge raw, whereas the ungrouped writer applies NJOY's
tolerance thinning to the whole list. A very low `threshold_eV` on a
material whose Debye-Waller factor suppresses the high-energy edges can
therefore produce a larger tape than grouping off. Use the default 1 eV
threshold unless the dense structure you want compressed actually sits
below it.

## Putting it together: a noncubic mode-2 deck

A minimal `iel=10` / `inelastic_mode=2` Card 5–6g block looks like this
(graphite, exact one-phonon, auto-sized multiphonon order):

```
 11.898 4.739 1 10 0 0 /       --- Card 5: awr, spr (FREE-atom sigma), npr, iel=10, ncold=0, nsk=0
 0 /                            --- Card 6: nss=0 (no secondary scatterer)
 1 1 0 2 /                      --- Card 6b: SEF, nat=1, nspec=0, inelastic_mode=2
 2.464 2.464 6.711 90 90 120 / --- Card 6c: hexagonal graphite lattice
 6 12 11.898 6.6484 0.0 4 /     --- Card 6d: C-12, b_coh [fm], sigma_inc [b], 4 positions
 0 0 0.25  0 0 0.75  ...  /     --- Card 6d: fractional positions
 'phonopy.yaml' /               --- Card 6f-1: phonopy.yaml path
 40 40 40 14 0 /                --- Card 6f-2: mesh, ncpu=14, use_born=0
 10000 1000 1 /                 --- Card 6g: ndir, mpdir, auto_order=1
```

The numbers in this snippet are illustrative: the lattice constants, cross
sections, and positions show the deck shape, not a validated evaluation.
The production-ready deck is `examples/tsl/graphite_mode2.input`, a
complete, runnable mode-2 deck at the recommended production sampling with
auto-sized multiphonon order (see `examples/tsl/README.md` for the run
command). The `tests/mode2_euphonic_n1_validation/*.template` decks are
the cross-code validation harness: they deliberately compute the
one-phonon term only and are not production evaluations.
