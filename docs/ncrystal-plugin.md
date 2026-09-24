# NCrystal data exporter

IRMA can export a mode-2 calculation (the exact coherent one-phonon
treatment with directional Debye-Waller factors; see
[Scattering modes](modes.md); mode 1 is also accepted) as per-temperature `.irmapack` files
that the **NCrystal IRMA plugin** samples at runtime. NCrystal is the
thermal neutron scattering library that Monte Carlo codes such as
McStas and OpenMC use for materials; the plugin adds two things
NCrystal's standard powder model lacks: an anisotropic-Debye-Waller Bragg-elastic
line, where NCrystal's core model uses a single scalar mean-squared
displacement (MSD), and a coherent one-phonon inelastic kernel, where
NCrystal's built-in inelastic model is its isotropic `vdos2sab`
construction (a DOS-driven S(α,β)). IRMA is
the single producer of that physics and the reference for it; NCrystal only
samples the result, and no part of IRMA runs at NCrystal runtime.

The exporter is the `irma.ncrystal` module, and it reimplements no physics:
the thermal scattering law `S(α,β)` it writes (α, β: the dimensionless
momentum and energy transfer; see [Theory](theory.md)) is exactly what
the mode-2 engine (`run_noncubic_standalone_sab`) returns, the same
engine output that feeds the ENDF tape NJOY processes. The exporter
calls that engine once per principal scatterer (the atom species an
evaluation is written for), converts conventions, and serializes the
result.

## What it exports

For one config (one temperature) the exporter writes one `.irmapack` data
file per principal scatterer, plus a complete, loadable
`<material_id>.ncmat` (the phonopy structure plus a `@CUSTOM_IRMA` section
that wires the data files in):

- **Graphite** (monoatomic) → 1 data file (`graphite__C.irmapack`).
- **BeO** (two species) → 2 data files (`beo__Be.irmapack`,
  `beo__O.irmapack`), summed by the plugin.

Each data file holds, at the configured temperature:

- the mode-2 inelastic `S(α,β)` table for that principal species (coherent
  one-phonon + anisotropic Debye-Waller + incoherent-approximation multiphonon,
  powder-averaged), stored as a scaled-symmetric downscatter half-table;
- the whole-crystal elastic line (anisotropic-DW Bragg edges + the per-site
  incoherent Debye-Waller), carried by exactly one data file (see
  [Polyatomic materials](#polyatomic-materials-beo) below);
- a record of the IRMA version and run settings that
  produced it, so the plugin's CI can regenerate and compare against the
  mode-2 tape.

The data is exported at one temperature: IRMA computes the full anisotropic
`S(α,β)` and the anisotropic-DW structure factors at
`material.temperature_K`, and the C++ plugin only samples them. Re-run the
exporter at each temperature you need. The per-temperature design is
deliberate. The mode-2 coherent one-phonon term re-solves the dynamical matrix at every
sampled `Q = G + q`, so a C++ path that worked at any temperature would have
to ship the force constants and port that dispersion re-solve: a large
effort, and unnecessary for the powder application. The data format is
versioned, so an any-temperature extension stays open.

## The YAML config

The config reuses IRMA's standard `material` block (the same phonon calculation +
per-species neutron data the ENDF and spectra workflows name) and adds an `export`
section. A complete graphite example, ready to copy:

```yaml
material:
  phonopy_yaml: graphite/phonopy.yaml
  mesh: [40, 40, 40]
  temperature_K: 296.0
  scatterers:
    - {symbol: C, sigma_bound_b: 5.551, awr: 11.898, b_coh_fm: 6.646, sigma_inc_b: 0.001}
export:
  material_id: graphite
  inelastic_mode: 2
  num_directions: 10000
  multiphonon_num_directions: 1000
  multiphonon_max_order: auto
  min_phonon_energy_meV: 0
  gain_side: scaled_sym
  elastic: true
```

### `material` section

The `material` section names the phonon calculation
([Preparing a phonopy calculation](phonopy-input.md) covers producing one
from your own force calculation) and the per-species scattering
data, identical to the [spectra](spectra.md) and ENDF workflows:

| Key | Meaning |
|-----|---------|
| `phonopy_yaml` | Path to the phonopy calculation (eigenvectors come from here; mode 2 needs it). |
| `born` | Optional BORN file for the non-analytical correction (polar crystals). |
| `force_constants` / `force_sets` | Optional explicit FORCE_CONSTANTS / FORCE_SETS file; by default they are embedded in the phonopy.yaml or found next to it. Embedded force constants win (phonopy's rule; a warning names the unused file). The file used is hashed into the data file's provenance. |
| `mesh` | Phonon mesh, e.g. `[40, 40, 40]`. |
| `temperature_K` | The single temperature this data is exported at (default `296.0`). |
| `scatterers` | One entry per chemical species in the structure (see below). |

Each `scatterers` entry carries the neutron data for that species:

| Key | Meaning |
|-----|---------|
| `symbol` | Element symbol, must match the phonopy primitive cell (e.g. `C`). |
| `sigma_bound_b` | Bound scattering cross section (barn), the `S(α,β)` normalization. **Required.** |
| `awr` | Atomic weight ratio (mass / neutron mass). Falls back to the phonopy mass / neutron mass if omitted. |
| `b_coh_fm` | Bound coherent scattering length (fm), drives the coherent Bragg edges, the coherent partition, and the coherent weights of the mode-1/2 inelastic calculation. **Required.** |
| `sigma_inc_b` | Bound incoherent cross section (barn), the incoherent Debye-Waller line and the mode-1/2 incoherent inelastic weight. **Required.** |

Even for a polyatomic material such as BeO, `scatterers` must list every
species the structure contains, not just one: each species gets its own data
file, and a species in the phonopy cell with no matching scatterer entry is
an error.

### `export` section

Every field, with its default read from `irma/ncrystal/config.py`:

| Key | Default | Meaning |
|-----|---------|---------|
| `material_id` | — (**required**) | Output set name; data files are `<material_id>__<symbol>.irmapack`. |
| `inelastic_mode` | `2` | Inelastic engine. `2` = coherent one-phonon + anisotropic DW (the point of the plugin). `1` (directional incoherent) is also accepted. |
| `num_directions` | `10000` | Powder-average directions for the coherent one-phonon term. |
| `multiphonon_num_directions` | `1000` | Powder-average directions for the multiphonon orders. |
| `multiphonon_max_order` | `auto` | Multiphonon order: an integer, or `auto` (the engine starts at 100 and sizes the order up to converge the high-Q Poisson sum, bounded by an internal safety cap of 2000). |
| `min_phonon_energy_meV` | `0` | Remove every phonon mode with energy at or below this value (meV) from all terms of the pack; 0 keeps the automatic floors. Nothing replaces the removed modes, and the pack's provenance records the value; see the input reference's optional minimum phonon energy card. |
| `jobs` | `null` | Worker processes; `null`/omitted uses all CPU cores. |
| `gain_side` | `scaled_sym` | `scaled_sym` (default) stores the downscatter half-table; NCrystal reconstructs the upscatter side by detailed balance. `scaled_sym` is the only accepted value (a full asymmetric table is not implemented). |
| `elastic` | `true` | Whether to attach the elastic line. When on, the data file carries the full physical elastic (coherent Bragg edges + incoherent Debye-Waller); isolate a component at scatter time with NCrystal's `comp=coh_elas` / `comp=incoh_elas`. With `elastic: false` the data files carry no elastic block, so NCrystal's standard elastic (Bragg and incoherent, from the placeholder Debye `@DYNINFO` and NCrystal's atom data) stays active; load with `;elas=0` for an inelastic-only material. |
| `coherent_partition_mode` | `principal-xs-weighted` | How the engine splits the total coherent cross section across principal sites for the *inelastic* `S(α,β)` (no double-counting): `principal-xs-weighted`, `exact-total` (single principal group only), or `auto` (`exact-total` for a single group, `principal-xs-weighted` otherwise). |
| `incoherent_elastic_mode` | `isotropic` | Debye-Waller treatment of the data file's incoherent-elastic component. `isotropic` collapses each site tensor to its trace/3 scalar (NCrystal's stock model). `directional` has the plugin sample the powder-averaged anisotropic `⟨exp(-Q² û·U·û)⟩` per site, larger at high Q for anisotropic crystals, and consistent with the directional multiphonon and the per-reflection coherent elastic. The ENDF tape path cannot represent this (its incoherent-elastic record stores a single scalar W′). |
| `alpha_grid` | `null` | Explicit `α` grid (ENDF dimensionless, `lat=1` → 0.0253 eV reference). Provide together with `beta_grid` for the **explicit** grid mode. |
| `beta_grid` | `null` | Explicit `β` grid (downscatter, starts at 0). Provide together with `alpha_grid`. |
| `lat` | `1` | `S(α,β)` grid convention flag. `lat=1`: the grid values are referenced to the fixed thermal kT = 0.0253 eV instead of the actual temperature. |
| `freq_max_eV` | `null` (auto) | **Automatic** grid: the maximum phonon frequency [eV] that sets the β span. Omit / `null` → estimated from the phonopy mesh. |
| `n_lower`,`n_phonon`,`n_upper` | `15`,`300`,`80` | **Automatic** grid: β-grid point counts, log low-β tail, linear phonon region, log high-β tail. `n_upper=80` keeps the high-β tail fine enough to bake an accurate high-energy `S(α,β)` (≈20 suffices for thermal-only work; check convergence for your energy range and adjust up or down). |
| `beta_max_eV` | `5.0` | **Automatic** grid: the upper β-tail cap [eV]. |
| `alpha_dq_invA`,`alpha_qcut_invA`,`alpha_nlog` | `0.05`,`12.0`,`160` | **Automatic** grid: α from a linear-in-Q segment (step `alpha_dq_invA` up to `alpha_qcut_invA` [1/Å]) plus `alpha_nlog` log-tail points. |

### Two ways to set the S(α,β) grid

Mirroring the ENDF-evaluation side, the grid is set one of two ways. The
automatic grid is the same converged grid the ENDF evaluator builds, from
the same shared code (`irma.core.grids.generate_beta_grid` /
`generate_alpha_grid`), so the two sides never drift:

- **Explicit**: provide both `alpha_grid` and `beta_grid` (dimensionless). Use
  this to match the exact grid behind a material's validated mode-2 tape.
- **Automatic** (default, omit both): IRMA builds its converged ENDF grid from
  the phonon spectrum: a β grid spanning the phonon region up to `freq_max_eV`
  (auto-estimated from the phonopy mesh when omitted), and a linear-in-Q α
  grid (α ∝ Q²) that avoids the low-Q thermal bias of a uniform-α grid. α is
  per-species (mass-dependent), β is shared. This linear-in-Q layout
  replaces the recoil-mirrored, uniform-α layout, which under-integrates the
  thermal cross section.

It is all-or-nothing: giving only one of `alpha_grid`/`beta_grid` is an error.
The automatic-grid knobs (`freq_max_eV`, `n_lower`/`n_phonon`/`n_upper`,
`beta_max_eV`, `alpha_dq_invA`/`alpha_qcut_invA`/`alpha_nlog`) tune the converged
grid but default to the ENDF-grid defaults.

## Running it

```bash
python -m irma.ncrystal graphite_export.yaml -o out/
```

For graphite this writes, into `out/`:

- **`graphite__C.irmapack`**: the exported NCrystal data (mode-2 `S(α,β)`
  half-table + anisotropic-DW elastic tensors + the `meta.*` record);
- **`graphite.ncmat`**, a complete, loadable material: the phonopy primitive
  cell (`@CELL` + `@ATOMPOSITIONS` + a placeholder `@DYNINFO`) followed by the
  `@CUSTOM_IRMA` section referencing the data file(s):

  ```text
  NCMAT v5
  @CELL
    lengths 2.4606 2.4606 6.705
    angles 90 90 120
  @ATOMPOSITIONS
    C 0 0 0
    C 0 0 0.5
    ...
  @DYNINFO
    element C
    fraction 1
    type vdosdebye
    debye_temp 1037.8
  @CUSTOM_IRMA
    pack /abs/out/graphite__C.irmapack
  ```

Load it directly with the `ncrystal_plugin_IRMA` plugin installed:

```python
import NCrystal as NC
sc = NC.createScatter("out/graphite.ncmat;temp=296.0K")
```

The exporter writes the structure from the same phonopy cell the data was
built from, so the per-site Debye-Waller tensors match the material's atom
sites exactly (the plugin pairs them by fractional position, tolerance
1e-6). A stock stdlib NCMAT for the same compound usually has a different
crystallographic origin and will not match. The placeholder `@DYNINFO` (a Debye VDOS
whose Debye temperature reproduces the phonopy MSD in NCrystal's Debye model)
lets NCrystal construct the crystal; the plugin overrides the inelastic
component with the exported `S(α,β)` and, on the coherent-bearing data file,
the coherent/incoherent elastic with the anisotropic-DW line. With
`elastic: false` NCrystal's own elastic uses the placeholder. With
`elastic: true` the plugin computes each Bragg plane's |F|² from the
data file's tensors, but takes the list of planes from NCrystal, which drops
planes whose |F|² with the placeholder's Debye-Waller factor falls below its
cutoff; the placeholder therefore still selects the weak high-Q planes, which
shows in the coherent elastic above about 1 eV (0.4% for graphite). The material temperature is set at load via `;temp=...` and
must match the export temperature; a mismatch is a hard error (see
[How NCrystal uses the data](#how-ncrystal-uses-the-data)).

## What's inside a data file

A `.irmapack` is a plain-text `key = value` container with no binary blob, so you
can read or `diff` one directly. Line 1 is the format magic; every other line is
either a data field or a `meta.*` provenance line (which the C++ loader skips).
All quantities are in the units named by `units` (`angstrom_meV_barn_K`: Å for
the U-tensors, meV for the grid, barn for σ, K for temperature).

```text
IRMAPACK_TEXT_V1
schema_version = 2
material_id = graphite__C
backend = precomputed_sab
units = angstrom_meV_barn_K
sab_representation = scaled_sym_sab
temperature_K = 296
bound_xs_barn = 5.551
element_mass_amu = 12.0107
elastic_u_tensors_a2 = <one 3×3 symmetric U per site, flattened>
elastic_u_symbols = C C C C
elastic_u_frac_positions = 0 0 0  0 0 0.5  0.3333 0.6667 0  0.6667 0.3333 0.5
elastic_u_coherent_scatlen_sqrtbarn = <one b_coh per site, √barn>
elastic_u_incoherent_xs_barn = <one σ_inc per site, barn>
alpha_grid = <n_alpha values>
beta_grid  = <n_beta values, downscatter, starts at 0>
sab_values = <n_alpha × n_beta values, beta-major>
meta.irma_git_sha = fef0e25
meta.irma_version = <IRMA version>
meta.phonopy_yaml_sha256 = 9364...
...
```

| Field | What it is |
|-------|-----------|
| `IRMAPACK_TEXT_V1` | Format magic (line 1). |
| `schema_version` | `2`, the current and only supported schema; the loader requires it. (`meta.*` provenance lines are loader-ignored.) |
| `backend` | `precomputed_sab`, the data file carries a ready-to-sample table. |
| `sab_representation` | `scaled_sym_sab`, the scaled-symmetric downscatter half-table (see [Conventions](#conventions-and-provenance)). |
| `temperature_K` | The export temperature; the entire data file is this one T. |
| `bound_xs_barn` | The per-atom weighted bound cross section, `atom_fraction × σ_b` (`meta.atom_fraction` records the fraction); the table itself stays normalized per principal atom to σ_b. |
| `element_mass_amu` | Species mass (sets the recoil kinematics / α scaling). |
| `alpha_grid` | Momentum-transfer grid in NCrystal's mass-scaled α (= IRMA α × AWR), `n_alpha` values. |
| `beta_grid` | Energy-transfer grid in β (downscatter, β ≥ 0), `n_beta` values. |
| `sab_values` | The S(α,β) table, `n_alpha × n_beta` values, **β-major**: β is the outer (slow) index and α the inner (fast) one, one full α-row per β, matching NCrystal's kernel layout (`idx = iβ·n_α + iα`). **Do not transpose:** the loader's only shape check is the `n_alpha × n_beta` product, which an α-major table also passes, so a transposed table loads silently and gives the wrong inelastic physics. |
| `elastic_u_tensors_a2` | The anisotropic Debye-Waller tensor **U** (3×3 symmetric, Å²) for each crystallographic site, the physics NCrystal core (scalar MSD) lacks. |
| `elastic_u_symbols` | The element at each site, parallel to the tensors. |
| `elastic_u_frac_positions` | Each site's fractional position, the key the plugin uses to pair tensors to the NCMAT atoms (tolerance 1e-6). |
| `elastic_u_coherent_scatlen_sqrtbarn` | Per-site coherent scattering length, in √barn (parallel to the tensors). **Required whenever `elastic_u_tensors_a2` is present** (the loader rejects a data file that omits it), and every value must be finite. |
| `elastic_u_incoherent_xs_barn` | Per-site incoherent cross section, in barn (parallel to the tensors). **Required whenever `elastic_u_tensors_a2` is present**; every value must be finite and ≥ 0. |
| `meta.*` | Provenance + convention notes (git SHA, version, phonopy hash, mesh, direction counts, the α-convention string, …). Loader-ignored; used by CI and for the record. |

An inelastic-only data file (the non-elastic members of a polyatomic set)
omits the five `elastic_u_*` blocks.

## How NCrystal uses the data

At runtime there is no IRMA. NCrystal loads the data through the compiled plugin
and samples it. The chain:

1. **Discovery.** The plugin is a CMake `MODULE` library (`libNCPlugin_IRMA.so`).
   NCrystal finds it either way:
    - **pip (default):** install the `ncrystal_plugin_IRMA` package; NCrystal's
      `ncrystal-pypluginmgr` scans installed modules named `ncrystal_plugin_*`
      and loads each one's `plugins/*NCPlugin*.so`. (The compiled plugin name,
      `IRMA`, comes from the `.so` itself, not the package name, which is why
      the package must keep the `ncrystal_plugin_` prefix.)
    - **explicit:** `export NCRYSTAL_PLUGIN_LIST=/abs/.../libNCPlugin_IRMA.so`
      forces an in-process load, the reliable path when NCrystal is driven from
      a compiled C application (e.g. a McStas instrument) rather than Python.
2. **Activation.** Creating a scatter from an NCMAT that contains a
   `@CUSTOM_IRMA` section makes the plugin's `IRMAFactory` declare itself
   applicable and bid priority 999, above NCrystal's standard factory (100),
   so it wins the request. An NCMAT without that section is untouched, so the
   plugin only ever activates on IRMA-produced materials.
3. **Load.** The factory reads the data-file path(s) from `@CUSTOM_IRMA`, parses
   the text fields, validates the magic / `schema_version` / `units`, and builds
   the NCrystal objects: the α and β grids, the S(α,β) table, the per-site
   U-tensors, σ_b, the element mass, and the temperature.
4. **Component routing (no double-counting).** The plugin returns a *combined*
   process: its own model **plus** NCrystal's standard scatter with the
   components the plugin supplies switched off, i.e.
   `combineProcs(globalCreateScatter(cfg; inelas=0; coh_elas=0; incoh_elas=0),
   pluginModel)`. The plugin supplies:
    - **inelastic**: an NCrystal `SABScatter` built from the data file's table
      (`knltype = SCALED_SYM_SAB`);
    - **coherent elastic**: an anisotropic-DW Bragg process assembled from the
      per-site U-tensors (paired to the NCMAT atoms by fractional position);
    - **incoherent elastic**: the per-site incoherent Debye-Waller line.

    So, for an export with `elastic: true`, `comp=inelas`, `comp=coh_elas`,
    and `comp=incoh_elas` all resolve to the IRMA physics, while anything the
    plugin does not provide still comes from NCrystal core. With
    `elastic: false` only `comp=inelas` is IRMA's.
5. **Sample.** `SABScatter` evaluates the cross section and samples the final
   `(E′, μ)` from S(α,β). The data stores only the downscatter half, so NCrystal
   reconstructs the energy-gain (up-scatter) side by detailed balance as
   it samples. That is why a `scaled_sym` data file is complete despite being a
   half table.

Because the data is exported at one temperature, you must load it at that
temperature (`;temp=<export T>`). The plugin checks the requested material
temperature against each data file's export temperature and throws
`BadInput` when they differ by more than a tight tolerance
(~`1e-3 + 1e-5·T` K): it samples the precomputed table and does not
interpolate across temperatures, so a mismatched request would silently be
the wrong physics. Omitting `;temp=` is not
neutral: NCrystal then defaults to 293.15 K, which is a hard error against
any data file not exported at that temperature.

## Polyatomic materials (BeO)

A two-species config writes two data files (one per principal scatterer), and
the `@CUSTOM_IRMA` section in `beo.ncmat` lists both:

```text
@CUSTOM_IRMA
  pack /abs/out/beo__Be.irmapack
  pack /abs/out/beo__O.irmapack
```

The inelastic side is cleanly per-species: each data file carries its own
species' mode-2 `S(α,β)`, and summing the files sums the per-species
inelastic kernels with no ambiguity. The `principal-xs-weighted` coherent
partition splits the total coherent cross section exactly, with no
double-counting: each pairwise one-phonon interference term is shared
between its two participants as `w_p/(w_p+w_o)` (the bound coherent
cross sections of the principal and the other species), so the per-species partials sum back to the exact total for any
number of site groups.

The elastic line is different. The coherent Bragg structure factor
`F(hkl) = Σ_sites b·e^{−W}·e^{iφ}` is a single whole-crystal quantity with
cross-species interference terms, so it cannot be split per species without
either dropping cross terms or double-counting when the data files are summed.
The exporter therefore carries the whole-crystal elastic line in exactly one
data file, the species with the largest coherent weight `n·b_coh²`, which
holds the full primitive-cell anisotropic U-tensor set; the other data files are
inelastic-only. Summing them then adds the per-species inelastic kernels to a
single whole-crystal elastic line, free of double-counting and exact for the
total. (Graphite, with one data file, is the trivial case and is unaffected.)

## Conventions and provenance

The exporter applies a few convention bridges between the engine's output
and the data file. Know them before reading a data file or comparing one
against an IRMA tape:

- **`α` units.** The engine's `α` is mapped to NCrystal's mass-scaled units by
  `alpha_ncrystal = alpha_irma × AWR` (the principal-scatterer atomic weight
  ratio). The data file stores the NCrystal-convention `alpha_grid`.
- **Scaled-symmetric half-table.** With `gain_side: scaled_sym` the data file
  stores only the downscatter (`β ≥ 0`) side as `S_scaled = S_downscatter ·
  e^{−β/2}`; NCrystal reconstructs the full table (including the energy-gain
  side) by detailed balance. This halves the table size.
- **Bound-XS normalization.** The table (`sab_values`) keeps the engine's
  normalization per principal atom to `sigma_bound_b`. The per-atom weight of
  a multi-species material goes only into the advertised `bound_xs_barn =
  atom_fraction × sigma_bound_b` (`meta.atom_fraction` records it), so
  NCrystal's `SABScatter` reproduces the per-atom cross section. When
  comparing a data file with an ENDF tape, divide `bound_xs_barn` by the
  atom fraction to get the σ_b the table is normalized to.
- **Negative-cell clip.** A species' share of the mode-2 coherent law can be
  negative where the interference is destructive. ENDF and NCrystal tables
  cannot hold negative values, so the exporter, like IRMA's ENDF writer, sets
  those cells to zero; the summed material law is then slightly larger than
  the exact total. A cell below -1% of the table maximum stops the export: the
  direction sampling is too coarse for that share, and a larger
  `export.num_directions` fixes it.
- **Cell setting.** NCrystal 4.4.6 builds a wrong reciprocal lattice for a
  cell with cos α − cos β cos γ ≠ 0 (a bug in its general lattice branch,
  `NCLatticeUtils.cc:75-78`), which misplaces every Bragg edge. The exporter
  refuses such a cell before it bakes anything. Build the phonon model on a cell
  with α = β = 90° or α = γ = 90°: orthorhombic, tetragonal, hexagonal and
  monoclinic cells pass, and for FCC or BCC use the conventional cell, not the
  primitive one.
- **Provenance pin.** Each data file records the IRMA git SHA (marked `-dirty`
  if the tree had uncommitted edits), IRMA version, the phonopy-yaml SHA-256,
  mesh, direction counts, multiphonon order, and temperature, as `meta.*` lines.
  This is the reference pin: the plugin's CI can regenerate the data and
  byte/tolerance compare it against the mode-2 tape to catch drift.

## Export verification

The exported representation was verified on the graphite mode-2 evaluation
at 296 K. A write-read-rewrite round trip of the data file is byte-exact,
and NCrystal returns identical component cross sections from the original
and round-tripped files, so serialization is lossless.

The consequential check compares the two programs that process the same
stored `S(α,β)` table: the NCrystal plugin and corrected NJOY THERMR (stock THERMR plus the
two patches of the [NJOY interoperability](njoy.md) page) each
reconstruct cross sections from it. With the ENDF evaluation's 399×497 grid
supplied explicitly to the exporter, the two routes carry the same table
(relative L2 difference 1.3×10⁻⁷ after the header's bound-cross-section
rounding is accounted for). The reconstructed coherent-elastic cross
sections differ by at most 0.044% away from Bragg edges. The reconstructed
inelastic cross sections differ by up to 11.5% pointwise, with NCrystal
systematically lower: about 7.2% in the 10 μeV–25 meV integral, and about
1.8% in the integral up to 0.5 eV. The origin is how each program turns the
table into a cross section: at low incident energy the kinematically allowed
(α, β) region is a small, sparsely tabulated corner of the table, and
THERMR's reconstruction agrees with an exact integration of the same table
to about 0.1% (+0.10%), so the spread reflects NCrystal's internal
interpolation and quadrature rules rather than the stored data.

The spread is a discretization sensitivity, not a fixed disagreement:
feeding the same physics tabulated on denser grids through the same
NCrystal kernel machinery moves the 10 μeV–25 meV ratio of NCrystal's
integral to the exact one (1 is perfect agreement) from 0.84 on the coarse dQ = 0.25 Å⁻¹ grid, through 0.93 on the
automatic and dQ = 0.05 grids, to 0.98 at dQ = 0.01 and 0.99 at
dQ = 0.005. NCrystal handles grids denser than THERMR can stably process, so an export intended for transport can simply use a denser
table than the ENDF file; the exporter accepts explicit grids (see
[the grid options](#two-ways-to-set-the-s-grid)).

The end-to-end transport test, an ARCS McStas simulation with the exported
graphite file against measured maps and cuts, is on the
[graphite validation page](validation/graphite.md#arcs-through-mcstas-the-transport-application).

## See also

- [Scattering modes](modes.md): what `inelastic_mode = 2` computes (coherent
  one-phonon + anisotropic Debye-Waller).
- [Neutron scattering spectra](spectra.md): the `irma.spectra` forward model,
  which shares the `material` config block used here.
- [NJOY interoperability](njoy.md): the other program that processes IRMA's
  mode-2 `S(α,β)`, via an ENDF MF7 tape.
