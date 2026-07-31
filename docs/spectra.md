# Neutron-scattering spectra (`irma spectra`)

An ENDF tape tells a transport code how a material scatters; it does not tell
*you* what your instrument will measure. IRMA's forward model closes that gap:
it takes the same phonon physics the ENDF side evaluates and turns it into a
measured-style observable (an instrument-resolved 1-D INS spectrum, or a
dense 2-D `S(Q,E)` powder map) with no ENDF tape involved. Under the hood it
computes a powder `S(Q,E)`, projects it onto the instrument's kinematic locus
(fixed final energy for indirect geometry, fixed incident energy for direct),
convolves with an energy-resolution model, and adds a tape-free elastic line
built from the same Debye-Waller factors as the inelastic part. Use it to
preview an experiment, to compare a phonon model against measured data, or to
sanity-check an evaluation by looking at it the way a beamline would.

The physics level is the `inelastic_mode` you already know from the ENDF
side: `0` builds `S(Q,E)` straight from a phonon **DOS** (no eigenvectors;
see [DOS-based spectra (mode 0)](spectra-mode0.md) for that whole workflow),
while `1` and `2` run the phonopy-backed engine (incoherent approximation,
or exact coherent one-phonon) on a `phonopy.yaml` + force constants. It
covers VISION, generic indirect, and direct (chopper) geometries, with an
automatic chopper resolution model validated against PyChop. Install the
extra first: `pip install -e ".[spectra]"` (plus `[phonopy]` for modes 1/2).

## An end-to-end example: graphite, mode 2, two instruments

The committed examples are the fastest orientation. From `examples/spectra/`
(the configs use paths relative to that directory), the highest-fidelity 1-D
spectrum (exact coherent one-phonon, anisotropic Debye-Waller, VISION
resolution) is one command:

```bash
cd examples/spectra
python -m irma spectra run graphite_mode2_vision.yaml -o graphite_vision.csv
```

and the same phonon model becomes a direct-geometry ARCS `S(Q,E)` map, with
the chopper resolution computed automatically and the map masked to the
detector coverage:

```bash
python -m irma spectra map graphite_arcs_map.yaml -o graphite_arcs.npz
```

These are the two product shapes (per-bank energy spectra and the 2-D map),
shown here as the GUI's Plot pane renders them (the CLI writes the same data
to CSV/npz):

![A finished VISION run: the 45-degree and 135-degree bank spectra of graphite, mode 2](assets/gui/gui_ns_plot_vision.png)

![A finished ARCS run: the mode-2 S(Q,E) map on a log scale, masked to the accessible (Q,E) arch, kinematic envelope overlaid](assets/gui/gui_ns_plot_map.png)

The rest of this page is the reference for driving the model: the CLI
subcommands, every flag, and the config-file schema. The GUI face of the same
model is the **Neutron Scattering Experiments** tab, covered in
[the GUI guide](gui.md#neutron-scattering-experiments-tab).

## Two ways to drive it

There are two entry points, and they share **exactly one** compute path:

- **Flag form**: `irma spectra vision|indirect|direct …` builds a config from
  command-line flags. Pure sugar for quick runs.
- **Config form**: `irma spectra run <config> -o out.csv` (or `map`) loads a
  `SpectraConfig` file (YAML / TOML / JSON). Reproducible; the GUI writes these.

All temperatures default to **296 K**; override with `--temperature` (flag form)
or `material.temperature_K` (config form).

## CLI subcommands

### `vision`: VISION preset (indirect)

```bash
irma spectra vision \
    --phonopy-yaml graphite/phonopy.yaml --mesh 40 40 40 \
    --inelastic-mode 2 \
    --scatterer C,5.551,11.898 \
    -o graphite_vision.csv
```

VISION fills its own `Ef = 3.5 meV`, 45°/135° banks and published resolution
polynomial; override `Ef` with `--ef`.

### `indirect`: generic fixed-final-energy geometry

```bash
irma spectra indirect --ef 3.5 --angles 45,90,135 \
    --phonopy-yaml model.yaml --inelastic-mode 2 \
    --scatterer C,5.551,11.898 -o spec.csv
```

`--angles` accepts a comma list (`45,90,135`) or an inclusive
`start:stop:step` range (`30:150:15`).

### `direct`: generic fixed-incident-energy geometry

```bash
irma spectra direct --ei 250 --e-max 245 --angles 5:140:5 \
    --resolution-model chopper --chopper-instrument ARCS \
    --chopper-package ARCS-700-1.5-AST --chopper-frequency 600 \
    --phonopy-yaml model.yaml --inelastic-mode 2 \
    --scatterer C,5.551,11.898 -o arcs.csv
```

Direct geometry requires `--e-max` strictly below `Ei` (a neutron cannot lose
its full incident energy and still reach the detector); since the `--e-max`
default is 250 meV, `--ei 250` needs an explicit `--e-max` (or a higher `Ei`).
`--resolution-model chopper` auto-computes the energy resolution for any
supported PyChop instrument (ARCS, SEQUOIA, MAPS, MARI, MERLIN, HYSPEC; CNCS,
LET). `--kinematic-factor` multiplies by `kf/ki` for a count-rate spectrum.

### `run`: execute a config file

```bash
irma spectra run material.yaml -o spec.csv --set material.temperature_K=500
```

`--set a.b.c=value` applies dotted overrides on top of the file (repeatable).

### `map`: dense 2-D `S(Q,E)` powder map

```bash
irma spectra map material.yaml --q-min 0.5 --q-max 12 --dq-map 0.1 \
    --angle-range 5 140 --mask -o map.npz
```

The map runs for **any** geometry; the kinematics follow the config's
`instrument.geometry`. `--angle-range MIN MAX` attaches the instrument
kinematic envelope; `--mask` blanks `S(Q,E)` outside the accessible "arch".
Output is `.npz` (default) or long-form `.csv`. With `physics.elastic: true`
(the default) the map carries the same elastic *model* as the 1-D spectra: the
per-Q elastic cross section appears as an `E = 0` ridge (Bragg comb +
incoherent Debye–Waller line, smeared over the map's `--dq-map` Q bin) and is
broadened by the energy-resolution model like every other feature. The two
products deliberately differ in how they normalize the elastic line when the
energy axis **starts at 0** (`e_min = 0`, the default): the 1-D spectra
renormalize the visible half-peak so its integral equals the FULL per-Q
elastic area, while the map deposits `area/dE` in the boundary bin, which
retains only HALF the area under trapezoidal integration (the run prints a
NOTE when this applies). For absolute-intensity comparisons between the map
and the 1-D spectra, use `e_min < 0` so the whole line is on-axis and the two
conventions agree.

### Common flags (all geometries)

| Flag | Meaning | Default |
|------|---------|---------|
| `--inelastic-mode {0,1,2}` | 0 = DOS + isotropic Debye-Waller; 1 = incoherent approximation; 2 = exact coherent 1-phonon + incoherent multiphonon | 1 |
| `--scatterer` | `SYMBOL,sigma_bound_b,awr[,b_coh_fm[,sigma_inc_b]]` plus mode-0 `key=value` tokens (`dos=`, `unit=`, `mult=`, `pos=`). Repeatable. | — |
| `--mesh NX NY NZ` | phonon q-mesh | `40 40 40` |
| `--temperature` | sample temperature (K) | 296 |
| `--de` / `--e-min` / `--e-max` | energy grid step / start / end (meV); negative `e-min` adds the energy-gain side | 0.5 / 0 / 250 |
| `--dq` | `S(Q,E)` Q-support spacing (1/Å) | 0.05 |
| `--max-phonon-order` | multiphonon order; `auto` sizes to convergence in every mode (mode 0 derives it from the DOS Debye-Waller lambda) | auto |
| `--directions` / `--mp-directions` | modes 1/2 coherent / multiphonon powder-average directions | 10000 / 1000 |
| `--elastic {on,off}` / `--elastic-kind {both,coherent,incoherent}` | elastic line | on / both |
| `--gain-side {direct,detailed_balance}` | energy-gain (E<0) evaluation: explicit Bose factors vs detailed-balance mirror | direct |
| `--resolution-shape {gaussian,lorentzian}` | line shape | gaussian |
| `--jobs` | worker processes | all cores |
| `-o, --output` | output `.csv` (default) / `.npz` / `.json` | required |

## Config file reference (`SpectraConfig`)

A config file is a mapping of four sections: `material`, `physics`, `grid`,
`instrument`. All keys below are optional except `material.scatterers` (and
`material.phonopy_yaml` for modes 1/2). Save with `irma spectra` GUI or hand-write
YAML/TOML/JSON.

### `material`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `phonopy_yaml` | str | — | phonopy model; required for modes 1/2 and mode-0 `dos_source: phonopy` |
| `force_constants` / `force_sets` / `born` | str | discovered | explicit FC / FORCE_SETS / BORN (NAC) files |
| `mesh` | `[nx,ny,nz]` | `[40,40,40]` | phonon q-mesh |
| `temperature_K` | float | 296 | sample temperature |
| `scatterers` | list | — | per-species scattering data (see below) |
| `lattice` | `[a,b,c,α,β,γ]` | — | mode-0 only, unit cell (Å, deg) for the coherent-elastic Bragg comb |

Each **scatterer** entry: `symbol`, `sigma_bound_b`, `awr`, optional `b_coh_fm`,
`sigma_inc_b`, and mode-0 fields `dos_file`, `dos_unit` (`meV|eV|cm-1|THz`),
`multiplicity`, `positions` (fractional `[x,y,z]` sites for the elastic comb).

### `physics`

| Key | Default | Meaning |
|-----|---------|---------|
| `inelastic_mode` | 1 | 0 DOS / 1 incoherent-approx / 2 coherent 1-phonon + incoherent multiphonon |
| `dos_source` | `file` | mode-0 DOS origin: per-scatterer `dos=` files or `phonopy` |
| `max_phonon_order` | `auto` | int ≥ 1 or `auto` |
| `n_directions` / `multiphonon_directions` | 10000 / 1000 | powder-average directions (modes 1/2) |
| `jobs` | null (all cores) | worker processes |
| `elastic` / `elastic_kind` | true / `both` | elastic line on/off and channel (`both`/`coherent`/`incoherent`) |
| `elastic_from_tape` | — | build the elastic line from an ENDF MF7/MT2 tape instead of from the phonon model |
| `incoherent_elastic_mode` | `isotropic` | Debye-Waller treatment of the incoherent elastic line: `isotropic` (trace/3 scalar W′ per species, the ENDF-convention form) or `directional` (powder-averaged anisotropic `⟨exp(-Q² û·U·û)⟩` per atom; modes 1/2 only; needs the engine's displacement tensors). Same option name as the NCrystal export. |
| `include_energy_gain` | true | add the energy-gain (E<0) side |
| `gain_side` | `direct` | energy-gain evaluation: `direct` computes E<0 with explicit Bose occupation factors (phonon-annihilation weights `n(ω)`, no mirror) for ALL modes, mode 0 sums every order in closed form, modes 1/2 deposit annihilation lines at `-ħω` plus the negative half of the signed multiphonon ladder (the loss/ENDF outputs stay bit-identical); or `detailed_balance` (mirror the loss side). The two agree to the sub-bin `O(dE/kT)` level, with `direct` the more accurate (it uses each line's true energy, not the loss bin center) |
| `kinematic_kf_ki` | false | multiply by `kf/ki` (count-rate spectrum) |

### `grid`

| Key | Default | Meaning |
|-----|---------|---------|
| `e_min_meV` / `e_max_meV` / `de_meV` | 0 / 250 / 0.5 | energy-transfer grid (negative `e_min` includes energy gain) |
| `dq_max_invA` | 0.05 | `S(Q,E)` Q-support spacing |
| `q_max_invA` | — | 2-D map Q-axis maximum (1-D runs derive Q from the instrument) |

### `instrument`

| Key | Default | Meaning |
|-----|---------|---------|
| `geometry` | `vision` | `vision` / `indirect` / `direct` |
| `e_fixed_meV` | 3.5 | `Ef` (vision/indirect) or `Ei` (direct) |
| `angles_deg` | preset | detector angles (deg) |
| `q_cuts` | — | constant-\|Q\| cuts (1/Å); honored whenever supplied, alongside the angle/bank spectra |
| `bank_halfwidth_deg` | 5 | detector-bank angular half-width |
| `sigma_coeffs` | preset | resolution width polynomial `c0,c1,c2` (meV) |
| `resolution_shape` / `resolution_model` | gaussian / poly | line shape; `poly` or `chopper` |
| `chopper_spec` | — | `{instrument, package, frequency}` for `resolution_model: chopper` |
| `combine` | mean | combine detector banks by `mean` or `sum` |
| `output_mode` / `cut_by` | cuts / angles | `cuts` (1-D spectra) or `map` (dense 2-D `S(Q,E)`, any geometry); `cut_by` (direct geometry only): `angles` (bank spectra, plus any `q_cuts`) or `q` (constant-Q cuts only) |
| `cut_dq_invA` | — | constant-Q cut band width (None → thin slice) |
| `map_coverage_deg` / `map_mask` | — / true | 2-D map kinematic envelope band + masking |
| `export_components` | false | also write the inelastic + elastic breakdown (else total only) |

!!! tip "Mode 0 vs modes 1/2"
    Mode 0 builds `S(Q,E)` straight from a phonon **DOS** (per species), with no
    eigenvectors. It is fast and ideal for incoherent / hydrogen-rich materials. See
    [DOS-based spectra (mode 0)](spectra-mode0.md). Modes 1/2 use the phonopy
    eigenvectors for the exact one-phonon term.

!!! note "Which chopper was in the beam? (finding `chopper_spec`)"
    `resolution_model: chopper` needs the Fermi package and frequency that were
    actually in the beam for the run you are modelling. Take them from the data,
    not from the nominal request, because they can differ and the choice matters
    (a 0.5 mm vs 1.5 mm slit package changes the elastic width roughly 2×). On
    **ARCS** (SNS), the in-beam Fermi package is recorded in the raw NeXus file
    as the DAS log `BL18:Chop:InUse:ChopperId`, and the Fermi speed is in the
    chopper-frequency log; the reduced `NXSPE`/processed file usually also
    carries the incident energy and chopper settings in its metadata. Other
    direct-geometry instruments expose the same information under their own log
    names. Look up the in-beam chopper id and speed for the specific run.

## What the forward model captures, and what it does not

The forward model is a **single-scattering, powder-averaged** `S(Q,E)`
convolved with an **energy-width** resolution model. That makes it fast and
makes its output a clean expression of the *phonon model*, but several real
measurement effects are deliberately absent. Read any comparison against
measured data with these in mind.

**Single scattering only.** IRMA scatters each neutron once. A real sample of
any appreciable thickness also multiply-scatters, and multiple scattering adds
a broad inelastic background that piles up under and beside the phonon
features. The visible symptom is the **elastic-to-inelastic balance**: the
single-scatter calculation keeps too much weight in the elastic line relative
to the inelastic continuum, and the gap grows with incident energy and sample
thickness. For a millimetre-thick graphite plate on a direct-geometry chopper
spectrometer the single-scatter elastic/inelastic area ratio runs roughly
**1.5–2.5× high** versus the measurement (growing with `Ei`), while the phonon
peak *positions* and *relative* one-phonon intensities stay correct. Treat
absolute inelastic intensities and the elastic/inelastic ratio as model
bounds, not predictions.

**Resolution is a Gaussian width, not a full lineshape.** The chopper model
(`resolution_model: chopper`) reproduces PyChop's energy-dependent FWHM to
~1%, and that width is applied as a **symmetric Gaussian**. PyChop itself
emits only a variance, so there is no lineshape to borrow. The true elastic
line is mildly **asymmetric**: the moderator emission pulse leaves a small
energy-*gain*-side tail, strongest at low incident energy (half-width
asymmetry ≈ 1.2–1.5 at `Ei` ≈ 30 meV) and fading to near-symmetric by a few
hundred meV. The Gaussian is a good approximation everywhere except the
low-`Ei` wings; it does not model the moderator-pulse, detector-depth, or
sample-size contributions to the line *shape* (only their contribution to the
*width*, through the chopper model).

**No sample or instrument geometry.** There is no self-shielding or
absorption, no sample-can or sample-environment scattering, no detector
efficiency or angular-coverage gaps, and no detector-binning kinematics. The
2-D map is an idealised powder `S(Q,E)` on the instrument's kinematic locus.

!!! tip "When you need the full instrument: use McStas / McVINE"
    For anything that depends on the effects above (multiple scattering, the
    real moderator-pulse lineshape, detector geometry and efficiency, sample
    self-shielding, or **absolute** intensities), model the instrument with a
    Monte-Carlo package such as **McStas** or **McVINE**, using IRMA (or an
    NCrystal / ENDF kernel built from the same physics) as the sample
    scattering kernel. IRMA's forward model is the right tool for previewing
    features and for comparing a *phonon model* to data on an equal footing;
    a Monte-Carlo instrument model is the right tool for reproducing a measured
    spectrum in full. The graphite validation in this project pairs the two
    exactly this way: IRMA for the phonon physics, a McStas virtual experiment
    for the instrument and multiple scattering.

    One caveat when reading such an overlay: the Monte-Carlo sample kernel and
    IRMA may not use the *same* inelastic physics. The McStas/NCrystal graphite
    kernel used here, for example, evaluates the inelastic scattering in the
    **incoherent approximation**, whereas IRMA mode 2 computes the **coherent
    one-phonon** term, so where the two inelastic continua differ, part of the
    difference is the scattering *model*, not just multiple scattering.
