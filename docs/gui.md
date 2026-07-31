# GUI guide

The IRMA graphical interface has four top-level tabs:

- **ENDF Evaluation** — a single-page front end for building a LEAPR-style deck
  and running a thermal-scattering-law calculation without hand-editing cards.
  The page is one scrolling form of five parts in deck-writing order
  (**Material**, **Scattering**, **Grids**, **Phonon**, and **Run**) that you
  fill in top to bottom; the **Jump to** bar above the form scrolls any part to
  the top.
- **Neutron Scattering Experiments** — the forward-spectra panel: it predicts
  instrument-resolved INS spectra and 2-D S(Q,E) maps from a phonopy model or a
  phonon DOS, with no ENDF tape involved. Covered in
  [its own section below](#neutron-scattering-experiments-tab).
- **NCrystal plugin** — exports the per-temperature material-data file that the
  companion transport plugin samples on the fly; the format and workflow are
  documented on the [NCrystal data exporter](ncrystal-plugin.md) page.
- **MLIP phonon models** — builds a phonon-model bundle from a structure file
  with a pretrained machine-learned interatomic potential (MLIP) and generates
  ready-to-edit inputs for the other tabs. Toured in
  [its own section below](#mlip-phonon-models-tab); the full reference is the
  [MLIP phonon models](mlip.md) page.

The defining behavior of every tab is *progressive disclosure*: a field appears only when the mode that consumes it is selected, so you never stare at controls that have no effect on your calculation. Importing an existing deck reveals exactly the sections that deck uses. This page walks through each part and calls out the pitfalls worth knowing before your first run.

## Launching

```bash
irma-gui
# or
python -m irma --gui
```

The window title bar shows the product name, and the bottom bar shows the installed IRMA version.

!!! note "Progressive disclosure, in one sentence"
    Choosing `iel=10` in the Material part unlocks the crystal-structure and phonopy sections; choosing `inelastic_mode=1` or `2` unlocks the phonopy mesh/control fields and flags the legacy Phonon cards as not read; choosing a special mode (`ncold`, `nsk`, `nss`) in the Scattering part unlocks just that mode's extra inputs. Everything else stays hidden.

---

## Material part

This is where you declare *what* you are modeling: the elastic treatment, and, for the generalized path, the crystal structure and the phonopy-backed inelastic mode.

![The Material part with iel=10 and inelastic mode 2 selected — the full crystal-structure and phonopy section is disclosed](assets/gui/gui_material_iel10.png)

Progressive disclosure in action: with `iel=10` and `inelastic_mode=2`, the lattice parameters, atom-type table, and the phonopy controls (mesh, `ndir`/`mpdir`, auto-size order, …) are all shown. Selecting a classic mode collapses them to a single dropdown:

![The Material part with iel=1 (classic graphite) — the crystal and phonopy sections are hidden](assets/gui/gui_material_iel1.png)

### Quick Start (deck import)

The **Quick Start** panel has a single **Import Input File...** button (also reachable from **File ▸ Import Input File**). Point it at an existing `.input` or `.leapr` deck and IRMA parses it and populates the whole form. Because import writes the same internal variables that your clicks do, the conditional sections reveal themselves automatically: a deck with `iel=10` shows the crystal cards, a deck with a secondary scatterer shows the secondary fields, and so on. This is the fastest way to start from a known-good evaluation and tweak it.

!!! note
    You can also fill in the fields by hand. Import is optional; it just saves typing.

### Elastic Scattering Mode (`iel`)

The `iel` dropdown selects the coherent-elastic (Bragg-edge) treatment.

| `iel` | Meaning |
|-------|---------|
| `0` | No coherent elastic scattering |
| `1`–`6` | Legacy built-in materials: graphite, Be, BeO, Al, Pb, Fe (hardcoded crystal structures) |
| `10` | **Generalized** — Bragg edges computed from the crystal structure you define in this part, for any material |

`iel=10` is the recommended option for new evaluations and is the default selection.

!!! warning "iel=10 unlocks the rest of the part"
    The **Elastic format** selector, **Lattice Parameters**, **Atom Types**, and the **Phonopy-Based Inelastic Mode** section appear *only* for `iel=10`. For the built-in materials (`iel=0`–`6`) those cards are not read, so the GUI hides them to keep you from filling in fields that have no effect.

### Elastic format (`elastic_mode`, iel=10 only)

| `elastic_mode` | Format | Behavior |
|----------------|--------|----------|
| `1` | SEF (single-channel elastic format) | One ENDF-legal elastic section per tape: a single-atom material gets coherent elastic scaled by the full strength when sigma_coh > sigma_inc, else incoherent elastic carrying the sum; a polyatomic cell routes the dominant channel's atom to coherent elastic and redistributes the rest incoherently (Eq. 26). See [the theory page](theory.md#elastic-format-sef-vs-mef-card-6b-field-1) for the exact rule. Standard format, supported by all transport codes. |
| `2` | MEF (Mixed Elastic Format) | Every atom gets both coherent and incoherent elastic (LTHR=3). More accurate for polyatomic materials, but needs downstream transport-code support for the mixed format. |

### Lattice Parameters (iel=10 only)

Six fields define the unit cell: edge lengths `a`, `b`, `c` in Ångström and angles `alpha`, `beta`, `gamma` in degrees. The defaults describe a hexagonal graphite cell. Common settings:

- Cubic: `a = b = c`, `alpha = beta = gamma = 90`
- Hexagonal: `a = b`, `alpha = beta = 90`, `gamma = 120`

### Atom Types (iel=10 only)

A free-text box, one line per distinct atom species, in the column order shown in the header:

```
Z  A  AWR  b_coh  sigma_inc  npos  x1 y1 z1  x2 y2 z2 ...
```

| Field | Meaning |
|-------|---------|
| `Z` | Atomic number |
| `A` | Mass number |
| `AWR` | Atomic weight ratio to the neutron mass |
| `b_coh` | Coherent scattering length (fm) |
| `sigma_inc` | Incoherent cross section (barn) |
| `npos` | Number of equivalent positions in the cell |
| `x y z ...` | Fractional coordinates of each position |

For a polyatomic material such as NaCl, use one line per species. Scattering lengths and cross sections come from the NIST neutron scattering-length tables.

### Coherent-Elastic Output (iel=10 only)

The **Bragg-edge grouping** controls live in their own frame between the atom-type table and the phonopy section, because they are a coherent-**elastic** MF7/MT2 output option that applies to *every* `iel=10` inelastic mode, including the classic mode 0, which needs no phonopy at all.

The **Bragg-edge grouping** checkbox is **on by default** (compact tapes); uncheck it to keep every Bragg edge. When checked, two fields control the grouping of dense high-energy edges, following ENDF-102 §7.2.2. Importing a deck sets the checkbox to match the deck (on only if it carries the Card 6b grouping fields).

| Field | Default | Meaning |
|-------|---------|---------|
| Bragg-edge grouping (checkbox) | **on** | Group dense high-energy edges. Uncheck to write every edge (the former default). |
| bins/decade | `50` | Above the threshold, the energy axis is split into this many log-uniform bins per decade; all edges in a bin merge into one step at the structure-factor-weighted log-mean energy. |
| above (eV) | `1.0` | Energy above which grouping may occur. Edges at or below it are always kept individually. |

Grouping preserves the cumulative S, the total bound cross section, and the high-energy 1/E tail; only the placement of merged high-energy steps is approximate (IRMA prints the resulting integral cross-section error per temperature). These map to optional fields 5–6 on Card 6b. Grouping **composes with extinction**: extinction refines the comb below ~0.1 eV while grouping compacts the edges above the threshold.

### Crystalline Extinction (iel=10 only)

The **Crystalline Extinction (Optional)** frame applies a dynamical-diffraction reduction to the coherent-elastic Bragg comb, the effect that makes a real crystallite's peaks weaker than ideal theory predicts. It is a *sample* property, off by default.

Tick **Enable extinction correction**, pick a **model** (five available; the coupled `BC_mix`/`BC_mod` need `l`, `g`, and `L` all > 0), and set the sample sizes: **l** (crystallite, primary), **g** (mosaic) and **L** (grain) for secondary, plus the **distribution**, **recipe**, and tabulation **rmse_tol**. Every control has an **ⓘ help glyph** (hover for a preview, click for the full text) explaining the parameter and linking the references; the attribution line credits the NCrystal CrysXT plugin. Changing the model snaps the distribution to that model's default (`Gauss` for Becker-Coppens, `rect` for Sabine). The section round-trips through deck export/import.

See **[Crystalline extinction](extinction.md)** for the full physics, the deck card, validation, and references.

### Phonopy-Based Inelastic Mode (iel=10 only)

This section chooses the inelastic model.

The `inelastic_mode` radio buttons:

| `inelastic_mode` | Label | What it does |
|------------------|-------|--------------|
| `0` | legacy cubic | Isotropic Debye-Waller, phonon expansion from a scalar DOS supplied in the Phonon part. |
| `1` | incoherent approx. | phonopy-backed MT4: directional Debye-Waller / MT2 handling plus an incoherent-approximation one-phonon term and incoherent-approximation multiphonons. |
| `2` | coherent (exact 1-phonon) | Same phonopy path, but injects the exact one-phonon term (coherent + incoherent) plus incoherent-approximation multiphonons. Matches the Neutron Scattering Experiments panel's "2 (coherent 1ph+multi)". |

!!! warning "Modes 1 and 2 need phonopy"
    Modes 1/2 require the optional `phonopy` package, a `phonopy.yaml` with force constants, and the Card 6g controls below. They do **not** read the legacy continuous-DOS, translational, or oscillator cards in the Phonon part — MT4 and the directional elastic Debye-Waller factors come straight from the phonon model. Selecting mode 1 or 2 reveals the phonopy sub-frame; mode 0 hides it.

#### Phonopy parameters (modes 1/2 only)

When `inelastic_mode` is 1 or 2, the sub-frame exposes:

| Control | Default | Purpose |
|---------|---------|---------|
| `phonopy.yaml` | — | Path to the phonopy YAML. Force constants are read from the YAML if embedded, otherwise discovered alongside it (`force_constants.hdf5`, `FORCE_CONSTANTS`, or `FORCE_SETS`). |
| Mesh `nx ny nz` | `40 40 40` | Monkhorst-Pack mesh for the BZ sampling behind the phonon DOS tensor (matches the Neutron Scattering Experiments tab and the validation suite). Anisotropic crystals may use an asymmetric mesh. |
| `ncpu` | `1` | Parallel worker processes for the phonopy-backed workflow. |
| `ndir` | `10000` | One-phonon powder-average direction count: the validation-campaign sampling, well converged for production runs; drop to ~4000 for a faster look or ~1000 for a quick one. Cost is roughly linear. |
| `mpdir` | `1000` | Multiphonon powder-average directions over the unit sphere (golden-spiral / Fibonacci); converged by ~50–100, so the default carries ample margin. |
| Auto-size multiphonon order | **on** | When on (the default), IRMA sizes the multiphonon order `nphon` from the anisotropic Debye-Waller physics. When off, your Card 3 `nphon` is honored exactly; see the warning below. |
| Use BORN corrections (NAC) | off | Non-analytical correction for polar materials; checking it reveals a BORN-file selector. Leave unchecked for non-polar materials such as graphite. |

!!! warning "Anisotropic Debye-Waller and high Q"
    With the anisotropic Debye-Waller factor the multiphonon order needed to reach the free-gas limit grows with Q. A hand-set `nphon` that is fine at low Q will silently truncate the high-Q rows of the cross section: with the default grids (`Beta max = 5 eV`, graphite AWR) the grid reaches Q ≈ 98 Å⁻¹, where an order near 223 is needed while the default `nphon` is 100. The GUI therefore ships **Auto-size multiphonon order** ON. If you turn it off (a deliberate low-order study), IRMA honors your `nphon` exactly and prints a terminal warning when it is too low.

!!! note "mpdir convergence"
    The multiphonon powder average converges quickly: for graphite-like crystals it is already converged by roughly 50–100 directions, so the default of 1000 is deep inside the converged regime. Cost scales linearly with `mpdir`, so larger values cost proportionally more without improving a converged result. Raise it only to verify convergence on a new material.

The coherent one-phonon powder averaging always uses the golden-spiral direction quadrature; it is not user-selectable.

The GUI defaults produce Card 6g `10000 1000 1`: 10000 coherent directions, 1000 multiphonon directions, auto-sized multiphonon order — the same sampling the validation campaign ran and `irma mlip emit` writes. (The powder averages converge well below these counts, so entering `4000 200` gives a faster exploratory run.)

---

## Scattering part

Here you describe the principal scatterer, control the ENDF output, and opt into any special moderator physics.

![The Scattering part — principal scatterer, ENDF output control, special modes, and secondary scatterer](assets/gui/gui_scattering.png)

### Principal Scatterer

| Field | Default | Meaning |
|-------|---------|---------|
| `ZA` | `6012` | Scatterer identity as Z×1000 + A. Must match an atom type from the Material part. |
| `AWR` | `11.898` | Atomic weight ratio to the neutron mass; drives recoil kinematics and alpha-grid scaling. |
| `sigma_free` (`spr`) | `4.739180` | **FREE-atom** scattering cross section (barn), exactly as on LEAPR Card 5. IRMA derives the bound cross section that normalizes S(α,β) internally as `sigma_b = sigma_free·((1+AWR)/AWR)²` and writes `B(1) = npr·sigma_free` to the tape. Do **not** enter the bound value (carbon: enter 4.739, not 5.55; hydrogen: enter ~20.45, not ~82; the bound value makes every MT4 cross section ~4× too large for H). |
| `npr` | `1` | Number of principal-scatterer atoms in the material unit (≥ 1). For `iel=1`–`6` it also scales the built-in Bragg-edge cross sections. |

The **Fill AWR + sigma_free from ZA** button looks the ZA up in IRMA's
built-in nuclear table (the Rauch–Waschkowski/Sears compilation, see
`irma.core.nuclear_data`) and overwrites AWR and `sigma_free` with the
tabulated values; `sigma_free` is derived from the bound cross section as
`sigma_b·(AWR/(1+AWR))²`, so the free/bound convention is always right.
Nuclides with energy-dependent scattering lengths (B, Cd, Gd, …) and
isotopes without measured constants are refused with a message; enter those
by hand.

!!! note "One principal MT4 law per deck"
    For `inelastic_mode=1/2`, IRMA writes one principal-scatterer MT4 law per deck even when the crystal has several atom types. If you need more than one principal (for example Be and O in BeO), run separate decks.

### ENDF Output Control

| Field | Default | Meaning |
|-------|---------|---------|
| `MAT number` | `28` | ENDF material number for the output library. |
| `nphon` | `100` | Maximum phonon expansion order. Light scatterers (H, D) reach much larger α for the same β grid and generally need *more* terms; heavy scatterers converge with fewer. For modes 1/2 this is also the multiphonon maximum order (unless Auto-size is on, the GUI default). |
| `isabt` | `0 — S(α,β)` | Stored law form. `0` is the standard symmetric S(α,β) required by THERMR; `1` writes the asymmetric S̃ form (diagnostics only, not for library production). |
| `ilog` | `0 — S values` | Storage form. `0` stores S directly; `1` stores ln(S) (ENDF LLN=1), useful for laws spanning many decades when the processor supports it. |
| `smin` | `1e-75` | Law values below this threshold are treated as zero (LEAPR convention). |

### Special Modes

These selectors stay collapsed by default; choosing a non-zero value reveals only that mode's extra inputs.

| Selector | Options | Reveals |
|----------|---------|---------|
| `ncold` | None / Ortho-H / Para-H / Ortho-D / Para-D | The S(κ) table (`dka`, S(κ) values) when non-zero. |
| `nsk` | None / Vineyard / Sköld | The S(κ) table, plus `cfrac` (coherent fraction) when non-zero. |

The S(κ) inputs (`dka` grid spacing and the space-separated S(κ) values) appear whenever `nsk > 0` **or** `ncold > 0`. For `nsk > 0` you must also give `cfrac`.

!!! warning "Not available with phonopy modes"
    `ncold` and `nsk` are rejected with `inelastic_mode=1/2`, because MT4 comes from the phonopy model rather than a tabulated pair-correlation treatment.

### Secondary Scatterer (optional)

Set `nss` to **1 — One secondary scatterer** to reveal the secondary-species fields:

| Field | Meaning |
|-------|---------|
| `b7` | Secondary model: `1` free gas (most common, e.g. O in H₂O), `2` diffusion, `0` bound two-pass (the secondary gets its own phonon-spectrum pass, e.g. O in BeO). |
| `AWS` | Secondary atomic weight ratio (> 0). |
| `sigma_s` | Secondary free-atom cross section in barn (> 0). |
| `mss` | Number of secondary atoms in the unit (≥ 1). |

Choosing `b7 = 0` additionally reveals the **Secondary phonon model (b7 = 0 two-pass only)** panel (its own DOS spacing/values, translational weights, and oscillator energies/weights), which IRMA emits as a complete second temperature pass and merges with bound-cross-section weighting. The two-pass merge exists only on the classic elastic paths: with generalized elastic (`iel = 10`) a bound `b7 = 0` secondary is rejected at parse time (use `b7 = 1`/`2` or `nss = 0`).

!!! note
    For most polyatomic materials evaluators generate a separate table per species (`nss = 0`) instead of using the mixed-moderator approach. A secondary scatterer is not supported with `inelastic_mode=1/2`, and combining `ncold`/`nsk` with the two-pass case is deck-file-only.

---

## Grids part

This part defines the temperatures and the α/β grids on which S(α,β) is tabulated.

![The Grids part — automatic α/β grid generation with the live size preview](assets/gui/gui_grids.png)

With **Automatic grid generation** selected, **Preview Grid Sizes** reports the resulting point counts and ranges without running the calculation.

### Temperatures

Enter temperatures in Kelvin, space-separated. The phonon spectrum is read only at the first temperature; subsequent temperatures reuse it and recompute S(α,β) for the new Boltzmann population.

The **LAT** selector controls α/β scaling:

| LAT | Meaning |
|-----|---------|
| `0` | α/β in units of kT = k_B·T (grid meaning changes per temperature). |
| `1` | α/β in units of a fixed kT_thermal = 0.0253 eV (T = 293.6 K). Standard ENDF TSL convention; the same grid serves all temperatures. **Recommended** (default). |

### Alpha / Beta grids — automatic vs manual

A radio pair switches between **Automatic grid generation** (default) and **Manual alpha/beta entry**; only the chosen mode's fields are shown.

#### Automatic — beta controls

The beta grid has three regions: a logarithmic low-β tail, a linear phonon region over [0, freq_max], and a logarithmic high-β tail up to Beta max.

| Field | Default | Meaning |
|-------|---------|---------|
| Max phonon freq | `0.20` | Upper bound of the linear region (eV). The **Detect from DOS** and **Detect from phonopy.yaml** buttons fill this in for you. |
| N lower (log) | `50` | Logarithmic points in the low-β (thermal quasi-elastic) tail. |
| N phonon (linear) | `300` | Linear points across [0, freq_max], where phonon features live. |
| N upper (log) | `80` | Logarithmic points in the high-β multiphonon tail. 80 is conservative: it keeps the high-β step fine enough that a lin-lin (INT=2) law does not overshoot the free-atom limit at high incident energy. Fewer (~20) suffice for log-lin or thermal-only runs; verify grid convergence for your energy range and adjust up or down. |
| Beta max | `5.0` | Maximum energy transfer (eV) for the upper tail. Hydrogen may need 10 eV or more. |

#### Automatic — alpha controls

The alpha grid is **linear in momentum transfer Q**: points every `dQ` out to `Q cut`, then a logarithmic tail to the beta grid's kinematic reach (α_max = 4·beta_max/AWR).

| Field | Default | Meaning |
|-------|---------|---------|
| Alpha dQ | `0.05` | Q spacing (1/Å) of the linear segment; α = Q²·(ℏ²/2m)/(AWR·kT). Sets how well the thermal upscatter windows (Q ≈ 1–6 1/Å) are resolved. |
| Alpha Q cut | `12.0` | Q (1/Å) where the grid switches from linear to a logarithmic tail. Fixed by neutron kinematics, not the material; rarely needs changing. |
| Alpha N log | `160` | Logarithmic points from Q cut to the grid maximum (feeds the epithermal cross section and the free-gas limit). |

!!! warning "Why the alpha grid is linear in Q"
    The thermal-energy inelastic cross section integrates S(α,β) over upscatter windows at small α (Q ≈ 1–6 1/Å). An alpha grid that simply mirrors the beta layout under-integrates that cross section by 10–20%. On graphite at 296 K the default `dQ = 0.05` grid lands within 3–5% of the converged result at only ~400 columns; a linear `dQ = 0.01` grid is converged (halving dQ again moves results ≤0.3%).

#### Manual entry

Two text boxes take space-separated α and β values, both ascending; the β list should start at 0. Use this to match a reference evaluation exactly or to control grid placement precisely.

### Preview

The **Preview Grid Sizes** button reports the resulting `nalpha` and `nbeta` (and the α/β endpoints in auto mode) without running the calculation, a quick sanity check on grid budget before you commit.

!!! note "nalpha is independent of nbeta"
    The alpha and beta point counts are decoupled, so you can resolve the thermal Q window densely without inflating the energy grid.

---

## Phonon part

This part supplies the scalar phonon model used by the **legacy cubic** path (`inelastic_mode = 0`).

![The Phonon part — with inelastic mode 1/2 selected, a banner notes these cards are not read](assets/gui/gui_phonon.png)

!!! warning "Ignored for phonopy modes"
    When `inelastic_mode = 1` or `2` is selected in the Material part, a red banner appears at the top of this part: the phonopy-backed modes compute MT4 directly from the force constants, so the phonon distribution, translational, and oscillator cards here are **not** read. They matter only for `inelastic_mode = 0`.

### Continuous Phonon Distribution

A radio pair chooses the DOS source, and only the selected source's fields appear directly beneath its button:

- **From phonopy total_dos.dat** (default) — a file selector for a phonopy `total_dos.dat` (two columns: frequency in THz and DOS). IRMA converts THz→eV and normalizes the spectrum.
- **Manual entry** — `delta_e` (uniform energy spacing, eV) plus a space-separated `rho` list on the equidistant grid starting at E = 0.

### Translational Mode

| Field | Default | Meaning |
|-------|---------|---------|
| `twt` | `0.0` | Weight of the translational (diffusive/free-gas) component. `0` for a crystalline solid. |
| `c (diffusion)` | `0.0` | Diffusion constant; `0` is free-gas translation, `> 0` is the Egelstaff-Schofield diffusion model. Used only when `twt > 0`. |
| `tbeta` | `1.0` | Weight of the continuous distribution. The weights satisfy `tbeta + twt + Σ(oscillator weights) = 1`. |

### Discrete Oscillators

Two text boxes take space-separated oscillator **energies** (eV) and **weights** for isolated vibrational modes treated analytically — typically intramolecular modes such as the H₂O bend and stretch. Leave them empty for most crystalline solids.

---

## Run part

The final part sets the output destination, optional ENDF documentation, and launches the job.

![The Run part — a completed graphite iel=10 mode-2 calculation with the streaming log](assets/gui/gui_run.png)

- **Output ENDF file** — destination for the ENDF-6 tape (S(α,β) in MF7/MT4, plus MF7/MT2 Bragg edges when `iel > 0`). This tape is what THERMR/ACER consume downstream.
- **ENDF Comment Cards (MF1/MT451)** — free-text box; each line becomes one comment record (truncated to 66 characters). Document the method and references here.
- **Run Calculation** — generates the deck from the current fields and runs it, streaming progress into the **Log** pane with an indeterminate progress bar, a status label, and a live phase readout showing the engine's current stage. **Cancel** terminates a running calculation; **Clear Log** empties the pane.
- **Export Input File...** — writes the generated deck to a `.input`/`.leapr` file without running, so you can inspect, archive, or hand it to the command-line tool.

!!! danger "Coherent mode-2 tapes need a stock-NJOY THERMR patch"
    A coherent `inelastic_mode = 2` tape (graphite and similar) processed through unpatched NJOY2016 THERMR comes out as ~`1e91`-barn garbage above ~`0.27 eV`, or stalls for hours. This is a `cliq` bug in stock `thermr.f90` (its guard tests decay along alpha but not beta), not an IRMA tape defect, and it fires on the law's *shape*, not its grid spacing, so no beta-grid choice avoids it. Apply the one-line two-axis guard patch first; `inelastic_mode = 0/1` tapes and most materials are unaffected. See [NJOY interoperability](njoy.md).

---

## Neutron Scattering Experiments tab

The second top-level tab is the forward-spectra panel. It answers a different
question than the deck editor: not "what is the scattering law?" but "what
would my instrument measure?": an instrument-resolved 1-D INS spectrum or a
2-D S(Q,E) powder map, computed directly from a phonon model with no ENDF tape
involved. It is the GUI face of the `irma spectra` command line; **Save
Config...** / **Open Config...** round-trip the same YAML files the CLI runs,
so you can prototype in the GUI and script the production run.

![Neutron Scattering Experiments tab with a phonopy model loaded — graphite, inelastic mode 2, 40³ mesh, with the per-element scattering table and the Physics direction counts](assets/gui/gui_ns_phonopy_mode2.png)

### Material (phonon model)

The first control, **phonon input**, gates the whole panel:

- **Phonopy model** (default) — point **phonopy.yaml** at your phonopy file
  (force constants are discovered next to it, or set **FORCE_CONSTANTS** /
  **FORCE_SETS** / **BORN** explicitly) and choose the **mesh**.
  **Auto-fill elements from phonopy.yaml** populates the element table with
  the cell's species and multiplicities.
- **DOS files (mode 0)** — the no-phonopy path: each element row carries its
  own 2-column phonon **DOS file** (frequency, intensity; meV/eV/cm⁻¹/THz)
  instead. See [DOS-based spectra (mode 0)](spectra-mode0.md).

![DOS files (mode 0) selected — the phonopy fields are gone; the element table gains the DOS file, unit, multiplicity and positions columns, and the lattice row enables the coherent-elastic comb (the committed graphite example)](assets/gui/gui_ns_dosfiles_mode0.png)

Below the gate sit the **inelastic mode** (`0` DOS + isotropic Debye-Waller,
`1` incoherent approximation, `2` exact coherent one-phonon; modes 1/2 need
the phonopy input), the **temperature**, the optional **lattice** (needed only
for the mode-0 coherent-elastic line), and the per-element table: symbol,
σ_bound, awr, b_coh, σ_inc; plus, in DOS-files mode, the multiplicity, DOS
file and frequency-unit columns, and, whenever a mode-0 elastic line is
requested, the fractional **positions** column (flat `x y z x y z …`, as in an
ENDF deck).

Typing a symbol (`Be`, `O`, …) and leaving the cell autofills the
still-empty nuclear columns from IRMA's built-in table
(`irma.core.nuclear_data`, the Rauch–Waschkowski/Sears compilation);
anything you typed yourself is never overwritten, and typing a new symbol
over a machine-filled row (including the default carbon row) refreshes the
prefilled constants. Nuclides with energy-dependent scattering lengths
(B, Cd, Gd, …) are left blank on purpose: enter constants appropriate for
your energy range by hand. **Auto-fill elements from phonopy.yaml** draws
from the same table, and the NCrystal-plugin tab's element table behaves
identically. Isotope labels (`C-13`, `D`, …) autofill too, but keep them
for the DOS-files mode where the symbol is a free label: phonopy-backed
modes and the NCrystal export match rows to the structure's chemical
species by exact symbol, so an enriched material there needs the bare
element symbol with hand-entered isotope constants.

### Physics, Grid, and the elastic line

**Physics** holds the engine controls: **max phonon order** (`auto`
recommended), and for modes 1/2 the powder-average **directions**,
**multiphonon dirs**, and worker **jobs**, plus the **elastic line** switch,
**elastic kind** (`both`, `coherent`, `incoherent`), and two toggles:
**include energy-gain side** (anti-Stokes by detailed balance, on by default)
and **kinematic kf/ki** (multiply by the flux factor to get the
double-differential cross section, off by default). The elastic line is
computed tape-free from the same Debye-Waller factors as the inelastic part.
**Grid** sets the energy window (**E min/E max/dE**) and the powder-average Q
resolution (**dQ**), with an optional **Q max** cap.

### Instrument geometry and resolution

Two sub-tabs select the geometry:

- **Indirect (VISION defaults)** — fixed final energy **Ef**; the VISION
  preset fills Ef = 3.5 meV and the 45°/135° banks, or enter your own
  **angles**.
- **Direct** — fixed incident energy **Ei** plus an **output** selector:
  *fixed cuts* (**cut by** detector **angles** or **constant-Q** values, with
  an optional **cut dQ** band average) or a *2-D S(Q,E) map* masked to the
  instrument's **detector coverage 2θ** arch.

![Direct sub-tab set up for an ARCS 2-D map: Ei = 300 meV, the automatic chopper resolution (ARCS-700-1.5-AST at 600 Hz), the pre-filled detector coverage, and the mask toggle](assets/gui/gui_ns_direct_map_chopper.png)

The **resolution model** is either a width polynomial (three labelled
coefficients, `sigma(E) = c0 + c1·|E| + c2·E²` (meV), mirrored on the
energy-gain side) or the automatic
**chopper** model: pick the **instrument** (ARCS, SEQUOIA, MAPS, MARI, MERLIN,
HYSPEC, CNCS, LET), the **chopper package**, and the **frequency**, and the
energy-dependent width is computed for you (validated against Mantid PyChop to
~1 %). **resolution shape** chooses Gaussian or Lorentzian.

!!! note "PyChop attribution"
    The chopper model is an independent BSD reimplementation written from the
    published resolution literature; no PyChop (GPL) source is included.
    PyChop serves as the black-box validation reference and as the collected
    source of the factual instrument parameters; see
    `THIRD_PARTY_NOTICES.md` in the repository for the full provenance
    statement.

![Indirect (VISION) geometry with the width-polynomial resolution: the three labelled c0/c1/c2 coefficient fields — leaving all three blank uses the VISION polynomial](assets/gui/gui_ns_indirect_resolution.png)

### Running and plotting

**Run** streams progress into the **Log** pane and enables **Plot** when done:
spectra plot per-bank/per-cut curves (linear or log y-axis), maps render with
the kinematic envelope and a **mask to accessible (q,E)** toggle, and **Save
map...** exports the 2-D map to CSV/npz. The spectrum itself is written to the
**output** path (CSV, npz, or JSON by extension); the **inelastic / elastic
breakdown** toggle next to it keeps each cut's component columns in the saved
file and overlays them on the 1-D plot.

![A finished VISION run in the Plot pane: the 45° and 135° bank spectra of graphite](assets/gui/gui_ns_plot_vision.png)

![A finished direct-geometry run: the ARCS S(Q,E) map on a log scale, masked to the accessible (Q,E) arch, with the kinematic envelope overlaid](assets/gui/gui_ns_plot_map.png)

---

## MLIP phonon models tab

When you have no phonon calculation at all, this tab builds one from a
structure file: pick the file and a potential, optionally set the supercell
and mesh, and click **Build bundle**. The form mirrors `irma mlip build`
field for field (every control has the ⓘ help glyph), the **Log** pane
streams the relaxation and force-evaluation progress, and the **DOS** pane
plots the bundle's quick-look DOS after a successful build (or on demand
via **Show DOS**). The sections below the build form operate on a finished
bundle, which is auto-filled after a build: **Validate** re-runs the bundle
checks, and **Generate IRMA inputs** is `irma mlip emit`: it writes the
ENDF deck, spectra YAML, or NCrystal exporter YAML for the selected
targets, ready for the other tabs. The **Potential environments** section
creates and removes the per-potential environments that isolate
conflicting package pins; it is only needed when a potential's packages
are not importable from the running environment. See
[MLIP phonon models](mlip.md) for the potential roster, the
conservative-force rule, and the disordered path, and
[MLIP examples](mlip-examples.md) for validated end-to-end builds.

![The MLIP phonon models tab staged for the graphite build of the examples page: nequip, a 6×6×1 supercell, a 40³ mesh, snap-to-symmetry, and 9 workers](assets/gui/gui_mlip_build.png)

---

## End-to-end example: graphite, iel=10, inelastic_mode=0

A minimal pass down the form for a crystalline-graphite evaluation:

1. **Material** — keep `iel = 10`, accept the hexagonal lattice defaults and the four-carbon atom line, leave `inelastic_mode = 0`.
2. **Scattering** — keep `ZA = 6012`, `AWR = 11.898`, `sigma_free = 4.739180` (the free-atom value; IRMA derives the bound normalization itself), `MAT = 28`.
3. **Grids** — enter `296.0`, leave LAT = 1, keep the automatic grid (click **Detect from DOS** to set Max phonon freq), and **Preview Grid Sizes**.
4. **Phonon** — point **From phonopy total_dos.dat** at your own phonopy `total_dos.dat` for graphite (two columns: frequency in THz and DOS, as written by `phonopy -p mesh.conf`; IRMA converts THz→eV on load).
5. **Run** — choose an output `.endf` path and click **Run Calculation**.

For reference, IRMA's mode-0 graphite result with the ENDF/B phonon spectrum overlays the ENDF/B-VIII.1 crystalline-graphite inelastic cross section:

![Graphite mode-0 inelastic cross section](assets/validation/graphite/graphite_mode0_xs.png)

and the linear-in-Q automatic alpha grid recovers the thermal inelastic cross section at a fraction of the column budget of a brute-force reference:

![Graphite automatic-grid inelastic cross section](assets/validation/graphite/fig_grid_autogrid.png)
