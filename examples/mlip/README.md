# MLIP front-end examples

From a bare crystal structure to all three IRMA outputs, with a
pretrained machine-learned interatomic potential standing in for the
first-principles calculation:

```bash
irma mlip build <structure> -o <bundle> --potential <name>
irma mlip emit <bundle> --to endf,spectra,ncrystal --mat 'SYM=MAT' ...
```

The example structure is `MgO.cif`, a conventional rocksalt MgO cell
(8 atoms, all sites listed explicitly, so any CIF reader accepts it).
Rocksalt symmetry reduces it to two displacements, which makes this
one of the fastest possible builds. Full documentation: the manual's
*MLIP phonon models* page.

## 1. One-time setup: build the potential's environment

Potentials are not installed with `pip install irma[mlip]`; each gets
a dedicated environment so their heavy dependency stacks never
conflict:

```bash
irma mlip env create nequip
irma mlip env list
```

This downloads the potential's dependency stack and, on first build,
its checkpoint (hundreds of MB; both are one-time costs). `nequip` is
the potential our validation ranks best all-around; `--potential`
defaults to `mattersim` and eight other backends are available (see
the manual's potential table, including which checkpoints carry an
academic-only license).

## 2. Build the phonon-model bundle — `MgO.cif`

```bash
irma mlip build examples/mlip/MgO.cif -o mgo_bundle --potential nequip
irma mlip validate mgo_bundle
```

The build relaxes the structure with the same potential (watch the
spacegroup report), generates the symmetry-reduced displacements on
the default supercell (3x3x3 of the conventional cell here, from the
12 A lattice-parameter rule), computes one force evaluation per
displacement, and writes a self-contained bundle: `phonopy.yaml` with
embedded force constants, the relaxed structure, a quick-look DOS
with an imaginary-mode census, and a manifest recording the full
provenance (resolved checkpoint identity, versions, arguments).
Runtime is seconds to minutes on a laptop CPU. `validate` re-checks
the bundle end to end, including a real phonopy reload.

MgO is polar, so its longitudinal-transverse optical splitting needs
Born effective charges: pass a phonopy-format BORN file from DFPT
with `--born PATH` to embed the non-analytic correction. Without it
the bundle still builds and this example stays self-contained; the
optical branches near Gamma are simply unsplit.

## 3. Emit ready-to-edit IRMA inputs

```bash
irma mlip emit mgo_bundle --to endf,spectra,ncrystal \
    --mat 'Mg=44' --mat 'O=48'
```

- **endf** — one ready-to-run deck per principal scatterer (`iel=10`,
  `inelastic_mode=2`, automatic grids), validated by the same parser
  the GUI uses. Run each with `python -m irma <deck> <out.endf>`.
- **spectra** — an `irma spectra` YAML with the bundle's phonon model
  wired in (see `../spectra/` for what to do with it).
- **ncrystal** — an exporter YAML for `irma ncrystal`.

Scattering constants come from the built-in nuclear-data table; the
`--mat` numbers are the ENDF MAT identifiers you assign to the
evaluations. The decks default to the mixed elastic format (MEF) and
the full mode-2 physics; `--elastic-format sef` and
`--inelastic-mode 0` (the fast classic path built from the bundle's
species-projected DOS, with a Card 6e partial spectrum for every
non-principal species) select the other conventions. Nothing runs the downstream calculations for you: you
inspect the emitted inputs and drive `irma`, `irma spectra`, or
`irma ncrystal` yourself, exactly as in the other example families.

## In the GUI

`python -m irma --gui`, *MLIP phonon models* tab: pick the structure
file and the potential, **Build**, then emit the inputs from the same
tab. The manual's GUI page walks through it with screenshots.
