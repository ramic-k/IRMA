# Changelog

Notable changes to IRMA. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

- User-trained MACE checkpoints in the MLIP front end. `--potential mace`
  or `mace-off` with `--model <file>` (the GUI's model field gains a
  Browse... button) loads the checkpoint once, from bytes verified
  against a digest pinned in the calculator specification; displacement
  workers and force servers verify the same digest before loading, and
  the manifest records it with the model class, cutoff, interaction
  layers, element table, stored dtype and head. The calculator refuses
  force calls on elements the checkpoint does not cover, for named MACE
  models too, and the build stops before relaxation with the uncovered
  elements named; multi-head checkpoints are refused; float32 weights
  are evaluated in float64 with a note in the log and the manifest. The
  force-cache fingerprint version is now 2, so the scratch caches of
  interrupted builds from earlier versions are recomputed.

## [1.0.3] — 2026-09-16

- Minimum phonon energy for the phonopy-backed modes 1 and 2. An optional
  one-value card before Card 6g (also `physics.min_phonon_energy_meV` in
  spectra configurations and `--min-phonon-energy` on the spectra CLI,
  `export.min_phonon_energy_meV` for the NCrystal exporter, and
  `irma mlip emit --min-phonon-energy`, with a field on each GUI form)
  removes every phonon mode with energy at or below the value from every
  term: the DOS, the Debye-Waller tensors, the coherent and incoherent
  one-phonon scattering, and the multiphonon expansion. Nothing replaces
  the removed modes. Blank or 0 keeps the automatic floors, which already
  exclude imaginary modes, so existing inputs are unchanged. The run log
  and the metadata (`min_phonon_energy_meV`, `phonon_cutoff`) report the
  removed weight and the change in the mean-square displacement, with a
  warning above 1%; NCrystal packs record the value in their provenance.


## [1.0.2] — 2026-09-16

Lin-lin (`iint=1`) grids, the multiphonon work grid, and the GUI's
principal-scatterer handling. Log-lin (`iint=0`) grids are byte-identical
to 1.0.1; evaluations on uniform output grids are byte-identical; a
log-lin evaluation on an automatic grid whose phonon region is not 300
steps gets that region's own step as its multiphonon work spacing (third
item below).

- The lin-lin beta grid keeps its 0.5 step out to a margin past the
  back-scatter alpha at the highest incident energy
  (`linlin_fine_beta_limit`: `RIDGE_MARGIN_SIGMAS` widths of the
  down-scattering kernel, with an upper bound on T_eff from `freq_max`
  and the hottest temperature in the deck) instead of stopping at that
  alpha. Stopping there left populated cells on the coarse log tail, and
  lin-lin interpolation across them raised the graphite total cross
  section between 2 eV and the requested energy (0.8% for a 5 eV grid,
  2.6% for a 10 eV grid, from the processed PENDF), while the same tape
  processed log-lin stayed flat; with the margin the rise is 0.09% (5 eV)
  and 0.10% (10 eV), the cost of the 0.5 step itself (calibration table
  in docs/grids.md). The wider fine region costs beta points (541 to 5 eV
  against 395 log-lin; 710 to 10 eV); the deck emitters, the GUI grid
  preview, and the NCrystal exporter print a line built from what was
  generated: the count, the log-lin count, the stored cap, and the energy
  the fine step reaches.
- The lin-lin step cap is scaled by the deck's lowest temperature
  (`evaluation_temperatures_K`) because a `lat=1` grid is stored in
  0.0253 eV units: a 0.5 stored step evaluated at 77 K was a 1.9 physical
  step. A seam node that would print as a duplicate at the deck's six
  decimals is dropped.
- The multiphonon work spacing for a non-uniform output grid is the output
  grid's own phonon-region step (the first run of at least ten equal
  spacings below the highest phonon energy), so the deck's phonon
  subdivision sets the multiphonon resolution and tail points cannot
  change it. The previous median rule coarsened from 0.67 meV to 12.7 meV
  when a graphite beta grid gained 235 tail points, which moved the cross
  section by 0.3% over 0.5 to 2 eV; it remains only for grids with no
  such run.
- GUI: choosing inelastic_mode 2 selects `iint=1` (lin-lin), the form the
  coherent one-phonon law needs; modes 0 and 1 select log-lin. A deck
  import keeps the deck's own `iint`.
- GUI: **Apply ZA** (the former "Fill AWR + sigma_free from ZA") also
  relabels the atom row of the ZA's element to the same nuclide, taking
  `A`, `AWR`, `b_coh` and `sigma_inc` from the one table entry and keeping
  the positions, and reports the change on a line under the button. Until
  now "Fill structure from phonopy.yaml" left every row as the natural
  element and a principal ZA naming an isotope (6012) was refused by the
  engine, with nothing in the GUI to make the two agree. Only the button
  changes a row; a row carrying constants other than the table's is replaced
  only after a dialog, other elements and rows sharing the element are
  never touched automatically, and nuclides without usable constants are
  refused before any change. Run and Save make the same offer once more
  when the deck is still inconsistent.
- The principal-scatterer membership rule and its message now live in one
  place (`irma.core.crystal_input.principal_mismatch_message`), used by
  the engine, the deck validator, and the GUI; the engine's error names
  the row it found and both ways to fix the deck.

## [1.0.1] — 2026-08-20

Hardening of MLIP environment provisioning, from the first field
reports. No physics or engine changes: evaluations are byte-identical
to 1.0.0.

- `irma mlip env create` now runs a torch/NumPy interop probe after
  the import check, retries once with `numpy<2` when a torch wheel
  built against NumPy 1.x is detected (Intel Macs: torch wheels
  stopped at 2.2.2), and refuses to register an environment that
  `pip check` reports inconsistent afterward.
- Shared environments (mace/mace-off) import-check every sibling
  before registering any of them.
- Install failures keep the part of pip's output that names the
  irreconcilable requirements; dispatch errors carry a one-sentence
  hint for the known failure signatures.
- With uv on PATH, potential environments are pinned to Python 3.12;
  potential packages lag new interpreters. Intel Macs get an up-front
  note about the platform.
- The nequip-compile bootstrap guards a torch.export crash on
  bleeding-edge torch (mixed cuDNN TF32 flags); verified against
  torch 2.13 (CUDA build).
- Windows: the provisioned environment's interpreter path now points
  at `Scripts\python.exe`.
- GUI help and manual updated to match, including the platform
  support note.

## [1.0.0] — 2026-08-03

Initial public release. IRMA turns one phonon calculation into three
outputs on one engine:

- **ENDF-6 File 7 evaluations** from LEAPR-style input files: the classic
  kernels (validated bit-for-bit against NJOY2016 reference tapes),
  generalized coherent elastic for arbitrary crystals (`iel=10`,
  SEF/MEF output conventions, Bragg-edge grouping, opt-in
  crystalline extinction), and the phonopy-backed noncubic inelastic
  modes with the exact coherent one-phonon term, anisotropic
  Debye-Waller tensors, a per-species partition for polyatomic
  materials, and automatic alpha/beta grids.
- **Instrument-resolved neutron spectra** (`irma spectra`):
  indirect- and direct-geometry spectrometers, chopper resolution,
  2-D S(Q,E) maps, and a DOS-driven mode 0.
- **NCrystal transport exports** (`irma ncrystal`) sampled by the
  companion `ncrystal_plugin_IRMA`, plus the standalone
  `ncrystal_plugin_ENDFTSL` plugin for sampling ENDF TSL files
  directly in NCrystal.

The **MLIP phonon front end** (`irma mlip`) builds the phonon calculation
itself from a bare crystal structure with one of nine pretrained
machine-learned interatomic potentials, and emits ready-to-edit
inputs for all three outputs (`--inelastic-mode 0|1|2`, mixed
elastic format by default).

A GUI (`python -m irma --gui`) drives every path. `examples/` maps
sixteen runnable examples to their calculation types; `skills/irma/`
ships an opt-in assistant skill for AI coding agents; the validation
record under `docs/validation/` documents the evidence behind the
physics (NJOY reference tapes, Euphonic and OCLIMAX cross-code
comparisons, and measured VISION, ARCS, and VENUS data).

The development history preceding this release is internal to ORNL.
