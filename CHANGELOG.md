# Changelog

Notable changes to IRMA. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

This release comes out of a full review of the code. Most of the changes
are internal and leave every output as it was: the `irma` package is about
9,200 lines shorter, and after each step the outputs of 24 reference cases
were compared with those of the code before the review. The classic LEAPR
kernels still reproduce the NJOY2016 reference tapes byte for byte in MF7.

Outputs that change, and what to regenerate:

- ENDF tapes: the MF1/MT451 record of every tape (EMAX, the library version
  and release, and the default header lines). MF7 changes only in the cases
  listed under *ENDF evaluations*. Regenerate a tape to get the new header.
- Modes 1 and 2: the MT4 effective temperature, by about 3e-5 relative.
- ENDFTSL packs: reconvert them. The graphite inelastic cross section drops
  4.2% at 25 meV, and packs grow by up to 19 times.
- NCrystal exports: the NCMAT Debye temperatures rise (graphite 827 K to
  923 K at 296 K), and with them the Bragg planes the plugin uses. The
  packs are unchanged. Re-export to update them.
- MLIP bundles: the DOS is a histogram of the phonon modes. Re-emit inputs
  from existing bundles; rebuild a bundle to refresh its own DOS file.
- Spectra: the bins near the ends of the energy axis change, and so do the
  chopper width on the energy-gain side, the kf/ki factor on 2-D maps,
  mode-0 runs whose DOS comes from phonopy, and runs with a minimum phonon
  energy.

### Security

- The phonopy.yaml guard refuses every YAML tag and `%TAG` directive.
  Percent-escaped tags (`!!%70ython/...`), verbatim tags
  (`!<tag:yaml.org%2C2002:python/...>`), a `%TAG` line after a UTF-8
  byte-order mark, and UTF-16 files used to pass the guard and reach
  phonopy's unsafe YAML loader, which runs the code such a tag names. The
  error quotes the refused line. phonopy writes none of these forms, and
  every phonopy.yaml in the repository and in the reference bundles still
  loads.

### ENDF evaluations

- MF1/MT451: EMAX is the MF7/MT4 upper energy B(4), as NJOY2016.79 writes
  it, instead of 0.0. NVER and LREL (library version and release: 8 and 1
  for ENDF/B-VIII.1) are optional trailing fields on Card 4, after `iint`,
  with defaults 8 and 1, where they were fixed at 6 and 0. When comment
  cards 3–5 are blank, the writer fills in the standard ENDF/B thermal
  header lines (`----ENDF/B-VIII.1     MATERIAL  mat`,
  `-----THERMAL NEUTRON SCATTERING DATA`, `------ENDF-6 FORMAT`); a deck
  that gives its own lines keeps them.
- The three `examples/tsl` decks carry an IRMA example header on comment
  card 1, blank cards 2–5, and their description from card 6. Two of them
  started their free text on card 1, where it filled the structured header
  fields, and `graphite_iel10_classic.input` carried the NCSU
  ENDF/B-VIII.0 header and description word for word.
- `examples/tsl/graphite_mode2.input`, the production mode-2 graphite deck,
  uses `iint=1` (lin-lin) on grids from the automatic generators: 399 α ×
  541 β, was 200 × 426. The tape grows from 2.7 MB to 6.2 MB.
- Modes 1 and 2: the per-atom DOS behind the MT4 effective temperature is a
  histogram of the mesh modes, each weighted by |e|², instead of a Gaussian
  as wide as two grid steps, which put weight below the lowest mode. The
  effective temperature moves by about 3e-5 relative (fast graphite test:
  707.2952 K to 707.2741 K).
- A principal split over several Card 6d rows is merged in every mode. The
  merge ran only for inelastic modes 1 and 2, so an `iel=10` deck in mode 0
  that listed the principal on two rows counted its coherent comb twice.
  One-row decks are unchanged.
- Directional Debye-Waller tensors are rotated into the frame of the Bragg
  comb (a along x, b in the xy-plane) when the phonopy lattice is oriented
  differently; they were applied in the model's own Cartesian frame, on the
  wrong axes. For ENDF MT2 the rotation applies only when Card 6c describes
  the phonopy cell (lengths within 1e-3 relative, angles within 0.01°);
  otherwise a NOTE line says the tensors are used as given, because Card 6c
  may describe another cell. Models already in the standard orientation,
  which includes every shipped example, are unchanged.
- The short-collision-time tail in `sint` divides by the square root of
  4π·wt·α·T̄, so the Gaussian integrates to 1; NJOY2016 leaves out the
  square root (leapr.f90:1892). docs/njoy.md lists this as a deliberate
  divergence. Only the discrete-oscillator and cold-hydrogen kernels reach
  the tail, past the tabulated β range, and the golden decks and NJOY
  minitapes stay byte-identical.
- The multiphonon energy-coverage warning for modes 1 and 2 now compares
  the reach of the truncated sum, the order times the highest phonon
  energy, with the recoil ridge at the largest Q plus six thermal widths,
  instead of with the top of the energy grid. The old comparison fired for
  heavy atoms at orders that already converge the sum, where the grid top
  lies far out on a negligible tail. Any order that meets the automatic
  order rule passes the new check, so it fires only for a lower order; the
  metadata gains `needed_multiphonon_beta_support`. The check on the
  computed S(α,β) array uses the same needed reach: it warns only
  when the array goes to zero before it, instead of whenever it goes to
  zero below the grid top.
- Modes 1 and 2 print the lowest kept mesh-mode energy and how many output
  energies lie below it, since the one-phonon term is zero there;
  docs/modes.md and docs/grids.md explain this.
- Force constants embedded in phonopy.yaml take precedence over an explicit
  FORCE_CONSTANTS or FORCE_SETS file (phonopy's rule). Every phonopy path
  now prints a warning naming the unused file. The docs had said the
  explicit file wins.
- The deck reader refuses:
    - a numeric card with an empty comma field (`,,` or a leading comma).
      NJOY keeps that item at its default, while IRMA dropped it and moved
      every later value one field left.
    - Card 10 temperatures that do not increase. THERMR reads the MT4
      temperatures in order, and the coherent-elastic edge thinning uses the
      first one, so a descending list gave wrong data above the first
      temperature.
    - a non-ASCII character in a comment card, which shifted the fixed
      MAT/MF/MT columns of its MF1 record. Tapes are written as ASCII.
    - a minimum phonon energy that removes every mode, such as a Card 6g
      missing its `mpdir` field (`200 /`), which reads as a 200 meV cutoff
      card.
- Run metadata: `one_phonon_energy_jacobian_meV_per_THz` is renamed
  `one_phonon_meV_per_THz` (the value is a unit factor, not a Jacobian),
  and `coherent_diagonal_term` says whether the `coherent_diagonal` arrays
  hold the per-site self term or the principal group's |F_p|².

### Neutron-scattering spectra

- Resolution broadening works on a padded grid. S is computed past both
  ends of the energy axis by the kernel's reach (6σ for a Gaussian; for a
  Lorentzian, until 1e-3 of its area lies outside), convolved and then
  cropped, and each kernel column and the elastic line are normalized over
  the whole uniform grid instead of over the window. Each line shape used
  to be rescaled to sum to 1 inside the window, so intensity that belonged
  past an axis end was pushed back in: at `e_min = 0` the full elastic line
  stayed on the axis, and the last bins below `e_max` were scaled up. The
  1-D spectra and the 2-D map now follow one convention. The E = 0 elastic
  value halves at `e_min = 0`, bins within about 15 meV of `e_max` change
  by up to 40%, and interior bins change by a median of 4e-7. A spectrum
  no longer depends on where `e_max` lies (2e-7; it was 23% at the last
  bin).
- Chopper resolution evaluates the energy-gain side at its own final
  energy. Every gain-side transfer had been given the elastic width. For
  ARCS-100-1.5-AST at 300 Hz and Ei = 30 meV the FWHM is 1.04 meV at E = 0
  and 2.24 meV at E = −30 meV. The loss side is unchanged.
- `physics.kinematic_kf_ki` applies to 2-D maps too, per energy column
  before the elastic line and the resolution pass, and the map metadata
  records it. Before, it weighted only the 1-D spectra.
- The minimum phonon energy reaches the cached engine context. The context
  was built with a cutoff of 0, so its mode data ignored the cutoff: for
  graphite in mode 1 with a 30 meV cutoff, 20% of the intensity stayed
  below 30 meV.
- The mode-0 phonopy bridge uses the per-atom DOS histogram described under
  *ENDF evaluations*. The Gaussian biased the Debye-Waller λ upward: on the
  graphite fixture's 12³ mesh at 300 K, λ goes from 1.153 to 0.891,
  against 0.894 from the exact mode sum.
  `examples/spectra/data/graphite_C_dos.txt` is regenerated.
- The CNCS and LET chopper packages are named High-Flux, after the PyChop
  settings their widths come from; they were Standard (CNCS) and
  High-Resolution (LET). The widths are unchanged. There are no aliases, so
  configs with the old names need editing.
- `start:stop:step` angle ranges end at `stop` (`0:1:0.6` gave 0, 0.6,
  1.2), and a step whose sign points away from `stop` is an error instead
  of an empty list. The GUI uses the same parser.
- Config validation catches five mistakes before the engine runs: a
  mode-1/2 species other than C without `b_coh_fm` and `sigma_inc_b` (the
  engine has built-in values only for carbon); numbers read as strings,
  since YAML 1.1 loads `2e2` as a string; a chopper setting that does not
  transmit Ei, a frequency above the instrument maximum, or an unknown
  `chopper_spec` key; angles other than the vision preset's 45° and 135°
  banks, which the 1-D path ignored and the map used; and a mode-0 species
  DOS that does not start at ω = 0.
- CLI: `--set KEY=VALUE` works with `vision`, `indirect` and `direct` and
  replaces six undocumented flags (`--force-constants`, `--force-sets`,
  `--elastic-from-tape`, `--bank-halfwidth`, `--combine`, `--components`)
  and the unused flag-form `--q-max`. Config errors in the flag form exit
  with status 2, as in the config form; they exited with 3.

### NCrystal export and plugins

- The NCMAT `debye_temp` placeholder reproduces the engine's mean-squared
  displacement in NCrystal's full Debye model, found by bisection. It used
  the high-temperature limit, which drops the zero-point term. At 296 K the
  reference exports move from 827 K to 923 K (graphite C), 1107 K to
  1351 K (BeO Be) and 959 K to 1113 K (BeO O); the packs are unchanged.
  The value matters in both modes: with `elastic: false` NCrystal's own
  elastic uses it, and with `elastic: true` NCrystal's |F|² cutoff,
  computed from it, chooses the Bragg planes whose structure factors the
  plugin computes from the pack's tensors. Coherent elastic through the
  plugin changes by up to 0.4% on graphite and 5e-5 on BeO.
- The exporter uses `material.force_constants` and `material.force_sets`.
  It accepted both keys and then read the files next to phonopy.yaml.
  Embedded force constants still win, with a warning, and the pack
  provenance records the name and SHA-256 of the file actually used
  (`force_constants_name`, `force_constants_sha256`).
- The exporter refuses a cell with |cos α − cos β cos γ| > 1e-9, such as
  an FCC or BCC primitive cell. NCrystal 4.4.6's general lattice branch
  (NCLatticeUtils.cc:75-78) builds a wrong reciprocal lattice for these
  cells and misplaces every Bragg plane. Use the conventional cell; the
  check can go once NCrystal is fixed.
- The −1% negative-cell error names the worst cell, says the direction
  sampling is too coarse, and points to `export.num_directions`.
- ENDFTSL converter: NCrystal interpolates S linearly in β, so every
  log-linear (INT=4) β interval of a tape is subdivided into
  n = ceil(max|ln(S_{j+1}/S_j)| / sqrt(8·tol)) parts, with tol = 1e-3. On
  the graphite example the inelastic cross section drops 4.2% at 25.3 meV
  (2.2% at 1 meV and at 0.1 eV), and BeO and Al2O3 drop 0.2–0.3%.
  From 1 meV to 0.5 eV graphite now agrees with a direct integration of
  the tape's own law to 0.04%, where it was 0.5–5.5% high. Packs grow 4.5
  to 19 times (graphite 2.0 MB to 37 MB, PMMA at 20 K 7.8 MB to 65 MB),
  and loading the graphite pack takes 7.3 s instead of 1.3 s. Lin-lin
  (INT=2) intervals are left as they are; see *Known limitations*.
- ENDFTSL converter: S values go into the pack straight from the stored
  table. The exp(+β/2)·exp(−β/2) round trip overflowed for LAT=1 tapes
  below about 41 K (PMMA at 20 K failed with OverflowError) and moved about
  a third of the values by 1 ulp.
- ENDFTSL converter: a sole coherent carrier (a hydride whose metal alone
  scatters coherently, or an IRMA SEF tape) is scaled by its atom
  fraction. The `per_atom` choice overstated its Bragg edges by 1/f, and
  the default refused to convert. The `coherent_convention` key is
  removed; a config that sets it is an error.
- ENDFTSL converter: LASYM=1 tapes (LEAPR writes them for cold H2 and D2)
  are refused; they converted into packs NCrystal could not load. A
  single-tape conversion needs `--symbol`, `--mass` and `--density`: the
  old defaults (C, 12.0107, 1 g/cm³) described any tape as carbon.
- Both plugins build with `BUILD_WITH_INSTALL_RPATH`, so installing copies
  the library unchanged. On macOS the install step deleted the build-tree
  RPATH with `install_name_tool`, which invalidates the linker's code
  signature unless that tool re-signs the library (conda's does, Apple's
  does not). macOS then killed any process that loaded the plugin, with
  exit 137 at the first `createScatter`; a pip build with CMake 4.4 on
  arm64 did this. Rebuild a plugin that fails this way.
- Plugin versions: `ncrystal_plugin_ENDFTSL` is 0.2.0 in `__version__`,
  in CMake and in the pack's converter stamp (they said 0.0.1, while
  pyproject.toml said 0.2.0), and a test checks that they agree. The CMake
  project of `ncrystal_plugin_IRMA` is 0.6.0, as in its pyproject.toml.

### MLIP front end

- User-trained MACE checkpoints. `--potential mace` or `mace-off` with
  `--model <file>` (the GUI's model field gains a Browse... button) loads
  the checkpoint in the parent process and pins its absolute path into the
  calculator specification that every displacement worker and force server
  rebuilds from. The manifest records the file's SHA-256 with the model
  class, cutoff, interaction layers, element table and head. If the
  checkpoint does not cover every element in the structure, the build stops
  before relaxation and names the missing elements; the same check applies
  to the named MACE models. Multi-head checkpoints are refused, and float32
  weights are evaluated in float64. The force-cache fingerprint version
  is now 2, so the scratch caches of interrupted builds from earlier
  versions are recomputed.
- The bundle's total DOS and the species DOS in emitted inputs are a
  histogram of the phonon modes, the same one the engine uses. A flat band
  counts in full, and the Gamma translations and imaginary modes are left
  out. phonopy's tetrahedron DOS gave a flat band zero width, so it fell
  back to a 1 meV Gaussian, and a Gamma-only disordered DOS kept the three
  translations as a smeared peak at 0. On the reference bundles the
  Debye-Waller λ of the disordered box goes from 146.4 to 3.75, and that of
  the crystal rises by 2%. All DOS files share one grid: 0 to the highest
  mode, at least 200 points, spacing at most 0.5 meV. `--dos-smearing` is
  removed.
- The THz-to-meV factor comes from `irma.core.constants`. The bundle code
  used an older CODATA value, 5e-7 lower, so bundle DOS energies, census
  frequencies and emitted spectra move by 5e-7 relative.
- `irma mlip emit` refuses `--min-phonon-energy` for the mode-0 deck and
  for disordered bundles, whose files do not apply it. The value had been
  dropped without a message.
- `irma mlip emit` prints the phonon mesh the emitted inputs use
  (int(98/a)+1 points per axis) next to the bundle's `--mesh`, which sets
  only the quick-look DOS and census mesh.
- SevenNet: the fidelity (`modal="mpa"`) is passed whenever the model is
  7net-mf-ompa or 7net-omni. Naming either one with `--model` failed with
  "modal argument missing". The manifest records the modal.
- `irma mlip build --born` parses the BORN file with phonopy's own parser,
  so files with phonopy's `# epsilon and Z* of atoms` header (written by
  phonopy-vasp-born and phonopy-qe-born) are accepted. The old count came
  out one row short and refused them.
- A pinned pet-mad checkpoint found in the cache is checked against its
  recorded SHA-256, a check the cache path had skipped. The force server
  computes stress only when asked; for a checkpoint without stress, ASE
  reran the whole model on every call. `irma mlip build` rejects a
  non-positive or non-finite `--delta`, `--fmax` or `--snap-symmetry` and a
  negative `--nmax` before any work.
- The manifest's `argv` records the command as given instead of the
  constant `irma mlip`, and a KeyError or TypeError message names its
  type.

### GUI

- ENDF form: NVER, LREL and three HSUB fields (comment cards 3–5) with
  help, in the MF1 section.
- ENDF form: a phonopy `total_dos.dat` is resampled onto the zero-anchored
  grid with the file's spacing. phonopy's default range starts below 0, and
  the reader kept the file's own points, which shifted the whole DOS down
  by that offset. A `--fmin=0` file is used as it is.
- Apply ZA moves an imported Card 6e spectrum with the atom row it
  relabels. Otherwise the engine refused the deck over a card the form
  does not show.
- Run and Save with a principal ZA that matches no atom row show the
  engine's message, which names Apply ZA, instead of offering to relabel
  the row.
- The atom block refuses extra positions and non-integral Z, A or npos,
  which were dropped or truncated; `npr` must be an integer; a non-numeric
  bins-per-decade value is an error instead of turning grouping off.
- Error dialogs show the whole message, and the runner keeps the last 4
  output lines. It kept 12 and then cut them to the first 500 characters,
  which dropped the line with the error.
- MLIP panel: the minimum phonon energy is passed for every emit target,
  not only for endf.
- Neutron-scattering panel: the 2-D map is computed unmasked and masked
  only for viewing and export, so with the mask toggle off the cells
  outside the kinematic envelope are shown. On the Direct tab the mask
  checkbox starts unticked. It started ticked, so a fresh map with the
  tab's default width-polynomial resolution was masked to a detector-angle
  band the tab does not show.
- Neutron-scattering panel: Open keeps the config fields the panel has no
  control for (`elastic_from_tape`, `q_pad_invA`, `bank_halfwidth_deg`,
  `combine`), lists them in the log, and writes them back. Running or
  saving had reset them to their defaults. A map config for indirect
  or vision geometry is refused on Open with a pointer to
  `irma spectra map`, since the panel runs maps for direct geometry only.

### Removed

- `irma mlip build --dos-smearing`.
- Spectra: the six flag-form options that `--set` replaces and the
  flag-form `--q-max` (see above), and the `IRMA_JOBS` and `IRMA_NCPU`
  environment overrides.
- NCrystal export: `export.site_groups`, which could only reorder the
  default one-group-per-species partition. A config that sets it fails
  with "unknown export key".
- ENDFTSL converter: `coherent_convention` (see above).
- Python API: `irma.core.engine` re-exports only `run_leapr`,
  `LeaprResult`, `DeckError`, `parse_leapr_input` and `TokenReader`.
  `irma.ncrystal` exports only `NCrystalExportConfig`, `build_packs`,
  `write_packs`, `IRMAPack`, `read_pack` and `write_pack`, and the
  `ncrystal_plugin_IRMA.pack` re-export of the last three is gone.
  `irma.spectra` no longer has `from_irma_cache`, `gaussian_resolution`,
  `Q_fit` or `bank_bragg_lines`. The noncubic engine's Gaussian line
  deposition (`sigma_mev`, and `--sigma-mev` on its diagnostic CLI) is
  gone; production runs always bin the lines.

### Internal

- Code: duplicated paths are merged and dead branches and unused options
  removed in the engine, the ENDF writer, the spectra model, the NCrystal
  exporter, the MLIP front end and the GUI, with no output change. The
  engine and the GUI share one set of Card 6b–6g readers.
- Tests: duplicates, tests of removed code, and tests that pinned wording
  or history are removed, and the remaining docstrings state what each
  test checks. The CrysXT extinction gate tightens from 5e-3 to 1e-10, and
  a test pins the Skold correction against an NJOY2016 tape.
- Documentation: the pages were checked against the code and corrected
  where they had drifted.

### Known limitations

- NCrystal evaluates S log-linearly in α (NCSABEval.cc instantiates only
  that form), while a lin-lin (INT=2) tape is linear in α. Through the
  ENDFTSL plugin, the inelastic cross section of such a tape falls below
  the tape's own law at low energy: on the mode-2 graphite example at
  296 K it is low by 4.6%, 8.0%, 10.2%, 4.5%, 2.7% and 0.9% at 1 meV,
  3 meV, 9.6 meV, 25.3 meV, 0.1 eV and 0.5 eV. THERMR processing of the
  same tape agrees with the law to 0.2%. The IRMA plugin passes its
  S(α,β) table through the same NCrystal evaluation, and this may account
  for part of the low-energy gap described in docs/ncrystal-plugin.md.
  Refining the α axis in the ENDFTSL converter, and checking the IRMA
  exporter's α spacing, may be necessary after this release. A prototype
  on the graphite tape brings the worst deficit to 0.9% with 2.6 times the
  α points (pack 4.3 MB to 11 MB) and to 0.33% with 4.8 times (20 MB).

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
