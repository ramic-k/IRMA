---
name: irma
description: Learn and drive IRMA, the thermal neutron scattering code. Use for running any IRMA calculation (ENDF/TSL evaluation from a deck, MLIP phonon-model bundles from a bare structure, instrument spectra, NCrystal transport exports), for choosing modes and options, for reading or writing LEAPR-style decks, for interpreting results and validation, and for physics questions about what IRMA computes (scattering laws, Debye-Waller treatments, coherent one-phonon, elastic formats), answered from the code and docs.
---

# IRMA

IRMA turns one phonon model into three outputs on one engine: an
ENDF-6 File 7 thermal scattering evaluation, instrument-resolved
neutron spectra, and per-temperature NCrystal transport kernels. The
phonon model comes from a phonopy calculation, a tabulated density of
states, or, with the MLIP front end, from a bare crystal structure and
a pretrained machine-learned interatomic potential.

This skill is for USING IRMA: running calculations, choosing options,
and understanding the physics it implements. The documentation under
`docs/` is authoritative; this file routes you to the right page,
example, and source anchor instead of duplicating them. When the docs
and this file disagree, trust the docs and say so.

## The four workflows

Classify the request as one of these before doing anything else.

1. **Deck -> ENDF evaluation** (`python -m irma <deck.input> <out.endf>`).
   Classic LEAPR decks run unchanged; extension cards select the
   generalized treatments (`iel=10` crystals, `inelastic_mode` 0/1/2,
   extinction, Bragg-edge grouping). Start from
   `examples/tsl/README.md` and the card-by-card
   `docs/input-reference.md`.
2. **Structure -> everything, via the MLIP front end** (`irma mlip`).
   One-time `irma mlip env create <potential>`, then
   `irma mlip build <structure> -o <bundle> --potential <name>`,
   `irma mlip validate <bundle>`, and
   `irma mlip emit <bundle> --to endf,spectra,ncrystal --mat 'SYM=N'`.
   Emitted ENDF decks default to the mixed elastic format (MEF) and
   mode-2 physics; `--elastic-format sef` and `--inelastic-mode 0|1|2`
   select the other conventions (mode 0 builds the classic
   DOS-driven deck from the bundle's species-projected DOS, with a
   Card 6e partial spectrum per non-principal species). Read
   `docs/mlip.md` (potentials table, licenses, model selection,
   what a build does) and `examples/mlip/README.md`.
3. **Spectra forward model** (`irma spectra run <config.yaml>`):
   instrument-resolved INS spectra and 2-D S(Q,E) maps for indirect-
   and direct-geometry spectrometers. `docs/spectra.md`,
   `docs/spectra-mode0.md`, `examples/spectra/README.md`.
4. **NCrystal export** (`irma ncrystal <config.yaml> -o <outdir>`): per-temperature
   material-data files plus a crystal-structure file for the companion
   `ncrystal_plugin_IRMA`, for Monte Carlo transport (McStas, OpenMC,
   ...). `docs/ncrystal-plugin.md`.

The GUI (`python -m irma --gui`) drives the same paths; `docs/gui.md`
has the tab-by-tab walkthrough with screenshots.

`examples/README.md` is a 16-row table mapping every calculation type
to a runnable example with its runtime; prefer starting a user from
the closest example over writing a deck from scratch.

## Driving calculations for the user

- Always show the exact command before running it, and show the run's
  key printed diagnostics after (grid sizes, Debye-Waller lambda,
  Bragg edge count, warnings). IRMA prints `WARNING:` at column 0;
  surface every one.
- Emitted MLIP inputs are STARTING POINTS by design: the banner in
  every emitted deck says to review the parameters. Walk the user
  through what to check (MAT numbers, temperature, sampling,
  isotope defaults printed by emit) before a production run.
- Do not silence the imaginary-mode gate. If a bundle records
  imaginary modes, the DOS-driven paths refuse; explain the fix
  (tighter `--fmax`, larger `--supercell`, different potential)
  before mentioning `--allow-unstable`.
- Respect the potentials' licenses: some checkpoints are academic-only
  (the table in `docs/mlip.md` marks them; the CLI prints a license
  note that travels in the bundle manifest).
- Long runs: a mode-2 evaluation for a small cell is minutes; direct
  runs of large cells or fine meshes can be hours. State the
  expectation before launching, and never run two engine calculations
  concurrently in one working directory.
- Validate outputs the way the project does: `irma mlip validate` for
  bundles; for tapes, the NJOY interoperability notes in
  `docs/njoy.md` (including the THERMR corrections a consumer needs).

## Learning IRMA

Route by topic: `docs/quickstart.md` (first run),
`docs/modes.md` (inelastic modes 0/1/2 and when each is right),
`docs/theory.md` (the thermal-scattering model IRMA implements),
`docs/grids.md` (automatic alpha/beta grids and convergence),
`docs/input-reference.md` (every card), `docs/mlip.md` +
`docs/mlip-examples.md` (front end), `docs/extinction.md`,
`docs/njoy.md`, `docs/troubleshooting.md`, and the validation record
under `docs/validation/` (methodology, graphite, beryllium,
beryllium-oxide): the evidence for what agrees with what, to which
number, and the comparison rules that keep such statements honest.

## Physics questions: answer from the code

For "what does IRMA actually compute" questions, do not answer from
memory. Read the implementing code and the matching docs section
first, cite what you read (module and function), and separate IRMA's
implemented conventions from textbook generalities. The conventions
that most often surprise people: only the downscatter side is
tabulated (detailed balance is applied downstream); `lat=1` grids are
expressed at 0.0253 eV; polyatomic evaluations are per principal
scatterer, normalized per atom; beta is positive for neutron energy
LOSS (the ENDF manual's sign convention is the opposite).

Routing table, concept -> where the physics lives:

| Concept | Code anchor | Docs |
|---|---|---|
| Classic phonon expansion, multiphonon orders | `irma/core/kernels.py` (`contin`, Poisson weights) | `docs/theory.md` |
| Translational / diffusion, discrete oscillators, cold H2, Skold | `irma/core/kernels.py` (`trans`, `discre`, `coldh`, `skold_approx`) | `docs/theory.md` |
| Displacement tensors, anisotropic Debye-Waller | `irma/core/phonopy_io.py` (thermal displacement matrices), `irma/core/crystal.py` | `docs/modes.md`, `docs/validation/methodology.md` |
| Isotropic vs per-species vs directional elastic DW | `irma/core/elastic_dw.py` (`resolve_species_dw`), `irma/core/crystal.py` (`_compute_per_species_msd`) | `docs/validation/methodology.md` (section "The Debye-Waller convention") |
| Coherent one-phonon term, interference, per-species partition | `irma/core/noncubic_inelastic.py` | `docs/modes.md`, `docs/validation/beryllium-oxide.md` |
| Bragg edges, structure factors, edge grouping | `irma/core/crystal.py` (`compute_bragg_edges_general`) | `docs/input-reference.md` |
| SEF vs MEF elastic formats | `irma/core/crystal_cards.py`, `irma/core/endf_writer.py` | `docs/validation/beryllium-oxide.md` |
| Crystalline extinction | `irma/core/extinction.py`, `irma/core/elastic_extinction.py` | `docs/extinction.md` |
| alpha/beta grids and conventions | `irma/core/grids.py`, `irma/core/sab_grids.py` | `docs/grids.md` |
| ENDF records, LTHR/LASYM/LLN, interpolation flags | `irma/core/endf_writer.py` | `docs/input-reference.md`, `docs/njoy.md` |
| Instrument resolution (indirect and chopper) | `irma/spectra/` (`chopper_resolution.py`) | `docs/spectra.md` |
| What the MLIP build computes | `irma/mlip/phonons.py`, `irma/mlip/emit.py` | `docs/mlip.md` |

When a physics question is really a comparison question ("why does
IRMA differ from X here?"), take the validation record's discipline:
name what each curve includes, the Debye-Waller convention, and the
metric, before judging agreement; and attribute discrepancies with
measurement neutrally (phonon model, sample, or measurement; a single
comparison usually cannot tell which).

## Boundaries

- This skill drives and explains IRMA; for modifying IRMA itself,
  follow `CONTRIBUTING.md` (the classic kernels are validated
  bit-for-bit against NJOY tapes, so numerical edits there are not
  casual changes).
- Never present an emitted starting-point deck, or a survey-grade
  MLIP evaluation, as production nuclear data without saying what
  separates them (reviewed parameters, converged phonon model,
  validation against references).
