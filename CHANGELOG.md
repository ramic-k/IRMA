# Changelog

Notable changes to IRMA. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

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
