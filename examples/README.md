# IRMA examples

Ready-to-run examples for every calculation type, from the command line and
from the GUI. Three families:

- **`tsl/`** — thermal-scattering-law **ENDF generation** (`python -m irma
  deck.input out.endf`): the LEAPR-style evaluations that feed NJOY
  THERMR/ACER.
- **`spectra/`** — the **neutron-scattering forward model** (`python -m irma
  spectra ...`): instrument-resolved INS spectra and 2-D S(Q,E) maps.
- **`mlip/`** — the **MLIP front end** (`irma mlip ...`): a phonon-model
  bundle from a bare crystal structure, emitted to ENDF, spectra, and
  NCrystal inputs.

| # | Calculation type | Example | Runtime |
|---|------------------|---------|---------|
| 1 | Classic LEAPR deck, built-in coherent elastic (`iel=1`) | `tsl/README.md` §2 → committed graphite deck under `tests/` | seconds |
| 2 | Incoherent + free-gas secondary scatterer (`iel=0`, `nss=1`) | `tsl/README.md` §2 → committed H-in-CH₂ deck | seconds |
| 3 | Multi-temperature, shared spectrum (negative-T convention) | `tsl/README.md` §2 → committed graphite/Al/Fe decks (10 temperatures) | seconds |
| 4 | Liquid: translation + oscillators + cold H₂ + Sköld | `tsl/README.md` §2 → committed l-CH₄ and ortho-/para-H₂ decks | seconds–minutes |
| 5 | Two-pass (bound) secondary scatterer (`b7=0`) | `tsl/README.md` §2 → committed BeO deck | seconds |
| 6 | **Generalized elastic, any crystal** (`iel=10`, `inelastic_mode=0`) | `tsl/graphite_iel10_classic.input` | ~1 min |
| 7 | Bragg-edge grouping (ENDF-102 §7.2.2) | `tsl/README.md` §1 (one-line variant of #6) | ~1 min |
| 8 | Phonopy-backed noncubic law (`iel=10`, `inelastic_mode=1/2`) | `tsl/graphite_mode2.input` (production: auto-sized multiphonon) | minutes |
| 9 | **Crystalline extinction** (`iel=10` + optional `extinction` card) | `tsl/be_iel10_extinction.input` | ~1 min |
| 10 | **INS spectrum from a DOS file** (spectra mode 0) | `spectra/graphite_mode0_dosfile.yaml` | seconds |
| 11 | INS spectrum, DOS derived from phonopy (mode 0) | `spectra/graphite_mode0_phonopy.yaml` | ~1 min |
| 12 | INS spectrum, full eigenvector engine (modes 1/2, VISION) | `spectra/graphite_mode2_vision.yaml` | ~1–2 min |
| 13 | Direct-geometry 2-D S(Q,E) map, chopper resolution (ARCS) | `spectra/graphite_arcs_map.yaml` | ~1–2 min |
| 14 | Direct-geometry constant-Q cuts | `spectra/graphite_direct_qcuts.yaml` | <1 min |
| 15 | **MLIP phonon-model bundle** from a structure file (`irma mlip build`) | `mlip/MgO.cif` + `mlip/README.md` | minutes (+ one-time env setup) |
| 16 | Bundle → ENDF / spectra / NCrystal inputs (`irma mlip emit`) | `mlip/README.md` §3 | seconds |

Every example file carries its exact run command — the spectra YAMLs in a
header comment, the TSL decks as trailing text on their first card (the deck
format has no leading-comment syntax) — and the `README.md` in each directory
adds the GUI walkthrough for the same calculation. Run the spectra examples
from inside `examples/spectra/` (their data paths are relative to that
directory) and the TSL decks from the repo root.

Also here: `lln_low_temperature_demo.py` — a 1-second numerical demo of why
cryogenic tapes need Card 4 `ilog=1` (ENDF LLN=1 log storage); run it with
`python examples/lln_low_temperature_demo.py`.
