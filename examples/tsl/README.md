# TSL (ENDF) examples

Thermal-scattering-law generation: a card deck in, an ENDF-6 MF7 tape out.

```bash
python -m irma <deck.input> <output.endf>
```

Full card-by-card documentation: the manual's *Input deck reference*.

## 1. The generalized path, any crystal — `graphite_iel10_classic.input`

The recommended starting point for a new evaluation. Crystalline graphite with
the **generalized coherent elastic** (`iel=10`, Cards 6b–6d: your lattice +
atom positions, valid for any crystal) and the classic deck-supplied phonon
DOS for the inelastic part (`inelastic_mode=0` — no phonopy needed). Derived
from the committed, NJOY-validated classic deck (same DOS, grids, and 10
shared-spectrum temperatures); only the elastic treatment changed.

```bash
python -m irma examples/tsl/graphite_iel10_classic.input graphite.endf
```

**In the GUI** (`python -m irma --gui`, *ENDF Evaluation* tab):
*Material* — keep `iel = 10`, fill the lattice constants and the atom-type
table (or just **File ▸ Import Input File...** on this deck to populate
everything); *Scattering* — ZA `6000` (natural carbon), AWR `11.898`, σ_free `4.73918`;
*Grids* — temperatures + automatic grids; *Phonon* — the carbon DOS;
*Run* — pick the output path and **Run Calculation**.

**Bragg-edge grouping** (dense high-energy edges → log-uniform bins): add the
two optional Card 6b fields, e.g. `1 1 0 0 50 1.0/` — 50 bins per decade above
1 eV. See *Scattering modes* in the manual.

## 2. Committed, validated reference decks

`tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/` ships eight real
evaluations. The `.input` decks are included in the source distribution; in
the repository checkout each additionally sits beside the NJOY LEAPR deck it
was derived from and the reference tape it reproduces:

| Deck | Demonstrates |
|------|--------------|
| `tsl-crystalline-graphite.input` | classic built-in coherent elastic (`iel=1`), 10 temperatures via the negative-temperature (shared-spectrum) convention |
| `tsl-013_Al_027.input`, `tsl-026_Fe_056.input` | the Al (`iel=4`) and Fe (`iel=6`) built-ins |
| `tsl-HinCH2.input` | `iel=0` with an analytic **free-gas secondary scatterer** (`nss=1`, `b7=1`) |
| `tsl-BeO.input` | **two-pass bound secondary** (`b7=0`): oxygen gets a full second phonon-spectrum pass |
| `tsl-l-CH4.input` | liquid methane: **translational diffusion + discrete oscillators** + free-gas secondary |
| `tsl-ortho-H.input`, `tsl-para-H.input` | **cold hydrogen** (`ncold`) ortho/para statistics + **Sköld** correction (`nsk=2`, S(κ) table) + per-temperature spectra |

```bash
python -m irma tests/native_LEAPR_NJOY_ENDF_validation/leapr_decks/tsl-HinCH2.input ch2.endf
```

All of these import cleanly into the GUI via **File ▸ Import Input File...** —
the conditional sections (secondary scatterer, S(κ) tables, ...) reveal
themselves automatically.

## 3. Phonopy-backed noncubic law (`inelastic_mode=1/2`) — `graphite_mode2.input`

The eigenvector-driven MT4: directional Debye-Waller + in-process noncubic
S(α,β) (`pip install -e ".[phonopy]"` first). The committed deck is a
**production-shaped evaluation** — Card 6g `10000 1000 1 /` (the recommended
production direction counts, matching the GUI form and `irma mlip emit`
defaults and the validation-campaign sampling, with `auto_order=1` so the
multiphonon order is auto-sized to converge the high-Q rows to the free-gas
limit):

```bash
python -m irma examples/tsl/graphite_mode2.input graphite_mode2.endf   # repo root
```

Set the Card 6f `ncpu` field (4th value) to your core count first; the
phonopy model it points at is vendored under
`tests/mode2_euphonic_n1_validation/graphite/`, so run it from the repo root.

Do **not** copy `tests/mode2_euphonic_n1_validation/*/irma_mode2_n1.input.template`
for a real evaluation: those are the Euphonic cross-validation decks, pinned
to Card 3 `nphon=1` with auto-sizing OFF, so they deliberately produce a
**one-phonon-only** law (no multiphonon background). `beryllium/` has the hcp
Be counterpart of the same model if you want a second material.

In the GUI: `iel = 10`, *inelastic mode 2*, point the phonopy section at the
`phonopy.yaml`, set the mesh and the Card 6g direction counts (the defaults
are the recommended production values), and tick *Auto-size multiphonon
order*.
Processing mode-2 tapes through stock NJOY THERMR needs the one-line `cliq`
patch — see *NJOY interoperability* in the manual.

## 4. Crystalline extinction (optional) — `be_iel10_extinction.input`

Realistic-sample beryllium: the ideal-crystal Be mode-0 deck plus the **optional
extinction card** (off by default), which reduces the coherent-elastic Bragg
peaks for dynamical-diffraction extinction (à la Xu 2025):

```
extinction BC_mix l=8550 g=170 L=75750 dist=Gauss rec=std rmse_tol=1e-3 /
```

placed as the last card of the `iel=10` elastic block (after Cards 6d/6e, before
Card 7). Extinction is a **sample** property: `l` = crystallite size [Å] (primary),
`g` = mosaic [rad⁻¹] and `L` = grain size [Å] (secondary) — fit them to
transmission or take them from microstructure; omit the card for the ideal tape.
Five models are available (`Sabine_uncorr/Sabine_corr`, `BC_pure/BC_mix/BC_mod`,
the last two requiring `g>0` and `L>0`).

```bash
python -m irma examples/tsl/be_iel10_extinction.input be_extinction.endf   # repo root
```

The models/recipes are ported from the NCrystal **CrysXT** plugin; see the
manual's *Crystalline extinction* page for the physics, parameters, and
references (Kittelmann 2026, Xu 2025).
