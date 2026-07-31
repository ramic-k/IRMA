# DOS-based spectra (mode 0)

The neutron-scattering forward model (`irma.spectra`) turns a phonon model into
an instrument-resolved 1-D INS spectrum or a 2-D `S(Q,E)` powder map. Its
`inelastic_mode` selects how the inelastic scattering is computed:

| Mode | Inelastic engine | Debye-Waller | Needs |
|------|------------------|--------------|-------|
| `0` | **DOS** incoherent-approximation phonon expansion | isotropic (scalar λ per element) | a phonon **DOS** (file or phonopy) |
| `1` | phonopy eigenvectors, incoherent approximation | anisotropic (per-atom tensors) | `phonopy.yaml` + mesh |
| `2` | phonopy eigenvectors, coherent one-phonon + multiphonon | anisotropic | `phonopy.yaml` + mesh |

**Mode 0** is the lightweight end of the range. It runs the same
incoherent-approximation phonon expansion LEAPR uses, straight from a phonon
density of states, with no eigenvectors and no engine. It is the right choice when:

- the material is **hydrogen-rich, incoherent or disordered** (where the
  incoherent approximation is already excellent and H dominates the signal);
- you only **have a DOS** (from MD/VACF, a measurement, or a quick calc);
- you want a **fast survey** before committing to a full mode-1/2 run.

It does *not* capture coherent **inelastic** (dispersion); use mode 2 for that.
This is the same modelling level as OCLIMAX `TASK=0`, but driven by IRMA's own
LEAPR kernel and wired into the full instrument/resolution/elastic forward model.

## Multi-element materials

Each element scatters by **its own** partial DOS, mass and Debye-Waller; the
result is the cross-section/multiplicity-weighted **per-atom** average

$$
\frac{d^2\sigma}{d\Omega\,dE'}(Q,E)=\frac{1}{N}\sum_d m_d\,\frac{\sigma_d}{4\pi}\,
e^{-2W_d(Q)}\,[\text{expansion of }\rho_d],\qquad
\alpha_d=\frac{C_E\,Q^2}{A_d\,k_BT},
$$

with $N=\sum_d m_d$ the atoms per cell. The $1/N$ makes the absolute scale
**per represented atom**, the *same* normalization as `inelastic_mode` 1 and 2
(the eigenvector engine also normalizes per represented atom), so a mode-0 and a
mode-1/2 spectrum are directly comparable. H is never blended away into a single
effective spectrum, and a neutron-weighted GDOS is also produced as a 1-D summary.

## Where the DOS comes from (`dos_source`)

- **`file`** (default): each element supplies a generic **2-column** phonon DOS
  (frequency, intensity). The unit is `meV` (default), `eV`, `cm-1` or `THz`.
- **`phonopy`**: the partial DOS (and per-element atom multiplicity) is derived
  from `material.phonopy_yaml` + `mesh`, the trace of IRMA's anisotropic DOS
  tensor, normalised to one phonon per atom. You still give the scattering data
  (`awr`, `sigma_bound_b`, …) per element.

## The elastic line (optional, iel=10 analogue)

Give a unit cell and mode 0 builds a rigorous elastic line from the **same**
Debye–Waller the inelastic uses:

- **coherent** Bragg comb from your `lattice` + per-element `b_coh_fm` and
  fractional `positions` (the same edge kernel the ENDF MF7/MT2 writer uses);
- **incoherent** Debye–Waller line, the per-atom average over **every** element,
  $\frac{1}{N}\sum_d m_d(\sigma_{\mathrm{inc},d}/4\pi)e^{-2W_d}$, so e.g. the
  hydrogen line in a CH$_2$ is kept even though carbon has the larger coherent
  length. Coherent and incoherent share the inelastic's per-atom scale.

Leave `lattice` blank and the coherent comb is skipped. With `elastic_kind:
both` (the default) you still get the lattice-free incoherent line; set
`physics.elastic: false` for a truly inelastic-only spectrum.

## Configuration

```yaml
material:
  temperature_K: 300.0
  # mode-0 elastic cell (omit for inelastic-only):
  lattice: [2.4612, 2.4612, 6.7079, 90.0, 90.0, 120.0]   # a,b,c,α,β,γ
  scatterers:
    - {symbol: C, sigma_bound_b: 5.551, awr: 11.898, b_coh_fm: 6.646,
       sigma_inc_b: 0.001, dos_file: c_dos.txt, dos_unit: meV, multiplicity: 4,
       positions: [[0,0,0.25],[0,0,0.75],[0.3333,0.6667,0.25],[0.6667,0.3333,0.75]]}
physics:
  inelastic_mode: 0
  dos_source: file        # or: phonopy  (then set material.phonopy_yaml + mesh)
  elastic: true
  elastic_kind: both      # both | coherent | incoherent
grid: {e_max_meV: 250.0, de_meV: 1.0, dq_max_invA: 0.1}
instrument: {geometry: vision, e_fixed_meV: 3.5}
```

### CLI

Run a config file (`irma spectra run config.yaml -o out.csv`), or use the flag
form directly: select `--inelastic-mode 0` and put the mode-0 extras on each
scatterer string as order-free `key=value` tokens:

```
SYMBOL,sigma_bound_b,awr[,b_coh_fm[,sigma_inc_b]][,dos=FILE,unit=U,mult=N,pos=x:y:z;...]
```

```bash
python -m irma spectra vision --inelastic-mode 0 --temperature 300 \
    --scatterer "C,5.551,11.898,6.646,0.001,dos=c_dos.txt,mult=4,pos=0:0:0.25;0:0:0.75;0.3333:0.6667:0.25;0.6667:0.3333:0.75" \
    --lattice 2.4612,2.4612,6.7079,90,90,120 -o graphite_vision.csv
```

`--lattice` (with per-scatterer `pos=` sites) enables the coherent-elastic
comb; `--dos-source phonopy` derives the partial DOS from `--phonopy-yaml` +
`--mesh` instead of `dos=` files.

### GUI

In the **Neutron Scattering Experiments** tab the first control, **phonon input**, gates
everything:

- **Phonopy model** (default): set *inelastic mode* to `0 (DOS + isotropic DW)`
  for the DOS-derived path (the partial DOS comes from the phonopy.yaml + mesh).
  *Auto-fill elements from phonopy.yaml* populates the element table for you.
- **DOS files (mode 0)**: the manual path, no phonopy. Build the element table
  with *+ Add element*; each row gets its own **DOS file** (Browse) and *unit*.

Fill the per-element scattering data (σ_bound, awr, b_coh, σ_inc) in the table.
For the coherent-elastic line, set *elastic kind* to include coherent and fill
the material *lattice* and each element's *positions* (flat `x y z x y z …`, the
same format as the ENDF deck; the triplet count must equal *mult*).

![The Neutron Scattering Experiments tab in DOS-files (mode 0) form: the per-element table with the DOS file column, and the lattice row that enables the coherent-elastic comb (the committed graphite example)](assets/gui/gui_ns_dosfiles_mode0.png)

## Validation

`tests/mode0_validation/validate_mode0_vs_mode1.py` cross-checks mode 0 against
the mode-1 engine on graphite (a sanity check, not a controlled proof). Because
both are now **per represented atom**, they agree in **absolute** scale (the
integral ratio is ≈ **1.02**) and the area-normalised shapes track closely
(cosine ≈ 0.92). The residual (the ~2 % integral excess and the shape difference)
is **consistent with** mode 0's isotropic Debye–Waller vs mode 1's anisotropic one
on this strongly anisotropic crystal, the same approximation OCLIMAX `TASK=0`
makes. A dedicated quantitative comparison against OCLIMAX `TASK=0` and against
measured hydrogen-rich INS spectra has not yet been performed.
