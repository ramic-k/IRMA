# Neutron-scattering spectra examples

The forward model: a phonon model in, what the instrument measures out.
Install the extra first (`pip install -e ".[spectra]"`), then **run these from
this directory** — the data paths inside the configs are relative to it.

```bash
cd examples/spectra
python -m irma spectra run <config>.yaml -o <output>.csv
```

Each YAML carries its exact run command and expected runtime in its header.

| Config | Calculation |
|--------|-------------|
| `graphite_mode0_dosfile.yaml` | **Mode 0 from a DOS file** — VISION spectrum from the committed 2-column carbon DOS (`data/graphite_C_dos.txt`); no phonopy needed. The template for "I only have a DOS" workflows (MD/VACF, measured GDOS, other phonon codes). |
| `graphite_mode0_phonopy.yaml` | Mode 0 with the partial DOS **derived from a phonopy model** (`dos_source: phonopy`). |
| `graphite_mode2_vision.yaml` | **Mode 2** (exact coherent one-phonon, anisotropic Debye-Waller) VISION spectrum — the highest-fidelity mode. Mode 1 = same file with `inelastic_mode: 1`. |
| `graphite_arcs_map.yaml` | **Direct-geometry 2-D S(Q,E) map** for ARCS at Ei = 250 meV with the automatic **chopper resolution** model, masked to the detector coverage (`irma spectra map`). |
| `graphite_direct_qcuts.yaml` | Direct-geometry **constant-Q cuts** with a finite detector Q-bin. |

The flag form needs no config file at all — for example, mode 0 in one line:

```bash
python -m irma spectra vision --inelastic-mode 0 --temperature 300 \
    --scatterer "C,5.551,11.898,6.646,0.001,dos=data/graphite_C_dos.txt,mult=4,pos=0:0:0.25;0:0:0.75;0.3333:0.6667:0.25;0.6667:0.3333:0.75" \
    --lattice 2.4612,2.4612,6.7079,90,90,120 -o graphite_vision.csv
```

and a generic indirect instrument is `irma spectra indirect --ef 4.0
--angles 30:150:15 ...`. See `python -m irma spectra vision --help` for every
flag.

## The same calculations in the GUI

Open `python -m irma --gui` → **Neutron Scattering** tab. **Open Config...**
loads any of these YAML files into the panel (and **Save Config...** writes
back what you build — the GUI and CLI run identical configs).

- **Mode 0, DOS files** (`graphite_mode0_dosfile.yaml`): set *phonon input* to
  **DOS files (mode 0)**, add a `C` row with *+ Add element*, fill the
  scattering columns, **Browse** its DOS file, set *mult* 4 and the positions;
  fill the *lattice* for the Bragg peaks.
- **Mode 0 from phonopy** (`graphite_mode0_phonopy.yaml`): keep *phonon input*
  = **Phonopy model**, point at the `phonopy.yaml`, set *inelastic mode* to
  `0 (DOS + isotropic DW)`, and use *Auto-fill elements from phonopy.yaml*.
- **Modes 1/2** (`graphite_mode2_vision.yaml`): *Phonopy model* + *inelastic
  mode* 1 or 2; the *Physics* box exposes the direction counts and worker
  processes.
- **Direct-geometry map / cuts** (`graphite_arcs_map.yaml`,
  `graphite_direct_qcuts.yaml`): pick the **Direct** geometry sub-tab, set Ei,
  choose *output* = 2-D map (with *detector coverage*) or fixed cuts
  (*cut by* angles or constant-Q), and pick the chopper instrument/package/
  frequency under *resolution model* = chopper.

**Run** streams the log; **Plot** displays the result; maps support
*mask to accessible (q,E)* and **Save map...**.
