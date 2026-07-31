# Installation

This page walks you through installing IRMA from source, confirming the install, and setting up the optional pieces you only need for some workflows. IRMA is a pure-Python package with a small, stable dependency footprint: the core engine needs only NumPy, `endf-parserpy`, and `threadpoolctl`, the GUI uses the standard-library `tkinter`, and three optional extras cover the rest: `[phonopy]` for the noncubic inelastic paths, `[spectra]` for the neutron-scattering forward model, and `[mlip]` for the MLIP phonon front end. The whole process is a single `pip install` plus the extras you need.

## Requirements

| Component | Requirement | When you need it |
|-----------|-------------|------------------|
| Python | >= 3.11 | Always |
| NumPy | >= 2.0 | Always (installed automatically) |
| `endf-parserpy` | >= 0.12, < 0.18 | Always (installed automatically; capped below the next unvalidated minor) |
| `threadpoolctl` | >= 3.0 | Always (installed automatically; pins native BLAS/OpenMP threads for the mode-1/2 workers) |
| `tkinter` | standard library | Only for the graphical interface |
| `phonopy` | >= 4.2 | For `iel=10` with `inelastic_mode=1` or `2`, and for `irma.spectra` inelastic modes 1/2 or `dos_source: phonopy` (4.x is the only tested major) |
| `scipy` + `PyYAML` | >= 1.10 / >= 6 | Only for the `irma.spectra` forward model (the `[spectra]` extra) |

!!! note "Where phonopy fits"
    `phonopy` powers the eigenvector-based workflows: the generalized noncubic inelastic paths of the ENDF generator (`iel=10` combined with `inelastic_mode=1`, the directional incoherent approximation, or `inelastic_mode=2`, the exact coherent one-phonon), and on the spectra side the inelastic modes 1/2 and the `dos_source: phonopy` option of mode 0. Everything else (all the classic LEAPR kernels, `iel=0`–`6`, `iel=10` with `inelastic_mode=0`, and spectra mode 0 from DOS files) runs without it. The `euphonic` package is used only by offline validation harnesses and is never imported by the engine, so you do not need it for normal use.

## Install from source

From the IRMA source directory, install the core package in editable (development) mode:

```bash
pip install -e .
```

This gives you the classic LEAPR paths (`iel=0`–`6`) and `iel=10` with `inelastic_mode=0`. NumPy and `endf-parserpy` are pulled in automatically.

!!! note "ENDF writer backend"
    `endf-parserpy` ships two interchangeable serializers, and IRMA writes
    the output tape with the **compiled backend** (`EndfParserCpp`) by
    default: it produces byte-identical tapes and is several times faster
    than the pure-Python writer, which dominates the wall time of classic
    runs. The compiled backend is included in the official `endf-parserpy`
    binary wheels (the normal `pip install`). On a source-only
    `endf-parserpy` build where the compiled module is unavailable, IRMA
    falls back to the pure-Python writer with a console warning; the
    output is identical, just slower. Set the environment variable
    `IRMA_ENDF_WRITER=py` to force the pure-Python writer (silences the
    fallback warning), or `IRMA_ENDF_WRITER=cpp` to make a missing
    compiled backend a hard error.

For a regular (non-editable) install:

```bash
pip install .
```

### Adding the phonopy extra

If you plan to run the noncubic inelastic modes (`iel=10` with `inelastic_mode=1` or `2`), install the `phonopy` extra:

```bash
pip install -e ".[phonopy]"
```

This adds `phonopy>=4.2` on top of the core dependencies, enabling the phonopy-backed S(α,β) calculation. Phonopy 4 is the only major version the suite is tested against.

### Adding the spectra extra

If you plan to use the neutron-scattering forward model (the `irma spectra` CLI or the GUI's **Neutron Scattering Experiments** tab), install the `spectra` extra:

```bash
pip install -e ".[spectra]"
```

This adds `scipy` (S(Q,E) interpolation) and `PyYAML` (the primary config format). The extras combine: `pip install -e ".[phonopy,spectra,mlip]"` covers every workflow, and the spectra inelastic modes 1/2 (and mode 0 with `dos_source: phonopy`) need both `[phonopy]` and `[spectra]`. See [DOS-based spectra (mode 0)](spectra-mode0.md) for what each spectra mode requires.

!!! warning "Mode 1/2 need more than the package"
    Installing the `phonopy` extra makes the code paths available, but a mode 1/2 deck also needs a phonopy model on disk: a `phonopy.yaml` plus force constants (named on `Card 6f`) and the `Card 6g` controls. The force constants are read from the YAML if embedded, otherwise discovered next to it (`force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`, in that order); the working directory is never consulted. See the input cards reference for the full `Card 6f`/`Card 6g` layout.

### Adding the mlip extra

The MLIP front end (`irma mlip`) builds phonon models directly from
pretrained machine-learned interatomic potentials (MLIPs):

```bash
pip install -e ".[mlip]"
```

This adds `ase`, `phonopy`, and `PyYAML`, deliberately **not** any
potential package. Each potential brings a heavy, mutually conflicting
dependency stack (different `e3nn` pins, an ABI-locked torch, one
TensorFlow backend), so you install only the ones you use, either into
this environment (`pip install nequip`, `pip install mattersim`, ...) or
into dedicated per-potential environments that IRMA provisions and uses
automatically:

```bash
irma mlip env create mace
```

The full support matrix (which potential needs which package, Python
floor, license, and known conflicts) lives in
[MLIP phonon models](mlip.md).

## The GUI and tkinter

IRMA ships a graphical interface built on `tkinter`, which is part of the Python standard library, so no extra GUI framework is required. Launch it either way:

```bash
python -m irma --gui   # module flag
irma-gui               # installed entry point
```

Both open the same window.

!!! note "If tkinter is missing"
    `tkinter` ships with most Python distributions, but some minimal builds omit it. If you see `ModuleNotFoundError: No module named 'tkinter'`, install the platform package:

    | Platform | Command |
    |----------|---------|
    | macOS (Homebrew Python) | `brew install python-tk` |
    | Ubuntu / Debian | `sudo apt install python3-tk` |
    | Fedora | `sudo dnf install python3-tkinter` |
    | Arch Linux | `sudo pacman -S tk` |
    | Conda | `conda install -c conda-forge tk` |

    On macOS, installing the official build from [python.org](https://www.python.org/downloads/) includes `tkinter`. On Windows it is included by default with the python.org installer (keep the "tcl/tk and IDLE" option checked). If the GUI does not appear under a virtual environment on macOS, recreate the environment with access to the system `tkinter`: `python -m venv --system-site-packages myenv`.

## Verify the install

Confirm the package imports and reports its version:

```bash
python -m irma --version
```

This prints the IRMA version number. You can also invoke the command-line tool directly:

```bash
python -m irma input_file output_file   # module form
irma input_file output_file             # installed console script
```

!!! note "If the import fails"
    `ModuleNotFoundError: No module named 'irma'` almost always means the install ran in a different environment than the one you are using now. Re-run `pip install -e .` from the IRMA source directory in the active environment.

## Developer install and running the tests

The editable install above (`pip install -e .`) is already a developer install: you can edit the source and the changes take effect without reinstalling. To run the fast regression suite, which does **not** require `phonopy`:

```bash
python -m pytest
```

Two deck-level validation harnesses live under `tests/` and are run manually (they are not part of CI):

| Harness | What it checks | Extra needs |
|---------|----------------|-------------|
| `tests/native_LEAPR_NJOY_ENDF_validation/` | Reproduces published ENDF/B-VIII.1 tapes (graphite, Fe, Al, polyethylene to 7e-5 or better; liquid methane, ortho-/para-hydrogen, and BeO exactly). Fully self-contained. | None |
| `tests/mode2_euphonic_n1_validation/` | Cross-validates the mode-2 exact one-phonon law against Euphonic (graphite, Be). | The committed Euphonic reference is frozen; regenerating it (only if the phonon model changes) requires the `euphonic` package |

!!! note "Renamed from THAWNE"
    IRMA was renamed from THAWNE in June 2026. The package is now `irma`; if you have older scripts, update imports and the CLI/GUI entry points (`python -m irma`, `irma`, `irma-gui`) accordingly.
