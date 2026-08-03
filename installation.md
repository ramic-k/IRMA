# Installation

This page covers the two ways to install IRMA (the released package from PyPI, or an editable install from source), the optional extras some workflows need, the separately built C++ NCrystal plugins, and how to confirm the install works. IRMA itself is a pure-Python package with a small, stable dependency footprint: the core engine needs only NumPy, `endf-parserpy`, and `threadpoolctl`, the GUI uses the standard-library `tkinter`, and three optional extras cover the rest: `[phonopy]` for the noncubic inelastic paths, `[spectra]` for the neutron-scattering forward model, and `[mlip]` for the MLIP phonon front end. Either route is one package-manager command; on PyPI you add the extras you need.

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

Most of this table is installed for you; the one dependency worth understanding is `phonopy`. It powers the eigenvector-based workflows: on the ENDF side, the generalized noncubic inelastic paths (`iel=10` combined with `inelastic_mode=1`, the directional incoherent approximation, or `inelastic_mode=2`, the exact coherent one-phonon), and on the spectra side, the inelastic modes 1/2 and the `dos_source: phonopy` option of mode 0. Everything else runs without it: all the classic LEAPR kernels, `iel=0`–`6`, `iel=10` with `inelastic_mode=0`, and spectra mode 0 from DOS files. The `euphonic` package is used only by offline validation harnesses and is never imported by the engine, so you do not need it for normal use.

## Install (released package)

From PyPI the extras are opt-in: `pip install irma` is the core, and `pip install "irma[phonopy]"`, `"irma[spectra]"`, and `"irma[mlip]"` add the pieces described below. One command gets everything:

```bash
pip install "irma[phonopy,spectra,mlip]"
```

A conda-forge package is on the way, under the name `irma-sqw`. bioconda already ships an unrelated `irma`, the CDC influenza assembler, and the two channels are routinely enabled together, so the conda package is named for the S(Q,ω) that underpins every IRMA calculation. Only the conda package name differs: the import and the command stay `irma`. That package is all of IRMA in one install, the core plus the dependencies of every extra (phonopy, scipy, PyYAML, ase), so the ENDF generator, the spectra forward model, the NCrystal exporter, and the MLIP phonon front end all work from it:

```bash
conda create -n irma -c conda-forge irma-sqw
conda activate irma
```

Until the recipe is merged, that command reports the package as missing; use pip in the meantime.

With either package manager, the pretrained potentials still get their own environments (`irma mlip env create <potential>`, described under the `[mlip]` extra), and the C++ NCrystal plugins are still built separately (see [NCrystal plugins](#ncrystal-plugins-built-separately)).

## Install from source

For development, or to run the committed examples and the test suites, clone the repository (`git clone https://github.com/ramic-k/IRMA.git`) and, from the source directory, install the core package in editable (development) mode:

```bash
pip install -e .
```

This gives you the classic LEAPR paths (`iel=0`–`6`) and `iel=10` with `inelastic_mode=0`. NumPy and `endf-parserpy` are pulled in automatically.

For a regular (non-editable) install:

```bash
pip install .
```

### The ENDF writer backend

IRMA writes the output tape through `endf-parserpy`, which ships two interchangeable serializers. By default IRMA uses the **compiled backend** (`EndfParserCpp`): it produces byte-identical tapes and is several times faster than the pure-Python writer, which dominates the wall time of classic runs. The compiled backend is included in the official `endf-parserpy` binary wheels, so a normal `pip install` has it. On a source-only `endf-parserpy` build where the compiled module is unavailable, IRMA falls back to the pure-Python writer with a console warning; the output is identical, just slower. Set the environment variable `IRMA_ENDF_WRITER=py` to force the pure-Python writer (this also silences the fallback warning), or `IRMA_ENDF_WRITER=cpp` to make a missing compiled backend a hard error.

### Adding the phonopy extra

If you plan to run the noncubic inelastic modes (`iel=10` with `inelastic_mode=1` or `2`), install the `phonopy` extra:

```bash
pip install -e ".[phonopy]"
```

This adds `phonopy>=4.2` on top of the core dependencies and enables the phonopy-backed S(α,β) calculation. Phonopy 4 is the only major version the suite is tested against.

It is important to note that installing the extra makes the code paths available but does not supply the physics input: a mode 1/2 deck also needs a phonopy model on disk, a `phonopy.yaml` plus force constants (named on `Card 6f`) and the `Card 6g` controls. The force constants are read from the YAML if embedded, otherwise discovered next to it (`force_constants.hdf5`, `FORCE_CONSTANTS`, `FORCE_SETS`, in that order); the working directory is never consulted. See the [input cards reference](input-reference.md) for the full `Card 6f`/`Card 6g` layout.

### Adding the spectra extra

If you plan to use the neutron-scattering forward model (the `irma spectra` CLI or the GUI's **Neutron Scattering Experiments** tab), install the `spectra` extra:

```bash
pip install -e ".[spectra]"
```

This adds `scipy` (S(Q,E) interpolation) and `PyYAML` (the primary config format). The extras combine: `pip install -e ".[phonopy,spectra,mlip]"` covers every workflow, and the spectra inelastic modes 1/2 (and mode 0 with `dos_source: phonopy`) need both `[phonopy]` and `[spectra]`. See [DOS-based spectra (mode 0)](spectra-mode0.md) for what each spectra mode requires.

### Adding the mlip extra

The MLIP front end (`irma mlip`) builds phonon models directly from
pretrained machine-learned interatomic potentials (MLIPs):

```bash
pip install -e ".[mlip]"
```

This adds `ase`, `phonopy`, and `PyYAML`, deliberately **not** any
potential package. Each potential brings a heavy, mutually conflicting
dependency stack (different `e3nn` pins, an ABI-locked torch, one
TensorFlow backend), so you install only the ones you use. You can
install a potential's package directly into this environment
(`pip install nequip`, `pip install mattersim`, ...), or keep each
potential in its own environment: `irma mlip env create <potential>`
builds a separate Python environment for that one potential, installs
that potential's packages into it, and registers it, after which
`irma mlip` uses the registered environment automatically:

```bash
irma mlip env create mace
```

The full support matrix (which potential needs which package, Python
floor, license, and known conflicts) lives in
[MLIP phonon models](mlip.md).

## NCrystal plugins (built separately)

Installing the `irma` package does not build the two C++ NCrystal plugins. Each is its own scikit-build-core package inside the repository and compiles against an existing NCrystal installation (NCrystal >= 4.3, with cmake, ninja, and scikit-build-core available in the same environment, for example from conda-forge):

```bash
pip install --no-build-isolation ./ncrystal_plugin_IRMA      # samples IRMA's exported kernels
pip install --no-build-isolation ./ncrystal_plugin_ENDFTSL   # reads ENDF thermal scattering files directly
ncrystal-pluginmanager --test IRMA                           # self-test of the installed plugins
ncrystal-pluginmanager --test ENDFTSL
```

NCrystal's plugin manager discovers both automatically once they are installed; no further configuration is needed. Each plugin's README (`ncrystal_plugin_IRMA/README.md`, `ncrystal_plugin_ENDFTSL/README.md`) documents its build environment and reference tests.

## The GUI and tkinter

IRMA ships a graphical interface built on `tkinter`, which is part of the Python standard library, so no extra GUI framework is required. Launch it either way:

```bash
python -m irma --gui   # module flag
irma-gui               # installed entry point
```

Both open the same window.

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

If the import fails with `ModuleNotFoundError: No module named 'irma'`, the install almost always ran in a different environment than the one you are using now. Re-run `pip install -e .` from the IRMA source directory in the active environment.

## Developer install and running the tests

The editable install above (`pip install -e .`) is already a developer install: you can edit the source and the changes take effect without reinstalling. To run the fast regression suite, which does **not** require `phonopy`:

```bash
python -m pytest
```

Two deck-level validation harnesses live under `tests/` and are run manually (they are not part of CI):

| Harness | What it checks | Extra needs |
|---------|----------------|-------------|
| `tests/native_LEAPR_NJOY_ENDF_validation/` | Reproduces published ENDF/B-VIII.1 tapes (graphite, Fe, Al, polyethylene to 7e-5 or better; liquid methane, ortho-/para-hydrogen, and BeO exactly). Fully self-contained. | None |
| `tests/mode2_euphonic_n1_validation/` | Cross-validates the mode-2 exact one-phonon S(α,β) against Euphonic (graphite, Be). | The committed Euphonic reference is frozen; regenerating it (only if the phonon model changes) requires the `euphonic` package |

One last note for long-time users: IRMA was renamed from THAWNE in June 2026, and the package is now `irma`. If you have older scripts, update imports and the CLI/GUI entry points (`python -m irma`, `irma`, `irma-gui`) accordingly.
