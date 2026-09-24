# Installing IRMA

The full installation guide — including the optional pieces and what each
workflow needs — lives in the manual at
[`docs/installation.md`](docs/installation.md). This file is the short
version.

## Prerequisites

- **Python 3.11 or later** — [Download Python](https://www.python.org/downloads/)
- **tkinter** — Required for the GUI. It is included with most Python
  installations. If you get an `ImportError: No module named tkinter`, see
  the [Tkinter installation notes](#tkinter) below.

## Install

From the IRMA source directory, pick the form that covers your workflows:

```bash
pip install -e .              # core (classic LEAPR paths, iel=0-6 and iel=10 mode 0)
pip install -e ".[phonopy]"   # + the phonopy-backed modes (inelastic_mode=1/2)
pip install -e ".[spectra]"   # + the neutron scattering forward model (irma spectra)
pip install -e ".[mlip]"      # + the MLIP phonon front end (irma mlip)
```

The extras combine: `pip install -e ".[phonopy,spectra,mlip]"` covers
everything.
For a regular (non-editable) install, drop the `-e`. The required
dependencies (**numpy**, **endf-parserpy**, and **threadpoolctl**, at the
versions validated in `pyproject.toml`) are resolved automatically — there is
no separate dependency-install step.

What each extra unlocks:

| Extra | Adds | Needed for |
|-------|------|------------|
| `[phonopy]` | `phonopy` | `iel=10` with `inelastic_mode=1` or `2` (the phonopy-backed MT4 modes), and `irma.spectra` inelastic modes 1/2 or `dos_source: phonopy` |
| `[spectra]` | `scipy`, `PyYAML` | the `irma.spectra` forward model (`irma spectra` CLI and the GUI's **Neutron Scattering Experiments** tab) |
| `[mlip]` | `ase`, `phonopy`, `PyYAML` | the `irma.mlip` phonon front end (`irma mlip` CLI and the GUI's **MLIP phonon models** tab). The potential packages themselves are not included: their pins conflict with each other, so `irma mlip env create` installs each one into its own dedicated environment |

The NCrystal pack exporter (`irma ncrystal`) needs `[phonopy,spectra]`: it
exports modes 1/2 and reads a YAML config. The C++ NCrystal plugins are
built separately (see the manual's installation page).

### Dependency documentation

If you need help installing or configuring the dependencies:

| Package | Documentation |
|---------|---------------|
| numpy | [numpy.org/install](https://numpy.org/install/) |
| endf-parserpy | [github.com/IAEA-NDS/endf-parserpy](https://github.com/IAEA-NDS/endf-parserpy) |
| phonopy | [phonopy.github.io/phonopy/install.html](https://phonopy.github.io/phonopy/install.html) |

For the phonopy-backed modes (`iel=10` with `inelastic_mode=1/2`), the input
file also needs the phonopy control cards (`Card 6f` and `Card 6g`):

- use `nspec=0` and omit Card `6e`
- keep the full crystal in Card `6d`, even for mixed materials
- keep one principal scatterer per input file; Card `4`'s ZA selects the
  principal atom type from Card `6d`, and Card `5` gives that atom's mass
  ratio and cross section
- if you need separate mixed-material tapes for different principals
  (for example Be and O in BeO), run IRMA once per principal

## Verify the installation

```bash
python -m irma --version
```

This should print the IRMA version number.

## Launching the GUI

There are two ways to launch the graphical interface:

```bash
# Option 1: Module flag
python -m irma --gui

# Option 2: Installed entry point
irma-gui
```

Both open the same GUI window. The GUI uses **tkinter**, which is part of
the Python standard library — no extra GUI framework is needed.

## Running from the command line

```bash
python -m irma input_file output_file
```

Or using the installed entry point:

```bash
irma input_file output_file
```

With the `[spectra]` extra installed, the neutron scattering forward model
is available as a subcommand:

```bash
irma spectra --help
```

With the `[mlip]` extra installed, the phonon front end is available the
same way:

```bash
irma mlip --help
```

## Platform support

| Capability | Linux | macOS | Windows |
| --- | --- | --- | --- |
| ENDF/TSL evaluation (the classic kernels, `iel` 0-6/10 with `inelastic_mode=0`) | yes | yes | yes |
| `inelastic_mode=1/2`, spectra forward model, NCrystal export | yes | yes | advisory CI* |
| GUI | yes | yes | advisory CI* |
| C++ NCrystal plugins (build) | yes | yes | untested |

CI runs the test suite on Linux (Python 3.11-3.13) and macOS as required
jobs, and on Windows as an advisory job; it builds both NCrystal plugins
on Linux and the IRMA plugin on macOS (advisory). The mode-1/2 worker
pool uses the
spawn start method with the compute context in shared memory, so
`ncpu > 1` (Card 6f) and `jobs > 1` work the same way on all three
platforms. \*Parallel runs (`ncpu`/`jobs` > 1) and GUI cancellation are
not verified on Windows: cancelling a parallel GUI run there stops only
the direct child process, and spawned workers keep running until they
finish their current work item. On Windows, prefer `jobs=1` for GUI runs.
One standard spawn requirement applies: a
script that drives IRMA programmatically must wrap its entry point in
`if __name__ == "__main__":` — the `irma` CLI and `python -m irma...`
entry points already do.

## Tkinter

The GUI requires tkinter, which ships with most Python distributions. If
it is missing on your system:

- **macOS** — Install the official Python from
  [python.org](https://www.python.org/downloads/). The Homebrew Python
  (`brew install python`) may not include tkinter; if so, run
  `brew install python-tk`.
- **Ubuntu/Debian** — `sudo apt install python3-tk`
- **Fedora** — `sudo dnf install python3-tkinter`
- **Arch Linux** — `sudo pacman -S tk`
- **Windows** — Tkinter is included by default with the standard Python
  installer from python.org. Make sure the "tcl/tk and IDLE" option is
  checked during installation.
- **Conda** — `conda install -c conda-forge tk`

## Troubleshooting

**`ModuleNotFoundError: No module named 'irma'`**
Make sure you ran `pip install -e .` from the IRMA directory, and that
you are using the same Python environment.

**`ModuleNotFoundError: No module named 'tkinter'`**
See the [Tkinter](#tkinter) section above.

**GUI does not appear on macOS**
A virtual environment uses its base interpreter's tkinter. If
`python -c 'import tkinter'` fails inside the venv, install Tk for the base
Python (the python.org build, or `brew install python-tk`) and recreate the
venv.
