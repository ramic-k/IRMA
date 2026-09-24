# Preparing a phonopy calculation

The eigenvector-based modes (1 and 2) of all three IRMA outputs read the
same phonon input: a `phonopy.yaml`, named on Card 6f of an ENDF input
file (cards are the numbered records of the input format; see the
[input file reference](input-reference.md)), in the
`material.phonopy_yaml` field of a spectra configuration, and in the
same field of an NCrystal export configuration. (Mode 0 of the ENDF and
spectra paths takes a DOS instead.)

If you have no force-constant calculation, use the [MLIP front
end](mlip.md), which writes a `phonopy.yaml` into every bundle. This page
is for users who ran their own calculation (DFT, AIMD, a classical or
machine-learned potential) and need to package it for IRMA.

## What IRMA reads

IRMA loads the `phonopy.yaml` for the cells and symmetry, then looks
for force constants in a fixed order: embedded in the yaml itself
first, and otherwise in a file sitting next to the yaml, trying
`force_constants.hdf5`, then `FORCE_CONSTANTS`, then `FORCE_SETS`.
Finding none of them is an error, and the working directory is never
consulted; only the yaml's own directory is searched. The simplest
arrangement is therefore a single self-contained file with the force
constants embedded, and phonopy will produce exactly that.

## From forces to a self-contained phonopy.yaml

The standard phonopy finite-displacement chain, shown here with VASP
as the force calculator:

```bash
phonopy -d --dim 4 4 4        # displaced supercells + phonopy_disp.yaml
# ... run the force calculator on each displaced supercell ...
phonopy -f disp-001/vasprun.xml disp-002/vasprun.xml   # -> FORCE_SETS
phonopy-load --include-fc
```

The `--include-fc` flag is the important one for IRMA: phonopy saves a
`phonopy.yaml` at the end of every run, and with `--include-fc` (or
`--include-all`) that file carries the force constants inside it,
which is all IRMA needs. The same chain works for the other
calculators phonopy supports; pass the calculator flag you used at the
`-d` step to the `-f` step as well (for example `phonopy --qe -d ...`
then `phonopy --qe -f ...` for Quantum ESPRESSO).

Any calculator interface will do. phonopy keeps the cell in the
*calculator's* native length unit (bohr for `qe`, `abinit`, `elk`,
`siesta`, `wien2k`, `DFTB+`, `TURBOMOLE`, `fleur`, `abacus` and `qlm`;
Ångström for `vasp`, `lammps`, `castep`, `aims`, `crystal`, `pwmat`,
`cp2k`) and converts frequencies with a factor that is likewise
calculator-specific. IRMA reads both from the phonopy.yaml and applies them at
load, so the interface you used changes nothing about the result. There
is nothing to convert by hand, and no reason to prefer one interface
over another on units grounds.

## If you already have a force-constant file

A `FORCE_CONSTANTS`, `force_constants.hdf5`, or `FORCE_SETS` from an
earlier phonopy run needs no conversion: place it in the same
directory as the `phonopy.yaml` and IRMA discovers it in the order
given above. With `FORCE_SETS`, phonopy's raw displacement-force data,
IRMA has phonopy rebuild the force constants at load time, which costs
a moment but changes nothing else.

## Polar crystals: BORN and the non-analytical correction

For polar materials (MgO, BeO, oxides in general) the long-range
dipole interaction splits the optical branches near the zone center,
and phonopy handles it with the non-analytical-term correction (NAC)
parameterized by a `BORN` file: the dielectric tensor and the Born
effective charges. IRMA honors NAC two ways. If the `phonopy.yaml`
already embeds the NAC parameters, IRMA applies them. With phonopy 4,
keep the `BORN` file in the directory where you run phonopy: phonopy
then applies NAC and saves the parameters in `phonopy.yaml` (use
`--include-fc` or `--include-all` so the force constants are embedded
too). On an ENDF
input file, Card 6f's `use_born=1` additionally names a `BORN` file
explicitly, and an unreadable file is an error rather than a silent
fallback. A `BORN` file merely sitting in
the working directory is never picked up on its own; the correction is
applied only through one of those two explicit routes.

## Checking the calculation before production

Whichever route produced the calculation, look at the phonon density of
states before spending compute on an evaluation (phonopy's own `-p`
plotting flag draws it, or plot the `total_dos.dat` it writes): imaginary modes or a
cut-off spectrum mean the force-constant calculation needs attention,
not the IRMA settings. The [scattering modes](modes.md) page explains
how the phonon q-mesh sets the lowest energy the one-phonon term
reaches, and the
[troubleshooting](troubleshooting.md) page lists the force-constant
pitfalls IRMA diagnoses at load time, including the supercell and
primitive-cell mismatches it rejects outright.
