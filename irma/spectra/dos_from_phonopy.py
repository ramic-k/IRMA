"""Derive per-species partial phonon DOS from a phonopy calculation (mode 0).

The DOS-based ("mode 0") path normally reads a 2-column DOS file per species
(:mod:`irma.spectra.dos_io`). This bridge instead computes the partial DOS
straight from a phonopy.yaml + mesh, so a user with a finished phonon
calculation can run the lightweight isotropic-DW incoherent-approximation kernel
without first exporting DOS files -- and without the heavier mode-1/2 eigenvector
engine.

The per-atom DOS is ``irma.core.phonopy_io.compute_atom_dos``, a histogram
of the mesh modes that integrates to 1 per atom -- exactly the ``tbeta=1`` normalization
``compute_mode0_sqe`` expects. Atoms of the same symbol are averaged into one
per-species partial DOS (still normalized to 1 per atom) on the kernel's uniform
omega grid (eV, starting at 0).
"""
from __future__ import annotations

import numpy as np

from irma.core.phonopy_io import load_phonopy_mesh, compute_atom_dos
from irma.spectra.elastic import _group_atoms_by_symbol


def partial_dos_from_phonopy(phonopy_yaml, mesh, *, born_path=None,
                             force_constants=None, force_sets=None):
    """Per-species partial phonon DOS from a phonopy.yaml + mesh.

    Parameters
    ----------
    phonopy_yaml : path to the phonopy.yaml (force constants embedded or beside).
    mesh : (n1, n2, n3) phonon q-mesh.
    born_path : optional BORN file for the non-analytic correction.
    force_constants, force_sets : optional explicit force-constants /
        force-sets file; overrides the yaml-adjacent discovery (same priority
        as the mode-1/2 engine, so an explicit override gives identical
        phonons in every inelastic mode).

    The DOS grid runs from 0 to 1.05 x the max mesh frequency with a 0.5 meV
    spacing.

    Returns a list of per-species dicts ``{symbol, omega_ev, rho, multiplicity}``
    in first-appearance order -- ready to merge with the per-species scattering
    data (awr, sigma_bound_b, ...) and hand to
    :func:`irma.spectra.dos_mode0.compute_mode0_sqe`.
    """
    mesh_data = load_phonopy_mesh(phonopy_yaml, tuple(int(m) for m in mesh),
                                  born_path=born_path,
                                  force_constants_filename=force_constants,
                                  force_sets_filename=force_sets)
    w_max = float(np.max(mesh_data.frequencies_ev))
    if not w_max > 0.0:
        raise ValueError(f"{phonopy_yaml}: mesh has no positive phonon frequencies")
    e_max_ev = w_max * 1.05
    n_freq = max(int(round(e_max_ev * 1000.0 / 0.5)) + 1, 16)
    g_atom, omega_ev = compute_atom_dos(mesh_data, e_max_ev, n_freq)

    omega_ev = np.ascontiguousarray(omega_ev, dtype=float)
    out = []
    for sym, idx in zip(*_group_atoms_by_symbol(list(mesh_data.atom_symbols))):
        rho = g_atom[idx].mean(axis=0)              # still normalized to 1/atom
        rho = np.clip(rho, 0.0, None)
        rho[0] = 0.0                                # kernel reconstructs omega=0
        out.append({"symbol": sym, "omega_ev": omega_ev,
                    "rho": np.ascontiguousarray(rho, dtype=float),
                    "multiplicity": len(idx)})
    return out
