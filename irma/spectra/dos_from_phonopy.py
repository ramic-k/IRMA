"""Derive per-species partial phonon DOS from a phonopy calculation (mode 0).

The DOS-based ("mode 0") path normally reads a 2-column DOS file per species
(:mod:`irma.spectra.dos_io`). This bridge instead computes the partial DOS
straight from a phonopy.yaml + mesh, so a user with a finished phonon
calculation can run the lightweight isotropic-DW incoherent-approximation kernel
without first exporting DOS files -- and without the heavier mode-1/2 eigenvector
engine.

Per-atom scalar DOS is the trace of IRMA's anisotropic DOS tensor,
``g_d(eps) = (1/3) Tr rho_{d,ij}(eps)`` (``irma.core.phonopy_io.compute_dos_tensor``),
which integrates to 1 per atom -- exactly the ``tbeta=1`` normalization
``compute_mode0_sqe`` expects. Atoms of the same symbol are averaged into one
per-species partial DOS (still normalized to 1 per atom) on the kernel's uniform
omega grid (eV, starting at 0).
"""
from __future__ import annotations

import numpy as np

from irma.core.phonopy_io import load_phonopy_mesh, compute_dos_tensor


def partial_dos_from_phonopy(phonopy_yaml, mesh, *, born_path=None,
                             force_constants=None, force_sets=None,
                             e_max_ev=None, n_freq=None, sigma_ev=None,
                             d_omega_mev=0.5):
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
    e_max_ev : top of the DOS grid [eV]; default = 1.05 x the max mesh frequency.
    n_freq : number of uniform grid points [0, e_max_ev]; default chosen so the
        spacing is ``d_omega_mev`` (0.5 meV).
    sigma_ev : Gaussian smearing [eV]; default = 2 x grid spacing (compute_dos_tensor).

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
    if e_max_ev is None:
        e_max_ev = w_max * 1.05
    if n_freq is None:
        n_freq = max(int(round(e_max_ev * 1000.0 / float(d_omega_mev))) + 1, 16)

    dos_tensor, omega_ev = compute_dos_tensor(mesh_data, float(e_max_ev),
                                              int(n_freq), sigma_ev=sigma_ev)
    # scalar per-atom DOS g_d = (1/3) Tr rho_{d,ij}  -> (n_atoms, n_freq), int = 1
    g_atom = np.einsum("diik->dk", dos_tensor) / 3.0

    species_order, groups = [], {}
    for d, sym in enumerate(mesh_data.atom_symbols):
        if sym not in groups:
            species_order.append(sym)
            groups[sym] = []
        groups[sym].append(d)

    omega_ev = np.ascontiguousarray(omega_ev, dtype=float)
    out = []
    for sym in species_order:
        idx = groups[sym]
        rho = g_atom[idx].mean(axis=0)              # still normalized to 1/atom
        rho = np.clip(rho, 0.0, None)
        rho[0] = 0.0                                # kernel reconstructs omega=0
        out.append({"symbol": sym, "omega_ev": omega_ev,
                    "rho": np.ascontiguousarray(rho, dtype=float),
                    "multiplicity": len(idx)})
    return out
