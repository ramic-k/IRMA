"""Shared Debye-Waller-weighted coherent-elastic edge evaluation (MF7/MT2).

Every IRMA coherent-elastic builder reduces one Bragg edge to a single
DW-attenuated contribution ``delta`` that ``_coherent_s_table`` accumulates into
the cumulative-S TAB1/LIST records. Three physical forms exist:

  * **isotropic** -- ``exp(-4 W E) * |F|^2`` with a scalar DW integral ``W``
    (classic LEAPR, iel=1-6; also the iel=10 fallback when no per-species data).
  * **per-species isotropic** -- ``Sigma_{s,t} b_s b_t exp(-2(W_s+W_t)E) D_{st}``
    with per-species scalar ``W_s = dwpix_s/(awr_s kT)`` (NCrystal-style).
  * **per-species directional** -- the same double sum but with a plane-by-plane
    anisotropic ``W_s(Ghat) = (Ghat.F_s.Ghat)/(awr_s kT)`` summed over the
    reciprocal-vector directions merged into the edge.

This module is the single source of that arithmetic for
``_build_coherent_elastic`` (iel=1-6) and ``_build_cef_coherent`` (iel=10
SEF, and MEF through it with scale 1), so the elastic Debye-Waller treatment
cannot drift between the classic and generalized paths.

BYTE-IDENTITY CONTRACT: the operation order here reproduces the original
closures exactly (the ``W_si``/``W_ti`` recompute pattern, the
``b_s * b_t * dw * D`` multiply order, the ``delta * scale`` placement). The
trailing ``* scale`` is an exact IEEE no-op when ``scale == 1.0``, so the
scale-free callers (iel=1-6 and MEF) reuse the scaled kernels without changing a
single output bit. Pinned by ``tests/test_elastic_dw.py`` and the full byte-
exact MF7/MT2 suite -- do not reorder operations. The directional kernel's
site-resolved branch (review finding P3) engages ONLY when the per-site
tensors differ within a Card 6d group (``dir_tensors_uniform`` false);
uniform-tensor materials keep the original loop verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp

from irma.core.constants import BK


@dataclass
class SpeciesDW:
    """Resolved per-species DW state shared by the iel=10 SEF/MEF builders.

    ``use_dir_dw`` selects the directional kernel, ``use_ps`` the per-species
    isotropic kernel; if both are false the caller falls back to the scalar
    isotropic form and no SpeciesDW is built.

    ``F_sites_per_temp[itemp]`` is the ``[n_sites, 3, 3]`` array of per-site
    displacement tensors in crystal.py's ``site_terms`` order (species, then
    position); ``dir_tensors_uniform`` keeps the species-averaged path when
    every group's site tensors are identical.
    """

    use_dir_dw: bool
    use_ps: bool
    nsp: int
    b_sqb: list
    awr_sp: list
    sc: object
    W_ps: object
    F_species_per_temp: object
    bragg_dir_terms: object
    atom_types: list
    F_sites_per_temp: object = None
    dir_tensors_uniform: bool = True


def resolve_species_dw(crystal_info, tempr, ntempr):
    """Resolve the per-species/directional DW state, or ``None`` for isotropic.

    Reproduces the detection both iel=10 builders performed inline: directional
    DW when ``F_species_per_temp`` and ``bragg_dir_terms`` are present (and every
    per-temperature F-matrix list is non-None), else per-species isotropic when
    ``species_corr`` is present. ``b_coh`` is converted fm->sqrt(barn) via /10 and
    ``W_ps`` is the per-temperature ENDF DW integral (1/eV) -- byte-identically to
    the originals.
    """
    if crystal_info is None:
        return None
    F_species_per_temp = crystal_info.get('F_species_per_temp')
    bragg_dir_terms = crystal_info.get('bragg_dir_terms')
    use_dir_dw = F_species_per_temp is not None and bragg_dir_terms is not None
    use_ps = not use_dir_dw and crystal_info.get('species_corr') is not None
    if not (use_dir_dw or use_ps):
        return None

    atom_types = crystal_info['atom_types']
    # species_corr is only required for the per-species isotropic mode (the
    # use_ps detection above already treats the key as optional); a
    # directional-only crystal_info must not KeyError here.
    sc = crystal_info.get('species_corr')
    nsp = len(atom_types)
    b_sqb = [at['b_coh'] / 10.0 for at in atom_types]
    awr_sp = [at['awr'] for at in atom_types]
    W_ps = None
    if use_ps:
        W_ps = [[at['dwpix'][it] / (at['awr'] * tempr[it] * BK)
                 for it in range(ntempr)] for at in atom_types]
    # Absent keys mean the uniform (species-averaged) path.
    return SpeciesDW(
        use_dir_dw=use_dir_dw, use_ps=use_ps, nsp=nsp, b_sqb=b_sqb,
        awr_sp=awr_sp, sc=sc, W_ps=W_ps,
        F_species_per_temp=F_species_per_temp, bragg_dir_terms=bragg_dir_terms,
        atom_types=atom_types,
        F_sites_per_temp=crystal_info.get('F_sites_per_temp'),
        dir_tensors_uniform=bool(crystal_info.get('dir_tensors_uniform', True)))


def isotropic_edge_delta(w, e, bragg_amp, scale=1.0):
    """Scalar-DW edge contribution ``exp(-4 w e) * bragg_amp * scale``."""
    return exp(-4.0 * w * e) * bragg_amp * scale


def per_species_edge_delta(e, sdw, itemp, sc_j, scale=1.0):
    """Per-species isotropic double sum for one edge (``W_s`` from ``W_ps``)."""
    nsp, b_sqb, W_ps = sdw.nsp, sdw.b_sqb, sdw.W_ps
    delta = 0.0
    for si in range(nsp):
        w_si = W_ps[si][itemp]
        for ti in range(nsp):
            dw = exp(-2.0 * (w_si + W_ps[ti][itemp]) * e)
            delta += b_sqb[si] * b_sqb[ti] * dw * sc_j[si, ti]
    return delta * scale


def directional_edge_delta(e, sdw, itemp, dir_terms_j, kT_j, scale=1.0):
    """Plane-by-plane anisotropic edge contribution.

    Uniform-tensor fast path (``sdw.dir_tensors_uniform``, i.e. every site of
    every Card 6d group carries the SAME tensor): the per-species double sum
    with ``W_s(Ghat) = (Ghat.F_s.Ghat)/(awr_s kT)`` recomputed inside the loops
    exactly as the originals (``W_ti`` is re-evaluated per (si,ti) pair) so the
    accumulation order -- and thus the rendered tape -- is bit-for-bit
    unchanged.

    Site-resolved path (review finding P3): when the tensors differ within a
    group, averaging them BEFORE exponentiation is wrong physics -- the
    correct per-plane form is the DW-attenuated complex amplitude sum
    ``|sum_i b_sp(i) exp(-2 W_i e) exp(i phi_i)|^2 * pref`` with
    ``W_i = (Ghat.F_i.Ghat)/(awr_sp(i) kT)`` and ``pref = d*mult*xsectfact``
    from the plane's ``site_terms``. Since
    ``D_st == pref * (fr_s fr_t + fi_s fi_t)`` with ``fr_s = sum_i cos_i``
    over the species' sites, the pair term
    ``b_s b_t exp(-2 (W_s+W_t) e) D_st`` equals this amplitude form exactly
    whenever W is constant within each species -- the same 2-factor
    convention, so the uniform limit matches.
    """
    nsp, b_sqb, awr_sp = sdw.nsp, sdw.b_sqb, sdw.awr_sp
    F_itemp = sdw.F_species_per_temp[itemp]
    if getattr(sdw, "dir_tensors_uniform", True):
        delta = 0.0
        for plane in dir_terms_j:
            G_hat, D_st_plane = plane[0], plane[1]
            for si in range(nsp):
                Fkk_si = float(G_hat @ F_itemp[si] @ G_hat)
                W_si = Fkk_si / (awr_sp[si] * kT_j)
                for ti in range(nsp):
                    Fkk_ti = float(G_hat @ F_itemp[ti] @ G_hat)
                    W_ti = Fkk_ti / (awr_sp[ti] * kT_j)
                    dw = exp(-2.0 * (W_si + W_ti) * e)
                    delta += b_sqb[si] * b_sqb[ti] * dw * D_st_plane[si, ti]
        return delta * scale

    F_flat = sdw.F_sites_per_temp[itemp]
    delta = 0.0
    for plane in dir_terms_j:
        G_hat = plane[0]
        cos_a, sin_a, sp_idx, pref = plane[2]
        acc_re = 0.0
        acc_im = 0.0
        for i in range(len(sp_idx)):
            sp = sp_idx[i]
            W_i = float(G_hat @ F_flat[i] @ G_hat) / (awr_sp[sp] * kT_j)
            amp = b_sqb[sp] * exp(-2.0 * W_i * e)
            acc_re += amp * float(cos_a[i])
            acc_im += amp * float(sin_a[i])
        delta += (acc_re * acc_re + acc_im * acc_im) * pref
    return delta * scale


def make_edge_delta(bragg, dwpix_iso, *, scale=1.0, species_dw=None, tempr=None):
    """Build the ``edge_delta(j, itemp, energy=None)`` closure for a builder.

    Dispatches per edge to the directional / per-species / isotropic kernel
    above. ``dwpix_iso[itemp]`` is the scalar isotropic DW integral (the iel=1-6
    builder may pass a pre-averaged SCT mixed-moderator value); ``species_dw`` is
    a :class:`SpeciesDW` or ``None``; ``scale`` is the SEF structure-factor scale
    (``1.0`` for the iel=1-6 and MEF callers, an exact no-op). ``tempr`` is only
    needed for the directional kernel's ``kT`` and may be ``None`` otherwise.
    """
    def edge_delta(j, itemp, energy=None):
        """Debye-Waller-attenuated cumulative-S increment of edge ``j``."""
        e = energy if energy is not None else bragg[j][0]
        if species_dw is not None and species_dw.use_dir_dw:
            kT_j = tempr[itemp] * BK
            return directional_edge_delta(
                e, species_dw, itemp, species_dw.bragg_dir_terms[j], kT_j, scale)
        if species_dw is not None and species_dw.use_ps:
            return per_species_edge_delta(
                e, species_dw, itemp, species_dw.sc[j], scale)
        return isotropic_edge_delta(dwpix_iso[itemp], e, bragg[j][1], scale)
    return edge_delta
