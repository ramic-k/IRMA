"""CEF branches that divide by the principal incoherent cross section must
refuse sigma_inc=0 with a clear message instead of crashing.

Eq. 25 (single-atom incoherent approximation) and Eq. 26 (polyatomic
redistribution onto a non-DC principal) both divide by sigma_inc_p; Card 6d
accepts sigma_inc=0, so an accepted deck could reach a ZeroDivisionError or
emit garbage redistribution.
"""
import numpy as np
import pytest

from irma.core.endf_writer import _build_generalized_elastic


def _ci(atom_types, elastic_mode=1, principal=0, dc=None):
    return {
        'elastic_mode': elastic_mode,
        'atom_types': atom_types,
        'nat': len(atom_types),
        'principal_atom_idx': principal,
        'dc_atom_idx': dc,
        'species_corr': None,
        'F_species_per_temp': None,
        'bragg_dir_terms': None,
    }


def test_single_atom_no_elastic_channel_raises_clearly():
    # b_coh=0 and sigma_inc=0: Eq 25 branch selected (coh <= inc), no
    # channel to scale — must raise, not divide by zero.
    at = {'Z': 6, 'A': 12, 'awr': 11.9, 'b_coh': 0.0, 'sigma_inc': 0.0,
          'sigma_coh': 0.0, 'fraction': 1.0, 'dwpix': [0.5]}
    with pytest.raises(ValueError, match="zero .*coherent AND incoherent"):
        _build_generalized_elastic(1, 6012.0, 11.9, [], 0, 1, [296.0],
                                   _ci([at]), [0.5], 4.74)


def test_polyatomic_nondc_principal_with_zero_sigma_inc_raises_clearly():
    # Principal (idx 0) has sigma_inc=0 and is NOT the DC atom: Eq 26
    # redistribution is undefined — must advise MEF, not crash.
    a0 = {'Z': 8, 'A': 16, 'awr': 15.86, 'b_coh': 5.803, 'sigma_inc': 0.0,
          'sigma_coh': 4.232, 'fraction': 0.5, 'dwpix': [0.4]}
    a1 = {'Z': 4, 'A': 9, 'awr': 8.93, 'b_coh': 7.79, 'sigma_inc': 0.0018,
          'sigma_coh': 7.63, 'fraction': 0.5, 'dwpix': [0.6]}
    with pytest.raises(ValueError, match="Eq. 26 .*undefined.*elastic_mode=2"):
        _build_generalized_elastic(1, 8016.0, 15.86, [], 0, 1, [296.0],
                                   _ci([a0, a1], dc=1), [0.4], 3.76)


# CX2-04: the coherent-elastic table builders index energies[-1]/E[-1]; with
# no surviving Bragg edge (compute_bragg_edges_general can return nedge=0) they
# crash on an empty list / emit a degenerate empty TAB1. They must instead
# reject the no-edge case with a clear message.

def _ci_coh(coh_group=None):
    # Single-atom CEF coherent branch (sigma_coh > sigma_inc) so the build
    # reaches the coherent-S table; no edges supplied.
    at = {'Z': 6, 'A': 12, 'awr': 11.9, 'b_coh': 6.6, 'sigma_inc': 0.001,
          'sigma_coh': 5.5, 'fraction': 1.0, 'dwpix': [0.5]}
    ci = _ci([at])
    if coh_group is not None:
        ci['coh_edge_group_bins_per_decade'] = coh_group
        ci['coh_edge_group_threshold_ev'] = 1.0
    return ci


def test_coherent_elastic_no_edges_ungrouped_raises_clearly():
    with pytest.raises(ValueError, match="no Bragg edge.*positive contribution"):
        _build_generalized_elastic(1, 6012.0, 11.9, [], 0, 1, [296.0],
                                   _ci_coh(), [0.5], 4.74)


def test_coherent_elastic_no_edges_grouped_raises_clearly():
    with pytest.raises(ValueError, match="no Bragg edge to group"):
        _build_generalized_elastic(1, 6012.0, 11.9, [], 0, 1, [296.0],
                                   _ci_coh(coh_group=20), [0.5], 4.74)


def test_coherent_s_table_all_edges_thinned_raises_clearly():
    # nedge > 0 but every edge contributes 0 -> jmax stays 0 -> energies[-1]
    # would index an empty list. Must raise, not IndexError.
    from irma.core.endf_writer import _coherent_s_table
    bragg = [(0.01, 0.0), (0.02, 0.0)]
    with pytest.raises(ValueError, match="no Bragg edge.*positive contribution"):
        _coherent_s_table(bragg, 2, 1, [296.0],
                          lambda j, it, energy=None: 0.0)


# QA2-026: iel=10 must never fall through to the built-in coherent-elastic
# builder. write_endf_output guards crystal_info=None with a fail-fast error.

def test_write_endf_output_iel10_requires_crystal_info():
    from irma.core.endf_writer import write_endf_output
    z = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="iel=10.*requires crystal_info"):
        write_endf_output(
            "/tmp/should_not_be_written.endf", 1, 6012.0, 11.9, 5.5, 1, 10,
            0, 0, 0.0, 0.0, 0.0, 0, 1, 1, 1,
            np.array([0.0]), np.array([0.0]), z, None, np.array([296.0]), 1,
            np.array([0.0]), np.array([0.0]), np.array([296.0]),
            np.array([296.0]), [], 0, 0, 0, 1.0e-6, 0,
            comments=None, crystal_info=None)
