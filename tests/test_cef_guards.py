"""SEF elastic builder refusals: a non-DC principal with sigma_inc=0 (Eq. 26
divides by it) and a crystal with no Bragg edge."""
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


# With no surviving Bragg edge the coherent table cannot be built.

def _ci_coh():
    # Single-atom CEF coherent branch (sigma_coh > sigma_inc) so the build
    # reaches the coherent-S table; no edges supplied.
    at = {'Z': 6, 'A': 12, 'awr': 11.9, 'b_coh': 6.6, 'sigma_inc': 0.001,
          'sigma_coh': 5.5, 'fraction': 1.0, 'dwpix': [0.5]}
    return _ci([at])


def test_coherent_elastic_no_edges_ungrouped_raises_clearly():
    with pytest.raises(ValueError, match="no Bragg edge.*positive contribution"):
        _build_generalized_elastic(1, 6012.0, 11.9, [], 0, 1, [296.0],
                                   _ci_coh(), [0.5], 4.74)
