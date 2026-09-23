"""The Bragg comb is built in the standard frame of the cell parameters (a along
x, b in the xy-plane), while the phonopy Debye-Waller tensors are in the
model's own Cartesian frame. A model whose lattice is rotated must give the
same elastic line as the same model in the standard frame.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from irma.core.crystal import standard_frame_rotation
from irma.core.crystal_cards import _dw_frame_rotation
from irma.spectra.elastic import from_engine_elastic_state


def _rotation():
    def rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        i, j = [k for k in range(3) if k != axis]
        R = np.eye(3)
        R[i, i] = R[j, j] = c
        R[i, j], R[j, i] = -s, s
        return R
    return rot(0, 20.0) @ rot(1, 35.0) @ rot(2, 50.0)


def _state(R=np.eye(3)):
    """Orthorhombic two-atom cell with anisotropic tensors, rotated by R
    (lattice rows L @ R, tensors R.T @ U @ R)."""
    U = np.array([np.diag([0.003, 0.006, 0.012]),
                  [[0.004, 0.001, 0.0], [0.001, 0.005, 0.0], [0.0, 0.0, 0.009]]])
    return {
        "thermal_displacement_matrices_ang2": R.T @ U @ R,
        "primitive_lattice_ang": np.diag([3.0, 4.0, 5.0]) @ R,
        "primitive_scaled_positions": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        "primitive_symbols": ["C", "O"],
        "primitive_masses_amu": np.array([12.0, 16.0]),
        "temperature_k": 296.0,
    }


def test_rotated_model_gives_the_same_bragg_peaks():
    assert standard_frame_rotation(_state()["primitive_lattice_ang"]) is None
    kw = dict(b_coh_fm=[6.646, 5.803], sigma_inc_b=[0.001, 0.0008],
              awr=[11.898, 15.858], elastic_kind="coherent", emax_eV=0.3)
    ref = from_engine_elastic_state(_state(), **kw)
    rot = from_engine_elastic_state(_state(_rotation()), **kw)
    np.testing.assert_allclose(rot.Q_bragg, ref.Q_bragg, rtol=1e-9)
    np.testing.assert_allclose(rot.f_bragg, ref.f_bragg, rtol=1e-9)


def test_pack_tensors_are_written_in_ncrystals_frame():
    from irma.ncrystal.build import _attach_elastic
    neutron = {"C": (0.6646, 0.001), "O": (0.5803, 0.0008)}
    ref, rot = SimpleNamespace(), SimpleNamespace()
    _attach_elastic(ref, _state(), neutron)
    _attach_elastic(rot, _state(_rotation()), neutron)
    assert rot.elastic_u_tensors_a2 == pytest.approx(ref.elastic_u_tensors_a2,
                                                     abs=1e-15)


def test_endf_rotates_only_when_card_6c_is_the_phonopy_cell(capsys):
    lattice = _state(_rotation())["primitive_lattice_ang"]
    M = _dw_frame_rotation((3.0, 4.0, 5.0, 90.0, 90.0, 90.0), lattice)
    np.testing.assert_allclose(M, _rotation().T, atol=1e-12)
    assert _dw_frame_rotation((6.0, 4.0, 5.0, 90.0, 90.0, 90.0), lattice) is None
    assert "NOTE" in capsys.readouterr().out
