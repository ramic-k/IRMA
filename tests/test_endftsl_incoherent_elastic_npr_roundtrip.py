"""A generalized SEF LTHR=2 tape with npr > 1 round-trips through the ENDFTSL
converter back to the per-principal incoherent-elastic cross section: the
writer stores SB = per-principal x npr and the converter divides by MT4
B(6)=npr once.

Imports the ENDFTSL python package from the sibling plugin tree; skipped when
its dependencies (endf-parserpy through the reader) are unavailable.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_ENDFTSL_PY = Path(__file__).resolve().parents[1] / "ncrystal_plugin_ENDFTSL" / "python"
sys.path.insert(0, str(_ENDFTSL_PY))

reader = pytest.importorskip("ncrystal_plugin_ENDFTSL.reader")
physics = pytest.importorskip("ncrystal_plugin_ENDFTSL.physics")

from irma.core.endf_writer import write_endf_output          # noqa: E402

_TEMPR = [296.0, 500.0]
_DWPIX = [0.8, 1.1]
_H = {'Z': 1, 'A': 1, 'awr': 0.99917, 'b_coh': -3.7406, 'sigma_inc': 80.27,
      'sigma_coh': 1.7568, 'fraction': 1.0, 'dwpix': list(_DWPIX)}


def _write_eq25_tape(path, npr):
    ssm = np.full((2, 2, 2), 0.1)
    crystal_info = {
        'elastic_mode': 1, 'atom_types': [_H], 'nat': 1,
        'principal_atom_idx': 0, 'dc_atom_idx': None,
        'species_corr': None, 'F_species_per_temp': None,
        'bragg_dir_terms': None,
    }
    write_endf_output(
        str(path), 1, 1001.0, _H['awr'], 5.5, npr, 10,
        0, 0.0, 0.0, 0.0, 0, 2, 2, 1,
        np.array([0.1, 1.0]), np.array([0.0, 1.0]), ssm, None,
        np.array(_TEMPR), 2, np.array(_DWPIX), np.array(_DWPIX),
        np.array([320.0, 520.0]), np.array([320.0, 520.0]),
        [], 0, 0, 0, 1.0e-6,
        comments=None, crystal_info=crystal_info)


@pytest.mark.parametrize("npr", [1, 2, 4])
def test_generalized_sef_converter_recovers_per_principal(tmp_path, npr):
    per_principal = _H['sigma_inc'] * (
        (_H['sigma_coh'] + _H['sigma_inc']) / _H['sigma_inc'])
    tape = tmp_path / f"eq25_npr{npr}.endf"
    _write_eq25_tape(tape, npr)
    ev = reader.read_tsl(str(tape))
    assert ev.lthr == 2
    _, sb = physics.incoherent_msd(ev, 296.0)
    assert sb == pytest.approx(per_principal, rel=1e-6), (
        f"converter recovered {sb} b, expected the per-principal "
        f"{per_principal} b (npr={npr})")
