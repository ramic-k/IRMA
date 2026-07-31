"""Engine-internal TDM fallback must route through the per-q DW hybrid.

When run_noncubic_sab_inprocess is called WITHOUT precomputed_thermal_mats
(the spectra forward model and the diagnostic CLI), the engine builds the
thermal-displacement matrices itself. A direct phonopy
ThermalDisplacementMatrices call with the GLOBAL tdm_freq_min_thz floor
would be wrong there: on a Gamma-containing (odd) mesh the global floor
collapses to the Gamma-tier cutoff for ALL q and drops the off-Gamma soft
modes in [1 ueV, GAMMA_ACOUSTIC_FLOOR] that the one-phonon/multiphonon mode
sums keep, making the DW tensor inconsistent with the mode sums. These
tests pin the wiring: the fallback must dispatch through
phonopy_io.compute_thermal_displacement_matrices, which applies the
explicit per-q sum on EVERY mesh (phonopy equivalence at ~1e-7 A^2 on
Gamma-free meshes is pinned in test_tdm_perq_floor).
"""
import os

import numpy as np
import pytest

pytest.importorskip("phonopy")

import irma.core.phonopy_io as pio  # noqa: E402
from irma.core.noncubic_engine import run_noncubic_sab_inprocess  # noqa: E402

_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mode2_euphonic_n1_validation", "graphite",
    "phonopy.yaml"))


def _tiny_run(tmp_path):
    """Smallest meaningful engine run on a Gamma-containing 3^3 mesh."""
    return run_noncubic_sab_inprocess(
        inelastic_mode=1,
        phonopy_yaml=_YAML,
        temperature_k=296.0,
        mesh=(3, 3, 3),                      # odd -> contains Gamma
        q_grid_ang_inv=np.array([1.0, 2.0, 4.0]),
        e_grid_mev=np.array([0.0, 40.0, 80.0, 120.0]),
        output_prefix=str(tmp_path / "tdm_fallback_unused"),
        sab_mass_ratio=11.898,
        num_directions=10,
        multiphonon_num_directions=10,
        jobs=1,
        multiphonon_max_order=1,             # one-phonon only: fastest
        write_output_files=False,
        # no precomputed_thermal_mats -> exercises the engine fallback
    )


def test_engine_tdm_fallback_routes_through_perq_hybrid(tmp_path, monkeypatch):
    calls = {}
    real_hybrid = pio.compute_thermal_displacement_matrices
    real_perq = pio.thermal_displacement_matrices_perq

    def spy_hybrid(mesh_data, temperature_k):
        calls["qpoints"] = np.asarray(mesh_data.qpoints, dtype=float)
        calls["temperature"] = float(temperature_k)
        return real_hybrid(mesh_data, temperature_k)

    def spy_perq(mesh_data, temperature_k):
        calls["perq"] = True
        calls.setdefault("perq_result", real_perq(mesh_data, temperature_k))
        return calls["perq_result"]

    monkeypatch.setattr(pio, "compute_thermal_displacement_matrices", spy_hybrid)
    monkeypatch.setattr(pio, "thermal_displacement_matrices_perq", spy_perq)

    result = _tiny_run(tmp_path)

    # The fallback went through the hybrid at the requested temperature...
    assert calls.get("temperature") == 296.0
    # ...on a mesh that really contains Gamma...
    assert bool(np.any(np.all(np.abs(calls["qpoints"]) < 1e-9, axis=1)))
    # ...and the hybrid dispatched to the per-q mode sum (the Gamma branch).
    assert calls.get("perq") is True

    # The U_ij the compute phase consumed (surfaced via elastic_state) is the
    # per-q hybrid result, not a separately-built global-floor TDM.
    U_used = np.asarray(
        result["elastic_state"]["thermal_displacement_matrices_ang2"])
    assert np.array_equal(U_used, np.asarray(calls["perq_result"]))
