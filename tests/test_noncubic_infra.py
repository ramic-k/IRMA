"""Noncubic engine infrastructure: native-thread pinning, the pool worker
initializer, the converter's temperature and mass-ratio checks, and the GUI
runner's macOS fork-safety variable. Headless (no Tk widgets).
"""
import os

import numpy as np
import pytest

from irma.core import noncubic_engine as ne


def test_pool_worker_init_forces_thread_env(monkeypatch):
    # the worker initializer overrides inherited values, not just setdefault
    for name in ne.NATIVE_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "8")
    ne._pool_worker_init()
    for name in ne.NATIVE_THREAD_ENV_VARS:
        assert os.environ[name] == "1", name


def test_parse_args_rejects_nonpositive_temperature():
    with pytest.raises(SystemExit):
        ne.parse_args(["--temperature", "0"])


def test_parse_args_rejects_nonpositive_mass_ratio():
    with pytest.raises(SystemExit):
        ne.parse_args(["--sab-mass-ratio", "-1"])


def test_infer_mass_ratio_rejects_nonpositive_override():
    with pytest.raises(ValueError):
        ne.infer_mass_ratio(np.array([12.0]), 0.0)


def test_convert_rejects_nonpositive_temperature():
    q = np.array([1.0, 2.0])
    e = np.array([1.0, 2.0])
    sqe = np.zeros((q.size, e.size))
    with pytest.raises(ValueError):
        ne.convert_sqe_to_asym_downscatter_sab(
            sqe, q, e, temperature_k=0.0, sigma_barn=5.0, mass_ratio=12.0
        )


def test_gui_runner_sets_fork_safety_env():
    import irma.gui.runner  # noqa: F401  (import is the trigger; headless-safe)

    assert os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") == "YES"
