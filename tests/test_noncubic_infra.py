"""Infrastructure guards for the noncubic engine and GUI runner.

Covers the concurrency / native-thread / CLI-validation findings:

* F15  -- the GUI runner sets OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES at import
          (setdefault) so a fork from the non-main GUI thread does not abort on
          macOS Cocoa.
* C8    -- the pool worker initializer FORCE-sets the native-thread env vars to
          "1" (overriding inherited values), while the parent-side limiter only
          setdefaults them.
* F17   -- the native-thread env vars are pinned when irma is imported (before numpy).
* F18   -- BrokenProcessPool is imported by name, not via the ``cf.process``
          attribute side-effect.
* C2    -- the converter CLI / conversion path validate temperature > 0 and
          sab mass ratio > 0 with clear errors.

All tests are headless: the GUI runner imports no tkinter at module level, and
no Tk widgets are instantiated here.
"""
import importlib
import inspect
import os

import numpy as np
import pytest

from irma.core import noncubic_engine as ne


# --------------------------------------------------------------------------- #
# F17 -- native-thread env pinned at import, before numpy
# --------------------------------------------------------------------------- #
def test_native_thread_env_pinned_at_import():
    # Importing irma (irma/__init__.py) sets every native-thread var before numpy.
    for name in ne.NATIVE_THREAD_ENV_VARS:
        assert os.environ.get(name) == "1", name


def test_limit_native_threads_setdefault_preserves_override(monkeypatch):
    # Parent-side limiter uses setdefault: a deliberate override survives.
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    ne.limit_native_threads_to_one()
    assert os.environ["OMP_NUM_THREADS"] == "8"


# --------------------------------------------------------------------------- #
# C8 -- worker initializer FORCE-sets thread env to 1 (overrides inherited)
# --------------------------------------------------------------------------- #
def test_pool_worker_init_forces_thread_env(monkeypatch):
    for name in ne.NATIVE_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "8")
    ne._pool_worker_init()
    for name in ne.NATIVE_THREAD_ENV_VARS:
        assert os.environ[name] == "1", name


# --------------------------------------------------------------------------- #
# F18 -- BrokenProcessPool imported by name, not via cf.process side-effect
# --------------------------------------------------------------------------- #
def test_broken_process_pool_imported_by_name():
    # The name is importable directly...
    from concurrent.futures.process import BrokenProcessPool  # noqa: F401

    # ...and the engine catches it by the imported name, not via cf.process.
    src = inspect.getsource(ne)
    assert "from concurrent.futures.process import BrokenProcessPool" in src
    assert "except BrokenProcessPool" in src
    # The except clause no longer dereferences cf.process (only the comment may).
    code_lines = [
        ln for ln in src.splitlines()
        if "cf.process" in ln and not ln.lstrip().startswith("#")
    ]
    assert code_lines == []


def test_pool_uses_worker_initializer():
    src = inspect.getsource(ne)
    assert "initializer=_pool_worker_init" in src


# --------------------------------------------------------------------------- #
# C2 -- converter CLI / conversion validate temperature and mass ratio
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_t", ["0", "-1", "-296.0"])
def test_parse_args_rejects_nonpositive_temperature(bad_t):
    with pytest.raises(SystemExit):
        ne.parse_args(["--temperature", bad_t])


@pytest.mark.parametrize("bad_m", ["0", "-1", "-11.9"])
def test_parse_args_rejects_nonpositive_mass_ratio(bad_m):
    with pytest.raises(SystemExit):
        ne.parse_args(["--sab-mass-ratio", bad_m])


def test_parse_args_accepts_positive_values():
    args = ne.parse_args(["--temperature", "296.0", "--sab-mass-ratio", "11.9"])
    assert args.temperature == pytest.approx(296.0)
    assert args.sab_mass_ratio == pytest.approx(11.9)


def test_parse_args_default_mass_ratio_is_none():
    # No --sab-mass-ratio supplied -> None, and the None guard must not fire.
    args = ne.parse_args([])
    assert args.sab_mass_ratio is None


@pytest.mark.parametrize("bad_m", [0.0, -1.0, -11.9])
def test_infer_mass_ratio_rejects_nonpositive_override(bad_m):
    with pytest.raises(ValueError):
        ne.infer_mass_ratio(np.array([12.0]), bad_m)


def test_infer_mass_ratio_accepts_positive_override():
    assert ne.infer_mass_ratio(np.array([12.0]), 11.9) == pytest.approx(11.9)


@pytest.mark.parametrize("bad_t", [0.0, -1.0])
def test_convert_rejects_nonpositive_temperature(bad_t):
    q = np.array([1.0, 2.0])
    e = np.array([1.0, 2.0])
    sqe = np.zeros((q.size, e.size))
    with pytest.raises(ValueError):
        ne.convert_sqe_to_asym_downscatter_sab(
            sqe, q, e, temperature_k=bad_t, sigma_barn=5.0, mass_ratio=12.0
        )


def test_convert_accepts_positive_temperature():
    q = np.array([1.0, 2.0])
    e = np.array([1.0, 2.0])
    sqe = np.zeros((q.size, e.size))
    alpha, beta, sab = ne.convert_sqe_to_asym_downscatter_sab(
        sqe, q, e, temperature_k=296.0, sigma_barn=5.0, mass_ratio=12.0
    )
    assert np.all(np.isfinite(alpha))
    assert np.all(np.isfinite(beta))


# --------------------------------------------------------------------------- #
# F15 -- GUI runner sets the macOS fork-safety env var at import (setdefault)
# --------------------------------------------------------------------------- #
def test_gui_runner_sets_fork_safety_env():
    import irma.gui.runner  # noqa: F401  (import is the trigger; headless-safe)

    assert os.environ.get("OBJC_DISABLE_INITIALIZE_FORK_SAFETY") == "YES"


def test_gui_runner_uses_setdefault_for_fork_safety(monkeypatch):
    # A pre-existing operator override must survive a fresh import.
    monkeypatch.setenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "NO")
    import irma.gui.runner as runner

    importlib.reload(runner)
    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "NO"
