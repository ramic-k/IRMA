"""GUI-owned temp files must not survive the window.

The unlink callbacks are queued via after() and die with the Tk interpreter,
so every panel exposes an idempotent cleanup_temp_files() that app close
calls synchronously. The cleanup methods only touch plain attributes, so they
are exercised unbound on stand-in objects: no display is needed, but the
modules import tkinter.
"""
import types

import pytest

pytest.importorskip("tkinter")

from irma.gui.app import IrmaApp  # noqa: E402
from irma.gui.endf_form import EndfFormMixin  # noqa: E402
from irma.gui.ncrystal_panel import NCrystalPanel  # noqa: E402


def _tmpfile(tmp_path, name):
    p = tmp_path / name
    p.write_text("x")
    return types.SimpleNamespace(name=str(p)), p


def test_endf_form_cleanup_idempotent(tmp_path):
    tmp, path = _tmpfile(tmp_path, "deck.input")
    fake = types.SimpleNamespace(_tmp_input=tmp)
    EndfFormMixin.cleanup_temp_files(fake)
    assert not path.exists() and fake._tmp_input is None
    EndfFormMixin.cleanup_temp_files(fake)          # second call: no-op


def _run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "config.yaml").write_text("x")
    return d


def test_ncrystal_panel_cleanup_idempotent(tmp_path):
    d = _run_dir(tmp_path)
    panel = object.__new__(NCrystalPanel)       # no Tk: only plain attributes
    panel._tmpdir = str(d)
    panel.cleanup_temp_files()
    assert not d.exists() and panel._tmpdir is None
    panel.cleanup_temp_files()


def test_app_close_calls_every_cleanup_synchronously(tmp_path):
    calls = []
    fake = types.SimpleNamespace(
        runner=types.SimpleNamespace(is_running=False),
        root=types.SimpleNamespace(destroy=lambda: calls.append("destroy")),
        cleanup_temp_files=lambda: calls.append("endf"),
        ns_panel=types.SimpleNamespace(
            cleanup_temp_files=lambda: calls.append("ns")),
        ncrystal_panel=types.SimpleNamespace(
            cleanup_temp_files=lambda: calls.append("nc")),
        mlip_panel=types.SimpleNamespace(
            cleanup_temp_files=lambda: calls.append("mlip")),
    )
    IrmaApp._on_close(fake)
    assert calls == ["endf", "ns", "nc", "mlip", "destroy"]
