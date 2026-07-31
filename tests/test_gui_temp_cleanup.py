"""GUI-owned temp files must not survive the window (review GUI-2).

The unlink callbacks are queued via after() and die with the Tk interpreter,
so every panel exposes an idempotent cleanup_temp_files() that app close
calls synchronously. Tk-free: the cleanup methods only touch plain
attributes, so they are exercised unbound on stand-in objects.
"""
import types

from irma.gui.app import IrmaApp
from irma.gui.endf_form import EndfFormMixin
from irma.gui.ncrystal_panel import NCrystalPanel
from irma.gui.ns_panel import NSPanel


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


def test_ncrystal_panel_cleanup_idempotent(tmp_path):
    tmp, path = _tmpfile(tmp_path, "cfg.yaml")
    fake = types.SimpleNamespace(_cfg_tmp=tmp)
    NCrystalPanel.cleanup_temp_files(fake)
    assert not path.exists() and fake._cfg_tmp is None
    NCrystalPanel.cleanup_temp_files(fake)


def test_ns_panel_cleanup_covers_cfg_and_maps(tmp_path):
    tmp, cfg_path = _tmpfile(tmp_path, "cfg.yaml")
    mp = tmp_path / "map.npz"
    mp.write_bytes(b"x")
    pending = tmp_path / "pending.npz"
    pending.write_bytes(b"x")
    fake = types.SimpleNamespace(
        _cfg_tmp=tmp, _map_path=str(mp), _pending_map=str(pending),
        _cleanup_cfg_tmp=lambda: NSPanel._cleanup_cfg_tmp(fake),
        _discard_map_file=NSPanel._discard_map_file)
    NSPanel.cleanup_temp_files(fake)
    assert not cfg_path.exists() and not mp.exists() and not pending.exists()
    assert fake._cfg_tmp is None and fake._map_path is None
    NSPanel.cleanup_temp_files(fake)                # idempotent


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
    )
    IrmaApp._on_close(fake)
    assert calls == ["endf", "ns", "nc", "destroy"]


def test_gui_map_uses_full_arch_q_min():
    """review GUI-MAP: the map command must not hardcode a 0.5 lower Q cut."""
    import inspect
    src = inspect.getsource(NSPanel._run_map)
    assert '"--q-min", "0.0"' in src
    assert '"--q-min", "0.5"' not in src
