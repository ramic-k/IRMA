"""SEC-1, GUI file-picker path: the ENDF form's freq_max detection worker
hands a user-picked phonopy.yaml to phonopy.load, whose unsafe YAML loader
executes ``!!python/`` tags at parse time. The file is untrusted by
construction (it comes from a file dialog), so reject_unsafe_phonopy_yaml
must fire BEFORE the load; the worker's except-handler then marshals the
rejection into the normal error dialog.

Requires tkinter and phonopy (skips without either). Uses a withdrawn
root, so no display is shown.
"""
import pytest

# importorskip FIRST -- a hard `import tkinter` at module top would error
# at collection on tkinter-less runners.
tk = pytest.importorskip("tkinter")
pytest.importorskip("phonopy")


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for tkinter")
    r.withdraw()
    yield r
    r.destroy()


def test_detect_freq_worker_guards_untrusted_yaml(root, tmp_path, monkeypatch):
    from irma.gui.app import IrmaApp
    app = IrmaApp(root)
    canary = tmp_path / "pwned"
    bad = tmp_path / "phonopy.yaml"
    bad.write_text(
        "phonopy:\n"
        "  version: 2.21.0\n"
        f'extra: !!python/object/apply:os.system ["touch {canary}"]\n'
        "force_constants:\n"
        "  format: full\n")
    calls = []
    monkeypatch.setattr(app, "_marshal",
                        lambda func, *a: calls.append((func, a)))
    # Run the worker synchronously (it normally runs on a daemon thread);
    # the guard must reject before phonopy parses anything.
    app._detect_freq_from_fc_worker(str(bad))
    assert not canary.exists(), "the !!python/ payload was executed"
    assert calls and calls[0][0] == app._detect_freq_from_fc_failed
    kind, detail = calls[0][1]
    assert kind == "error" and "!!python/" in detail
