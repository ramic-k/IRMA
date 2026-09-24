"""Regression tests for irma.cli.

* `python -m irma --gui` must print a friendly tkinter-installation hint
  (and exit nonzero) instead of dumping a bare ImportError traceback when
  the GUI module cannot be imported (e.g. tkinter missing).
* the `irma ncrystal` subcommand is advertised in --help and dispatches to
  irma.ncrystal.__main__ with the remaining argv.
"""

import builtins
import sys


from irma import cli


def _block_gui_import(monkeypatch):
    """Make importing irma.gui.app raise ImportError, as if tkinter is gone."""
    # Ensure the module is not already cached as importable.
    for name in list(sys.modules):
        if name == "irma.gui.app" or name == "irma.gui":
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "irma.gui.app" or name.startswith("irma.gui"):
            raise ImportError("No module named 'tkinter'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_gui_missing_tkinter_friendly_message(monkeypatch, capsys):
    _block_gui_import(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["irma", "--gui"])

    # main() returns the exit code (wrappers do sys.exit(main())); a missing
    # GUI -> nonzero (4), not the success (0) path.
    rc = cli.main()
    assert rc == 4

    err = capsys.readouterr().err
    assert "tkinter" in err.lower()
    assert "INSTALL.md" in err


def test_cli_help_mentions_ncrystal(capsys):
    assert cli.main(["--help"]) == 0
    assert "ncrystal" in capsys.readouterr().out


def test_cli_dispatches_ncrystal_subcommand(monkeypatch):
    import irma.ncrystal.__main__ as ncmain
    seen = {}

    def fake(argv=None):
        seen["argv"] = argv
        return 7

    monkeypatch.setattr(ncmain, "main", fake)
    assert cli.main(["ncrystal", "-o", "outdir", "cfg.yaml"]) == 7
    assert seen["argv"] == ["-o", "outdir", "cfg.yaml"]
