"""Deck-CLI message pins.

Three UX requirements:

- A directory passed as ``input_file`` must fail with a clean "is a
  directory, not a file" message (exit 1), not leak an
  ``IsADirectoryError`` traceback out of the deck parser's ``open()``.
- ``irma evaluate --help`` must print the usage text and exit 0 — asking
  for help is never an error.
- Unexpected ``OSError`` escapes from ``run_leapr`` get the clean
  "IRMA failed:" treatment (exit 3) instead of a traceback.

Pure stdlib -> runs in the bare gate.
"""
from irma import cli


# ---------------------------------------------------------------------------
# Directory passed as input_file -> clean preflight error, no traceback
# ---------------------------------------------------------------------------

def test_directory_input_clean_error(tmp_path, capsys):
    """`irma /tmp out.endf` (a tab-completion slip) must return 1 with a
    message naming the path, WITHOUT raising IsADirectoryError."""
    rc = cli.main([str(tmp_path), str(tmp_path / "out.endf")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a file" in err
    assert str(tmp_path) in err


def test_missing_input_still_reported(tmp_path, capsys):
    rc = cli.main([str(tmp_path / "nope.input"), str(tmp_path / "out.endf")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "input file not found" in err
    assert "nope.input" in err


# ---------------------------------------------------------------------------
# `irma evaluate --help` -> usage text, exit 0
# ---------------------------------------------------------------------------

def test_evaluate_help_prints_usage(capsys):
    rc = cli.main(["evaluate", "--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert captured.err == ""


# ---------------------------------------------------------------------------
# OSError backstop: filesystem failures out of run_leapr get exit 3, clean
# ---------------------------------------------------------------------------

def test_oserror_from_run_leapr_is_clean(tmp_path, capsys, monkeypatch):
    """Anything the preflights miss (file vanishes mid-run, permission lost)
    must surface as 'IRMA failed: ...' with exit 3, not a traceback."""
    import irma.core.engine as engine

    def boom(input_file, output_file):
        raise IsADirectoryError(21, "Is a directory", str(tmp_path))

    monkeypatch.setattr(engine, "run_leapr", boom)
    deck = tmp_path / "in.input"
    deck.write_text("leapr\n")
    rc = cli.main([str(deck), str(tmp_path / "out.endf")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "IRMA failed:" in err
    # OSError embeds the filename as its repr; on Windows that doubles
    # the backslashes, so compare against the de-escaped message
    assert str(tmp_path) in err.replace("\\\\", "\\")
