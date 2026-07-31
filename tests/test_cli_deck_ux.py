"""Deck-CLI and driver message-hygiene pins.

Four UX requirements:

- A directory passed as ``input_file`` must fail with a clean "is a
  directory, not a file" message (exit 1), not leak an
  ``IsADirectoryError`` traceback out of the deck parser's ``open()``.
- ``irma evaluate --help`` / ``irma run --help`` must print the usage
  text and exit 0 — asking for help is never an error.
- Unexpected ``OSError`` escapes from ``run_leapr`` get the clean
  "IRMA failed:" treatment (exit 3) instead of a traceback.
- driver.py's log severity tags (WARNING:/NOTE:) start at column 0 so
  ``grep '^WARNING'`` over a run log finds every warning, and the note
  tag is uniformly uppercase; the input reference documents that the
  FIRST temperature always supplies a detail block even when negative.

Pure stdlib + text assertions -> runs in the bare gate.
"""
import re
from pathlib import Path

from irma import cli

REPO = Path(__file__).resolve().parent.parent
DRIVER_SRC = (REPO / "irma" / "core" / "driver.py").read_text(encoding="utf-8")
INPUT_REFERENCE = (REPO / "docs" / "input-reference.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Directory passed as input_file -> clean preflight error, no traceback
# ---------------------------------------------------------------------------

def test_directory_input_clean_error(tmp_path, capsys):
    """`irma /tmp out.endf` (a tab-completion slip) must return 1 with a
    message naming the path, WITHOUT raising IsADirectoryError."""
    rc = cli.main([str(tmp_path), str(tmp_path / "out.endf")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "directory, not a file" in err
    assert str(tmp_path) in err


def test_directory_input_via_evaluate_subcommand(tmp_path, capsys):
    rc = cli.main(["evaluate", str(tmp_path), str(tmp_path / "out.endf")])
    assert rc == 1
    assert "directory, not a file" in capsys.readouterr().err


def test_missing_input_still_reported(tmp_path, capsys):
    """The isfile() tightening must not change the missing-file message."""
    rc = cli.main([str(tmp_path / "nope.input"), str(tmp_path / "out.endf")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "input file not found" in err
    assert "nope.input" in err


# ---------------------------------------------------------------------------
# `irma evaluate --help` / `irma run --help` -> usage text, exit 0
# ---------------------------------------------------------------------------

def test_evaluate_help_prints_usage(capsys):
    rc = cli.main(["evaluate", "--help"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert captured.err == ""


def test_run_dash_h_prints_usage(capsys):
    rc = cli.main(["run", "-h"])
    assert rc == 0
    assert "Usage:" in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# driver.py log message style: severity tags at column 0, uppercase
# ---------------------------------------------------------------------------

def test_driver_severity_tags_start_message_strings():
    """No WARNING:/NOTE: tag may be indented inside its message string,
    otherwise `grep '^WARNING'` over a run log misses it."""
    bad = re.findall(r'["\']\s+(?:WARNING|NOTE|Note):', DRIVER_SRC)
    assert not bad, (
        f"driver.py has indented severity tags {bad}; severity tags must "
        "begin the printed string (column 0)"
    )


def test_driver_note_tag_is_uppercase():
    """'Note:' (title case) was driver.py's one deviation from the repo-wide
    'NOTE:' convention; it must not come back."""
    assert "Note:" not in DRIVER_SRC, (
        "driver.py uses title-case 'Note:'; the repo convention is 'NOTE: '"
    )


# ---------------------------------------------------------------------------
# input-reference.md: first temperature always supplies a detail block
# ---------------------------------------------------------------------------

def test_doc_first_negative_temperature_exception():
    """The Card 10 section must state the LEAPR convention the parser
    (driver.py temperature loop) actually implements: the FIRST temperature
    is always followed by a detail block, even if written negative."""
    text = INPUT_REFERENCE.lower()
    assert "first" in text and "temperature always supplies a detail" in text, (
        "docs/input-reference.md must document that the first temperature "
        "always supplies a detail block, even when written negative"
    )
