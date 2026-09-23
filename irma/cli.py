"""Command-line interface for IRMA.

Four capabilities behind one ``irma`` entry point:

    irma <deck> <out.endf>          ENDF/TSL evaluation (legacy, unchanged)
    irma evaluate <deck> <out.endf> same, explicit (alias: run)
    irma spectra vision|indirect|direct ...   neutron-scattering forward spectra
    irma spectra run <config> -o out.csv
    irma ncrystal -o <outdir> config.yaml     NCrystal .irmapack export
    irma --gui | --version

If the first argument is not a subcommand keyword, it is the deck path of
the 2-positional form ``irma deck out.endf``.

``main(argv=None)`` returns a process exit code; the console-script and
``python -m irma`` wrappers both ``sys.exit(main())``.
"""

import os
import sys


def _print_help():
    """Print the top-level CLI usage text."""
    print("IRMA — (In)elastic Representation of Materials "
          "As S(alpha,beta) evaluations")
    print()
    print("Usage:")
    print("  irma <input_file> <output_file>            "
          "Run ENDF/TSL evaluation from an input deck")
    print("  irma evaluate <input_file> <output_file>   "
          "Same, explicit (alias: run)")
    print("  irma spectra vision|indirect|direct ...    "
          "Neutron-scattering forward spectrum")
    print("  irma spectra run <config> -o out.csv       "
          "Run a spectra config file")
    print("  irma ncrystal -o <outdir> config.yaml      "
          "Bake NCrystal .irmapack set (alias: python -m irma.ncrystal)")
    print("  irma mlip build <structure> -o <outdir>    "
          "MLIP phonon model bundle (no DFT needed)")
    print("  irma mlip emit <bundle> --to endf,...      "
          "Generate IRMA inputs from a bundle")
    print("  irma mlip validate <bundle>                "
          "Check a bundle end to end")
    print("  irma --gui                                 "
          "Launch graphical interface")
    print("  irma --version                             "
          "Show version")


def _launch_gui():
    """Start the Tk GUI, with a clear message if tkinter is unavailable."""
    try:
        from irma.gui.app import main as gui_main
    except ImportError as exc:
        print(f"\nThe GUI could not be started ({exc}); it needs tkinter, "
              "see the Tkinter section of INSTALL.md.", file=sys.stderr)
        return 4
    gui_main()
    return 0


def validate_output_path(path):
    """Pre-flight an output file path BEFORE a long compute (shared by the deck
    and spectra CLIs). Catches the common fail-late cases: the target is an
    existing directory, the parent directory is missing, or the parent is not
    writable. Returns an error message on failure, else None (each caller picks
    its own exit code / stream)."""
    p = os.path.abspath(str(path))
    if os.path.isdir(p):
        return f"output path is a directory, not a file: {p}"
    if os.path.exists(p) and not os.access(p, os.W_OK):
        # An existing read-only target otherwise fails only at the
        # post-compute open (review S10).
        return f"output file exists and is not writable: {p}"
    parent = os.path.dirname(p)
    if not os.path.isdir(parent):
        return f"output directory does not exist: {parent}"
    if not os.access(parent, os.W_OK):
        return f"output directory is not writable: {parent}"
    return None


def _run_deck(args):
    """Legacy 2-positional ENDF/TSL evaluation (run_leapr), unchanged."""
    # `irma evaluate --help` / `irma run --help` is the natural discovery
    # command; asking for help must never be answered with an error.
    if args and args[0] in ("-h", "--help"):
        _print_help()
        return 0
    if len(args) < 2:
        print("Error: both input_file and output_file are required.", file=sys.stderr)
        print("Usage: irma <input_file> <output_file>", file=sys.stderr)
        return 1
    input_file, output_file = args[0], args[1]
    if not os.path.isfile(input_file):
        print(f"Error: input file not found or not a file: {input_file}",
              file=sys.stderr)
        return 1
    # Pre-validate the output path BEFORE the (multi-minute) compute, so a typo'd
    # output path / directory target / read-only parent fails fast instead of only
    # after the whole run finishes.
    err = validate_output_path(output_file)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    from irma.core.engine import run_leapr, DeckError
    try:
        run_leapr(input_file, output_file)
    except DeckError as exc:
        print(f"\nInput deck error:\n  {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError, ValueError) as exc:
        # User-meaningful failures (phonopy model loading, missing/unreadable
        # files, semantic problems) get a clean message; a traceback is
        # reserved for genuinely unexpected errors. OSError covers the
        # filesystem backstops (FileNotFoundError, IsADirectoryError,
        # PermissionError) for anything the preflights above did not catch.
        print(f"\nIRMA failed: {exc}", file=sys.stderr)
        return 3
    return 0


def main(argv=None):
    """Console entry point: dispatch to help, GUI, spectra, ncrystal, or a
    deck evaluation depending on the arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        _print_help()
        return 0
    if argv[0] == "--version":
        from irma import __version__
        print(f"IRMA v{__version__}")
        return 0
    if argv[0] in ("--help", "-h"):
        _print_help()
        return 0
    if argv[0] == "--gui":
        return _launch_gui()
    if argv[0] == "spectra":
        from irma.spectra.cli import main as spectra_main
        return spectra_main(argv[1:])
    if argv[0] == "mlip":
        from irma.mlip.cli import main as mlip_main
        return mlip_main(argv[1:])
    if argv[0] == "ncrystal":
        # Same code path as `python -m irma.ncrystal` (which keeps working);
        # surfaced here so all three capabilities live behind one command.
        from irma.ncrystal.__main__ import main as ncrystal_main
        return ncrystal_main(argv[1:])
    if argv[0] in ("evaluate", "run"):
        return _run_deck(argv[1:])

    tok = argv[0]
    if tok.startswith("-") and not os.path.exists(tok):
        print(f"Error: unknown option {tok!r}", file=sys.stderr)
        _print_help()
        return 1
    return _run_deck(argv)


if __name__ == "__main__":
    sys.exit(main())
