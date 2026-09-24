#!/usr/bin/env python3
"""Derive an IRMA ``.input`` deck from an NJOY LEAPR deck.

WHY THIS EXISTS
---------------
Evaluated ``.leapr`` reference decks are *NJOY job streams*: a sequence of
module blocks (``reconr``/``broadr``/``leapr``/``thermr``/``acer``/``plotr`` …)
each introduced by its module name and terminated by the next module name or
``stop``. IRMA is not NJOY — it runs only the LEAPR scattering-law calculation
and expects a *standalone* deck whose very first card is the output unit
(``nout``), not a ``leapr`` module header.

Feeding a raw NJOY deck to IRMA is the thing we must NOT do. Instead this
tool performs the small, well-defined translation from the NJOY LEAPR block to
an IRMA deck:

  1. Locate the ``leapr`` module line and take the LEAPR block that follows it,
     stopping at the next NJOY module name or ``stop`` (so a full multi-module
     job stream yields only its LEAPR part — e.g. the aluminum deck).
  2. Reformat Card 1: NJOY writes the output unit on its own line right after
     ``leapr`` (``25``); IRMA wants it as a normal terminated card (``25 /``).
  3. Quote Card 2 as one title line, and copy every other card unchanged:
     Cards 3-9, the per-temperature
     detail blocks (continuous phonon spectrum, translational/oscillator data),
     the temperature cards (INCLUDING the negative-temperature "reuse the
     previous spectrum" cards, which IRMA now honors), and the MF1/MT451
     comment cards. The classic-path (iel<10) card grammar is identical between
     NJOY LEAPR and IRMA, so no field-level rewriting is needed.

The result is a faithful, runnable IRMA deck: same physics inputs, same
comment/provenance block, just packaged as an IRMA deck instead of an NJOY job
stream.

USAGE
-----
    python leapr_to_irma_input.py tsl-crystalline-graphite.leapr [out.input]

If the output path is omitted it is written next to the input with a
``.input`` extension.
"""
import os
import sys

# NJOY module names that delimit blocks within a job stream. Kept in sync with
# irma.core.deck._NJOY_MODULES; duplicated here so the tool has no import
# dependency on the package internals.
NJOY_MODULES = {
    'moder', 'reconr', 'broadr', 'unresr', 'heatr', 'thermr', 'groupr',
    'errorr', 'covr', 'acer', 'powr', 'wimsr', 'plotr', 'viewr', 'mixr',
    'dtfr', 'ccccr', 'matxsr', 'resxsr', 'purr', 'gaspr', 'leapr', 'stop',
}


def extract_leapr_block(lines):
    """Return the IRMA-deck lines derived from an NJOY deck's LEAPR block.

    Raises ValueError if no ``leapr`` block is present.
    """
    # 1. Find the 'leapr' module line.
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == 'leapr':
            start = i + 1
            break
    if start is None:
        raise ValueError("no 'leapr' module line found in deck")

    # 2. First non-blank line after 'leapr' is Card 1 (nout).
    j = start
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        raise ValueError("LEAPR block ended before Card 1 (nout)")
    nout_tok = lines[j].strip().split()[0].rstrip('/').strip()
    out = [f"{nout_tok} /\n"]
    j += 1

    # 2b. Card 2 is the title. Some decks (e.g. the H-in-H2O CAB deck) give it
    #     UNQUOTED with embedded spaces and a trailing "/ TITLE" annotation;
    #     IRMA reads the title as a single token, so an unquoted multi-word
    #     title would be truncated to its first word. Re-emit it quoted so the
    #     full title survives into MF1/MT451. Single-token or already-quoted
    #     titles are passed through unchanged.
    while j < len(lines) and not lines[j].strip():
        out.append(lines[j] if lines[j].endswith('\n') else lines[j] + '\n')
        j += 1
    if j < len(lines):
        out.append(_normalize_title(lines[j]))
        j += 1

    # 3. Copy the rest of the LEAPR block verbatim, stopping at the next module
    #    name or 'stop'. Quoted comment cards are never bare module names, so
    #    they are copied through; the trailing blank '/' comment terminator is
    #    preserved.
    while j < len(lines):
        if lines[j].strip().lower() in NJOY_MODULES:
            break
        out.append(lines[j] if lines[j].endswith('\n') else lines[j] + '\n')
        j += 1
    return out


def _normalize_title(line):
    """Return Card 2 as a single quoted title line.

    Leaves already-quoted titles untouched (they round-trip verbatim). For an
    unquoted title, strips the card terminator and any trailing "/ comment",
    then wraps the remaining text in single quotes. Embedded apostrophes are
    Fortran-doubled ("'" -> "''") so IRMA's quote scanner round-trips them
    instead of treating the first apostrophe as the closing quote and
    truncating the title.
    """
    s = line.strip()
    if s.startswith("'") or s.startswith('"'):
        return line if line.endswith('\n') else line + '\n'
    # Unquoted: content is everything up to the first '/' card terminator.
    content = s.split('/', 1)[0].strip()
    content = content.replace("'", "''")
    return f"'{content}'/\n"


def convert(in_path, out_path=None):
    with open(in_path, 'r') as f:
        lines = f.readlines()
    out_lines = extract_leapr_block(lines)
    if out_path is None:
        base, _ = os.path.splitext(in_path)
        out_path = base + '.input'
    with open(out_path, 'w') as f:
        f.writelines(out_lines)
    return out_path


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    in_path = argv[1]
    out_path = argv[2] if len(argv) > 2 else None
    written = convert(in_path, out_path)
    print(f"wrote {written}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
