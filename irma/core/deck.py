"""LEAPR-style input-deck parsing for IRMA.

NJOY free-format tokenization (with '/' card terminators), the TokenReader
with card/line-context DeckError reporting, and the per-temperature
detail-block reader. The card-by-card deck format reference is
``docs/input-reference.md`` ("Input deck reference" in the manual).
"""

import os
from math import isfinite

import numpy as np

# ============================================================================
CARD_END = 'CARD_END'  # Sentinel token marking end of an NJOY input card (/)
LINE_END = 'LINE_END'  # Sentinel for a line WITHOUT '/' (array continuation)
_RECORD_ENDS = (CARD_END, LINE_END)


class DeckError(ValueError):
    """A problem in the user's input deck, reported with card/line context.

    Raised instead of letting malformed input cascade into cryptic numerical
    errors (ZeroDivisionError, NaN conversions, silent garbage output). The
    CLI and GUI present the message without a traceback.
    """

# Known NJOY module names (used to detect end of LEAPR input block)
_NJOY_MODULES = {
    'moder', 'reconr', 'broadr', 'unresr', 'heatr', 'thermr', 'groupr',
    'errorr', 'covr', 'acer', 'powr', 'wimsr', 'plotr', 'viewr', 'mixr',
    'dtfr', 'ccccr', 'matxsr', 'resxsr', 'purr', 'gaspr', 'leapr', 'stop',
}


# ============================================================================
# Input parser
# ============================================================================
def parse_leapr_input(filename):
    """Parse a LEAPR input file (NJOY free-format style).

    Returns (tokens, lines, start, token_lines) where token_lines holds the
    1-based source line number of every token (for deck-error reporting).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the 'leapr' line
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped == 'leapr':
            start = i + 1
            break

    # Collect all tokens from remaining lines, stopping at module names or 'stop'
    tokens = []
    token_lines = []

    i = start
    cards_seen = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for module name or stop (end of LEAPR input block).
        # Cards 1-2 (nout, title) are exempt: an unquoted title that
        # happens to be a module word must not truncate the deck.
        if line.lower() in _NJOY_MODULES and cards_seen >= 2:
            break

        i += 1
        tokens_in_line = _parse_line(line)
        cards_seen += sum(1 for t in tokens_in_line if t == CARD_END)
        tokens.extend(tokens_in_line)
        token_lines.extend([i] * len(tokens_in_line))   # i has been incremented, so it is the 1-based source line number

    return tokens, lines, start, token_lines


def _parse_line(line):
    """Parse a single NJOY free-format line into tokens.

    In NJOY free-format input, '/' terminates the current card.
    Values not provided before '/' take their defaults.
    A CARD_END sentinel is inserted at the end of every line to match
    Fortran's record-based I/O (each read() consumes one complete line).
    """
    stripped = line.strip()
    if not stripped:
        return []

    # Check if it's a quoted string line (NJOY supports ' and " as delimiters)
    if stripped.startswith("'") or stripped.startswith('"'):
        quote_char = stripped[0]
        # Scan for the closing quote, honoring Fortran doubled-quote
        # escaping ('' inside a '-quoted string is a literal quote).
        chars = []
        close_idx = -1
        j = 1
        while j < len(stripped):
            if stripped[j] == quote_char:
                if j + 1 < len(stripped) and stripped[j + 1] == quote_char:
                    chars.append(quote_char)
                    j += 2
                    continue
                close_idx = j
                break
            chars.append(stripped[j])
            j += 1
        if close_idx < 0:
            # No closing quote found: keep the chars already decoded (with
            # doubled-quote escapes resolved) so the result is consistent
            # whether or not a closing quote was located. Re-slicing
            # stripped[1:] here would have discarded the escape processing.
            pass
        text = ''.join(chars)
        result = [('string', text)]
        result.append(CARD_END)
        return result

    # Lines starting with * are comment/title lines in NJOY.
    # Fortran's list-directed `read(nsysi,*) text` reads only the first
    # free-format token (terminated by space, comma, or /).
    # The rest of the line is discarded.
    if stripped.startswith('*'):
        token = ''
        for ch in stripped:
            # Break on the same whitespace/separators the numeric value-line
            # path (.split()) treats as delimiters — the tab included, so the
            # two tokenizers can never disagree. The leading '*' is kept verbatim:
            # reference-deck comment cards begin '*...' and NJOY preserves it.
            if ch in (' ', '\t', ',', '/'):
                break
            token += ch
        result = [('string', token)]
        result.append(CARD_END)
        return result

    # Check for / card terminator
    has_slash = '/' in stripped
    if has_slash:
        idx = stripped.index('/')
        stripped = stripped[:idx]

    # Split on whitespace and commas
    parts = stripped.replace(',', ' ').split()
    tokens = []
    for p in parts:
        if '_' in p:
            # Python's int()/float() accept underscore digit grouping
            # (1_000), which is not Fortran numeric syntax; keep the raw
            # token so the card read fails loudly instead of silently
            # admitting it as a number.
            tokens.append(p)
            continue
        try:
            if '.' not in p and 'e' not in p.lower() and 'd' not in p.lower():
                tokens.append(int(p))
            else:
                # Fortran D-exponent notation (1.0d-5, 2D3) is legitimate in
                # NJOY-style decks: map d/D to e for the float parse. A token
                # that is not a number either way falls through to a string.
                val = float(p.lower().replace('d', 'e'))
                if not isfinite(val):
                    # An overflowing exponent (1e999) must not silently
                    # inject inf into the numeric stream; keep the raw
                    # token so the card read fails loudly.
                    tokens.append(p)
                else:
                    tokens.append(val)
        except ValueError:
            tokens.append(p)

    # End-of-record sentinel: CARD_END for a '/'-terminated card, LINE_END for
    # a bare continuation line (multi-line arrays). Card reads treat both as
    # the record boundary (Fortran record-based I/O); array reads skip
    # LINE_END but treat CARD_END as the array's hard terminator, so a value
    # count that disagrees with the deck is reported at the right card.
    tokens.append(CARD_END if has_slash else LINE_END)

    return tokens


class TokenReader:
    """Sequential reader for parsed tokens with NJOY card boundary support.

    NJOY free-format input uses '/' to terminate cards. When a card provides
    fewer values than expected, the remaining values take their defaults.
    CARD_END sentinels in the token stream mark these boundaries.
    """

    def __init__(self, tokens, token_lines=None, filename=None, raw_lines=None):
        self.tokens = tokens
        self.pos = 0
        self.token_lines = token_lines
        self.raw_lines = raw_lines
        self.filename = filename
        self._card = None

    def card(self, label):
        """Set the current card label used in deck-error messages."""
        self._card = label
        self._card_line = self._line()   # where this card starts
        return self

    def _line(self):
        """Source line of the current (or last) token, if line info exists."""
        if not self.token_lines:
            return None
        idx = min(self.pos, len(self.token_lines) - 1)
        return self.token_lines[idx] if self.token_lines else None

    def _fail(self, msg, line=None):
        """Raise a DeckError carrying the current card and input line."""
        where = f" while reading {self._card}" if self._card else ""
        if line is None:
            line = self._line()
        at = f" (input line {line}" if line is not None else ""
        if at and self.filename:
            at += f" of {os.path.basename(self.filename)}"
        at += ")" if at else ""
        raise DeckError(f"{msg}{where}{at}")

    def require(self, cond, msg):
        """Semantic deck validation tied to the current card context.

        Reports the line where the card STARTED (the reader has already
        consumed the card by the time semantic checks run).
        """
        if not cond:
            self._fail(msg, line=getattr(self, "_card_line", None))

    @staticmethod
    def _token_repr(t):
        """Readable form of a token for error messages."""
        return repr(t[1]) if isinstance(t, tuple) and t[0] == 'string' else repr(t)

    def _consume_card_end(self):
        """Consume remaining tokens on the current card, including CARD_END.

        Matches Fortran's record-based I/O where each read() consumes one
        complete line. Any unread numeric values on the card are discarded.
        """
        while self.pos < len(self.tokens):
            if self.tokens[self.pos] in _RECORD_ENDS:
                self.pos += 1
                return
            # Skip every remaining token on this record, numeric or not:
            # Fortran record-based I/O never re-reads leftover line content,
            # and a leaked stray token would misattribute the next card's
            # error.
            self.pos += 1

    def read_ints(self, n, defaults=None):
        """Read up to n integers from the current card, with defaults.

        Stops at CARD_END and consumes it. Unread values get defaults.
        Raises DeckError if the input is exhausted (no card at all) or a
        non-numeric token appears where a number is expected.
        """
        if defaults is None:
            defaults = [0] * n
        if len(defaults) < n:
            raise ValueError(
                f"defaults list has {len(defaults)} entries but {n} "
                f"values were requested")
        if self.pos >= len(self.tokens):
            self._fail("input ended before this card")
        result = list(defaults[:n])
        for i in range(n):
            if self.pos < len(self.tokens) and self.tokens[self.pos] not in _RECORD_ENDS:
                t = self.tokens[self.pos]
                if isinstance(t, (int, float)):
                    if isinstance(t, float) and not t.is_integer():
                        # int() would silently truncate 1.9 -> 1 on an
                        # integer-coded LEAPR field, hiding a malformed
                        # deck. Exactly-integral floats (200.0) stay
                        # accepted for NJOY list-directed compatibility.
                        self._fail(f"expected an integer for field {i+1}, "
                                   f"got the non-integral value {t!r}")
                    result[i] = int(t)
                    self.pos += 1
                else:
                    self._fail(f"expected a number for field {i+1}, "
                               f"got {self._token_repr(t)}")
            else:
                break
        self._consume_card_end()
        return result

    def to_int(self, value, what):
        """Coerce a float read by read_floats/read_card_floats to int.

        Integer-coded fields read through the float path must not be
        silently truncated: 1.9 is a malformed deck, 200.0 is fine.
        """
        f = float(value)
        if not f.is_integer():
            self._fail(f"{what} must be an integer, "
                       f"got the non-integral value {value!r}")
        return int(f)

    def read_floats(self, n, defaults=None):
        """Read up to n floats from the current card, with defaults.

        Stops at CARD_END and consumes it. Unread values get defaults.
        Raises DeckError if the input is exhausted (no card at all) or a
        non-numeric token appears where a number is expected.
        """
        if defaults is None:
            defaults = [0.0] * n
        if len(defaults) < n:
            raise ValueError(
                f"defaults list has {len(defaults)} entries but {n} "
                f"values were requested")
        if self.pos >= len(self.tokens):
            self._fail("input ended before this card")
        result = list(defaults[:n])
        for i in range(n):
            if self.pos < len(self.tokens) and self.tokens[self.pos] not in _RECORD_ENDS:
                t = self.tokens[self.pos]
                if isinstance(t, (int, float)):
                    result[i] = float(t)
                    self.pos += 1
                else:
                    self._fail(f"expected a number for field {i+1}, "
                               f"got {self._token_repr(t)}")
            else:
                break
        self._consume_card_end()
        return result

    def read_card_floats(self):
        """Read all numeric values remaining on the current card.

        A non-numeric token where a numeric field is expected (e.g. the
        method selector written as the word 'numerical' instead of its code
        0 on a variable-length card) is a malformed deck: it is reported
        here rather than silently dropped by _consume_card_end, which would
        let the surviving numeric values pass a downstream field-count check
        with a field defaulted away.
        """
        result = []
        while self.pos < len(self.tokens) and self.tokens[self.pos] not in _RECORD_ENDS:
            t = self.tokens[self.pos]
            if not isinstance(t, (int, float)):
                self._fail(f"expected a number (field {len(result)+1}), "
                           f"got {self._token_repr(t)} — numeric-coded fields "
                           f"must use their codes, not words")
            result.append(float(t))
            self.pos += 1
        self._consume_card_end()
        return result

    def peek_card_floats(self):
        """Peek numeric values remaining on the current card.

        Non-consuming lookahead: a public probing API for callers that need
        to inspect a variable-length card's field count before deciding how
        to read it (no production caller yet; exercised by the unit tests).
        """
        result = []
        pos = self.pos
        while pos < len(self.tokens) and self.tokens[pos] not in _RECORD_ENDS:
            t = self.tokens[pos]
            if not isinstance(t, (int, float)):
                break
            result.append(float(t))
            pos += 1
        return result

    def peek_token(self):
        """Non-consuming peek at the next raw token (string or number), or None
        at end of input. Used to detect an optional free-form card by keyword."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def read_card_tokens(self):
        """Read all raw tokens remaining on the current card (strings AND numbers),
        consuming the card terminator. This serves free-form key=value cards that
        mix a keyword/model name with numeric fields (e.g. the optional extinction
        card) -- cases the numeric ``read_floats`` path would reject."""
        if self.pos >= len(self.tokens):
            self._fail("input ended before this card")
        result = []
        while self.pos < len(self.tokens) and self.tokens[self.pos] not in _RECORD_ENDS:
            result.append(self.tokens[self.pos])
            self.pos += 1
        self._consume_card_end()
        return result

    def read_float_array(self, n):
        """Read exactly n floats (possibly spanning multiple lines).

        Skips CARD_END markers between lines since arrays can span multiple
        records. Consumes the trailing CARD_END after the last value.

        Deck-format requirement: the array's '/' terminator must sit on the
        last DATA line (after the n-th value), matching every native NJOY
        deck. If the n-th value ends a bare continuation line and the '/'
        is placed alone on the FOLLOWING line, _consume_card_end consumes
        only that continuation's LINE_END; the lone '/' is left as an
        orphan CARD_END that the next scalar read sees as an empty card
        (its fields take defaults), shifting every later card by one. This
        matches NJOY's list-directed read, which also rolls a lone '/' into
        the next read. (Pinned by test_lone_slash_array_terminator_*.)
        """
        result = []
        for _ in range(n):
            # LINE_END = continuation line inside a multi-line array (or a
            # blank line): skip. CARD_END = the '/' terminator: hitting it
            # mid-array means the deck supplies fewer values than the count
            # card promised; hitting it FIRST means a stray empty '/' card sits
            # where the array should start, which is reported as a deck error
            # rather than silently consuming values from the next card.
            while self.pos < len(self.tokens) and self.tokens[self.pos] == LINE_END:
                self.pos += 1
            if self.pos >= len(self.tokens):
                self._fail(f"input ended after {len(result)} of {n} "
                           f"expected values")
            t = self.tokens[self.pos]
            if t == CARD_END and not result:
                self._fail("an empty '/' card appeared where this array "
                           "was expected — remove the stray terminator")
            if t == CARD_END:
                self._fail(f"the '/' terminator appeared after {len(result)} "
                           f"of {n} expected values — check the count given "
                           f"on the preceding card")
            if not isinstance(t, (int, float)):
                self._fail(f"expected a number (value {len(result)+1} of {n}), "
                           f"got {self._token_repr(t)}")
            result.append(float(t))
            self.pos += 1
        self._consume_card_end()
        return np.array(result)

    def read_string(self, allow_numeric=False):
        """Read a quoted-string token. Consumes trailing CARD_END if present.

        By default a numeric token or a bare record end where a string
        card is expected is a malformed deck (typically a missing path
        card whose slot was taken by the next card's numbers) and fails
        loudly instead of being stringified. Free-text cards (the deck
        title) pass allow_numeric=True: a numeric token is stringified
        and an empty card reads as ''.
        """
        if self.pos >= len(self.tokens):
            self._fail("input ended where a quoted string card was expected")
        t = self.tokens[self.pos]
        if isinstance(t, (int, float)) and not allow_numeric:
            self._fail(f"expected a quoted string, got the number {t!r} "
                       f"(missing string card?)")
        if t in _RECORD_ENDS:
            if allow_numeric:
                self._consume_card_end()
                return ''
            self._fail("expected a quoted string, found an empty card "
                       "(write '' for an intentionally empty string)")
        self.pos += 1
        if isinstance(t, tuple) and t[0] == 'string':
            self._consume_card_end()
            return t[1]
        # An UNQUOTED string card: if the raw source line shows the token was
        # truncated by a glued '/' (e.g. `dir/file.yaml` tokenizes to just `dir`,
        # because an unquoted '/' ends the card), the user almost certainly meant
        # a path -- tell them to quote it instead of silently using the fragment.
        if not allow_numeric and self.raw_lines and self.token_lines:
            ln = self.token_lines[min(self.pos - 1, len(self.token_lines) - 1)]
            if ln is not None and 1 <= ln <= len(self.raw_lines):
                raw = self.raw_lines[ln - 1].strip()
                ts = str(t)
                # Only flag a TRUNCATED path: the token glued to '/' with MORE
                # non-space content after it (e.g. `dir/file.yaml`). A bare
                # filename followed by the legal glued terminator (`file.yaml/`)
                # or `file.yaml/ comment` is NOT truncated -- nothing path-like
                # follows the slash.
                if (raw.startswith(ts) and len(raw) > len(ts) + 1
                        and raw[len(ts)] == '/'
                        and not raw[len(ts) + 1].isspace()):
                    self._fail(
                        f"path card looks truncated at '/': read {ts!r} from the "
                        f"line {raw!r}. An unquoted '/' ends the card -- quote the "
                        f"whole path, e.g. '{raw}'")
        self._consume_card_end()
        return str(t)

    def read_comment_strings(self):
        """Read comment string cards until a bare '/' (CARD_END with no data).

        Returns a list of comment strings.

        Handles both quoted string cards (e.g., ' text '/) and unquoted
        cards (e.g., 0/). In Fortran, all are read as character strings
        via read(nsysi,*) text.
        """
        comments = []
        while self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            if isinstance(t, tuple) and t[0] == 'string':
                self.pos += 1
                self._consume_card_end()
                comments.append(t[1])
            elif t == LINE_END:
                self.pos += 1
            elif t == CARD_END:
                # Bare '/' - end of comment section
                self.pos += 1
                break
            else:
                # Non-string token (e.g., numeric '0/')
                # Fortran reads these as character strings too
                val = str(t)
                self.pos += 1
                self._consume_card_end()
                comments.append(val)
        return comments


def _read_temperature_detail_cards(reader, nsk, ncold):
    """Read the legacy LEAPR temperature-detail block for one temperature."""
    reader.card("Card 11 (delta ni — continuous-spectrum grid)")
    fvals = reader.read_floats(2)
    delta1 = fvals[0]
    ni = reader.to_int(fvals[1], "ni")
    reader.require(delta1 > 0.0, f"delta (spectrum spacing, eV) must be > 0, "
                                 f"got {delta1:g}")
    reader.require(ni >= 2, f"ni (number of spectrum points) must be >= 2, "
                            f"got {ni}")
    reader.card(f"Card 12 ({ni} phonon-spectrum rho values)")
    p1 = reader.read_float_array(ni)
    reader.require(np.all(p1 >= 0.0), "rho values must be >= 0")
    reader.require(np.any(p1 > 0.0), "rho values are all zero")
    np1 = ni

    reader.card("Card 13 (twt c tbeta)")
    fvals = reader.read_floats(3)
    twt = fvals[0]
    c_diff = fvals[1]
    tbeta = fvals[2]
    reader.require(twt >= 0.0, f"twt (translational weight) must be >= 0, got {twt:g}")
    reader.require(tbeta > 0.0, f"tbeta (continuous weight) must be > 0, got {tbeta:g}")

    reader.card("Card 14 (nd — discrete oscillator count)")
    nd = reader.read_ints(1)[0]
    reader.require(nd >= 0, f"nd must be >= 0, got {nd}")
    bdel = None
    adel = None
    if nd > 0:
        reader.card(f"Card 15 ({nd} oscillator energies)")
        bdel = reader.read_float_array(nd)
        reader.require(bool(np.all(bdel > 0.0)),
                       "oscillator energies must be > 0 (a zero energy "
                       "makes the discrete-oscillator kernel divide by zero)")
        reader.card(f"Card 16 ({nd} oscillator weights)")
        adel = reader.read_float_array(nd)
        reader.require(bool(np.all(adel >= 0.0)),
                       "oscillator weights must be >= 0")

    ska = None
    nka = 0
    dka = 0.0
    if nsk > 0 or ncold > 0:
        reader.card("Card 17 (nka dka — Skold S(kappa) grid)")
        fvals = reader.read_floats(2)
        nka = reader.to_int(fvals[0], "nka")
        dka = fvals[1]
        reader.require(nka >= 1, f"nka must be >= 1, got {nka}")
        reader.require(dka > 0.0, f"dka must be > 0, got {dka:g}")
        reader.card(f"Card 18 ({nka} S(kappa) values)")
        ska = reader.read_float_array(nka)
        reader.require(bool(np.all(ska >= 0.0)),
                       "Skold S(kappa) values must be >= 0 (a negative static "
                       "structure factor is unphysical and corrupts the Skold "
                       "coherent kernel, which divides alpha by S(kappa))")

    cfrac = 0.0
    if nsk > 0:
        reader.card("Card 19 (cfrac — coherent fraction)")
        cfrac = reader.read_floats(1)[0]

    return (
        delta1, np1, p1, twt, c_diff, tbeta,
        nd, bdel, adel, ska, nka, dka, cfrac,
    )
