"""Regression tests for TokenReader card parsing.

This is the mechanism behind optional/backward-compatible card fields (e.g. the
Card 6b Bragg-edge grouping fields, the variable-length Card 6g). The stale-deck
breakages we hit were exactly here, so pin the behavior.
"""
import pytest

from irma.core.engine import TokenReader, CARD_END


def test_read_floats_fills_defaults_for_missing_trailing_fields():
    """A 4-token Card 6b read as 6 floats -> 4 values + 2 defaults (grouping OFF)."""
    r = TokenReader([1, 1, 0, 2, CARD_END])
    vals = r.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    assert vals == [1.0, 1.0, 0.0, 2.0, 0, 0]


def test_read_floats_reads_all_present_fields():
    """A 6-token Card 6b (grouping ON) reads all six."""
    r = TokenReader([1, 1, 0, 2, 20, 1.0, CARD_END])
    vals = r.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    assert vals == [1.0, 1.0, 0.0, 2.0, 20.0, 1.0]


def test_read_floats_consumes_card_end():
    """After a card is read (including its terminator), the next read starts on the
    following card."""
    r = TokenReader([1, 2, CARD_END, 7, 8, 9, CARD_END])
    first = r.read_floats(4, defaults=[0, 0, 0, 0])
    assert first == [1.0, 2.0, 0, 0]
    second = r.read_floats(3, defaults=[0, 0, 0])
    assert second == [7.0, 8.0, 9.0]


def test_read_card_floats_reads_variable_length_card():
    """Card 6g is variable length (2-4 fields); read_card_floats takes all of them."""
    r = TokenReader([5000, 200, 0, 0, 0, CARD_END])
    assert r.read_card_floats() == [5000.0, 200.0, 0.0, 0.0, 0.0]


def test_read_card_floats_three_field_form():
    r = TokenReader([5000, 200, 0, CARD_END])
    assert r.read_card_floats() == [5000.0, 200.0, 0.0]


def test_read_ints_with_defaults():
    """Card 7 (nalpha nbeta lat) parsed as ints with defaults."""
    r = TokenReader([200, 426, 1, CARD_END])
    assert r.read_ints(3, defaults=[0, 0, 0]) == [200, 426, 1]
    r2 = TokenReader([200, 426, CARD_END])
    assert r2.read_ints(3, defaults=[0, 0, 0]) == [200, 426, 0]




def test_fortran_d_exponents_tokenize_as_numbers():
    """NJOY-style decks legitimately use Fortran D-exponent notation
    (1.0d-5, 2D3); these used to tokenize as strings and fail every
    numeric read."""
    from irma.core.deck import _parse_line, CARD_END
    tokens = _parse_line("1.0d-5 2D3 -3.5d+2 0.25 /")
    assert tokens == [1.0e-5, 2000.0, -350.0, 0.25, CARD_END]


def test_non_numeric_d_tokens_stay_strings():
    from irma.core.deck import _parse_line
    tokens = _parse_line("deck d5 1d /")
    assert tokens[:3] == ["deck", "d5", "1d"]


def test_doubled_quote_escaping_in_strings():
    """Fortran '' escaping inside a quoted string is a literal quote; it
    used to truncate the string at the first inner quote."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("'it''s a graphite deck' /")
    assert tokens[0] == ("string", "it's a graphite deck")




def test_stray_text_on_card_is_discarded_not_leaked():
    """Leftover non-numeric tokens on a card are consumed with the record
    (Fortran semantics), never leaked into the next card's read."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("1 2 stray words") + _parse_line("7 8 /")
    r = TokenReader(tokens)
    assert r.read_floats(2, defaults=[0, 0]) == [1.0, 2.0]
    assert r.read_floats(2, defaults=[0, 0]) == [7.0, 8.0]


def test_read_card_floats_rejects_trailing_word():
    """A word in a numeric-coded field (method selector written 'numerical'
    instead of its code 0) used to be silently dropped, letting [5000, 200]
    pass a downstream 2/3/4-field count check with the method defaulted.
    It must now fail loudly (QA2-032)."""
    from irma.core.deck import _parse_line, DeckError
    r = TokenReader(_parse_line("5000 200 numerical /"))
    with pytest.raises(DeckError, match="numerical"):
        r.read_card_floats()


def test_read_card_floats_still_takes_all_numeric_fields():
    """The trailing-word guard must not disturb the all-numeric path."""
    r = TokenReader([5000, 200, 0, 1, CARD_END])
    assert r.read_card_floats() == [5000.0, 200.0, 0.0, 1.0]


def test_lone_slash_array_terminator_leaves_orphan_card_end():
    """Deck-format requirement pin (QA2-031): when an array's '/' sits ALONE
    on the line after the last data line, read_float_array consumes only the
    data line's LINE_END; the lone '/' is left as an orphan CARD_END that the
    next scalar read sees as an empty (all-default) card. This matches NJOY's
    list-directed read; native decks always put '/' on the last data line."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("1.0 2.0 3.0") + _parse_line("/") + _parse_line("5.0 6.0 /")
    r = TokenReader(tokens)
    arr = r.read_float_array(3)
    assert list(arr) == [1.0, 2.0, 3.0]
    # The orphan '/' card is read next as an empty card -> defaults, NOT
    # [5.0, 6.0]; that value pair is only reached by the following read.
    assert r.read_floats(2, defaults=[0, 0]) == [0, 0]
    assert r.read_floats(2, defaults=[0, 0]) == [5.0, 6.0]


def test_array_slash_on_last_data_line_aligns():
    """The supported form ('/' on the last data line) keeps cards aligned."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("1.0 2.0 3.0 /") + _parse_line("5.0 6.0 /")
    r = TokenReader(tokens)
    arr = r.read_float_array(3)
    assert list(arr) == [1.0, 2.0, 3.0]
    assert r.read_floats(2, defaults=[0, 0]) == [5.0, 6.0]


def test_star_comment_breaks_on_tab():
    """The '*'-comment token must break on a tab, matching the .split()
    semantics the numeric value-line path uses (QA2-033). The leading '*'
    is kept verbatim (expected comment cards begin '*...')."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("*hello\tworld")
    assert tokens[0] == ("string", "*hello")


def test_unterminated_doubled_quote_keeps_decoded_chars():
    """An open quote whose only inner '' is at end-of-line with no real
    closing quote falls into the no-close fallback. It must keep the chars
    already decoded (doubled '' -> single ') rather than re-slicing the raw
    remainder (QA2-035)."""
    from irma.core.deck import _parse_line
    tokens = _parse_line("'abc''")
    assert tokens[0] == ("string", "abc'")
