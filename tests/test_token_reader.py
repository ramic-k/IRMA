"""TokenReader card parsing: defaults for missing trailing fields (the optional
Card 6b Bragg-edge grouping fields, the 2- or 3-field Card 6g), variable-length
cards, Fortran number and string forms, and card alignment.
"""
import pytest

from irma.core.engine import TokenReader, CARD_END


@pytest.mark.parametrize("tokens, want", [
    ([1, 1, 0, 2], [1.0, 1.0, 0.0, 2.0, 0, 0]),                # Card 6b, grouping off
    ([1, 1, 0, 2, 20, 1.0], [1.0, 1.0, 0.0, 2.0, 20.0, 1.0]),  # grouping on
])
def test_read_floats_fills_defaults_for_missing_trailing_fields(tokens, want):
    r = TokenReader(tokens + [CARD_END])
    assert r.read_floats(6, defaults=[0, 0, 0, 0, 0, 0]) == want


def test_read_floats_consumes_card_end():
    """After a card is read (including its terminator), the next read starts on the
    following card."""
    r = TokenReader([1, 2, CARD_END, 7, 8, 9, CARD_END])
    first = r.read_floats(4, defaults=[0, 0, 0, 0])
    assert first == [1.0, 2.0, 0, 0]
    second = r.read_floats(3, defaults=[0, 0, 0])
    assert second == [7.0, 8.0, 9.0]


@pytest.mark.parametrize("tokens", [[5000, 200, 0], [5000, 200, 0, 1],
                                    [5000, 200, 0, 0, 0]])
def test_read_card_floats_reads_variable_length_card(tokens):
    """read_card_floats takes every field on the card; the caller checks the
    count (Card 6g allows 2 or 3)."""
    r = TokenReader(tokens + [CARD_END])
    assert r.read_card_floats() == [float(t) for t in tokens]


def test_read_ints_with_defaults():
    """Card 7 (nalpha nbeta lat) parsed as ints with defaults."""
    r = TokenReader([200, 426, 1, CARD_END])
    assert r.read_ints(3, defaults=[0, 0, 0]) == [200, 426, 1]
    r2 = TokenReader([200, 426, CARD_END])
    assert r2.read_ints(3, defaults=[0, 0, 0]) == [200, 426, 0]


def test_fortran_d_exponents_tokenize_as_numbers():
    """NJOY-style decks use Fortran D-exponent notation (1.0d-5, 2D3); these
    tokenize as numbers, not strings."""
    from irma.core.deck import _parse_line, CARD_END
    tokens = _parse_line("1.0d-5 2D3 -3.5d+2 0.25 /")
    assert tokens == [1.0e-5, 2000.0, -350.0, 0.25, CARD_END]


def test_non_numeric_d_tokens_stay_strings():
    from irma.core.deck import _parse_line
    tokens = _parse_line("deck d5 1d /")
    assert tokens[:3] == ["deck", "d5", "1d"]


def test_doubled_quote_escaping_in_strings():
    """Fortran '' escaping inside a quoted string is a literal quote and does
    not end the string."""
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
    """A word in a numeric field ('numerical' where a code belongs) raises
    instead of being dropped, which would let [5000, 200] pass a later
    field-count check."""
    from irma.core.deck import _parse_line, DeckError
    r = TokenReader(_parse_line("5000 200 numerical /"))
    with pytest.raises(DeckError, match="numerical"):
        r.read_card_floats()


def test_lone_slash_array_terminator_leaves_orphan_card_end():
    """Deck-format requirement: when an array's '/' sits ALONE
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


def test_fortran_null_value_is_refused_not_shifted():
    # NJOY keeps a ',,' item at its default; dropping it would shift fields
    from irma.core.deck import DeckError, _parse_line
    with pytest.raises(DeckError, match="null value"):
        _parse_line("1.0,,3.0 /")
    with pytest.raises(DeckError, match="null value"):
        _parse_line(", 2 3 /")
    assert _parse_line("1, 2, /")[:2] == [1, 2]          # trailing comma is fine
    assert _parse_line("'a,, title' /")[0] == ("string", "a,, title")
