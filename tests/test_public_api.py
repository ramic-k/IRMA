"""The stable top-level public API of the ``irma`` package.

``from irma import run_leapr, parse_leapr_input, DeckError`` is the supported
surface that decouples callers from the internal module layout. This file pins
that contract; it needs no optional extras, so it runs in the bare-install gate
too.
"""
def test_public_symbols_importable_from_top_level():
    from irma import run_leapr, parse_leapr_input, DeckError, __version__
    assert run_leapr.__name__ == "run_leapr"
    assert parse_leapr_input.__name__ == "parse_leapr_input"
    assert issubclass(DeckError, ValueError)
    assert isinstance(__version__, str) and __version__


def test_public_symbols_match_canonical_definitions():
    import irma
    from irma.core.driver import run_leapr as canon_run
    from irma.core.deck import parse_leapr_input as canon_parse, DeckError as canon_err
    assert irma.run_leapr is canon_run
    assert irma.parse_leapr_input is canon_parse
    assert irma.DeckError is canon_err


def test_all_lists_the_public_surface():
    import irma
    assert set(irma.__all__) == {
        "run_leapr", "LeaprResult", "parse_leapr_input", "DeckError",
        "__version__", "__author__"}
    # every advertised name resolves
    for name in irma.__all__:
        assert getattr(irma, name) is not None
    # and they show up in dir() for autocomplete/introspection
    assert {"run_leapr", "LeaprResult", "parse_leapr_input",
            "DeckError"} <= set(dir(irma))
