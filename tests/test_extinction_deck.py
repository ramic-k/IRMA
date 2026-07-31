"""Deck-card parsing for the optional crystalline-extinction card
(``extinction <model> l= g= L= [dist=] [rec=] [rmse_tol=]``)."""
import pytest

from irma.core.deck import _parse_line, DeckError
from irma.core.engine import TokenReader
from irma.core.crystal_cards import _parse_extinction_card


def _rd(line):
    return TokenReader(_parse_line(line))


def test_absent_card_returns_none():
    # the next card is Card 7 (numbers) or end of input -> no extinction
    assert _parse_extinction_card(_rd("150 400 1/"), elastic_mode=1) is None
    assert _parse_extinction_card(TokenReader([]), elastic_mode=1) is None


def test_valid_bc_mix_full():
    cfg = _parse_extinction_card(
        _rd("extinction BC_mix l=8550 g=170 L=75750 dist=Gauss rec=std rmse_tol=1e-3 /"),
        elastic_mode=1)
    assert cfg == {"model": "BC_mix", "l": 8550.0, "g": 170.0, "L": 75750.0,
                   "dist": "Gauss", "recipe": "std", "rmse_tol": 1e-3}


def test_l_vs_L_are_case_sensitive_and_distinct():
    cfg = _parse_extinction_card(
        _rd("extinction BC_mix l=8550 g=170 L=75750 /"), elastic_mode=1)
    assert cfg["l"] == 8550.0 and cfg["L"] == 75750.0    # not collided by lowercasing


def test_defaults_dist_recipe_tol():
    cfg = _parse_extinction_card(_rd("extinction BC_pure l=8550 /"), elastic_mode=1)
    assert cfg["dist"] == "Gauss" and cfg["recipe"] == "std" and cfg["rmse_tol"] == 1e-3
    cfg = _parse_extinction_card(_rd("extinction Sabine_corr l=8550 /"), elastic_mode=1)
    assert cfg["dist"] == "rect"                          # Sabine default differs


def test_bc_pure_primary_only_one_knob():
    cfg = _parse_extinction_card(_rd("extinction BC_pure l=8550 /"), elastic_mode=1)
    assert cfg["model"] == "BC_pure" and cfg["g"] == 0.0 and cfg["L"] == 0.0


def test_fortran_d_exponent_accepted():
    # numeric extinction fields accept Fortran D-exponent notation, like the rest
    # of the deck (irma/core/deck.py). 1.0d4 == 1e4, 5.0D-1 == 0.5.
    cfg = _parse_extinction_card(
        _rd("extinction BC_pure l=1.0d4 L=7.575D4 rmse_tol=1.0d-3 /"),
        elastic_mode=1)
    assert cfg["l"] == 1.0e4 and cfg["L"] == 7.575e4 and cfg["rmse_tol"] == 1.0e-3


def test_provenance_comment_records_model_and_params():
    from irma.core.driver import extinction_provenance_comment
    line = extinction_provenance_comment(
        {"model": "BC_mix", "l": 8550.0, "g": 170.0, "L": 75750.0,
         "dist": "Gauss", "recipe": "std"})
    assert "extinction:" in line and "BC_mix" in line
    assert "l=8550" in line and "g=170" in line and "L=75750" in line
    assert "Gauss/std" in line and "CrysXT" in line


@pytest.mark.parametrize("line,msg", [
    ("extinction /", "needs a model name"),
    ("extinction NotAModel l=1 /", "model must be one of"),
    ("extinction BC_mix l=8550 /", "needs l>0, g>0 and L>0"),
    ("extinction BC_mod g=170 L=75750 /", "needs l>0, g>0 and L>0"),
    ("extinction BC_pure l=8550 g=170 L=75750 /", "primary OR secondary, not both"),
    ("extinction BC_mix g=170 L=75750 rec=lux /", "rec=lux is not implemented"),
    ("extinction BC_pure /", "no active mechanism"),
    ("extinction BC_pure l=1 l=2 /", "given more than once"),
    ("extinction BC_pure l=8550 dist=Bogus /", "dist for BC_pure must be one of"),
    ("extinction BC_pure l=8550 foo=3 /", "unknown extinction field"),
    ("extinction BC_pure l=abc /", "must be a number"),
    ("extinction BC_pure l=-5 /", "must be finite and >= 0"),
    ("extinction BC_pure l=8550 rmse_tol=0 /", "rmse_tol must be a positive, finite number"),
    ("extinction BC_pure l=8550 rec=bogus /", "rec must be one of"),
])
def test_invalid_cards_raise(line, msg):
    with pytest.raises(DeckError, match=msg):
        _parse_extinction_card(_rd(line), elastic_mode=1)


def test_accepted_for_mef():
    cfg = _parse_extinction_card(_rd("extinction BC_pure l=8550 /"), elastic_mode=2)
    assert cfg["model"] == "BC_pure"


def test_rejected_for_invalid_elastic_mode():
    with pytest.raises(DeckError, match="elastic_mode=1 SEF or 2 MEF"):
        _parse_extinction_card(_rd("extinction BC_pure l=8550 /"), elastic_mode=3)


def _tokens(*lines):
    toks = []
    for ln in lines:
        toks.extend(_parse_line(ln))
    return toks


# a minimal iel=10 mode-0 crystal-card block (Cards 6b-6d) for Be
_BE_CARDS = (
    "1 1 0 0/",                                                  # 6b: CEF, 1 atom, mode 0
    "2.2866 2.2866 3.5833 90 90 120/",                          # 6c: lattice
    "4 9 8.93478 7.79 0.0018 2/",                               # 6d: Be atom
    "0.33333333 0.66666667 0.75 0.66666667 0.33333333 0.25/",   # 6d: positions
)


def test_parse_crystal_cards_sets_coherent_extinction():
    from irma.core.crystal_cards import _parse_crystal_cards
    toks = _tokens(*_BE_CARDS, "extinction BC_pure l=8550 /")
    ci = _parse_crystal_cards(TokenReader(toks), za=4009, nphon=100)
    assert ci["coherent_extinction"]["model"] == "BC_pure"
    assert ci["coherent_extinction"]["l"] == 8550.0


def test_parse_crystal_cards_without_card_has_no_key():
    from irma.core.crystal_cards import _parse_crystal_cards
    ci = _parse_crystal_cards(TokenReader(_tokens(*_BE_CARDS)), za=4009, nphon=100)
    assert "coherent_extinction" not in ci


# ENG-3: extinction is applied only by the coherent-carrying MT2 builders.
# When SEF (elastic_mode=1) routes to the INCOHERENT elastic builder — a
# single-atom principal with sigma_coh <= sigma_inc, or a polyatomic whose
# principal is not the designated-coherent (DC) atom — the config would be a
# silent no-op on a tape still stamped as extinction-corrected. Both routing
# conditions are fully determined by Card 6b/6d data, so the parser must
# reject the combination; the coherent-routing configurations must keep
# parsing.

# single-atom V-like material: sigma_coh = 4*pi*0.3824^2*0.01 ~ 0.018 b
# << sigma_inc = 5.08 b -> SEF routes to the incoherent builder
_V_CARDS = (
    "1 1 0 0/",                       # 6b: SEF, 1 atom type, mode 0
    "3.03 3.03 3.03 90 90 90/",       # 6c: bcc lattice
    "23 51 50.5063 -0.3824 5.08 1/",  # 6d: V atom
    "0.0 0.0 0.0/",                   # 6d: position
)

# BeO-like polyatomic: DC selection minimizes f/(1-f)*sigma_inc, so the
# O type (sigma_inc=0.0008 < Be's 0.0018) is the DC atom. Principal Be
# (za=4009) -> incoherent routing; principal O (za=8016) -> coherent.
_BEO_CARDS = (
    "1 2 0 0/",                                              # 6b: SEF, 2 atom types, mode 0
    "2.698 2.698 4.359 90 90 120/",                          # 6c: wurtzite lattice
    "4 9 8.93478 7.79 0.0018 2/",                            # 6d: Be
    "0.33333333 0.66666667 0.0 0.66666667 0.33333333 0.5/",
    "8 16 15.8575 5.803 0.0008 2/",                          # 6d: O
    "0.33333333 0.66666667 0.375 0.66666667 0.33333333 0.875/",
)

_EXT_CARD = "extinction BC_pure l=8550 /"


def test_extinction_rejected_when_sef_single_atom_routes_incoherent():
    from irma.core.crystal_cards import _parse_crystal_cards
    toks = _tokens(*_V_CARDS, _EXT_CARD)
    with pytest.raises(DeckError, match="silent no-op.*incoherent elastic builder"):
        _parse_crystal_cards(TokenReader(toks), za=23051, nphon=100)


def test_extinction_rejected_when_sef_principal_is_not_dc():
    from irma.core.crystal_cards import _parse_crystal_cards
    toks = _tokens(*_BEO_CARDS, _EXT_CARD)
    with pytest.raises(DeckError, match="not the.*designated-coherent atom"):
        _parse_crystal_cards(TokenReader(toks), za=4009, nphon=100)


def test_extinction_accepted_when_sef_principal_is_dc():
    # near-miss: the SAME polyatomic cell with the DC atom (O) as principal
    # routes coherent and must still parse, carrying the extinction config
    from irma.core.crystal_cards import _parse_crystal_cards
    toks = _tokens(*_BEO_CARDS, _EXT_CARD)
    ci = _parse_crystal_cards(TokenReader(toks), za=8016, nphon=100)
    assert ci["coherent_extinction"]["model"] == "BC_pure"


def test_extinction_accepted_for_mef_single_atom():
    # MEF (elastic_mode=2) always carries the coherent Bragg edges -> must parse
    from irma.core.crystal_cards import _parse_crystal_cards
    cards = ("2 1 0 0/",) + _BE_CARDS[1:]
    toks = _tokens(*cards, _EXT_CARD)
    ci = _parse_crystal_cards(TokenReader(toks), za=4009, nphon=100)
    assert ci["coherent_extinction"]["model"] == "BC_pure"


def test_incoherent_routing_without_extinction_still_parses():
    # the guard is extinction-conditional: the same incoherent-routing decks
    # WITHOUT the card stay legal (incoherent elastic is a valid SEF output)
    from irma.core.crystal_cards import _parse_crystal_cards
    ci = _parse_crystal_cards(TokenReader(_tokens(*_V_CARDS)),
                              za=23051, nphon=100)
    assert "coherent_extinction" not in ci
    ci = _parse_crystal_cards(TokenReader(_tokens(*_BEO_CARDS)),
                              za=4009, nphon=100)
    assert "coherent_extinction" not in ci
