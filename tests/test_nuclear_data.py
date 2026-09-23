"""Spot checks of the generated nuclide table against published values.

References: Sears, Neutron News 3 (1992) 26 (b_coh, sigma_inc) and the
Atominstitut/Rauch-Waschkowski compilation the table is generated from.
Tolerances are loose enough to survive compilation-revision differences
(e.g. Sears-vs-Rauch b_c updates) but tight enough to catch unit mistakes,
Z/A mixups, or a broken generator.
"""
import math

import pytest

from irma.core.nuclear_data import NUCLIDES, isotopes, lookup, most_abundant_a


def test_natural_carbon_matches_irma_convention():
    c = lookup("C")
    # The GUI example row and every graphite deck use 6.646 fm / 0.001 b /
    # sigma_bound 5.551 b; the table must land on the same values.
    assert c.Z == 6 and c.A == 0
    assert c.b_coh_fm == pytest.approx(6.646, abs=0.005)
    assert c.sigma_inc_b == pytest.approx(0.001, abs=0.0005)
    assert c.sigma_bound_b == pytest.approx(5.551, rel=0.002)
    # awr from the natural mass; evaluations may pin their own (11.898)
    assert c.awr == pytest.approx(11.908, abs=0.002)


@pytest.mark.parametrize("key,z,a,b_coh,sigma_inc", [
    ("H", 1, 0, -3.741, 80.26),
    ("D", 1, 2, 6.671, 2.05),
    ("Be", 4, 0, 7.79, 0.0018),
    ("O", 8, 0, 5.803, 0.0),
    ("V", 23, 0, -0.443, 5.08),
])
def test_published_values(key, z, a, b_coh, sigma_inc):
    n = lookup(key)
    assert (n.Z, n.A) == (z, a)
    assert n.b_coh_fm == pytest.approx(b_coh, rel=0.02, abs=0.01)
    assert n.sigma_inc_b == pytest.approx(sigma_inc, rel=0.02, abs=0.001)


def test_sigma_bound_is_consistent_with_b_coh_everywhere():
    for n in NUCLIDES.values():
        derived = 0.04 * math.pi * n.b_coh_fm**2 + n.sigma_inc_b
        assert n.sigma_bound_b == pytest.approx(derived, rel=1e-9), n


def test_isotope_label_aliases():
    assert lookup("60-Ni") == lookup("Ni-60") == lookup((28, 60))
    assert lookup("D") == lookup("2-H") == lookup((1, 2))
    assert lookup("13-C").A == 13


def test_most_abundant_defaults():
    assert most_abundant_a(6) == 12    # C
    assert most_abundant_a(4) == 9     # Be
    assert most_abundant_a(8) == 16    # O
    # elements whose element-level mass table lacks abundances (U) or whose
    # isotopes carry no scattering constants (Ru, Xe) — review finding 1
    assert most_abundant_a(92) == 238  # U, from the neutron-table abundances
    assert most_abundant_a(44) == 102  # Ru
    assert most_abundant_a(54) == 132  # Xe
    assert most_abundant_a(26) == 56   # Fe
    assert most_abundant_a(28) == 58   # Ni


def test_synthetic_elements_have_no_default_identity():
    with pytest.raises(KeyError, match="no natural isotopic composition"):
        most_abundant_a(43)            # Tc
    with pytest.raises(KeyError, match="no natural isotopic composition"):
        most_abundant_a(94)            # Pu


def test_identity_inputs_are_strictly_integer():
    with pytest.raises(KeyError, match="must be an integer"):
        lookup((6.0, 13))              # e.g. Z read from a float array
    with pytest.raises(KeyError, match="A >= 1"):
        lookup("C-0")                  # the natural element is the bare symbol


def test_energy_dependent_flag_is_carried():
    assert lookup("199-Hg").energy_dependent is True
    assert lookup("C").energy_dependent is False
    assert lookup("D").energy_dependent is False


def test_natural_entries_never_carry_isotope_identity():
    for (z, a), n in NUCLIDES.items():
        assert (n.Z, n.A) == (z, a)
        if a == 0:
            assert n.abundance_pct is None


def test_awr_is_mass_over_neutron_mass():
    d = lookup("D")
    assert d.awr == pytest.approx(2.014102 / 1.008664916, rel=1e-5)


def test_isotopes_enumerates_an_element():
    """The table is keyed (Z, A) with no per-element index; isotopes() is
    the accessor a caller offering an isotope CHOICE needs."""
    assert [n.A for n in isotopes("C")] == [12, 13]
    assert [n.A for n in isotopes("Li")] == [6, 7]
    assert [n.A for n in isotopes(6)] == [12, 13]          # by Z too
    assert isotopes("D") == isotopes("H")                  # alias -> element
    # the natural-composition entry is keyed A = 0 and is NOT an isotope
    assert all(n.A >= 1 for n in isotopes("C"))
    assert all(n.symbol == "C" and n.Z == 6 for n in isotopes("C"))
    # every returned label round-trips through lookup()
    for n in isotopes("Sn"):
        assert lookup(f"{n.A}-{n.symbol}") == n
    # unknown elements are empty, not an exception: a caller walking a
    # species list must not have to guard every symbol
    assert isotopes("Xx") == []


def test_isotopes_carries_the_energy_dependent_flag_per_nuclide():
    """The flagged set is per NUCLIDE: natural Li is a safe prefill and
    6-Li is not; natural B is flagged and 11-B is not."""
    flags = {f"{n.A}-{n.symbol}": n.energy_dependent for n in isotopes("Li")}
    assert flags == {"6-Li": True, "7-Li": False}
    assert lookup("Li").energy_dependent is False
    assert lookup("B").energy_dependent is True
    assert {f"{n.A}-{n.symbol}": n.energy_dependent
            for n in isotopes("B")} == {"10-B": True, "11-B": False}


def test_unknown_keys_raise_with_guidance():
    with pytest.raises(KeyError, match="species override"):
        lookup("Xx")
    with pytest.raises(KeyError, match="species override"):
        lookup((6, 99))
    with pytest.raises(KeyError, match="cannot parse"):
        lookup("foo-bar")


def test_core_module_imports_nothing_heavy():
    # Behavior check, not text scanning (review finding 6): importing the
    # generated module in a fresh interpreter must not load periodictable or
    # any other heavy package.
    import subprocess
    import sys
    probe = ("import sys; import irma.core.nuclear_data; "
             "bad = [m for m in ('periodictable','ase','torch','phonopy',"
             "'scipy','yaml') if m in sys.modules]; "
             "print('LOADED:' + ','.join(bad))")
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, check=True)
    assert "LOADED:\n" in out.stdout or out.stdout.strip() == "LOADED:", out.stdout
