"""GUI deck generation and import (export -> engine -> import round trips).

Covers the QA GUI slice: export/import previously crashed with NameError for
every iel != 10 (inelastic_mode_val / inelastic_mode_loaded unbound), the
secondary scatterer exported an invalid Card 6 (mss=0, b7 hardcoded 0, nss=2
offered), ncold/nsk decks were exported without their Cards 17-19 S(kappa)
records, Card 6g's auto-order field and Card 6e partial spectra were dropped,
multi-positive-temperature decks were silently collapsed, and isabt/ilog/smin
were reset on export.

Requires a display (Tk); skipped headless (CI).
"""

import pytest

tk = pytest.importorskip("tkinter")

from irma.core.engine import run_leapr  # noqa: E402


@pytest.fixture(scope="module")
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    from irma.gui.app import IrmaApp
    application = IrmaApp(root)
    yield application
    root.destroy()


def _set_text(widget, text):
    widget.delete("1.0", "end")
    widget.insert("1.0", text)


def _reset(app):
    """Put every field the tests rely on into a known classic-deck state."""
    app.inelastic_mode_var.set(0)       # first: mode 1/2 pins iel to 10
    app.iel_var.set("0 — None")
    app.za.set("125")
    app.awr.set("0.99917")
    app.spr.set("20.449")
    app.npr.set("1")
    app.mat.set("1")
    app.nphon.set("4")
    app.isabt.set("0 — S(α,β) (standard)")
    app.ilog.set("0 — S values")
    app.iint.set("0 — log-lin (INT=4)")
    app.smin.set("1e-75")
    app.ncold.set("0 — None")
    app.nsk.set("0 — None")
    app.ska_dka.set("0")
    _set_text(app.ska_text, "")
    app.cfrac.set("0")
    app.nss.set("0 — None")
    app.b7.set("1 — Free gas")
    app.aws.set("0")
    app.sps.set("0")
    app.mss.set("1")
    app.sec_dos_delta.set("0")
    _set_text(app.sec_dos_rho_text, "")
    app.sec_twt.set("0.0")
    app.sec_c_diff.set("0.0")
    app.sec_tbeta.set("1.0")
    _set_text(app.sec_osc_energies, "")
    _set_text(app.sec_osc_weights, "")
    app.temps_var.set("296.0")
    app.lat.set("1 — alpha/beta in kT_thermal (0.0253 eV)")
    app.grid_mode.set("manual")
    _set_text(app.alpha_text, "0.05 1.0 8.0")
    _set_text(app.beta_text, "0.0 0.6 2.0 6.0")
    app.dos_source.set("manual")
    app.dos_delta.set("0.005")
    _set_text(app.dos_rho_text, "0.0 0.20 0.45 0.55 0.30 0.0")
    app.twt.set("0.0")
    app.c_diff.set("0.0")
    app.tbeta.set("1.0")
    _set_text(app.osc_energies, "")
    _set_text(app.osc_weights, "")
    app.comments_text.clear()
    app._imported_partial_spectra = []


def _run_engine(deck_text, tmp_path):
    inp = tmp_path / "gui.input"
    inp.write_text(deck_text)
    run_leapr(str(inp), str(tmp_path / "gui.endf"))


# ---------- H6/H7: every iel must export and import ----------

def test_export_classic_iel0_runs_engine(app, tmp_path):
    _reset(app)
    text = app._generate_input_text()      # NameError before the fix
    assert "0 0 0 0 0 /" in text           # Card 6: no secondary
    _run_engine(text, tmp_path)


def test_export_legacy_iel1_graphite_runs_engine(app, tmp_path):
    _reset(app)
    app.iel_var.set("1 — Graphite (legacy)")
    app.za.set("6012")
    app.awr.set("11.907856")
    app.spr.set("4.724629")
    text = app._generate_input_text()
    _run_engine(text, tmp_path)


def test_import_classic_deck_populates_fields(app, tmp_path):
    _reset(app)
    deck = tmp_path / "classic.input"
    deck.write_text(
        "20 /\n'import test'/\n1 1 4/\n7 125./\n"
        "0.99917 20.449 2 0 0 0/\n0/\n3 4 1/\n"
        "0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n0.005 6/\n"
        "0.0 0.20 0.45 0.55 0.30 0.0/\n0. 0. 1./\n0/\n/\n")
    summary = app._import_leapr_from_path(str(deck))   # NameError before fix
    assert "iel=0" in summary
    assert app.npr.get() == "2"
    assert app._code(app.iel_var) == 0
    assert app.temps_var.get().startswith("300")


def test_iint_default_omits_field_and_roundtrips(app, tmp_path):
    _reset(app)
    text = app._generate_input_text()
    # default iint=0 -> classic 5-field Card 4 (no trailing flag)
    assert "0 0 1e-75 /" in text
    assert "0 0 1e-75 0 /" not in text and "0 0 1e-75 1 /" not in text


def test_iint_linlin_exports_and_roundtrips(app, tmp_path):
    _reset(app)
    app.iint.set("1 — lin-lin (INT=2)")
    text = app._generate_input_text()
    assert "0 0 1e-75 1 /" in text             # iint appended on Card 4
    _run_engine(text, tmp_path)                # engine accepts iint=1
    deck = tmp_path / "iint.input"
    deck.write_text(text)
    _reset(app)                                # back to default (iint=0)
    app._import_leapr_from_path(str(deck))
    assert app._code(app.iint) == 1  # survived the round trip
    assert "0 0 1e-75 1 /" in app._generate_input_text()


# ---------- H8/H9: secondary scatterer ----------

def test_secondary_free_gas_roundtrip(app, tmp_path):
    _reset(app)
    app.nss.set("1 — One secondary scatterer")
    app.b7.set("1 — Free gas")
    app.aws.set("15.85316")
    app.sps.set("3.8883")
    app.mss.set("1")
    text = app._generate_input_text()
    assert "1 1 15.85316 3.8883 1 /" in text
    _run_engine(text, tmp_path)             # engine accepts mss>=1, b7>0

    deck = tmp_path / "sec.input"
    deck.write_text(text)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app._code(app.nss) == 1
    assert app._code(app.b7) == 1
    assert app.mss.get() == "1"
    assert "1 1 15.85316 3.8883 1 /" in app._generate_input_text()


def _setup_two_pass(app):
    """A BeO-like bound secondary: O gets its own phonon model (b7=0)."""
    app.za.set("4009")
    app.awr.set("8.93478")
    app.spr.set("6.15")
    app.nss.set("1 — One secondary scatterer")
    app.b7.set("0 — Bound (two-pass)")
    app.aws.set("15.858")
    app.sps.set("3.7481")
    app.mss.set("1")
    app.sec_dos_delta.set("0.006")
    _set_text(app.sec_dos_rho_text, "0.0 0.15 0.40 0.60 0.35 0.0")
    app.sec_twt.set("0.0")
    app.sec_c_diff.set("0.0")
    app.sec_tbeta.set("1.0")


def test_two_pass_secondary_roundtrip(app, tmp_path):
    """b7=0: the secondary scatterer's own phonon model is emitted as a
    complete second temperature pass, the engine merges the two laws, and
    import recovers every secondary field (refusing these decks
    decks entirely)."""
    _reset(app)
    _setup_two_pass(app)
    app.temps_var.set("296.0 400.0")
    text1 = app._generate_input_text()
    assert "1 0 15.858 3.7481 1 /" in text1            # Card 6, b7=0
    assert text1.count("296.0000 /") == 2              # both passes
    assert text1.count("-400.0000 /") == 2             # shared-spectrum conv.
    _run_engine(text1, tmp_path)                       # two-pass merge runs

    deck = tmp_path / "twopass.input"
    deck.write_text(text1)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app._code(app.b7) == 0
    assert len(app.sec_dos_rho_text.get("1.0", "end").split()) == 6
    assert float(app.sec_tbeta.get()) == pytest.approx(1.0)
    assert app._generate_input_text() == text1         # idempotent


def test_two_pass_with_ncold_refused(app):
    _reset(app)
    _setup_two_pass(app)
    app.ncold.set("1 — Ortho-H")
    app.ska_dka.set("0.5")
    _set_text(app.ska_text, "0.1 0.4 0.9 1.1")
    with pytest.raises(ValueError, match="two-pass"):
        app._generate_input_text()


def test_two_pass_without_secondary_spectrum_refused(app):
    _reset(app)
    _setup_two_pass(app)
    _set_text(app.sec_dos_rho_text, "")
    with pytest.raises(ValueError, match="secondary phonon"):
        app._generate_input_text()


# ---------- H11: ncold/nsk need Cards 17-19 ----------

def test_skold_exports_cards_17_18_19_and_roundtrips(app, tmp_path):
    _reset(app)
    app.nsk.set("2 — Skold")
    app.ska_dka.set("0.5")
    _set_text(app.ska_text, "0.1 0.4 0.9 1.1 1.0 1.0")
    app.cfrac.set("0.3")
    text = app._generate_input_text()
    assert "6 5.000000e-01 /" in text       # Card 17: nka dka
    assert "0.3 /" in text                  # Card 19: cfrac
    _run_engine(text, tmp_path)             # skold path actually runs

    deck = tmp_path / "skold.input"
    deck.write_text(text)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app._code(app.nsk) == 2
    assert float(app.ska_dka.get()) == pytest.approx(0.5)
    assert len(app.ska_text.get("1.0", "end").split()) == 6
    assert float(app.cfrac.get()) == pytest.approx(0.3)


def test_nsk_without_ska_table_is_rejected(app):
    _reset(app)
    app.nsk.set("2 — Skold")
    with pytest.raises(ValueError, match=r"S\(kappa\)"):
        app._generate_input_text()


def test_ncold_emits_skappa_but_no_cfrac(app):
    _reset(app)
    app.ncold.set("2 — Para-H")
    app.ska_dka.set("0.5")
    _set_text(app.ska_text, "0.1 0.4 0.9 1.1")
    text = app._generate_input_text()
    assert "4 5.000000e-01 /" in text       # Card 17 present
    lines = text.splitlines()
    i17 = next(i for i, l in enumerate(lines) if l == "4 5.000000e-01 /")
    # Card 18 (one line of 4 values) then NO Card 19 — comments follow
    assert lines[i17 + 2] == "/"


# ---------- #30: multi-positive-temperature decks refused ----------

def test_multi_positive_temperature_import_refused(app, tmp_path):
    _reset(app)
    detail = ("0.005 6/\n0.0 0.20 0.45 0.55 0.30 0.0/\n0. 0. 1./\n0/\n")
    deck = tmp_path / "multit.input"
    deck.write_text(
        "20 /\n'multi T'/\n2 1 4/\n7 125./\n"
        "0.99917 20.449 1 0 0 0/\n0/\n3 4 1/\n"
        "0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n"
        "300/\n" + detail + "600/\n" + detail + "/\n")
    with pytest.raises(ValueError, match="more than one temperature"):
        app._import_leapr_from_path(str(deck))


# ---------- #28: isabt / ilog / smin round-trip ----------

def test_isabt_ilog_smin_roundtrip(app, tmp_path):
    _reset(app)
    app.isabt.set("1 — S̃ tilde (asymmetric)")
    app.ilog.set("1 — ln(S) values")
    app.smin.set("1e-30")
    text = app._generate_input_text()
    assert "1 125 1 1 1e-30 /" in text      # Card 4
    deck = tmp_path / "card4.input"
    deck.write_text(text)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app._code(app.isabt) == 1
    assert app._code(app.ilog) == 1
    assert float(app.smin.get()) == pytest.approx(1e-30)


# ---------- #29 + #32 + mode-1/2 validation (iel=10) ----------

def _setup_iel10(app):
    app.iel_var.set("10 — Generalized (crystal structure)")
    app.za.set("6012")
    app.awr.set("11.907856")
    app.spr.set("4.724629")
    app.elastic_mode.set("1 — SEF (Single-channel Elastic Format)")
    _set_text(app.atoms_text,
              "6 12 11.907856 6.6484 0.001 4 "
              "0 0 0.25  0 0 0.75  0.333333 0.666667 0.25  "
              "0.666667 0.333333 0.75")
    # Normalized float strings (matching import's str(float(...))) so the
    # export -> import -> export identity can hold byte-for-byte.
    app.latt_a.set("2.46")
    app.latt_b.set("2.46")
    app.latt_c.set("6.7")
    app.latt_alpha.set("90.0")
    app.latt_beta.set("90.0")
    app.latt_gamma.set("120.0")
    app.coh_edge_group_bpd.set("0")


def test_card6e_partial_spectra_preserved(app, tmp_path):
    _reset(app)
    _setup_iel10(app)
    app._imported_partial_spectra = [{
        'Z': 6, 'A': 12, 'delta': 0.005, 'ni': 6,
        'rho': [0.0, 0.20, 0.45, 0.55, 0.30, 0.0],
    }]
    text = app._generate_input_text()
    assert " 1 0 /" in text                 # Card 6b: nspec=1, mode 0
    assert "6 12 5.000000e-03 6 /" in text  # Card 6e header
    _run_engine(text, tmp_path)             # exercises the fsum DW branch

    deck = tmp_path / "spectra.input"
    deck.write_text(text)
    _reset(app)
    summary = app._import_leapr_from_path(str(deck))
    assert "Preserved 1 Card 6e" in summary
    assert len(app._imported_partial_spectra) == 1
    assert app._imported_partial_spectra[0]['ni'] == 6


def test_card6g_auto_order_field_roundtrip(app, tmp_path):
    _reset(app)
    _setup_iel10(app)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    app.nc_mesh_nx.set("8")
    app.nc_mesh_ny.set("8")
    app.nc_mesh_nz.set("8")
    app.nc_ncpu.set("1")
    app.nc_use_born_var.set(0)
    app.nc_num_directions.set("100")
    app.nc_multiphonon_num_directions.set("100")
    app.nc_auto_order_var.set(1)
    text = app._generate_input_text()
    assert "100 100 1 /" in text            # 3-field Card 6g, auto-order kept

    deck = tmp_path / "mode2.input"
    deck.write_text(text)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert int(app.nc_auto_order_var.get()) == 1
    assert "100 100 1 /" in app._generate_input_text()


def test_min_phonon_energy_gui_deck_roundtrip(app, tmp_path):
    _reset(app)
    _setup_iel10(app)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    app.nc_min_phonon_energy.set("0.5")
    text = app._generate_input_text()
    assert "\n0.5 /\n" in text
    assert text.index("\n0.5 /\n") < text.index(" 1 /\n", text.index("\n0.5 /\n"))

    deck = tmp_path / "cutoff.input"
    deck.write_text(text)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app.nc_min_phonon_energy.get() == "0.5"
    assert app._generate_input_text() == text


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_min_phonon_energy_gui_rejects_invalid_values(app, value):
    _reset(app)
    _setup_iel10(app)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    app.nc_min_phonon_energy.set(value)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        app._generate_input_text()


def test_iel10_empty_atom_table_rejected_up_front(app):
    """QA2-042: iel=10 with no atom types emits an invalid Card 6b
    ('elastic_mode 0 0 ...') that the engine later rejects; the GUI must
    refuse it up front like its other preconditions, not defer to a temp-file
    engine run."""
    _reset(app)
    _setup_iel10(app)
    _set_text(app.atoms_text, "")      # no atom types
    with pytest.raises(ValueError, match="at least one atom type"):
        app._generate_input_text()


def test_nsk_with_empty_cfrac_rejected(app):
    _reset(app)
    app.nsk.set("2 — Skold")
    app.ska_dka.set("0.5")
    _set_text(app.ska_text, "0.1 0.4 0.9 1.1")
    app.cfrac.set("")
    with pytest.raises(ValueError, match="cfrac"):
        app._generate_input_text()


def test_import_rejects_invalid_nss(app, tmp_path):
    """The old GUI offered nss=2 ('Free gas'); such decks are invalid
    (the engine requires nss in (0,1)) and must not round-trip."""
    _reset(app)
    deck = tmp_path / "nss2.input"
    deck.write_text(
        "20 /\n'bad nss'/\n1 1 4/\n7 125./\n"
        "0.99917 20.449 1 0 0 0/\n2 1. 15.85 3.88 1/\n")
    with pytest.raises(ValueError, match="nss must be 0 or 1"):
        app._import_leapr_from_path(str(deck))


# ---------- export -> import -> export idempotence ----------

def test_full_roundtrip_idempotent_classic_everything_on(app, tmp_path):
    """A maximal classic deck (Skold + free-gas secondary + oscillators +
    isabt/ilog/smin + multiple temperatures) must survive
    export -> import -> export byte-for-byte."""
    _reset(app)
    app.temps_var.set("296.0 400.0 600.0")
    app.isabt.set("1 — S̃ tilde (asymmetric)")
    app.ilog.set("1 — ln(S) values")
    app.smin.set("1e-30")
    app.nsk.set("2 — Skold")
    app.ska_dka.set("0.5")
    _set_text(app.ska_text, "0.1 0.4 0.9 1.1 1.0 1.0")
    app.cfrac.set("0.3")
    app.nss.set("1 — One secondary scatterer")
    app.b7.set("2 — Diffusion")
    app.aws.set("15.85316")
    app.sps.set("3.8883")
    app.mss.set("2")
    _set_text(app.osc_energies, "0.205")
    _set_text(app.osc_weights, "0.5")
    text1 = app._generate_input_text()

    deck = tmp_path / "maximal.input"
    deck.write_text(text1)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    text2 = app._generate_input_text()
    assert text2 == text1


def test_full_roundtrip_idempotent_mode2(app, tmp_path):
    _reset(app)
    _setup_iel10(app)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    app.nc_mesh_nx.set("8")
    app.nc_mesh_ny.set("8")
    app.nc_mesh_nz.set("8")
    app.nc_ncpu.set("4")
    app.nc_use_born_var.set(1)
    app.nc_born_path.set("/nonexistent/BORN")
    app.nc_num_directions.set("100")
    app.nc_multiphonon_num_directions.set("200")
    app.nc_auto_order_var.set(1)
    text1 = app._generate_input_text()
    assert "100 200 1 /" in text1          # 3-field Card 6g

    deck = tmp_path / "mode2max.input"
    deck.write_text(text1)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    text2 = app._generate_input_text()
    assert text2 == text1


# ---------- oscillator half-filled / mismatched inputs ----------

def test_oscillator_energies_without_weights_rejected(app):
    """Energies with an empty weights box must not silently emit '0 /' --
    the discrete modes vanished from the law with no error."""
    _reset(app)
    _set_text(app.osc_energies, "0.205 0.436")
    with pytest.raises(ValueError, match="Discrete oscillators.*weights"):
        app._generate_input_text()


def test_oscillator_weights_without_energies_rejected(app):
    _reset(app)
    _set_text(app.osc_weights, "0.166 0.389")
    with pytest.raises(ValueError, match="Discrete oscillators.*energies"):
        app._generate_input_text()


def test_oscillator_count_mismatch_rejected(app):
    _reset(app)
    _set_text(app.osc_energies, "0.205 0.436 0.480")
    _set_text(app.osc_weights, "0.166 0.389")
    with pytest.raises(ValueError, match="3 energies but 2 weights"):
        app._generate_input_text()


def test_secondary_oscillator_half_filled_rejected(app):
    _reset(app)
    _setup_two_pass(app)
    _set_text(app.sec_osc_energies, "0.2")
    with pytest.raises(ValueError, match="Secondary oscillators.*weights"):
        app._generate_input_text()


def test_secondary_oscillator_count_mismatch_rejected(app):
    _reset(app)
    _setup_two_pass(app)
    _set_text(app.sec_osc_energies, "0.2 0.4")
    _set_text(app.sec_osc_weights, "0.5")
    with pytest.raises(ValueError, match="2 energies but 1 weights"):
        app._generate_input_text()


# ---------- safe-by-construction mode-1/2 defaults ----------

def test_reset_defaults_match_production_recommendation(app):
    """The reset/fresh-form sampling defaults must be the documented
    recommended production sampling (Card 6g '10000 1000 1', mesh 40^3,
    the validation-campaign sampling and the `irma mlip emit` sampling) --
    the pre-0.17 defaults (ndir=5000, mesh 20^3, Auto-size OFF with
    nphon=100) were the manual's own documented high-Q truncation case.
    LAT must match the builder default (1) too."""
    _reset(app)
    app._reset_form_to_defaults()
    assert app.nc_num_directions.get() == "10000"
    assert (app.nc_mesh_nx.get(), app.nc_mesh_ny.get(),
            app.nc_mesh_nz.get()) == ("40", "40", "40")
    assert app.nc_multiphonon_num_directions.get() == "1000"
    assert int(app.nc_auto_order_var.get()) == 1
    assert app.lat.get().startswith("1 —")
    assert app.coh_edge_group_enable_var.get() is True     # grouping on
    assert app.coh_edge_group_bpd.get() == "50"
    # Methodology stays; material IDENTITY clears to blank, exactly as a
    # freshly built form ships it (compared field by field in
    # test_gui_widgets.test_reset_lands_in_the_same_blank_state_as_a_fresh_form).
    assert (app.za.get(), app.mat.get(),
            app.awr.get(), app.spr.get()) == ("", "", "", "")


def test_mode2_default_sampling_emits_recommended_deck(app):
    _reset(app)
    app._reset_form_to_defaults()
    app.mat.set("1")            # reset blanks the identity fields; MAT is ours
    _setup_iel10(app)
    app.temps_var.set("296.0")
    app.grid_mode.set("manual")
    _set_text(app.alpha_text, "0.05 1.0 8.0")
    _set_text(app.beta_text, "0.0 0.6 2.0 6.0")
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    text = app._generate_input_text()
    assert "40 40 40 1 0 /" in text         # Card 6f: 40^3 mesh, ncpu=1, no BORN
    assert "10000 1000 1 /" in text         # Card 6g: the recommended defaults


def test_import_keeps_explicit_auto_order_off(app, tmp_path):
    """Import stays faithful to the deck: a 2-field Card 6g means
    auto_order=0, which must override the safe-form default of 1."""
    _reset(app)
    deck = tmp_path / "noauto.input"
    deck.write_text(
        "20 /\n'no auto'/\n1 0 4/\n1 6012./\n"
        "11.907856 4.724629 1 10 0 0/\n0 0 0 0 0/\n"
        "1 1 0 2/\n"                        # Card 6b: mode 2
        "2.46 2.46 6.7 90 90 120/\n"
        "6 12 11.9 6.65 0.001 1/\n0 0 0/\n"
        "'/nonexistent/phonopy.yaml'/\n"
        "8 8 8 1 0/\n"
        "100 100/\n"                        # 2-field Card 6g -> auto_order 0
        "3 4 1/\n0.05 1.0 8.0/\n0.0 0.6 2.0 6.0/\n300/\n/\n")
    app._import_leapr_from_path(str(deck))
    assert int(app.nc_auto_order_var.get()) == 0
    assert "100 100 /" in app._generate_input_text()   # still 2-field on export


# ---------- spr is the FREE-atom cross section ----------

def test_spr_field_labeled_free_atom(app):
    """The deck field is LEAPR's free-atom spr; the GUI must not label it
    sigma_b (bound) and offer H-1 = 81.67 b (a bound value) as an example,
    a ~4x cross-section error for anyone who followed it."""
    from irma.gui.widgets import InfoLabel
    assert app.spr.label.cget("text") == "sigma_free [barn]:"
    helps = [w for w in app.spr.winfo_children()
             if isinstance(w, InfoLabel)]
    assert helps, "spr field lost its help widget"
    msg = helps[0]._message
    assert "FREE-atom" in msg
    assert "20.45" in msg                   # H-1 FREE value, not 81.67
    assert "do NOT enter" in msg            # explicit bound-value warning
    assert "81.67" not in msg


def test_comment_apostrophe_and_slash_roundtrip(app, tmp_path):
    """Comment cards containing ' and / must survive export -> import ->
    export (the emit sites double embedded quotes per the Fortran
    convention; a raw ' silently truncated the card)."""
    _reset(app)
    tricky = "Evaluator: O'Brien's data / IRMA test 'quoted'"
    app.comments_text.clear()
    app.comments_text.append(tricky)
    text1 = app._generate_input_text()
    assert "''" in text1                      # doubling actually happened

    deck = tmp_path / "apostrophe.input"
    deck.write_text(text1)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app.comments_text.get_text().strip() == tricky
    text2 = app._generate_input_text()
    assert text2 == text1                     # idempotent
    _run_engine(text1, tmp_path)              # and the engine accepts it


# ---------- Crystalline extinction (iel=10 optional card) ----------

def _repo_example(name):
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "tsl" / name)


def test_extinction_import_populates_fields(app):
    """Importing the Be extinction example fills the GUI extinction section."""
    _reset(app)
    app._import_leapr_from_path(_repo_example("be_iel10_extinction.input"))
    assert app.ext_enable_var.get() is True
    assert app.ext_model.get() == "BC_mix"
    assert app.ext_l.get() == "8550"
    assert app.ext_g.get() == "170"
    assert app.ext_L.get() == "75750"
    assert app.ext_dist.get() == "Gauss"
    assert app.ext_recipe.get() == "std"


def test_extinction_reexport_emits_card(app):
    """After importing the extinction example, re-export emits the same card."""
    _reset(app)
    app._import_leapr_from_path(_repo_example("be_iel10_extinction.input"))
    text = app._generate_input_text()
    assert "extinction BC_mix l=8550 g=170 L=75750 dist=Gauss rec=std" in text


def test_no_extinction_card_leaves_toggle_off(app):
    """A deck without the card imports with extinction disabled and emits none."""
    _reset(app)
    app._import_leapr_from_path(_repo_example("graphite_iel10_classic.input"))
    assert app.ext_enable_var.get() is False
    assert "extinction " not in app._generate_input_text()


def test_extinction_rmse_tol_round_trips_by_value(app):
    """rmse_tol survives import -> re-export -> re-parse with VALUE identity (the
    GUI reformats it via :g, so the text may change 1e-3 -> 0.001, but the number
    must not)."""
    from irma.core.deck import _parse_line
    from irma.core.engine import TokenReader
    from irma.core.crystal_cards import _parse_extinction_card

    _reset(app)
    app._import_leapr_from_path(_repo_example("be_iel10_extinction.input"))
    assert float(app.ext_rmse_tol.get()) == 1e-3
    text = app._generate_input_text()
    card = next(ln for ln in text.splitlines() if ln.strip().startswith("extinction "))
    cfg = _parse_extinction_card(TokenReader(_parse_line(card.strip())), elastic_mode=1)
    assert cfg["rmse_tol"] == 1e-3                     # exact value, not text


def test_extinction_dist_dropdown_restricted_to_model_family(app):
    """The distribution dropdown only offers the active model's family, so a
    cross-family value (e.g. Sabine 'rect' under a Becker-Coppens model) can't be
    selected from the GUI."""
    _reset(app)
    app.ext_model.set("BC_pure")
    app._on_ext_model_change()
    assert list(app.ext_dist.combo["values"]) == ["Gauss", "Lorentz", "Fresnel"]
    assert app.ext_dist.get() == "Gauss"
    app.ext_model.set("Sabine_uncorr")
    app._on_ext_model_change()
    assert list(app.ext_dist.combo["values"]) == ["rect", "tri"]
    assert app.ext_dist.get() == "rect"


# ---------- Bragg-edge grouping default-on toggle ----------

def test_grouping_off_deck_unchecks_and_emits_four_field(app):
    # a deck without grouping -> checkbox off on import, 4-field Card 6b on export
    _reset(app)
    app._import_leapr_from_path(_repo_example("graphite_iel10_classic.input"))
    assert app.coh_edge_group_enable_var.get() is False
    text = app._generate_input_text()
    assert "1 1 0 0 /" in text and " 50 1.0 /" not in text


def test_grouping_checked_emits_grouping_fields(app):
    # turn grouping ON over an imported deck -> Card 6b gains the grouping fields
    _reset(app)
    app._import_leapr_from_path(_repo_example("graphite_iel10_classic.input"))
    app.coh_edge_group_enable_var.set(True)
    app.coh_edge_group_bpd.set("50")
    assert "1 1 0 0 50 1.0 /" in app._generate_input_text()


# ---------- the phonopy modes clear the special modes ----------

def test_switching_to_a_phonopy_mode_clears_the_special_modes(app):
    """Hiding the sections CLEARS them, so the invalid state is unreachable.

    Before: nsk set in mode 0 survived the switch to mode 2 invisibly and
    Run failed with 'ncold/nsk pair-correlation options are not available
    with inelastic_mode=1/2' -- pointing at a control that was off screen.
    The refusal is correct physics and stays (Skold corrects the incoherent
    approximation, which modes 1/2 do not use); the fix is to clear the
    values when the section goes away.
    """
    _reset(app)
    app.nsk.set("2 — Skold")
    app.ncold.set("1 — Ortho-H")
    app.ska_dka.set("0.1")
    _set_text(app.ska_text, "1.0 1.1 0.9")
    app.cfrac.set("0.3")
    app.nss.set("1 — One secondary scatterer")
    app.aws.set("15.858")
    app.sps.set("3.761")

    app.iel_var.set("10 — Generalized (crystal structure)")
    app.inelastic_mode_var.set(2)
    assert app._code(app.ncold) == 0 and app._code(app.nsk) == 0
    assert app._code(app.nss) == 0
    assert app.ska_dka.get() == "0" and app.cfrac.get() == "0"
    assert app.ska_text.get("1.0", "end").strip() == ""
    assert app.aws.get() == "0" and app.sps.get() == "0"
    # the note says the values are gone and why
    note = app._special_modes_note.cget("text")
    assert "CLEARED" in note and "force constants" in note
    assert "does not restore" in note

    # ... and the values do NOT come back with the sections
    app.inelastic_mode_var.set(0)
    assert app._code(app.nsk) == 0 and app._code(app.nss) == 0
    _reset(app)


def test_mode0_deck_with_nsk_then_mode2_generates_cleanly(app, tmp_path):
    """The reproducer end to end: a valid mode-0 nsk deck, switched to
    mode 2, generates a runnable deck instead of raising on hidden state."""
    _reset(app)
    app.nsk.set("2 — Skold")
    app.ska_dka.set("0.1")
    _set_text(app.ska_text, "1.0 1.1 0.9")
    app.cfrac.set("0.3")
    text_mode0 = app._generate_input_text()        # valid as it stands
    assert "1 0 0 2 /" in text_mode0               # Card 5: iel=0, nsk=2
    _run_engine(text_mode0, tmp_path)

    # switch to the phonopy-backed mode: iel=10 + a structure + Cards 6f/6g
    _setup_iel10(app)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set("/nonexistent/phonopy.yaml")
    text = app._generate_input_text()              # raised before the fix
    assert "1 10 0 0 /" in text                    # Card 5: ncold=nsk=0
    assert any(ln.strip().startswith("1 1 0 2")
               for ln in text.splitlines())        # Card 6b: mode 2
    _reset(app)


# ---------- iel coupling ----------

def test_phonopy_modes_restrict_iel_choices_to_generalized(app):
    """Modes 1/2 are only defined for iel=10 (generation coerces the mode to
    0 otherwise), so selecting one narrows the iel menu to the generalized
    entry and switches the selection to it; mode 0 restores the full menu."""
    _reset(app)
    app.iel_var.set("1 — Graphite (legacy)")
    assert len(app._iel_combo.cget("values")) == 8

    app.inelastic_mode_var.set(2)
    assert list(app._iel_combo.cget("values")) == [
        "10 — Generalized (crystal structure)"]
    assert app._code(app.iel_var) == 10

    app.inelastic_mode_var.set(0)
    assert len(app._iel_combo.cget("values")) == 8
    assert app._code(app.iel_var) == 10       # mode 0 leaves the selection alone

    app.iel_var.set("3 — BeO (legacy)")
    app.inelastic_mode_var.set(1)       # mode 1 couples the same way
    assert list(app._iel_combo.cget("values")) == [
        "10 — Generalized (crystal structure)"]
    assert app._code(app.iel_var) == 10
    app.iel_var.set("0 — None")         # a classic iel under mode 1/2 snaps back
    assert app._code(app.iel_var) == 10
    _reset(app)


def test_classic_deck_import_restores_full_iel_choices(app, tmp_path):
    """Importing a classic deck out of a mode-1/2 form must restore the full
    iel menu and land on the deck's own iel, not leave the menu pinned to
    the generalized entry."""
    _reset(app)
    app.iel_var.set("1 — Graphite (legacy)")
    app.za.set("6012")
    app.awr.set("11.907856")
    app.spr.set("4.724629")
    deck = tmp_path / "classic_iel1.input"
    deck.write_text(app._generate_input_text())

    _reset(app)
    app.inelastic_mode_var.set(2)                  # menu now restricted
    assert len(app._iel_combo.cget("values")) == 1
    app._import_leapr_from_path(str(deck))
    assert len(app._iel_combo.cget("values")) == 8
    assert app._code(app.iel_var) == 1
    assert int(app.inelastic_mode_var.get()) == 0
    _reset(app)


def test_classic_iel1_roundtrip_unaffected_by_restructure(app, tmp_path):
    """The section move is layout only: a classic legacy deck must still
    survive export -> import -> export byte-for-byte."""
    _reset(app)
    app.iel_var.set("1 — Graphite (legacy)")
    app.za.set("6012")
    app.awr.set("11.907856")
    app.spr.set("4.724629")
    text1 = app._generate_input_text()

    deck = tmp_path / "iel1_roundtrip.input"
    deck.write_text(text1)
    _reset(app)
    app._import_leapr_from_path(str(deck))
    assert app._generate_input_text() == text1


def test_clicking_mode_2_selects_linlin_and_setting_the_variable_does_not(app):
    """A click on the inelastic_mode radio buttons sets the Card 4 iint
    default that suits the mode (lin-lin for the coherent law, log-lin
    otherwise); a programmatic set of the variable, which is what a deck
    import does before it applies the deck's own iint, changes nothing."""
    app.inelastic_mode_var.set(0)
    app.iint.set(app.IINT_LOGLIN)
    app.inelastic_mode_var.set(2)                 # trace only, no click
    assert app._code(app.iint) == 0
    app._on_inelastic_mode_click()                # the click
    assert app._code(app.iint) == 1
    app.inelastic_mode_var.set(1)
    app._on_inelastic_mode_click()
    assert app._code(app.iint) == 0
    # the user's later choice is kept until the next click
    app.inelastic_mode_var.set(2)
    app._on_inelastic_mode_click()
    app.iint.set(app.IINT_LOGLIN)
    app.inelastic_mode_var.set(2)
    assert app._code(app.iint) == 0
    app.inelastic_mode_var.set(0)
    app._on_inelastic_mode_click()
