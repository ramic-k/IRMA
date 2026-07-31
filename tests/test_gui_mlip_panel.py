"""MLIP GUI panel: widget->argv mapping and DOS plumbing, headless.

Requires tkinter (skips without it). Uses a withdrawn root, so no display
is needed; the runner is a stub -- no subprocess ever starts.
"""
import json

import pytest

# importorskip FIRST -- a hard `import tkinter` at module top would error
# at collection on ase-only environments.
tk = pytest.importorskip("tkinter")
pytest.importorskip("ase")
pytest.importorskip("yaml")

from irma.gui.mlip_panel import MlipPanel  # noqa: E402
from irma.mlip.calculators import POTENTIALS  # noqa: E402


class StubRunner:
    is_running = False

    def __init__(self):
        self.commands = []

    def run_command(self, cmd, **kw):
        self.commands.append((cmd, kw))


@pytest.fixture()
def panel():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display / Tk unavailable: {exc}")
    root.withdraw()
    p = MlipPanel(root, runner=StubRunner())
    yield p
    root.destroy()


def test_panel_builds_and_lists_all_potentials(panel):
    assert tuple(panel.potential.combo["values"]) == POTENTIALS


def test_build_command_minimal_and_full(panel, tmp_path):
    with pytest.raises(ValueError, match="structure"):
        panel.build_command()
    struct = tmp_path / "cell.vasp"
    struct.write_text("x")
    panel.structure.set(str(struct))
    with pytest.raises(ValueError, match="output directory"):
        panel.build_command()
    panel.outdir.set(str(tmp_path / "bundle"))

    cmd = panel.build_command()
    assert cmd[3:6] == ["irma", "mlip", "build"]
    assert "--potential" in cmd
    # defaults are OMITTED from the argv (the CLI owns its defaults)
    for flag in ("--jobs", "--worker-threads", "--delta", "--fmax",
                 "--model", "--supercell", "--mesh", "--born",
                 "--relax-cell", "--disordered", "--overwrite"):
        assert flag not in cmd, flag

    panel.model.set("medium-omat-0")
    panel.supercell.set("4 4 3")
    panel.jobs.set("4")
    panel.worker_threads.set("2")
    panel.relax_cell.set(True)
    panel.disordered.set(True)
    panel.overwrite.set(True)
    cmd = panel.build_command()
    for flag in ("--model", "--supercell", "--jobs", "--worker-threads",
                 "--relax-cell", "--disordered", "--overwrite"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--jobs") + 1] == "4"
    assert cmd[cmd.index("--worker-threads") + 1] == "2"


def test_emit_command_targets_and_pairs(panel, tmp_path):
    with pytest.raises(ValueError, match="bundle"):
        panel.emit_command()
    panel.bundle.set(str(tmp_path))
    panel.mats.set("Be=26, O=48")
    cmd = panel.emit_command()
    assert "--to" in cmd and cmd[cmd.index("--to") + 1] == "endf"
    assert cmd.count("--mat") == 2
    assert "Be=26" in cmd and "O=48" in cmd

    for name, var in panel.targets.items():
        var.set(True)
    cmd = panel.emit_command()
    assert cmd[cmd.index("--to") + 1] == "endf,spectra,ncrystal"

    for var in panel.targets.values():
        var.set(False)
    with pytest.raises(ValueError, match="target"):
        panel.emit_command()
    panel.targets["endf"].set(True)

    panel.mats.set("C31")                     # malformed pair
    with pytest.raises(ValueError, match="SYM=VALUE"):
        panel.emit_command()


def test_run_build_goes_through_the_stub_runner(panel, tmp_path):
    struct = tmp_path / "cell.vasp"
    struct.write_text("x")
    panel.structure.set(str(struct))
    panel.outdir.set(str(tmp_path / "b"))
    panel._run_build()
    assert len(panel.runner.commands) == 1
    cmd, kw = panel.runner.commands[0]
    assert cmd[3:6] == ["irma", "mlip", "build"]
    assert "error_label" in kw


def test_dos_plot_reads_bundle_data(panel, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    # errors surface as dialogs; capture instead of popping UI
    seen = {}
    monkeypatch.setattr("irma.gui.mlip_panel.messagebox",
                        type("MB", (), {"showerror":
                                        staticmethod(lambda *a: seen.update(
                                            err=a))})())
    panel._plot_dos(str(tmp_path))            # no dos.dat yet
    assert "err" in seen

    (tmp_path / "dos.dat").write_text(
        "\n".join(f"{e} {max(0.0, 10 - abs(e - 10))}"
                  for e in range(0, 21)))
    json.dump({"phonons": {"freq_max_meV": 20.0, "n_imaginary": 0,
                           "mesh": [4, 4, 4]},
               "calculator": {"potential": "emt"}},
              open(tmp_path / "manifest.json", "w"))
    panel._plot_dos(str(tmp_path))
    assert panel._dos_data is not None
    assert "emt" in panel._dos_data[1]
    assert panel._dos_canvas is not None
    # log-scale replot survives zero bins
    panel.dos_yscale.set("log")
    panel._replot_dos()


def test_cleanup_temp_files_is_a_safe_noop(panel):
    panel.cleanup_temp_files()


def test_env_summary_never_raises(panel, monkeypatch, tmp_path):
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))
    text = panel._env_summary()
    assert "registered" in text


def test_build_command_jitter_cycles(panel):
    """--jitter-cycles maps onto the CLI: absent at default 0, valued
    when set."""
    panel.structure.set("x.vasp")
    panel.outdir.set("out")
    cmd = panel.build_command()
    assert "--jitter-cycles" not in cmd
    panel.jitter_cycles.set("3")
    cmd = panel.build_command()
    assert cmd[cmd.index("--jitter-cycles") + 1] == "3"


def test_emit_command_deck_and_ncrystal_selectors(panel, tmp_path):
    """SPG-4: --inelastic-mode / --elastic-format / --material-id are
    selectable from the panel ('default' omits the flag; explicit values --
    including an explicit 'mef', which the CLI distinguishes from the absent
    flag -- are forwarded), and the argv parses on the real CLI surface."""
    panel.bundle.set(str(tmp_path))
    cmd = panel.emit_command()
    for flag in ("--inelastic-mode", "--elastic-format", "--material-id"):
        assert flag not in cmd, flag           # defaults omit the flags

    panel.emit_inelastic_mode.set("0")
    panel.emit_elastic_format.set("sef")
    panel.material_id.set("graphite_296K")
    panel.targets["ncrystal"].set(True)
    cmd = panel.emit_command()
    assert cmd[cmd.index("--inelastic-mode") + 1] == "0"
    assert cmd[cmd.index("--elastic-format") + 1] == "sef"
    assert cmd[cmd.index("--material-id") + 1] == "graphite_296K"

    # explicit 'mef' names the CLI default but is NOT the same as omitting
    # the flag (the CLI rejects it for disordered bundles): forwarded.
    panel.emit_elastic_format.set("mef")
    cmd = panel.emit_command()
    assert cmd[cmd.index("--elastic-format") + 1] == "mef"

    # the emitted argv must agree with the current CLI surface
    from irma.mlip.cli import _build_parser
    ns = _build_parser().parse_args(cmd[5:])
    assert ns.inelastic_mode == 0
    assert ns.elastic_format == "mef"
    assert ns.material_id == "graphite_296K"
    # and with the selectors at 'default', argparse keeps its None defaults
    panel.emit_inelastic_mode.set("default")
    panel.emit_elastic_format.set("default")
    panel.material_id.set("")
    ns = _build_parser().parse_args(panel.emit_command()[5:])
    assert ns.inelastic_mode is None
    assert ns.elastic_format is None
    assert ns.material_id is None


# ---- progressive disclosure: emit rows scoped to the targets that read them
def test_emit_rows_endf_only_fields(panel):
    """inelastic mode is forwarded to the ENDF deck emitter only, and
    allow-unstable is read by the endf and spectra emitters (never
    ncrystal): each shows exactly for the targets that read it."""
    # default: endf ticked
    assert panel.emit_inelastic_mode.winfo_manager() == "pack"
    assert panel._allow_unstable_row.winfo_manager() == "pack"
    panel.targets["endf"].set(False)                 # nothing ticked
    assert panel.emit_inelastic_mode.winfo_manager() == ""
    assert panel._allow_unstable_row.winfo_manager() == ""
    panel.targets["spectra"].set(True)               # spectra: unstable only
    assert panel.emit_inelastic_mode.winfo_manager() == ""
    assert panel._allow_unstable_row.winfo_manager() == "pack"
    panel.targets["spectra"].set(False)
    panel.targets["ncrystal"].set(True)              # ncrystal: neither
    assert panel.emit_inelastic_mode.winfo_manager() == ""
    assert panel._allow_unstable_row.winfo_manager() == ""
    panel.targets["endf"].set(True)
    assert panel.emit_inelastic_mode.winfo_manager() == "pack"
    assert panel._allow_unstable_row.winfo_manager() == "pack"


def test_emit_inelastic_mode_help_is_endf_scoped():
    from irma.gui.mlip_panel import HELP
    assert HELP["emit_inelastic_mode"].startswith("ENDF target only. ")


# ---- the argv must follow the disclosure, not just the widget contents -----
def test_emit_argv_drops_fields_the_targets_do_not_read(panel, tmp_path):
    """Hidden fields are neither validated nor forwarded.

    Filling the ENDF/NCrystal fields and then narrowing the targets used to
    leave the stale values in the argv: an NCrystal-only emit carried
    --mat/--inelastic-mode/--elastic-format, from rows the user could no
    longer see. The spectra target has no scoped field of its own, so a
    spectra-only emit must carry none of them.
    """
    panel.bundle.set(str(tmp_path))
    for var in panel.targets.values():
        var.set(True)
    panel.mats.set("C=31")
    panel.emit_inelastic_mode.set("1")
    panel.emit_elastic_format.set("sef")
    panel.material_id.set("graphite_296K")
    cmd = panel.emit_command()                    # everything applies
    for flag in ("--mat", "--inelastic-mode",
                 "--elastic-format", "--material-id"):
        assert flag in cmd, flag

    panel.targets["endf"].set(False)
    panel.targets["spectra"].set(False)           # NCrystal only
    cmd = panel.emit_command()
    assert cmd[cmd.index("--to") + 1] == "ncrystal"
    for flag in ("--mat", "--inelastic-mode", "--elastic-format"):
        assert flag not in cmd, flag
    assert cmd[cmd.index("--material-id") + 1] == "graphite_296K"

    panel.targets["ncrystal"].set(False)
    panel.targets["spectra"].set(True)            # spectra only
    cmd = panel.emit_command()
    assert cmd[cmd.index("--to") + 1] == "spectra"
    for flag in ("--mat", "--inelastic-mode", "--elastic-format",
                 "--material-id"):
        assert flag not in cmd, flag

    # always-applicable fields are unaffected by the target narrowing
    panel.temperature.set("500")
    cmd = panel.emit_command()
    assert cmd[cmd.index("--temperature") + 1] == "500"


def test_emit_argv_ignores_a_malformed_hidden_mat(panel, tmp_path):
    """A malformed MAT left over from an ENDF emit must not block an
    NCrystal-only emit -- the validation error names a hidden field."""
    panel.bundle.set(str(tmp_path))
    panel.mats.set("C31")                          # missing '='
    with pytest.raises(ValueError, match="SYM=VALUE"):
        panel.emit_command()                       # endf ticked: still fatal
    panel.targets["endf"].set(False)
    panel.targets["ncrystal"].set(True)
    cmd = panel.emit_command()                     # no raise
    assert "--mat" not in cmd
    assert cmd[cmd.index("--to") + 1] == "ncrystal"


def test_build_argv_forwards_only_the_live_thread_field(panel, tmp_path):
    """serial threads applies at jobs = 1, threads/worker at jobs > 1.

    The CLI reads --threads for the parent relaxation whatever jobs says, so
    a stale serial value forwarded from the hidden field silently capped the
    relaxation (and a nonnumeric one blocked Build from an unreachable
    field).
    """
    struct = tmp_path / "cell.vasp"
    struct.write_text("x")
    panel.structure.set(str(struct))
    panel.outdir.set(str(tmp_path / "bundle"))

    panel.threads.set("2")                         # serial field, jobs = 1
    panel.worker_threads.set("4")                  # hidden at jobs = 1
    assert panel.threads.winfo_manager() == "pack"
    assert panel.worker_threads.winfo_manager() == ""
    cmd = panel.build_command()
    assert cmd[cmd.index("--threads") + 1] == "2"
    assert "--worker-threads" not in cmd

    panel.jobs.set("4")                            # parallel: swaps the rows
    assert panel.worker_threads.winfo_manager() == "pack"
    assert panel.threads.winfo_manager() == ""
    cmd = panel.build_command()
    assert cmd[cmd.index("--jobs") + 1] == "4"
    assert cmd[cmd.index("--worker-threads") + 1] == "4"
    assert "--threads" not in cmd                  # stale serial value dropped

    # and a nonnumeric leftover in the hidden field cannot block Build
    panel.threads.set("two")
    cmd = panel.build_command()
    assert "--threads" not in cmd
    panel.jobs.set("1")                            # visible again: it must raise
    with pytest.raises(ValueError, match="serial threads"):
        panel.build_command()


# ---------------------------------------------------------------------------
# Per-species nuclear-data editor (replaces the `nuclides` / `species
# overrides` free-text fields). The panel stays a pure widget->argv mapping:
# the CLI's --nuclide / --species flags are unchanged and assembled here.
# ---------------------------------------------------------------------------
def _write_bundle_yaml(tmp_path, points):
    """A bundle directory carrying only a phonopy.yaml.

    Species discovery reads that file with the plain-YAML parser the NS
    tab uses, never ase and never the structure file, so a bundle stub
    this thin is exactly what the panel sees.
    """
    lines = ["primitive_cell:", "  lattice:", "  - [2.5, 0.0, 0.0]",
             "  points:"]
    for sym, mass in points:
        lines += [f"  - symbol: {sym}",
                  "    coordinates: [0.0, 0.0, 0.0]",
                  f"    mass: {mass}"]
    (tmp_path / "phonopy.yaml").write_text("\n".join(lines) + "\n")
    return str(tmp_path)


def _row(panel, symbol):
    for r in panel.species_table.rows:
        if r["_var"]["symbol"].get().strip() == symbol:
            return r
    raise AssertionError(f"no row for {symbol}")


def _set_mode(panel, symbol, mode, nuclide=None):
    r = _row(panel, symbol)
    r["_var"]["mode"].set(mode)
    panel.species_table.sync_nuclide_row(r)
    if nuclide is not None:
        r["_var"]["nuclide"].set(nuclide)
        panel.species_table.sync_nuclide_row(r)
    return r


def test_species_table_hints_until_a_bundle_is_set(panel, tmp_path):
    """No bundle: a short hint, not an empty grid."""
    assert panel.species_table.nuclide_rows() == []
    assert "select a bundle" in panel.species_table.hint

    bundle = _write_bundle_yaml(tmp_path, [("C", 12.011)])
    panel.bundle.set(bundle)
    assert [r["symbol"] for r in panel.species_table.nuclide_rows()] == ["C"]
    assert panel.species_table.hint == ""


def test_species_discovery_degrades_without_a_readable_phonopy_yaml(
        panel, tmp_path):
    """A missing, unparseable, or species-less phonopy.yaml gives a hint and
    no rows -- never an exception on the UI thread."""
    empty = tmp_path / "empty"
    empty.mkdir()
    panel.bundle.set(str(empty))                       # no phonopy.yaml
    assert panel.species_table.nuclide_rows() == []
    assert "could not read a species list" in panel.species_table.hint

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "phonopy.yaml").write_text("not: [a, valid, cell\n")
    panel.bundle.set(str(broken))
    assert panel.species_table.nuclide_rows() == []
    assert "could not read a species list" in panel.species_table.hint

    panel.bundle.set(str(tmp_path / "does-not-exist"))
    assert panel.species_table.nuclide_rows() == []


def test_species_table_rebuilds_when_the_bundle_changes(panel, tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    one = _write_bundle_yaml(tmp_path / "one", [("Be", 9.012)])
    two = _write_bundle_yaml(tmp_path / "two",
                             [("Be", 9.012), ("O", 15.999)])
    panel.bundle.set(one)
    assert [r["symbol"] for r in panel.species_table.nuclide_rows()] == ["Be"]
    panel.bundle.set(two)
    assert [r["symbol"] for r in panel.species_table.nuclide_rows()] \
        == ["Be", "O"]


def test_default_natural_rows_emit_no_identity_flags(panel, tmp_path):
    """The DEFAULT path must stay byte-identical to what the CLI already
    emitted: every row starts Natural and Natural contributes nothing, so
    the argv carries neither --nuclide nor --species."""
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("Be", 9.012),
                                                   ("O", 15.999)]))
    panel.mats.set("Be=26, O=48")
    rows = panel.species_table.nuclide_rows()
    assert [r["mode"] for r in rows] == ["natural", "natural"]
    cmd = panel.emit_command()
    assert "--nuclide" not in cmd and "--species" not in cmd
    # the constants that WILL be emitted are on screen, read-only
    from irma.core.nuclear_data import lookup
    be = _row(panel, "Be")
    assert be["_var"]["b_coh_fm"].get() == f"{lookup('Be').b_coh_fm:.6g}"
    assert str(be["_w"]["b_coh_fm"].cget("state")) == "readonly"
    # and the argv still parses on the real CLI surface
    from irma.mlip.cli import _build_parser
    ns = _build_parser().parse_args(cmd[5:])
    assert ns.nuclide is None and ns.species is None


def test_isotope_row_emits_nuclide(panel, tmp_path):
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("C", 12.011)]))
    r = _row(panel, "C")
    # the selector is disclosed only once the row asks for an isotope
    assert panel.species_table._cell_hidden(r, "nuclide") is True
    _set_mode(panel, "C", "isotope")
    assert panel.species_table._cell_hidden(r, "nuclide") is False
    assert "13-C" in r["_w"]["nuclide"].cget("values")

    r["_var"]["nuclide"].set("13-C")
    panel.species_table.sync_nuclide_row(r)
    cmd = panel.emit_command()
    assert cmd[cmd.index("--nuclide") + 1] == "C=13-C"
    assert "--species" not in cmd
    # identity AND constants come from the same isotope entry
    from irma.core.nuclear_data import lookup
    assert r["_var"]["b_coh_fm"].get() == f"{lookup('13-C').b_coh_fm:.6g}"

    from irma.mlip.cli import _build_parser
    assert _build_parser().parse_args(cmd[5:]).nuclide == ["C=13-C"]


def test_custom_row_emits_species(panel, tmp_path):
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("C", 12.011)]))
    r = _set_mode(panel, "C", "custom")
    assert str(r["_w"]["b_coh_fm"].cget("state")) == "normal"
    r["_var"]["b_coh_fm"].set("6.646")
    r["_var"]["sigma_inc_b"].set("0.001")
    r["_var"]["awr"].set("11.898")
    cmd = panel.emit_command()
    assert "--nuclide" not in cmd
    assert cmd[cmd.index("--species") + 1] == \
        "C:b_coh_fm=6.646,sigma_inc_b=0.001,awr=11.898"

    from irma.mlip.cli import _build_parser, _parse_species
    ns = _build_parser().parse_args(cmd[5:])
    assert _parse_species(ns.species) == {
        "C": {"b_coh_fm": 6.646, "sigma_inc_b": 0.001, "awr": 11.898}}

    # a non-numeric entry is caught by NAME here: the group syntax is
    # comma-delimited, so it would reach the CLI as a malformed group
    r["_var"]["b_coh_fm"].set("6,646")
    with pytest.raises(ValueError, match="species C b_coh_fm"):
        panel.emit_command()

    # a custom row with nothing in it is refused by name, not by flag
    for key in ("b_coh_fm", "sigma_inc_b", "awr"):
        r["_var"][key].set("")
    with pytest.raises(ValueError, match="species C"):
        panel.emit_command()


def test_energy_dependent_species_opens_its_own_row(panel, tmp_path):
    """The recovery happens in the row that caused it.

    Gd's tabulated scattering length is a resonance-region value; the CLI
    refuses to prefill it, and because the cross-target preflight resolves
    species before publishing anything, a failed emit used to leave no
    files to edit afterwards. Here the row opens itself instead.
    """
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("Gd", 157.25),
                                                   ("O", 15.999)]))
    rows = {r["symbol"]: r for r in panel.species_table.nuclide_rows()}
    assert rows["Gd"]["mode"] == "custom" and rows["Gd"]["flagged"] is True
    assert rows["Gd"]["b_coh_fm"] == ""       # never a silent prefill
    assert rows["Gd"]["complete"] is False
    assert rows["O"]["mode"] == "natural" and rows["O"]["flagged"] is False
    note = _row(panel, "Gd")["_note"].cget("text")
    assert "ENERGY-DEPENDENT" in note and "still needed" in note

    panel.mats.set("Gd=1, O=2")
    with pytest.raises(ValueError, match="energy-dependent"):
        panel.emit_command()

    gd = _row(panel, "Gd")
    gd["_var"]["b_coh_fm"].set("6.5")
    gd["_var"]["sigma_inc_b"].set("151.0")
    cmd = panel.emit_command()
    assert cmd[cmd.index("--species") + 1] == \
        "Gd:b_coh_fm=6.5,sigma_inc_b=151.0"


def test_energy_dependent_isotope_keeps_the_identity_it_recovered_from(
        panel, tmp_path):
    """The flag is per NUCLIDE, not per element: natural Li is fine and
    6-Li is not. Picking 6-Li opens the row for values but must not throw
    the isotope choice away -- the CLI composes identity (--nuclide) and
    constants (--species) independently."""
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("Li", 6.94)]))
    assert panel.species_table.nuclide_rows()[0]["mode"] == "natural"
    r = _set_mode(panel, "Li", "isotope", "7-Li")
    assert panel.species_table.nuclide_rows()[0]["mode"] == "isotope"

    r["_var"]["nuclide"].set("6-Li")
    panel.species_table.sync_nuclide_row(r)
    row = panel.species_table.nuclide_rows()[0]
    assert (row["mode"], row["nuclide"], row["flagged"]) \
        == ("custom", "6-Li", True)
    # the selector stays on screen so the identity is still visible
    assert panel.species_table._cell_hidden(r, "nuclide") is False
    r["_var"]["b_coh_fm"].set("2.0")
    r["_var"]["sigma_inc_b"].set("0.46")
    cmd = panel.emit_command()
    assert cmd[cmd.index("--nuclide") + 1] == "Li=6-Li"
    assert cmd[cmd.index("--species") + 1] == "Li:b_coh_fm=2.0,sigma_inc_b=0.46"


def test_isotope_mass_mismatch_warns_but_does_not_block(panel, tmp_path):
    """H/D: emit keeps pointing at the bundle's phonopy.yaml, so deuterium
    constants would ride on hydrogen phonon masses."""
    panel.bundle.set(_write_bundle_yaml(tmp_path, [("H", 1.008)]))
    r = _set_mode(panel, "H", "isotope", "1-H")
    assert r["_note"].cget("text") == ""            # matching mass: silent

    r["_var"]["nuclide"].set("2-H")
    panel.species_table.sync_nuclide_row(r)
    note = r["_note"].cget("text")
    assert "mass check" in note and "2-H" in note and "1.008" in note
    cmd = panel.emit_command()                      # warning only
    assert cmd[cmd.index("--nuclide") + 1] == "H=2-H"


def test_mass_check_is_silent_without_model_masses(panel, tmp_path):
    """A phonopy.yaml with no mass entries still yields species rows; the
    mass check simply has nothing to compare against."""
    (tmp_path / "phonopy.yaml").write_text(
        "primitive_cell:\n  points:\n  - symbol: H\n"
        "    coordinates: [0.0, 0.0, 0.0]\n")
    panel.bundle.set(str(tmp_path))
    r = _set_mode(panel, "H", "isotope", "2-H")
    assert r["_note"].cget("text") == ""


def test_bundle_species_reads_symbols_and_masses(tmp_path):
    from irma.gui.mlip_panel import bundle_species
    path = _write_bundle_yaml(tmp_path, [("Be", 9.012218), ("O", 15.9994)])
    symbols, masses = bundle_species(path)
    assert symbols == ["Be", "O"]
    assert masses == {"Be": 9.012218, "O": 15.9994}
    assert bundle_species(str(tmp_path / "nope")) == ([], {})
    assert bundle_species("") == ([], {})


def test_bundle_species_skips_the_embedded_force_constants(tmp_path):
    """A real bundle embeds its force constants, which are most of the
    file; the cells are written first, and the mass read stops there."""
    from irma.gui.mlip_panel import bundle_species
    yaml_path = tmp_path / "phonopy.yaml"
    _write_bundle_yaml(tmp_path, [("Be", 9.012218), ("O", 15.9994)])
    bulk = "\n".join(f"- - [{i}.0, 0.0, 0.0]" for i in range(2000))
    yaml_path.write_text(yaml_path.read_text()
                         + "force_constants:\n" + bulk + "\n")
    symbols, masses = bundle_species(str(tmp_path))
    assert symbols == ["Be", "O"]
    assert masses == {"Be": 9.012218, "O": 15.9994}
