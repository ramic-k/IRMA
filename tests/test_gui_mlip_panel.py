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
