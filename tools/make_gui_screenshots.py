"""Regenerate the GUI screenshots referenced by ``docs/gui.md``.

Each scene stages one documented GUI state in a live Tk window (the real
:class:`irma.gui.app.IrmaApp`, not a mock-up) and captures the window with the
macOS ``screencapture`` utility, writing the PNGs that ``docs/gui.md`` embeds
from ``docs/assets/gui/``.

Requirements
------------
* macOS with an unlocked display session (the window must actually render).
* Screen Recording permission for the terminal running this script
  (System Settings > Privacy & Security > Screen Recording). Without it,
  ``screencapture`` fails with "could not create image from rect".
* The ``euphonic_env`` interpreter (matplotlib for the plot scenes).

Usage
-----
::

    python tools/make_gui_screenshots.py --list
    python tools/make_gui_screenshots.py                  # static scenes
    python tools/make_gui_screenshots.py --with-runs      # + the scenes that
                                                          #   need a finished run
    python tools/make_gui_screenshots.py --only gui_scattering gui_grids
    python tools/make_gui_screenshots.py --dry-run        # stage, skip capture

The ``--with-runs`` scenes execute real calculations (the committed graphite
examples): the classic ``iel=10`` TSL deck for the Run tab (~1-2 min), the
mode-0 VISION spectrum (~seconds) and the ARCS map example (~1-2 min).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "assets" / "gui"
GRAPHITE_PHONOPY = (REPO / "tests" / "mode2_euphonic_n1_validation"
                    / "graphite" / "phonopy.yaml")
TSL_EXAMPLE = REPO / "examples" / "tsl" / "graphite_iel10_classic.input"
SPECTRA_DIR = REPO / "examples" / "spectra"

# One window size for every scene: the docs page reads as a set, and a
# column of images that each jump to a different aspect ratio does not.
# Sized for the widest content, the Neutron Scattering plot pane, which
# carries the VISION spectrum and the 2-D S(Q,E) map.
SHOT_SIZE = "2000x1150"

IEL10_LABEL = "10 — Generalized (crystal structure)"
RUN_TIMEOUT_S = 30 * 60


def _silence_dialogs():
    """Route modal message boxes to stderr so automation never blocks."""
    def report(kind):
        def _show(title="", message="", **_kw):
            print(f"[{kind}] {title}: {message}", file=sys.stderr)
            return "ok"
        return _show
    messagebox.showinfo = report("info")
    messagebox.showwarning = report("warning")
    messagebox.showerror = report("error")


def _pump(root, seconds=0.3):
    """Run the Tk event loop for ``seconds`` so Aqua finishes drawing."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def _form_canvas(panel):
    """The NS panel's scrolling form canvas (first Canvas descendant)."""
    stack = list(panel.winfo_children())
    while stack:
        w = stack.pop(0)
        if isinstance(w, tk.Canvas):
            return w
        stack.extend(w.winfo_children())
    raise RuntimeError("form canvas not found")


def _scroll_to(panel, widget, root):
    """Scroll the form so ``widget`` sits near the top of the visible area —
    each capture must SHOW the section it documents, not the form top."""
    root.update()
    canvas = _form_canvas(panel)
    bbox = canvas.bbox("all")
    total = max(1, bbox[3] - bbox[1])
    y = widget.winfo_rooty() - canvas.winfo_rooty() + canvas.canvasy(0)
    canvas.yview_moveto(max(0.0, (y - 24) / total))


def _wait_for_runner(root, runner, timeout_s=RUN_TIMEOUT_S):
    """Wait for the shared ComputationRunner inside a REAL mainloop.

    The runner's reader thread delivers its done-callback via ``after()``,
    which only registers while the main thread is actually inside
    ``mainloop()`` -- a pump loop (update() + sleep) raises
    'main thread is not in main loop' in the worker and the plot is never
    drawn. So: poll on a timer and quit the loop when the run ends."""
    t0 = time.time()

    def _check():
        if not runner.is_running or time.time() - t0 > timeout_s:
            root.quit()
        else:
            root.after(50, _check)

    root.after(50, _check)
    root.mainloop()
    if runner.is_running:
        runner.cancel()
        raise TimeoutError(f"calculation exceeded {timeout_s} s")
    _pump(root, 1.0)        # let the done-callbacks (plotting) land


# Greedy up to the last separator, so the whole temp directory goes and the
# bare filename stays.
_TEMP_PATH = re.compile(r"/(?:private/)?(?:var/folders|tmp)/[^\s'\"]*/")
_REPO_PATH = re.compile(re.escape(str(REPO)) + r"/[^\s'\"]*")


def _repo_relative(match):
    """Repo-relative form of an absolute path, with any ``..`` resolved:
    the scenes rebase the examples' relative paths onto the repo root, and
    ``examples/spectra/../../tests/...`` is not what a reader would type."""
    return os.path.relpath(os.path.normpath(match.group(0)), REPO)


def _tidy_paths(app):
    """Rewrite this machine's paths out of every visible field and log.

    Scenes hand the GUI absolute paths so a run resolves whatever the cwd
    is, and runs write into a temp directory. Neither belongs in a
    published image: a reader should see the repo-relative path they would
    type themselves, and a temp directory both looks like debris and dates
    the screenshot. Only widgets whose text actually changes are touched,
    so scroll positions elsewhere in the form are left alone.
    """
    def clean(text):
        return _TEMP_PATH.sub("", _REPO_PATH.sub(_repo_relative, text))

    def walk(widget):
        for child in widget.winfo_children():
            try:
                kind = child.winfo_class()
                if kind in ("Entry", "TEntry"):
                    name = child.cget("textvariable")
                    if name:
                        value = child.getvar(name)
                        if isinstance(value, str) and clean(value) != value:
                            child.setvar(name, clean(value))
                elif kind == "Text":
                    body = child.get("1.0", "end-1c")
                    if clean(body) != body:
                        state = child.cget("state")
                        child.configure(state="normal")
                        child.delete("1.0", "end")
                        child.insert("1.0", clean(body))
                        child.see("end")
                        child.configure(state=state)
            except tk.TclError:
                pass
            walk(child)

    walk(app.root)


def _flat_frame(path, threshold=0.65):
    """True when one colour covers most of the image.

    A notebook page that has not redrawn when the frame is grabbed comes out
    as a flat grey rectangle. Measured over the committed set, a real scene
    tops out near 46% single-colour and a blank one sits above 85%.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    with Image.open(path) as im:
        small = im.convert("RGB").resize((160, 100))
        counts = small.getcolors(160 * 100) or []
    return bool(counts) and max(counts)[0] / 16000.0 >= threshold


def _window_id():
    """CGWindowID of this process's own app window, or None.

    Used to capture the window's backing store instead of a screen
    rectangle. A rectangle grab takes whatever is on screen, so a system
    dialog appearing over the app (a helper-tool authorization prompt, say)
    is baked into the published image; capturing the window by id cannot
    see anything in front of it, and leaves the rounded corners
    transparent rather than showing the desktop through them.
    """
    try:
        import Quartz
    except ImportError:
        return None                     # fall back to the rectangle grab
    pid = os.getpid()
    options = (Quartz.kCGWindowListOptionOnScreenOnly
               | Quartz.kCGWindowListExcludeDesktopElements)
    best, best_area = None, 0
    for info in Quartz.CGWindowListCopyWindowInfo(
            options, Quartz.kCGNullWindowID) or []:
        if info.get("kCGWindowOwnerPID") != pid:
            continue
        # No layer filter: capture() raises the window to topmost, which
        # moves it off layer 0. Largest area picks the app window over any
        # transient panel Tk may have open.
        bounds = info.get("kCGWindowBounds") or {}
        area = bounds.get("Width", 0) * bounds.get("Height", 0)
        if area > best_area:
            best, best_area = info.get("kCGWindowNumber"), area
    return best


def capture(root, out_path, attempts=3):
    """Capture the app window (including its title bar) to ``out_path``."""
    root.lift()
    root.attributes("-topmost", True)
    root.focus_force()          # colour the title bar: an unfocused window
    _pump(root, 0.8)            # is greyed out and looks like a dead capture
    root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    title_h = y - root.winfo_y()
    if not 0 < title_h <= 50:
        title_h = 28            # standard macOS title-bar height in points
    rect = f"{x},{y - title_h},{w},{h + title_h}"
    if os.environ.get("IRMA_SHOT_DEBUG"):
        print(f"    rect={rect} screen={root.winfo_screenwidth()}x"
              f"{root.winfo_screenheight()} state={root.state()}")
    win = _window_id()
    grab = (["screencapture", "-x", "-o", f"-l{win}", str(out_path)] if win
            else ["screencapture", "-x", f"-R{rect}", str(out_path)])
    if os.environ.get("IRMA_SHOT_DEBUG"):
        print(f"    {'window id ' + str(win) if win else 'rect ' + rect}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for attempt in range(1, attempts + 1):
            subprocess.run(grab, check=True, capture_output=True, text=True)
            if not _flat_frame(out_path):
                return
            if attempt < attempts:
                print(f"    blank frame, forcing a relayout "
                      f"(attempt {attempt}/{attempts})")
                # A notebook page that was unmapped and remapped can stay
                # undrawn until a <Configure> reaches it, and a window that
                # never changes size never sends one. Nudge the height by a
                # pixel and put it back: same geometry, two Configure events.
                root.geometry(f"{w}x{h + 1}")
                _pump(root, 0.6)
                root.geometry(f"{w}x{h}")
                _pump(root, 0.9)
                root.update()
        raise RuntimeError(
            f"{out_path.name} came out blank after {attempts} attempts: the "
            "notebook page never redrew")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "screencapture failed"
            f" ({(exc.stderr or '').strip() or exc}).\n"
            "This usually means the terminal lacks Screen Recording "
            "permission (System Settings > Privacy & Security > Screen "
            "Recording) or the display session is locked."
        ) from exc
    finally:
        root.attributes("-topmost", False)
    try:
        shown = out_path.relative_to(REPO)
    except ValueError:                  # --out outside the repo
        shown = out_path
    print(f"  wrote {shown}")


# ---------------------------------------------------------------------------
# Scenes. Each function stages one documented state; staging is idempotent
# and self-contained (every scene sets the fields it depends on).
# ---------------------------------------------------------------------------

def _endf_part(app, title):
    """Select the ENDF page and scroll the named part to the top (the five
    former sub-tabs are one scrolling page since the single-page merge)."""
    app.top_notebook.select(0)
    app.root.update()
    app.scroll_to_part(title)


def _ns_tab(app):
    app.top_notebook.select(1)
    return app.ns_panel


def scene_material_iel10(app, run=False):
    """Material tab: iel=10, inelastic_mode=2, phonopy section disclosed."""
    _endf_part(app, "Material")
    app.iel_var.set(IEL10_LABEL)
    app.inelastic_mode_var.set(2)
    app.nc_phonopy_yaml.set(str(GRAPHITE_PHONOPY))


def scene_material_iel1(app, run=False):
    """Material tab: classic iel=1 — crystal/phonopy sections collapsed."""
    _endf_part(app, "Material")
    app.iel_var.set("1 — Graphite (legacy)")


def scene_scattering(app, run=False):
    """Scattering tab with the default principal-scatterer fields."""
    app.iel_var.set(IEL10_LABEL)
    _endf_part(app, "Scattering")


def scene_grids(app, run=False):
    """Grids tab in automatic mode with the live size preview filled."""
    _endf_part(app, "Grids")
    # Max phonon freq now ships blank (only the user's model knows it), and
    # the preview refuses to run without it. Take the value the way the form
    # intends, from the model, rather than typing one in: the worker is what
    # the Detect button starts, minus the file dialog.
    if not app.freq_max.get().strip():
        app._detect_freq_from_fc_worker(str(GRAPHITE_PHONOPY))
        _pump(app.root, 0.5)
    # The preview also needs the mass, and material identity ships blank.
    # These are the graphite values of examples/tsl/graphite_mode2.input.
    if not app.awr.get().strip():
        app.awr.set("11.898")
    app._preview_grids()


def scene_phonon(app, run=False):
    """Phonon tab with the modes-1/2 'cards are not read' banner shown."""
    app.iel_var.set(IEL10_LABEL)
    app.inelastic_mode_var.set(2)
    _endf_part(app, "Phonon")


def scene_run(app, run=False):
    """Run tab; with --with-runs, a completed classic graphite calculation."""
    if run:
        app._import_leapr_from_path(str(TSL_EXAMPLE))
        out_dir = tempfile.mkdtemp(prefix="irma_shot_")
        app.output_file.set(str(Path(out_dir) / "graphite.endf"))
        app._generate_input_text()      # raise config problems here, not in a dialog
        _endf_part(app, "Run")
        app._run()
        _wait_for_runner(app.root, app.runner)
    else:
        _endf_part(app, "Run")


def _stage_ns_phonopy_mode2(panel):
    panel.input_source.set("phonopy")
    panel.phonopy_yaml.set(str(GRAPHITE_PHONOPY))
    panel.inelastic_mode.set("2 (coherent 1ph+multi)")
    panel._sync_ns_context()


def scene_ns_phonopy_mode2(app, run=False):
    """Neutron Scattering tab: phonopy input, mode 2, default physics."""
    panel = _ns_tab(app)
    _stage_ns_phonopy_mode2(panel)


def scene_ns_dosfiles_mode0(app, run=False):
    """Neutron Scattering tab: the committed DOS-files (mode 0) example."""
    from irma.spectra.config import load
    panel = _ns_tab(app)
    cfg = _absolutize(load(str(SPECTRA_DIR / "graphite_mode0_dosfile.yaml")),
                      SPECTRA_DIR)
    panel.load_config(cfg)
    panel.input_source.set("dos_files")
    panel._sync_ns_context()


def scene_ns_indirect_resolution(app, run=False):
    """Indirect (VISION) geometry with the width-polynomial c0/c1/c2 fields."""
    panel = _ns_tab(app)
    _stage_ns_phonopy_mode2(panel)
    for widget, value in ((panel.e_min, "0.0"), (panel.e_max, "250.0"),
                          (panel.de, "0.5"), (panel.dq, "0.05")):
        widget.set(value)
    panel.geom_nb.select(0)
    _scroll_to(panel, panel.geom_nb, app.root)   # subject: the resolution fields


def scene_ns_direct_map_chopper(app, run=False):
    """Direct tab: Ei=300, chopper resolution (ARCS), 2-D map output."""
    panel = _ns_tab(app)
    _stage_ns_phonopy_mode2(panel)
    panel.geom_nb.select(1)
    panel.dir_ei.set("300.0")
    panel.e_min.set("-100")
    panel.dir_res_model.set("chopper (auto, real instrument)")
    panel._on_res_model_change()        # chopper coverage pre-fill + mask default
    panel.dir_output.set("2-D map")
    panel._sync_output()
    _scroll_to(panel, panel.geom_nb, app.root)   # subject: chopper + map controls



def scene_mlip_build(app, run=False):
    """MLIP phonon models tab: the build form staged with the graphite
    settings of the examples page (nequip, 6x6x1 supercell, 40^3 mesh)."""
    app.top_notebook.select(3)
    panel = app.mlip_panel
    panel.structure.set("graphite.vasp")
    panel.potential.set("nequip")
    panel.supercell.set("6 6 1")
    panel.mesh.set("40 40 40")
    panel.snap_symmetry.set(True)
    panel.jobs.set("9")
    panel.outdir.set("bundle_nequip")


def _absolutize(cfg, base):
    """Rebase the relative path fields of a loaded example config onto its
    own directory: the GUI child process anchors at the launch cwd (the repo
    root here), not at the config file's location, so the committed examples'
    relative dos_file/phonopy paths would not resolve and the run would fail
    before plotting."""
    m = cfg.material
    for field in ("phonopy_yaml", "born", "force_constants", "force_sets"):
        v = getattr(m, field)
        if v and not os.path.isabs(v):
            setattr(m, field, str(base / v))
    for sc in m.scatterers:
        if sc.dos_file and not os.path.isabs(sc.dos_file):
            sc.dos_file = str(base / sc.dos_file)
    return cfg

def scene_ns_plot_vision(app, run=False):
    """Plot pane: a finished VISION run of the mode-2 graphite example
    (the full coherent treatment -- the production-quality spectrum)."""
    from irma.spectra.config import load
    panel = _ns_tab(app)
    cfg = _absolutize(load(str(SPECTRA_DIR / "graphite_mode2_vision.yaml")),
                      SPECTRA_DIR)
    panel.load_config(cfg)
    if not run:
        panel.build_config()            # config must at least validate
        return
    out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out.close()
    panel.output.set(out.name)
    panel._run()
    _wait_for_runner(app.root, app.runner)
    panel._out_nb.select(1)             # Plot pane (auto-plotted on success)
    panel.output.set("graphite_mode2_vision.csv")   # display name for capture
    panel._form_canvas.yview_moveto(0.0)            # form top, not mid-scroll


def scene_ns_plot_map(app, run=False):
    """Plot pane: a finished ARCS 2-D map (the committed example upgraded to
    inelastic_mode=2 -- plots should showcase the full coherent physics)."""
    from irma.spectra.config import load
    panel = _ns_tab(app)
    cfg = _absolutize(load(str(SPECTRA_DIR / "graphite_arcs_map.yaml")),
                      SPECTRA_DIR)
    cfg.physics.inelastic_mode = 2
    cfg.material.mesh = [40, 40, 40]    # production grid: prettier map
    cfg.grid.de_meV = 0.5
    cfg.grid.dq_max_invA = 0.05
    panel.load_config(cfg)
    panel.dir_output.set("2-D map")
    panel._sync_output()
    if not run:
        panel.build_config()
        return
    panel._run()
    _wait_for_runner(app.root, app.runner)
    panel._out_nb.select(1)
    panel.yscale.set("log")
    panel._replot()


# scene name (= PNG basename) -> (stager, needs --with-runs)
def scene_ncrystal_export(app, run=False):
    """NCrystal plugin tab: the export form staged with the graphite
    mode-2 settings of the tutorial (vendored phonopy model, 40^3 mesh,
    296 K, material_id graphite)."""
    app.top_notebook.select(2)
    panel = app.ncrystal_panel
    panel.phonopy_yaml.set(
        "tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml")
    panel.mesh.set("40 40 40")
    panel.temperature.set("296")
    panel.material_id.set("graphite")


def scene_mlip_build_mgo(app, run=False):
    """MLIP phonon models tab staged as the tutorial's MgO build: the
    committed MgO.cif, nequip, everything else left at the defaults."""
    app.top_notebook.select(3)
    panel = app.mlip_panel
    panel.structure.set("examples/mlip/MgO.cif")
    panel.potential.set("nequip")
    panel.outdir.set("mgo_bundle")


SCENES = {
    "gui_material_iel10": (scene_material_iel10, False),
    "gui_material_iel1": (scene_material_iel1, False),
    "gui_scattering": (scene_scattering, False),
    "gui_grids": (scene_grids, False),
    "gui_phonon": (scene_phonon, False),
    "gui_run": (scene_run, True),
    "gui_mlip_build": (scene_mlip_build, False),
    "gui_ncrystal_export": (scene_ncrystal_export, False),
    "gui_mlip_build_mgo": (scene_mlip_build_mgo, False),
    "gui_ns_phonopy_mode2": (scene_ns_phonopy_mode2, False),
    "gui_ns_dosfiles_mode0": (scene_ns_dosfiles_mode0, False),
    "gui_ns_indirect_resolution": (scene_ns_indirect_resolution, False),
    "gui_ns_direct_map_chopper": (scene_ns_direct_map_chopper, False),
    "gui_ns_plot_vision": (scene_ns_plot_vision, True),
    "gui_ns_plot_map": (scene_ns_plot_map, True),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the docs/gui.md screenshots.")
    parser.add_argument("--only", nargs="+", metavar="SCENE",
                        help="capture only these scenes (see --list)")
    parser.add_argument("--with-runs", action="store_true",
                        help="also execute the scenes that need a finished "
                             "calculation (Run tab and the two Plot panes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="stage every scene but skip the capture call")
    parser.add_argument("--out", type=Path, default=OUT_DIR,
                        help=f"output directory (default {OUT_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="list scene names and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, (fn, needs_run) in SCENES.items():
            tag = "  [--with-runs]" if needs_run else ""
            print(f"{name:30s} {fn.__doc__.splitlines()[0]}{tag}")
        return 0

    names = args.only or list(SCENES)
    unknown = [n for n in names if n not in SCENES]
    if unknown:
        parser.error(f"unknown scene(s): {', '.join(unknown)} (see --list)")

    if sys.platform != "darwin" and not args.dry_run:
        parser.error("window capture uses macOS screencapture; "
                     "use --dry-run elsewhere")

    _silence_dialogs()
    from irma.gui.app import IrmaApp
    root = tk.Tk()
    app = IrmaApp(root)
    root.geometry(f"{SHOT_SIZE}+30+30")
    _pump(root, 0.5)

    failures = []
    for name in names:
        fn, needs_run = SCENES[name]
        run = needs_run and args.with_runs and not args.dry_run
        if needs_run and not args.with_runs and not args.dry_run:
            print(f"skipping {name} (needs --with-runs)")
            continue
        print(f"scene {name}{' [run]' if run else ''} ...")
        try:
            root.geometry(SHOT_SIZE)
            fn(app, run=run)
            root.geometry(SHOT_SIZE)   # a scene must not resize the set
            _tidy_paths(app)
            _pump(root, 0.5)
            if args.dry_run:
                print(f"  staged ok ({root.winfo_width()}x{root.winfo_height()})")
            else:
                capture(root, args.out / f"{name}.png")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)

    root.destroy()
    if failures:
        print(f"\n{len(failures)} scene(s) failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
