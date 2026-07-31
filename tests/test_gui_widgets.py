"""GUI widget behavior.

Covers the HelpButton popup sizing -- a fixed 12-line, non-resizable,
scrollbar-less Text would clip the longest help texts (Bragg-edge grouping,
mpdir, ilog, ...) mid-sentence -- and the fresh-form builder defaults that
must match the documented production defaults (Card 6g '10000 1000 1',
mesh 40^3, Auto-size ON, LAT=1).

Requires a display (Tk); skipped headless (CI).
"""
import sys

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

from irma.gui.widgets import HelpButton  # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    r.withdraw()
    yield r
    r.destroy()


def _open_popup(root, message):
    btn = HelpButton(root, "Test Help", message)
    btn._show()
    top = next(w for w in btn.winfo_children() if isinstance(w, tk.Toplevel))
    top.update_idletasks()
    return top


def _find_all(widget, klass):
    found = []

    def _rec(w):
        for child in w.winfo_children():
            if isinstance(child, klass):
                found.append(child)
            _rec(child)

    _rec(widget)
    return found


def _resizable_flags(top):
    res = top.wm_resizable()
    if isinstance(res, str):
        res = res.split()
    return tuple(int(v) for v in res)


# ---------------- HelpButton popup sizing ----------------

@pytest.mark.skipif(sys.platform == "win32", reason="Tk display-line metrics differ on Windows; the popup sizing is cosmetic and pinned on X11/aqua")
def test_short_help_stays_compact(root):
    top = _open_popup(root, "one short line")
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) <= 5
    assert not _find_all(top, ttk.Scrollbar)     # no pointless scrollbar
    top.destroy()


@pytest.mark.skipif(sys.platform == "win32", reason="Tk display-line metrics differ on Windows; the popup sizing is cosmetic and pinned on X11/aqua")
def test_medium_help_sized_to_content_not_12_lines(root):
    # 20 short logical lines: the old fixed height=12 clipped lines 13-20.
    msg = "\n".join(f"line {i}" for i in range(1, 21))
    top = _open_popup(root, msg)
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) == 20
    assert not _find_all(top, ttk.Scrollbar)
    top.destroy()


def test_long_help_caps_height_and_gets_scrollbar(root):
    # Far beyond the cap (and wider than the wrap width, so display lines
    # exceed logical lines): height caps at _MAX_VISIBLE_LINES and a
    # scrollbar appears so the rest is discoverable.
    msg = "\n".join(f"line {i} " + "word " * 12 for i in range(1, 61))
    top = _open_popup(root, msg)
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) == HelpButton._MAX_VISIBLE_LINES
    assert _find_all(top, ttk.Scrollbar), "long help text needs a scrollbar"
    top.destroy()


def test_popup_is_vertically_resizable(root):
    top = _open_popup(root, "anything")
    assert _resizable_flags(top) == (0, 1)       # fixed width, free height
    top.destroy()


# ---------------- fresh-form builder defaults (IrmaApp) ----------------

@pytest.fixture(scope="module")
def fresh_app(root):
    from irma.gui.app import IrmaApp
    app_root = tk.Toplevel(root)
    app_root.withdraw()
    app = IrmaApp(app_root)
    yield app
    app_root.destroy()


def test_fresh_app_sampling_defaults_are_production_recommended(fresh_app):
    """A user who accepts every mode-1/2 default must get the documented
    recommended production sampling (Card 6g 10000 1000 1), not the manual's
    truncation example."""
    assert fresh_app.nc_num_directions.get() == "10000"
    assert fresh_app.nc_multiphonon_num_directions.get() == "1000"
    assert (fresh_app.nc_mesh_nx.get(), fresh_app.nc_mesh_ny.get(),
            fresh_app.nc_mesh_nz.get()) == ("40", "40", "40")
    assert int(fresh_app.nc_auto_order_var.get()) == 1   # safe by construction
    assert fresh_app.lat.get().startswith("1 —")         # matches reset default


def test_fresh_app_mode_radio_labels_match_ns_panel_wording(fresh_app):
    """One physics mode, one name: the ENDF tab must not call mode 2
    'coh. approx.' while the NS panel calls it exact/coherent."""
    radios = _find_all(fresh_app.root, ttk.Radiobutton)
    labels = {str(r.cget("text")) for r in radios}
    assert "2 — coherent (exact 1-phonon)" in labels
    assert "1 — incoherent approx." in labels
    assert not any("coh. approx" in lb for lb in labels)


def test_bragg_grouping_default_on_and_help_agrees(fresh_app):
    """SPG-5: the Bragg-edge grouping checkbox is DELIBERATELY constructed
    checked with 50 bins/decade prefilled (docs/gui.md: 'on by default');
    the in-app help must describe that default, not claim 'Off by default'
    / '0 (default) = OFF' as it used to."""
    import inspect
    from irma.gui import endf_form
    assert fresh_app.coh_edge_group_enable_var.get() is True
    assert fresh_app.coh_edge_group_bpd.get() == "50"
    assert fresh_app.coh_edge_group_thr.get() == "1.0"
    src = inspect.getsource(endf_form)
    assert "0 (default) = OFF" not in src
    assert "option). Off by default" not in src
    assert "The GUI enables it by default" in src
