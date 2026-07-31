"""IRMA GUI — Main application window.

Provides a tabbed interface for configuring and running thermal scattering
law calculations. Uses tkinter (Python standard library) for zero extra
GUI dependencies.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from irma.gui.runner import ComputationRunner

from irma.gui.endf_form import (   # noqa: F401  (re-exported for tests)
    EndfFormMixin, _phase_from_log_line,
)



class IrmaApp(EndfFormMixin):
    """Main IRMA GUI application."""

    def __init__(self, root):
        self.root = root
        self.root.title("IRMA — (In)elastic Representation of Materials "
                        "As S(α,β) evaluations")
        self.root.minsize(800, 600)

        # Card 6e partial spectra carried through import -> export verbatim
        # (the GUI has no editor for them; they are preserved, not edited).
        self._imported_partial_spectra = []

        # Card 2 title and Card 3 iprint have no widgets; they are carried
        # through import -> export so a round trip preserves them.
        self._imported_title = "IRMA calculation"
        self._imported_iprint = 0

        # The runner is shared (serialized) between the ENDF and neutron-
        # scattering panels; create it before building so the NS panel can bind
        # to it. Its callbacks resolve at run time, so the log widgets need not
        # exist yet.
        self.runner = ComputationRunner(
            log_callback=self._log_threadsafe,
            done_callback=self._done_threadsafe,
        )

        self._build_menu()
        self._build_notebook()
        self._build_bottom_bar()

        # Closing the window must not strand a running compute tree (the child
        # is a detached session leader): confirm, shut down (blocking until
        # the tree is reaped), then destroy.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Open at the size the built widgets actually request (the fixed
        # 900x750 default clipped the lattice row's third column and the
        # trailing help buttons), clamped to the screen.
        self.root.update_idletasks()
        width = min(self.root.winfo_reqwidth() + 40,
                    self.root.winfo_screenwidth() - 80)
        height = min(self.root.winfo_reqheight() + 20,
                     self.root.winfo_screenheight() - 120)
        self.root.geometry(f"{max(width, 900)}x{max(height, 750)}")

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu(self):
        """Build the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import Input File...",
                              command=self._import_leapr)
        file_menu.add_command(label="Export Input File...",
                              command=self._export_input)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    # ------------------------------------------------------------------
    # Notebook (tabs)
    # ------------------------------------------------------------------
    def _build_notebook(self):
        # Two top-level capabilities: ENDF/TSL evaluation and neutron-scattering
        # forward spectra. The ENDF form is a single scrolling page (jump bar +
        # five parts in deck order); all deck-generation wiring lives on self.
        """Build the main tab notebook (ENDF Evaluation, Neutron Scattering
        Experiments, NCrystal plugin)."""
        self.top_notebook = ttk.Notebook(self.root)
        self.top_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        endf_page = ttk.Frame(self.top_notebook)
        self.top_notebook.add(endf_page, text="ENDF Evaluation")
        self._page_header(
            endf_page,
            "Build a thermal-scattering-law deck and generate an ENDF-6 TSL "
            "file: the inelastic law S(α,β) (MF7/MT4) plus coherent + "
            "incoherent elastic (MF7/MT2), for neutron-transport codes "
            "(NJOY/THERMR, OpenMC, MCNP).")
        self._build_endf_form(endf_page)

        # Neutron-scattering panel (its own VISION/indirect/direct sub-tabs).
        ns_page = ttk.Frame(self.top_notebook)
        self.top_notebook.add(ns_page, text="Neutron Scattering Experiments")
        self._page_header(
            ns_page,
            "Forward-model an inelastic-neutron-scattering spectrum or S(Q,E) "
            "map from a phonon model with instrument resolution (VISION "
            "indirect, or direct-geometry chopper) — produces 1-D spectra and "
            "2-D S(Q,E) maps to compare against measured INS data.")
        from irma.gui.ns_panel import NSPanel
        self.ns_panel = NSPanel(ns_page, runner=self.runner,
                                status_setter=self._set_ns_status)

        # NCrystal-plugin panel: exports the per-temperature NCrystal data set.
        nc_page = ttk.Frame(self.top_notebook)
        self.top_notebook.add(nc_page, text="NCrystal plugin")
        self._page_header(
            nc_page,
            "Export per-temperature NCrystal scattering data (.irmapack) + a "
            "matching material .ncmat so NCrystal-based codes (McStas, OpenMC) can "
            "sample IRMA's coherent-one-phonon + anisotropic-Debye-Waller "
            "S(alpha,beta): one data file per principal scatterer, per temperature.")
        from irma.gui.ncrystal_panel import NCrystalPanel
        self.ncrystal_panel = NCrystalPanel(nc_page, runner=self.runner,
                                            status_setter=self._set_ns_status)

        # MLIP front-end panel: structure + pretrained potential -> phonon
        # model bundle -> prefilled inputs for the three capabilities above.
        mlip_page = ttk.Frame(self.top_notebook)
        self.top_notebook.add(mlip_page, text="MLIP phonon models")
        self._page_header(
            mlip_page,
            "Build a phonon model for a crystal from a pretrained "
            "machine-learned interatomic potential (no DFT needed): relax, "
            "compute finite-displacement force constants, inspect the DOS "
            "— then "
            "generate ready-to-edit inputs for the ENDF evaluation, "
            "neutron-scattering spectra, and NCrystal export tabs.")
        from irma.gui.mlip_panel import MlipPanel
        self.mlip_panel = MlipPanel(mlip_page, runner=self.runner,
                                    status_setter=self._set_ns_status)

    def _page_header(self, page, text):
        """Short 'what this panel does / what it produces' banner at the top of
        a top-level tab, set off from the content by a separator. Wraps to the
        page width so it reflows on resize."""
        head = ttk.Frame(page)
        head.pack(fill=tk.X, side=tk.TOP)
        lbl = ttk.Label(head, text=text, justify=tk.LEFT, padding=(10, 8, 10, 6))
        lbl.pack(fill=tk.X, anchor="w")
        head.bind("<Configure>",
                  lambda e: lbl.configure(wraplength=max(e.width - 24, 200)))
        ttk.Separator(head, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10)
        return head

    def _set_ns_status(self, msg):
        # The NS panel manages its own status label; the bottom bar mirrors it.
        """Update the Neutron-Scattering panel status line."""
        if hasattr(self, "status_label"):
            try:
                self.status_label.config(text=msg)
            except tk.TclError:
                pass

    def _on_close(self):
        """WM_DELETE_WINDOW / Quit: confirm, then shut the runner down and
        destroy. shutdown() (not cancel()) is required here: a plain cancel
        runs the SIGTERM->SIGKILL escalation on a daemon thread that dies
        with the interpreter, and the compute child is a detached session
        leader -- it would outlive the GUI at full CPU. shutdown() blocks
        (bounded) until the process tree is actually reaped."""
        if self.runner.is_running:
            if not messagebox.askyesno(
                    "Quit IRMA",
                    "A calculation is running — quit anyway?\n\n"
                    "Quitting cancels the calculation and terminates its "
                    "worker processes."):
                return
            self.runner.shutdown()
        # GUI-owned temp files must not survive the window (review GUI-2):
        # the unlink callbacks queued via after() die with the Tk
        # interpreter, so clean synchronously on every close path. The ENDF
        # form is a mixin on the app itself.
        for panel in (self, getattr(self, "ns_panel", None),
                      getattr(self, "ncrystal_panel", None),
                      getattr(self, "mlip_panel", None)):
            cleanup = getattr(panel, "cleanup_temp_files", None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    pass
        self.root.destroy()

    def _show_about(self):
        """Show the About dialog.

        The third-party list mirrors THIRD_PARTY_NOTICES.md (the
        authoritative attribution surface); keep the two in sync.
        """
        from irma import __version__
        messagebox.showinfo(
            "About IRMA",
            f"IRMA v{__version__}\n\n"
            "(In)elastic Representation of Materials\n"
            "As S(α,β) evaluations\n\n"
            "Calculates thermal neutron scattering law\n"
            "S(alpha, beta) using the phonon expansion method.\n\n"
            "Licensed under the BSD 3-Clause License.\n\n"
            "This software derives code from:\n"
            "  - NJOY2016 (BSD-3, LANL)\n"
            "  - NCrystal (Apache-2.0, NCrystal developers)\n"
            "  - ncplugin-CrysXT (Apache-2.0) — extinction models\n"
            "and uses:\n"
            "  - phonopy (BSD-3, A. Togo)\n"
            "  - endf-parserpy (MIT, G. Schnabel)\n\n"
            "See THIRD_PARTY_NOTICES.md for the full notices.")


def main():
    """Launch the IRMA GUI application."""
    root = tk.Tk()
    app = IrmaApp(root)  # noqa: F841  -- keep a ref alive for the mainloop
    root.mainloop()


if __name__ == "__main__":
    main()
