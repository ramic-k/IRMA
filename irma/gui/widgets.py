"""Reusable custom tkinter widgets for IRMA GUI."""

import os
import shutil
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox


def parse_float(label, text):
    """``float(text)`` that names the originating GUI field on failure, so the
    error dialog says WHICH entry is bad instead of a bare conversion message."""
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label}: not a number: {str(text).strip()!r}") from None


def parse_int(label, text):
    """``int(text)`` that names the originating GUI field on failure."""
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label}: not an integer: {str(text).strip()!r}") from None


def fixed_font():
    """The platform's pleasant monospace font for code-like text fields.

    ('Courier', 10) renders poorly on retina macOS; Tk substitutes a default
    if the named family is missing, so this degrades safely."""
    import sys
    if sys.platform == "darwin":
        return ("Menlo", 12)
    if sys.platform == "win32":
        return ("Consolas", 10)
    return "TkFixedFont"


def is_dark_theme(widget):
    """True when the effective theme background is dark (e.g. macOS dark
    mode). ``winfo_rgb`` resolves symbolic system colors such as
    ``systemWindowBackgroundColor``, so the luminance test works on every
    platform; unresolvable colors fall back to the light palette."""
    bg = ttk.Style().lookup("TFrame", "background") or "#ececec"
    try:
        r, g, b = widget.winfo_rgb(bg)
    except tk.TclError:
        return False
    return (0.299 * r + 0.587 * g + 0.114 * b) / 65535.0 < 0.5


# The help popup grows with its content up to this many text lines; longer
# help texts get a scrollbar instead of being clipped mid-sentence.
_HELP_MAX_VISIBLE_LINES = 30


def show_help_dialog(anchor, title, message):
    """Open a help popup centered on ``anchor``'s window, sized to its
    wrapped content. Shared by :class:`HelpButton` and :class:`InfoLabel`."""
    top = tk.Toplevel(anchor)
    top.title(title)
    # Vertical resizing stays available so a user can enlarge a long
    # (scrolled) help text; the width is fixed to keep the wrap stable.
    top.resizable(False, True)

    frame = ttk.Frame(top, padding=15)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text=title,
              font=("TkDefaultFont", 12, "bold")).pack(anchor=tk.W)
    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(
        fill=tk.X, pady=(5, 10))

    body = ttk.Frame(frame)
    body.pack(fill=tk.BOTH, expand=True)
    msg = tk.Text(body, wrap=tk.WORD, width=55, height=12,
                  font=("TkDefaultFont", 11), relief=tk.FLAT,
                  background=frame.winfo_toplevel().cget("background"),
                  padx=5, pady=5)
    msg.insert("1.0", message)
    msg.config(state=tk.DISABLED)
    msg.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Size the popup to its content: a fixed-height window would clip
    # the longer help texts (Bragg-edge grouping, mpdir, ilog, ...) with
    # no scrollbar and no resize handle -- the message would just stop
    # mid-sentence. Count the WRAPPED display lines and grow the widget
    # up to _HELP_MAX_VISIBLE_LINES; beyond that, attach a scrollbar.
    msg.update_idletasks()
    n_lines = msg.count("1.0", "end", "displaylines")
    if isinstance(n_lines, (tuple, list)):     # tkinter < 3.13 returns (n,)
        n_lines = n_lines[0]
    n_lines = int(n_lines or 1)
    msg.configure(height=max(3, min(n_lines, _HELP_MAX_VISIBLE_LINES)))
    if n_lines > _HELP_MAX_VISIBLE_LINES:
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL, command=msg.yview)
        msg.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    ttk.Button(frame, text="OK", command=top.destroy).pack(pady=(10, 0))

    # Center on parent
    top.update_idletasks()
    pw = anchor.winfo_toplevel()
    x = pw.winfo_x() + (pw.winfo_width() - top.winfo_width()) // 2
    y = pw.winfo_y() + (pw.winfo_height() - top.winfo_height()) // 2
    top.geometry(f"+{x}+{y}")
    try:
        top.grab_set()
    except tk.TclError:
        pass    # not viewable yet (e.g. headless tests) -- modality only
    top.focus_set()
    return top


class HelpButton(ttk.Button):
    """A small '?' button that shows a help popup when clicked."""

    _MAX_VISIBLE_LINES = _HELP_MAX_VISIBLE_LINES

    def __init__(self, parent, title, message):
        super().__init__(parent, text="?", width=2, command=self._show)
        self._title = title
        self._message = message

    def _show(self):
        """Open the help popup, sized to its wrapped content."""
        show_help_dialog(self, self._title, self._message)


class InfoLabel(ttk.Label):
    """A flat 'ⓘ' glyph: hover previews the help text, click opens the full
    scrollable dialog. A quieter alternative to :class:`HelpButton` for
    dense forms, where a boxed button per field reads as visual noise."""

    def __init__(self, parent, title, message):
        super().__init__(parent, text="ⓘ", padding=(2, 0))
        try:
            self.configure(foreground="systemLinkColor", cursor="pointinghand")
        except tk.TclError:            # not aqua: portable fallbacks
            self.configure(foreground="#4a90d9", cursor="hand2")
        self._title = title
        self._message = message
        self.bind("<Button-1>", self._show)
        preview = message.split("\n\n", 1)[0]
        if len(preview) > 240:
            preview = preview[:240].rsplit(" ", 1)[0] + " ..."
        ToolTip(self, preview + "\n\n(click for details)")

    def _show(self, _event=None):
        """Open the full help dialog."""
        show_help_dialog(self, self._title, self._message)


def init_form_styles():
    """Named fonts and ttk styles for the flat form sections.

    Idempotent: the GUI tests build several panels in one interpreter, and
    a named font may only be created once."""
    import tkinter.font as tkfont
    if "IrmaSectionFont" not in tkfont.names():
        base = tkfont.nametofont("TkDefaultFont")
        tkfont.Font(name="IrmaSectionFont", family=base.cget("family"),
                    size=base.cget("size") + 1, weight="bold")
    style = ttk.Style()
    style.configure("Section.TLabel", font="IrmaSectionFont")
    style.configure("Hint.TLabel", foreground="gray")


def form_section(parent, title, help_title=None, help_text=None,
                 expand=False):
    """A flat form section: bold title, optional 'ⓘ' help, separator, and
    an indented body frame (returned).

    Replaces the boxed LabelFrame look. The body and outer frames carry
    ``_irma_section_title`` so tests can locate the section a widget
    belongs to without relying on LabelFrame. ``expand=True`` lets the
    section grow with the window (log panes)."""
    outer = ttk.Frame(parent)
    outer.pack(fill=tk.BOTH if expand else tk.X, expand=expand,
               pady=(2, 12))
    head = ttk.Frame(outer)
    head.pack(fill=tk.X)
    ttk.Label(head, text=title, style="Section.TLabel").pack(side=tk.LEFT)
    if help_text:
        InfoLabel(head, help_title or title, help_text).pack(
            side=tk.LEFT, padx=(6, 0))
    ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(3, 8))
    body = ttk.Frame(outer)
    body.pack(fill=tk.BOTH if expand else tk.X, expand=expand, padx=(4, 0))
    outer._irma_section_title = title
    body._irma_section_title = title
    return body


def scrolled_columns(parent):
    """A vertically scrolling form column beside a column that fills the
    rest of ``parent``. The canvas takes the form's natural width, so
    nothing is clipped horizontally. Returns (form, side, canvas)."""
    top = ttk.Frame(parent)
    top.pack(fill=tk.BOTH, expand=True)
    left_outer = ttk.Frame(top)
    left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
    bg = ttk.Style().lookup("TFrame", "background")
    canvas = tk.Canvas(left_outer, highlightthickness=0, borderwidth=0,
                       background=bg or None)
    vsb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    form = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=form, anchor="nw")
    form.bind("<Configure>", lambda _e: canvas.configure(
        scrollregion=canvas.bbox("all"), width=form.winfo_reqwidth()))

    def _wheel(event):
        """Scroll on the mouse wheel (platform-normalized delta)."""
        delta = event.delta
        step = delta // 120 if abs(delta) >= 120 else delta
        canvas.yview_scroll(-int(step), "units")
    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    side = ttk.Frame(top)
    side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    return form, side, canvas


class RunPanel(ttk.Frame):
    """Run, cancel and completion handling shared by the NS, NCrystal and
    MLIP panels. A subclass provides ``runner``, ``log``, ``cancel_btn``,
    ``_status`` and ``_action_buttons()``."""

    error_title = "Error"
    _tmpdir = None

    def _action_buttons(self):
        """The buttons that start a run; disabled while one is active."""
        return ()

    def _set_busy(self, busy):
        """Enable or disable the action buttons and Cancel as one unit."""
        for btn in self._action_buttons():
            btn.config(state=tk.DISABLED if busy else tk.NORMAL)
        self.cancel_btn.config(state=tk.NORMAL if busy else tk.DISABLED)

    def _temp_path(self, name):
        """A path in a fresh panel-owned temp directory, removed when the
        run ends."""
        self._drop_tmpdir()
        self._tmpdir = tempfile.mkdtemp(prefix="irma_gui_")
        return os.path.join(self._tmpdir, name)

    def _drop_tmpdir(self):
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def cleanup_temp_files(self):
        """Remove the panel-owned temp files (idempotent; app close calls it)."""
        self._drop_tmpdir()

    def _start(self, argv, banner, success_msg, error_label, header=None,
               running=None, done="Done", error_title=None,
               output_path=None, on_ok=None):
        """Run ``argv`` through the shared runner. ``header`` follows the
        banner in the log (default: the command line); ``running`` and
        ``done`` are the status texts; ``on_ok`` runs after a success."""
        if self.runner.is_running:
            messagebox.showwarning("Running",
                                   "A calculation is already in progress.")
            return
        self.log.clear()
        self.log.append(f"=== {banner} ===\n")
        self.log.append("$ " + " ".join(argv[3:]) + "\n\n"
                        if header is None else header)
        self._status(running or banner + "...")
        self._set_busy(True)
        self.runner.run_command(
            argv, success_msg=success_msg, on_log=self._log_ts,
            on_done=lambda ok, msg: self.after(
                0, self._finish, ok, msg, done, error_title, on_ok),
            output_path=output_path, error_label=error_label)

    def _finish(self, ok, msg, done, error_title, on_ok):
        """Completion, on the Tk thread."""
        self._set_busy(False)
        self.log.append(f"\n{msg}\n")
        if ok:
            self._status(done)
            if on_ok is not None:
                on_ok()
        elif msg.startswith("Calculation cancelled"):
            self._status("Cancelled")
        else:
            self._status("Error")
            messagebox.showerror(error_title or self.error_title, msg[:500])
        self._drop_tmpdir()

    def _cancel(self):
        """Cancel the running calculation."""
        if not self.runner.is_running:
            return
        self._status("Cancelling...")
        self.cancel_btn.config(state=tk.DISABLED)
        self.log.append("\n=== Cancelling (terminating workers) ===\n")
        self.runner.cancel()

    def _log_ts(self, text):
        """Append a log line from the worker thread (via after())."""
        self.after(0, self.log.append, text)


def check_with_help(parent, text, var, help_text):
    """Checkbox with an attached ⓘ help glyph; returns the row frame so
    callers can show or hide it."""
    row = ttk.Frame(parent)
    row.pack(anchor=tk.W, fill=tk.X, pady=2)
    ttk.Checkbutton(row, text=text, variable=var).pack(side=tk.LEFT)
    InfoLabel(row, text, help_text).pack(side=tk.LEFT, padx=(4, 0))
    return row


class LabeledEntry(ttk.Frame):
    """A label + entry field + optional help ('?' button or 'ⓘ' glyph).

    ``label_width`` fixes the label column (right-aligned) so stacked fields
    align; 0 gives a natural-width label for inline sub-fields.
    ``compact_help=False`` renders the help as the boxed
    :class:`HelpButton` instead of the flat :class:`InfoLabel` default."""

    def __init__(self, parent, label, default="", width=12, tooltip=None,
                 help_title=None, help_text=None, label_width=20,
                 compact_help=True):
        super().__init__(parent)
        self.label = ttk.Label(self, text=label, width=label_width or None,
                               anchor="e" if label_width else "w")
        self.label.pack(side=tk.LEFT, padx=(0, 5))
        self.var = tk.StringVar(value=str(default))
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side=tk.LEFT)
        if help_text:
            helper = InfoLabel if compact_help else HelpButton
            helper(self, help_title or label.rstrip(":"),
                   help_text).pack(side=tk.LEFT, padx=(4, 0))
        if tooltip:
            ToolTip(self.entry, tooltip)
            ToolTip(self.label, tooltip)

    def get(self):
        """Return the current value."""
        return self.var.get()

    def set(self, value):
        """Set the value."""
        self.var.set(str(value))


class LabeledCombobox(ttk.Frame):
    """A label + combobox + optional help ('?' button or 'ⓘ' glyph).

    ``label_width`` / ``compact_help`` as in :class:`LabeledEntry`."""

    def __init__(self, parent, label, values, default=None, width=None,
                 tooltip=None, help_title=None, help_text=None,
                 label_width=20, compact_help=True):
        super().__init__(parent)
        self.label = ttk.Label(self, text=label, width=label_width or None,
                               anchor="e" if label_width else "w")
        self.label.pack(side=tk.LEFT, padx=(0, 5))
        if width is None:
            # fit the longest option so neither the field nor the
            # dropdown list truncates its text
            width = max(12, max(len(str(v)) for v in values) + 1)
        self.var = tk.StringVar(value=str(default or values[0]))
        self.combo = ttk.Combobox(
            self, textvariable=self.var, values=values,
            width=width, state="readonly")
        self.combo.pack(side=tk.LEFT)
        if help_text:
            helper = InfoLabel if compact_help else HelpButton
            helper(self, help_title or label.rstrip(":"),
                   help_text).pack(side=tk.LEFT, padx=(4, 0))
        if tooltip:
            ToolTip(self.combo, tooltip)

    def get(self):
        """Return the current value."""
        return self.var.get()

    def set(self, value):
        """Set the value."""
        self.var.set(str(value))


class FileSelector(ttk.Frame):
    """A label + entry + browse button + optional help ('?' or 'ⓘ').

    ``label_width`` / ``compact_help`` as in :class:`LabeledEntry`."""

    def __init__(self, parent, label, mode="open", filetypes=None,
                 defaultextension=None, tooltip=None, help_title=None,
                 help_text=None, label_width=20, compact_help=True):
        super().__init__(parent)
        self.mode = mode
        self.filetypes = filetypes or [("All files", "*.*")]
        # save-dialog extension follows the offered filetypes unless given
        self.defaultextension = (defaultextension if defaultextension is not None
                                 else self._default_extension(self.filetypes))

        self.label = ttk.Label(self, text=label, width=label_width or None,
                               anchor="e" if label_width else "w")
        self.label.pack(side=tk.LEFT, padx=(0, 5))
        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var, width=40)
        self.entry.pack(side=tk.LEFT, padx=(0, 5))
        self.btn = ttk.Button(self, text="Browse...", command=self._browse)
        self.btn.pack(side=tk.LEFT)
        if help_text:
            helper = InfoLabel if compact_help else HelpButton
            helper(self, help_title or label.rstrip(":"),
                   help_text).pack(side=tk.LEFT, padx=(4, 0))
        if tooltip:
            ToolTip(self.entry, tooltip)

    @staticmethod
    def _default_extension(filetypes):
        """Save-dialog default extension from the first concrete filetype
        pattern ('*.csv' -> '.csv'); '' when only wildcards are offered."""
        for _label, patterns in filetypes:
            for pat in str(patterns).split():
                ext = os.path.splitext(pat)[1]
                if ext and "*" not in ext and "?" not in ext:
                    return ext
        return ""

    def _browse(self):
        """Open a file dialog and put the chosen path in the entry."""
        from tkinter import filedialog
        if self.mode == "directory":
            path = filedialog.askdirectory()
        elif self.mode == "open":
            path = filedialog.askopenfilename(filetypes=self.filetypes)
        else:
            path = filedialog.asksaveasfilename(
                filetypes=self.filetypes,
                defaultextension=self.defaultextension)
        if path:
            self.var.set(path)

    def get(self):
        """Return the current value."""
        return self.var.get()

    def set(self, value):
        """Set the value."""
        self.var.set(str(value))


class ToolTip:
    """Tooltip shown below a widget after a short hover delay.

    The delay keeps casual mouse travel from flashing popups; the palette
    follows the effective theme so the tip is legible in dark mode."""

    def __init__(self, widget, text, delay_ms=450, wraplength=440):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self.tipwindow = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<Button>", self.hide, add="+")

    def _schedule(self, event=None):
        """Arm the show timer on pointer entry."""
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self.show)

    def _cancel(self):
        """Disarm a pending show timer."""
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def show(self, event=None):
        """Show the tooltip below the widget."""
        if self.tipwindow or not self.widget.winfo_ismapped():
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        if is_dark_theme(self.widget):
            bg, fg, border = "#3a3a3c", "#e8e8e8", "#5c5c5e"
        else:
            bg, fg, border = "#ffffe0", "#1a1a1a", "#8a8a6a"
        # 1px contrasting frame stands in for a border (tk.Label's SOLID
        # relief draws in the label background on some platforms).
        box = tk.Frame(tw, background=border, padx=1, pady=1)
        box.pack()
        tk.Label(box, text=self.text, justify=tk.LEFT,
                 background=bg, foreground=fg, padx=8, pady=6,
                 font=("TkDefaultFont", 11),
                 wraplength=self.wraplength).pack()

    def hide(self, event=None):
        """Cancel a pending tooltip and destroy a visible one."""
        self._cancel()
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ScrolledText(ttk.Frame):
    """A text widget with scrollbar."""

    def __init__(self, parent, height=15, **kwargs):
        super().__init__(parent)
        self.text = tk.Text(self, height=height, wrap=tk.WORD,
                            font=fixed_font(), **kwargs)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL,
                                  command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def append(self, text):
        """Append text and scroll to the end."""
        self.text.insert(tk.END, text)
        self.text.see(tk.END)

    def clear(self):
        """Delete all text."""
        self.text.delete("1.0", tk.END)

    def get_text(self):
        """Return the full log contents."""
        return self.text.get("1.0", tk.END)
