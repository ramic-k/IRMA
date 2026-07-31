"""Phonon-DOS readers for the DOS-based ("mode 0") spectra path.

v1 supports a GENERIC 2-column text DOS: column 0 = phonon frequency, column 1 =
DOS intensity (arbitrary units; the mode-0 kernel renormalizes it). Comments
(``#``), blank lines and non-numeric header rows are skipped; whitespace- or
comma-separated. The frequency unit is given by the caller. The DOS is returned
on a UNIFORM omega grid in eV (the kernel's units), ready for
:func:`irma.spectra.dos_mode0.compute_mode0_sqe`.
"""
from __future__ import annotations

import numpy as np

# frequency-unit -> eV
_TO_EV = {
    "ev": 1.0,
    "mev": 1.0e-3,
    "cm-1": 1.239841984e-4,     # hc/e in eV*cm
    "cm^-1": 1.239841984e-4,
    "thz": 4.135667696e-3,      # h*1e12/e
}


def _parse_2col(path):
    """Robustly read (freq, intensity) rows -- skip comments/headers, accept
    whitespace OR comma separation."""
    freq, dos = [], []
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                f, d = float(parts[0]), float(parts[1])
            except ValueError:
                continue                          # a header / non-numeric row
            freq.append(f)
            dos.append(d)
    if len(freq) < 2:
        raise ValueError(f"{path}: need >= 2 numeric (frequency, DOS) rows")
    return np.asarray(freq, float), np.asarray(dos, float)


def read_dos_2col(path, *, unit="meV", resample=True, d_omega_mev=None):
    """Read a generic 2-column phonon DOS into ``(omega_ev, rho)`` on a UNIFORM
    omega grid (eV).

    Parameters
    ----------
    path : the 2-column text file (freq, DOS).
    unit : frequency unit of column 0 -- 'meV' (default), 'eV', 'cm-1', or 'THz'.
    resample : if True (default), interpolate onto a uniform grid from 0 to the
        max frequency (the kernel requires uniform spacing starting at omega=0).
        If False, the input must already be uniform and start at ~0.
    d_omega_mev : target uniform spacing [meV] when resampling; default = the
        input's median spacing, clamped to 0.01--0.5 meV (the upper clamp
        preserves structure; the lower keeps a very finely gridded file --
        e.g. an MD/VACF export -- from blowing up the O(npt^2) phonon-expansion
        kernel).

    Rows with negative frequency (imaginary modes) are dropped with a warning;
    DOS values below zero are clipped to zero; rho(omega=0) is set to 0 (the
    kernel reconstructs that endpoint).
    """
    key = unit.strip().lower()
    if key not in _TO_EV:
        raise ValueError(f"unknown frequency unit {unit!r}; use one of "
                         f"{sorted(_TO_EV)}")
    freq, dos = _parse_2col(path)
    w_ev = freq * _TO_EV[key]
    order = np.argsort(w_ev)
    w_ev, dos = w_ev[order], dos[order]
    neg = w_ev < 0.0
    if neg.any():
        # folding imaginary modes onto omega=0 would hand their weight to the
        # first positive interval through the interpolation; drop them instead
        import warnings
        warnings.warn(
            f"{path}: dropped {int(neg.sum())} negative-frequency "
            f"(imaginary-mode) rows carrying total DOS weight "
            f"{float(dos[neg].sum()):.3g}; fix the phonon model if this "
            "weight is significant", stacklevel=2)
        w_ev, dos = w_ev[~neg], dos[~neg]
    # Clip negative DOS to 0 AFTER the imaginary-row drop+warning: clipping first
    # would zero any negative DOS on the dropped rows and understate the reported
    # weight. The retained positive-frequency rows clip identically either way.
    dos = np.clip(dos, 0.0, None)
    if w_ev.size < 2:
        raise ValueError(f"{path}: fewer than 2 rows remain after dropping "
                         "negative frequencies")

    if not resample:
        d = np.diff(w_ev)
        if w_ev[0] > 1e-9 or d.size == 0 or not np.allclose(d, d[0], rtol=1e-4):
            raise ValueError(f"{path}: resample=False requires a uniform grid "
                             f"starting at omega=0; pass resample=True")
        rho = dos.copy()
        rho[0] = 0.0
        return w_ev, rho

    w_max = float(w_ev.max())
    if not w_max > 0.0:
        raise ValueError(f"{path}: DOS has no positive frequencies")
    pos = np.diff(np.unique(w_ev))
    med_mev = float(np.median(pos)) * 1000.0 if pos.size else 0.5
    if d_omega_mev:
        step_mev = d_omega_mev
    else:
        step_mev = min(max(med_mev, 0.01) if med_mev > 0 else 0.5, 0.5)
    n = max(int(round(w_max * 1000.0 / step_mev)) + 1, 4)
    omega_ev = np.linspace(0.0, w_max, n)
    rho = np.interp(omega_ev, w_ev, dos, left=0.0, right=0.0)
    rho[0] = 0.0
    return omega_ev, rho
