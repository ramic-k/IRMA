"""Elastic-line physics for the spectra forward model.

The inelastic forward model in :mod:`irma.spectra.sqe` adds an optional elastic
line at zero energy transfer.  This module supplies that line from real physics
rather than a schematic area, building the powder elastic *differential* cross
section dsigma/dOmega(Q) [barn/sr] for the coherent (Bragg) and incoherent
(Debye-Waller) channels.  Several builders return the same
:class:`ElasticModel`: :func:`from_endf_mf7mt2` reads the MF7/MT2 section of an
ENDF tape (the one IRMA itself writes); :func:`from_engine_elastic_state`
builds the line tape-free from the noncubic engine's Debye-Waller state; and
the mode-0 :func:`from_dos_elastic` builds it from a phonon DOS (isotropic
Debye-Waller).

OCLIMAX can also add an elastic line, but its elastic treatment is not
documented. IRMA computes the coherent Bragg structure factors and the
incoherent Debye-Waller term itself; the forms below follow NJOY THERMR (the
reference processor):

Coherent elastic  (THERMR ``sigcoh``; ENDF-102 LTHR=1)
    The tape stores the cumulative structure factor  S(E) = sum_{E_i<=E} f_i
    (units eV*barn).  The angle-integrated cross section is
        sigma_coh(E) = S(E) / E                                  [barn]
    Each Bragg edge E_i corresponds to a reciprocal-lattice shell tau_i with
        tau_i = 2*sqrt(E_i / C_E)        [1/A]   (E_i in meV, C_E=hbar^2/2m_n)
    and scatters at fixed momentum transfer Q = tau_i (independent of E).  The
    powder differential cross section is therefore a set of Bragg peaks
        dsigma_coh/dOmega(Q) = (1/(2*pi*C_E)) * sum_i (f_i/tau_i) delta(Q-tau_i)
    which integrates back to sigma_coh(E) exactly (verified analytically and in
    ``selftest`` below):  2*pi * int_0^{2k} (dsigma/dOmega)(Q*C_E/E) dQ = S(E)/E.

Incoherent elastic  (THERMR ``iel``; ENDF-102 LTHR=2)
    The tape stores the bound xs sigma_b and the Debye-Waller integral W'(T)
    (units eV^-1).  THERMR uses
        dsigma_inc/dOmega = (sigma_b/4pi) * exp(-2*E*W'*(1-mu))
    and with Q^2 = 2 k^2 (1-mu), k^2 = E/C_E this is the E-independent
        dsigma_inc/dOmega(Q) = (sigma_b/4pi) * exp(-W' * C_E * Q^2)     [barn/sr]
    whose angle integral is the THERMR result
        sigma_inc(E) = (sigma_b/2) * (1 - exp(-4 E W')) / (2 E W').

Units throughout: energies in meV, Q in 1/A, cross sections in barn. The
normalization follows the source: per principal atom (the ENDF scatterer) for
:func:`from_endf_mf7mt2`, per atom of the cell for the tape-free builders.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from irma.core.constants import HBAR2_OVER_2MN_MEV_A2 as C_E  # CODATA, shared with core
from irma.core.constants import BK                            # eV/K
EV_PER_MEV = 1.0e-3   # 1 meV = 1e-3 eV


# -----------------------------------------------------------------------------
# Minimal ENDF-6 record reader (column based, MF7/MT2 only)
# -----------------------------------------------------------------------------
def _endf_float(s: str) -> float:
    """Parse an 11-char ENDF field, including Fortran exponents like '1.23-4'."""
    s = s.strip()
    if not s:
        return 0.0
    # already a normal float?
    try:
        return float(s)
    except ValueError:
        pass
    # insert the missing 'E' before a +/- exponent (not the leading sign)
    mant = s[0]
    rest = s[1:].replace("+", "E+").replace("-", "E-")
    return float(mant + rest)


def _mf7_rows(path, mt):
    """The six 11-character data fields of every MF7 record with this MT."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if len(line) >= 75 and line[70:75] == f" 7{mt:3d}":
                rows.append([line[i:i + 11] for i in range(0, 66, 11)])
    return rows


def _mf7mt2_lines(path):
    """The data fields of every MF7/MT2 record, 6 per source line."""
    return _mf7_rows(path, 2)


def _mf7mt4_npr(path):
    """npr (principal-atom count) from MF7/MT4 B(6) of the same tape.

    MF7/MT2 does not carry npr, but every IRMA writer (classic iel<0, SEF/CEF
    and MEF generalized) stores the molecular incoherent-elastic
    ``SB = per-principal sigma x npr`` and records npr in MF7/MT4 B(6)
    (endf_writer: ``mf7mt4['B'][6]``), so the same tape's MT4 is the
    authoritative source. Returns 1.0 when the tape has no MF7/MT4 section or
    B(6) is not positive -- a bare MT2-only tape genuinely cannot say, and 1.0
    leaves SB as stored (correct only for npr=1 tapes; see _parse_incoherent).
    """
    rows = _mf7_rows(path, 4)
    if not rows:
        return 1.0
    cur = _Cursor(rows)
    cur.head()                                   # HEAD: ZA, AWR, 0, LAT, LASYM, 0
    _c1, _c2, _lln, _l2, NI, _ns = cur.head()    # LIST: B(NI) values follow
    if NI < 6:
        return 1.0
    B = cur.reals(NI)
    return float(B[5]) if B[5] > 0 else 1.0      # B(6), 1-based ENDF indexing


@dataclasses.dataclass
class _Cursor:
    """Position while walking ENDF TAB1 rows (row index, field index)."""

    rows: list
    i: int = 0      # row index
    j: int = 0      # field index within current row (0..5)

    def head(self):
        """Read a 6-field control record; advance to next row."""
        r = self.rows[self.i]
        self.i += 1
        self.j = 0
        c1 = _endf_float(r[0]); c2 = _endf_float(r[1])
        l1 = int(round(_endf_float(r[2]))); l2 = int(round(_endf_float(r[3])))
        n1 = int(round(_endf_float(r[4]))); n2 = int(round(_endf_float(r[5])))
        return c1, c2, l1, l2, n1, n2

    def reals(self, n):
        """Read n consecutive real fields, flowing across rows (6/line)."""
        out = []
        while len(out) < n:
            r = self.rows[self.i]
            while self.j < 6 and len(out) < n:
                out.append(_endf_float(r[self.j]))
                self.j += 1
            if self.j >= 6:
                self.i += 1
                self.j = 0
        # ENDF-6 arrays are line-aligned: every record starts on a fresh line
        # and the last line of an array is blank-padded to 6 fields. Always
        # row-align here, or the next head()/reals() re-reads this row's
        # padding as data (wrong-temperature blocks, W'=0, lost top edges).
        if self.j > 0:
            self.i += 1
            self.j = 0
        return np.asarray(out, float)


# -----------------------------------------------------------------------------
# Elastic model
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class ElasticModel:
    """Powder elastic cross sections: coherent (Bragg) and/or incoherent
    (Debye-Waller) channels, read from an MF7/MT2 section (LTHR = 1 / 2 / 3) or
    built tape-free by one of the builders above.
    """
    T_K: float
    # coherent (Bragg peaks)
    Q_bragg: np.ndarray          # tau_i  [1/A], sorted ascending
    f_bragg: np.ndarray          # per-edge structure factor f_i  [meV*barn]
    E_edge_meV: np.ndarray       # Bragg edge energies E_i  [meV]
    # incoherent (Debye-Waller)
    sigma_b: float               # bound xs SB  [barn]   (0 if absent)
    Wprime_invmeV: float         # Debye-Waller integral W'(T) [1/meV] (0 if absent)
    has_coherent: bool
    has_incoherent: bool
    label: str = ""
    # optional multi-species incoherent: ((sigma_b_d, Wprime_d), ...) already
    # multiplicity-weighted (the DOS/mode-0 sample-total line). When non-empty the
    # incoherent methods sum these channels instead of the scalar sigma_b/Wprime
    # (which then carry the total sigma_b and a representative W' for display).
    incoherent_channels: tuple = ()
    # optional directional incoherent channels: ((sigma_b_d, (u1,u2,u3)), ...)
    # with per-atom U-tensor eigenvalues [Ang^2], already multiplicity-weighted.
    # When non-empty the incoherent methods use the orientation-averaged
    # <exp(-Q^2 uhat.U.uhat)> (irma.core.incoherent_dw) instead of the
    # isotropic trace/3 exponent; the isotropic fields above stay populated
    # for display. The ENDF tape path cannot carry this (scalar W' only).
    incoherent_channels_dir: tuple = ()

    # --- coherent ---------------------------------------------------------
    def coherent_dsigma_dOmega(self, Q, q_res=0.0):
        """Powder coherent-elastic dsigma/dOmega(Q) [barn/sr].

        q_res>0 broadens each Bragg peak with a normalized Gaussian of that
        sigma (1/A), modelling finite instrument Q-resolution; q_res=0 returns
        the bare peaks (zero except exactly on a peak -> use q_res for spectra).
        """
        Q = np.atleast_1d(np.asarray(Q, float))
        if not self.has_coherent or self.Q_bragg.size == 0:
            return np.zeros_like(Q)
        amp = self.f_bragg / self.Q_bragg / (2.0 * np.pi * C_E)  # [barn/sr * (1/A)]
        if q_res <= 0.0:
            out = np.zeros_like(Q)
            # nearest-peak assignment
            for qi, ai in zip(self.Q_bragg, amp):
                out[np.isclose(Q, qi)] += ai
            return out
        # sum of normalized Gaussians; vectorized over peaks
        d = Q[:, None] - self.Q_bragg[None, :]
        g = np.exp(-0.5 * (d / q_res) ** 2) / (np.sqrt(2 * np.pi) * q_res)  # [A]
        return g @ amp

    def sigma_coherent(self, E_meV):
        """Angle-integrated coherent-elastic cross section sigma_coh(E) [barn].

        sigma_coh(E) = (1/E) * sum_{E_i <= E} f_i  (THERMR/ENDF form).
        """
        E = np.atleast_1d(np.asarray(E_meV, float))
        if not self.has_coherent or self.E_edge_meV.size == 0:
            return np.zeros_like(E)
        order = np.argsort(self.E_edge_meV)
        Es = self.E_edge_meV[order]
        cum = np.cumsum(self.f_bragg[order])
        idx = np.searchsorted(Es, E, side="right") - 1
        s = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(E > 0, s / E, 0.0)
        return out

    # --- incoherent -------------------------------------------------------
    def _inc_channels(self):
        """The incoherent channels to sum: explicit multi-species list, else the
        single scalar (sigma_b, W')."""
        return self.incoherent_channels or ((self.sigma_b, self.Wprime_invmeV),)

    def incoherent_dsigma_dOmega(self, Q):
        """Incoherent-elastic dsigma/dOmega(Q) [barn/sr]
        = sum_d (sigma_b_d/4pi) exp(-W'_d C_E Q^2), or, with directional
        channels, sum_d (sigma_b_d/4pi) <exp(-Q^2 uhat.U_d.uhat)>."""
        Q = np.atleast_1d(np.asarray(Q, float))
        if not self.has_incoherent:
            return np.zeros_like(Q)
        out = np.zeros_like(Q)
        if self.incoherent_channels_dir:
            from irma.core.incoherent_dw import dw_orientation_average
            for sb, eig in self.incoherent_channels_dir:
                out += (sb / (4.0 * np.pi)) * dw_orientation_average(eig, Q)
            return out
        for sb, wp in self._inc_channels():
            out += (sb / (4.0 * np.pi)) * np.exp(-wp * C_E * Q ** 2)
        return out

    def sigma_incoherent(self, E_meV):
        """Angle-integrated incoherent-elastic cross section [barn]
        = sum_d (sigma_b_d/2)(1-exp(-4 E W'_d))/(2 E W'_d), or the
        orientation-averaged equivalent with directional channels."""
        E = np.atleast_1d(np.asarray(E_meV, float))
        if not self.has_incoherent:
            return np.zeros_like(E)
        if self.incoherent_channels_dir:
            from irma.core.incoherent_dw import sigma_elinc_directional
            return sigma_elinc_directional(E / C_E, self.incoherent_channels_dir)
        out = np.zeros_like(E)
        for sb, wp in self._inc_channels():
            a = 2.0 * E * wp
            with np.errstate(divide="ignore", invalid="ignore"):
                out += np.where(a > 0,
                                (sb / 2.0) * (1.0 - np.exp(-2.0 * a)) / a, sb)
        return out

    # --- combined ---------------------------------------------------------
    def elastic_dsigma_dOmega(self, Q, q_res=0.0):
        """Total elastic dsigma/dOmega(Q) = coherent + incoherent [barn/sr]."""
        return (self.coherent_dsigma_dOmega(Q, q_res=q_res)
                + self.incoherent_dsigma_dOmega(Q))


# -----------------------------------------------------------------------------
# Parser: ENDF MF7/MT2 -> ElasticModel
# -----------------------------------------------------------------------------
def _parse_coherent(cur, T_target):
    """Parse an LTHR=1 coherent-elastic block at the cursor; return
    (E_edge_meV, f_bragg, Q_bragg)."""
    # TAB1: C1=T0, C2=0, L1=LT, L2=0, N1=NR, N2=NP
    T0, _c2, LT, _l2, NR, NP = cur.head()
    cur.reals(2 * NR)               # interpolation table (NBT, INT) pairs
    pairs = cur.reals(2 * NP)
    E_eV = pairs[0::2]
    S = pairs[1::2]                 # cumulative S(E,T0)  [eV*barn]
    chosen_T = T0
    # additional temperatures as LIST records: C1=T, L1=LI, N1=NP, then S values
    for _ in range(LT):
        Tn, _c2, _LI, _l2, NPn, _n2 = cur.head()
        Sn = cur.reals(NPn)
        if abs(Tn - T_target) < abs(chosen_T - T_target):
            chosen_T, S = Tn, Sn
    # de-cumulate -> per-edge structure factors
    f = np.diff(np.concatenate([[0.0], S]))     # [eV*barn]
    keep = f > 0
    E_edge_meV = E_eV[keep] / EV_PER_MEV         # eV -> meV
    f_bragg = f[keep] / EV_PER_MEV               # eV*barn -> meV*barn
    Q_bragg = 2.0 * np.sqrt(E_edge_meV / C_E)    # tau_i [1/A]
    order = np.argsort(Q_bragg)
    return chosen_T, E_edge_meV[order], f_bragg[order], Q_bragg[order]


def _parse_incoherent(cur, T_target, npr=1.0):
    """Parse an LTHR=2/3 incoherent-elastic block; return
    (sigma_b [per principal atom], W'(T)[1/meV]).

    The tape SB is molecular: per-principal sigma x npr (the convention every
    IRMA writer -- classic iel<0, SEF/CEF, MEF -- uses; NJOY THERMR divides it
    back out the same way). Dividing by ``npr`` converts to this module's
    per-principal-atom convention. CAVEAT: MF7/MT2 alone does not carry npr,
    so a standalone parse must be handed it explicitly (``from_endf_mf7mt2``
    reads it from the same tape's MF7/MT4 B(6)); the default npr=1.0 keeps
    SB as stored, which is only correct for npr=1 tapes.

    W' is linearly interpolated in the tabulated temperatures; a request
    outside the tabulated range is clamped to the nearest endpoint with a
    warning.
    """
    # TAB1: C1=SB, C2=0, L1=0, L2=0, N1=NR, N2=NP, then (T, W') pairs
    SB, _c2, _l1, _l2, NR, NP = cur.head()
    cur.reals(2 * NR)
    pairs = cur.reals(2 * NP)
    T = np.asarray(pairs[0::2], dtype=float)
    Wp_eV = np.asarray(pairs[1::2], dtype=float)   # W'(T) [eV^-1]
    order = np.argsort(T)          # np.interp requires ascending x; ENDF T need not be
    Ts = T[order]
    if T_target < Ts[0] - _T_MATCH_TOL_K or T_target > Ts[-1] + _T_MATCH_TOL_K:
        import warnings
        warnings.warn(
            f"MF7/MT2 incoherent elastic: requested T={T_target:g} K lies "
            f"outside the tabulated range [{Ts[0]:g}, {Ts[-1]:g}] K; W' is "
            f"clamped to the nearest endpoint.", stacklevel=3)
    Wp = np.interp(T_target, Ts, Wp_eV[order])
    return SB / npr, Wp / 1.0e3      # eV^-1 -> meV^-1  (W' has units 1/energy)


# Tolerance for the "the tape's temperature grid does not actually cover the
# requested temperature" warnings below: absolute Kelvin, generous enough that
# a 293.15 K vs 296 K convention difference never warns.
_T_MATCH_TOL_K = 10.0


def from_endf_mf7mt2(path, T_K=296.0, label=None):
    """Build an :class:`ElasticModel` from the MF7/MT2 section of an ENDF tape.

    Handles LTHR = 1 (coherent), 2 (incoherent), 3 (both). Temperature
    handling per channel: the coherent channel selects the tabulated
    temperature closest to ``T_K`` (Bragg-edge tables cannot be
    interpolated safely), while the incoherent W'(T) is linearly
    interpolated. Both channels warn when the request is further than
    ``_T_MATCH_TOL_K`` from what the tape can provide, so a 296 K request
    against a 1000 K-only tape is loud instead of silently wrong.

    The returned ``sigma_b`` is per principal atom (the tape convention):
    the tape's molecular ``SB = per-principal sigma x npr`` is divided by the
    npr read from the same tape's MF7/MT4 B(6). A tape with no MF7/MT4
    section is taken as npr=1 (see :func:`_mf7mt4_npr`).
    """
    rows = _mf7mt2_lines(path)
    if not rows:
        raise ValueError(f"no MF7/MT2 section found in {path}")
    cur = _Cursor(rows)
    ZA, AWR, LTHR, _l2, _n1, _n2 = cur.head()

    E_edge = np.array([]); f_b = np.array([]); Q_b = np.array([])
    SB = 0.0; Wp = 0.0
    has_coh = has_inc = False
    Tsel = T_K

    if LTHR in (1, 3):
        Tsel, E_edge, f_b, Q_b = _parse_coherent(cur, T_K)
        has_coh = E_edge.size > 0
        if abs(Tsel - T_K) > _T_MATCH_TOL_K:
            import warnings
            warnings.warn(
                f"MF7/MT2 coherent elastic: requested T={T_K:g} K but the "
                f"nearest tabulated temperature is {Tsel:g} K; using the "
                f"{Tsel:g} K Bragg edges.", stacklevel=2)
    if LTHR in (2, 3):
        SB, Wp = _parse_incoherent(cur, T_K, npr=_mf7mt4_npr(path))
        has_inc = SB > 0

    return ElasticModel(
        T_K=Tsel, Q_bragg=Q_b, f_bragg=f_b, E_edge_meV=E_edge,
        sigma_b=SB, Wprime_invmeV=Wp,
        has_coherent=has_coh, has_incoherent=has_inc,
        label=label or f"MF7/MT2 ZA={int(ZA)} LTHR={LTHR}")


# -----------------------------------------------------------------------------
# Instrument-reach-aware Bragg-enumeration cutoff
# -----------------------------------------------------------------------------
def instrument_reach_emax_eV(*, q_max_invA, e_fixed_meV=None, q_cuts=None,
                             cut_dq=None, q_res_invA=0.0, q_margin_invA=0.5,
                             cap_eV=5.0):
    """Bragg-edge enumeration cutoff [eV] covering everything the instrument
    forward model can evaluate, so the tape-free elastic builders stop enumerating
    at the instrument's reach instead of the ENDF tape writer's 5 eV
    (~ Q <= 98 1/A; the MF7/MT2 path keeps its own full-range call).

    The elastic line is only ever evaluated at
      * each bank's elastic-Q window (``bank_elastic_area``), bounded above by
        the full backscattering ring Q = 2 k_el -- i.e. E_edge <= E_fixed;
      * the simulated Q-support (``q_max_invA``), which spans the bank loci
        (including E = 0) plus any constant-Q-cut bands;
      * the constant-Q cuts at Q0 +/- cut_dq, where the Bragg peaks are broadened
        by a Gaussian of sigma ``q_res_invA``: a peak more than ~39 sigma beyond
        the last evaluated Q underflows exp() to exactly 0.0, so 40 sigma of
        headroom keeps the result bit-identical to a full enumeration.

    Edges beyond that reach contribute exactly zero to the spectrum by
    construction. ``q_margin_invA`` adds absolute headroom on top (boundary
    edges, bank half-width beyond the locus envelope) and the result is
    capped at ``cap_eV``, the reach-aware cutoff's hard upper bound.
    """
    q_need = max(float(q_max_invA), 0.0)
    if e_fixed_meV is not None:
        q_need = max(q_need, 2.0 * np.sqrt(float(e_fixed_meV) / C_E))
    qs = [float(q) for q in q_cuts] if q_cuts else []
    if qs:
        band = float(cut_dq) if cut_dq else 0.0
        tail = 40.0 * max(float(q_res_invA), 0.0)      # broadened-peak underflow reach
        q_need = max(q_need, max(qs) + band + tail)
    q_need += max(float(q_margin_invA), 0.0)
    emax = C_E * (0.5 * q_need) ** 2 * EV_PER_MEV      # E[meV] -> eV  (Q = 2 sqrt(E/C_E))
    return float(min(float(cap_eV), emax))


# -----------------------------------------------------------------------------
# Tape-free elastic: build the SAME ElasticModel from the engine's elastic_state
# -----------------------------------------------------------------------------
# The engine's thermal-displacement matrices U_ij and primitive geometry, with
# the caller's scattering data, give the elastic line without an ENDF tape and
# with the same Debye-Waller state as the inelastic kernel. The Bragg peaks use
# the irma.core.elastic_dw kernels of the MF7/MT2 writer.
def _group_atoms_by_symbol(symbols):
    """First-appearance-ordered species grouping -> (species_symbols, groups)."""
    order, groups = [], {}
    for i, s in enumerate(symbols):
        if s not in groups:
            order.append(s)
            groups[s] = []
        groups[s].append(i)
    return order, [np.asarray(groups[s], dtype=int) for s in order]


def _finalize_bragg_peaks(E_edge_meV, f_bragg):
    """Edge-energy [meV] + structure-factor lists -> arrays sorted by Q, with the
    matching ``Q_bragg = 2 sqrt(E/C_E)``."""
    E_edge_meV = np.asarray(E_edge_meV, float)
    f_bragg = np.asarray(f_bragg, float)
    Q_bragg = 2.0 * np.sqrt(E_edge_meV / C_E) if E_edge_meV.size else np.array([])
    order = np.argsort(Q_bragg) if Q_bragg.size else slice(None)
    return E_edge_meV[order], f_bragg[order], Q_bragg[order]


def _bragg_peaks(crystal, awr, b_coh_fm, f0, T_K, emax_eV, directional=None):
    """Debye-Waller-attenuated Bragg peaks of ``crystal`` up to ``emax_eV``
    -> (E_edge_meV, f_bragg, Q_bragg), shared by the engine and DOS builders.

    ``awr``, ``b_coh_fm`` and ``f0`` (the dimensionless isotropic DW
    coefficient) are per species in ``crystal.sites`` order. ``directional``
    holds the engine's F-matrices (``F_species_per_temp``, ``F_sites_per_temp``,
    ``dir_tensors_uniform``) and selects the directional DW kernel; without it
    the per-species isotropic kernel is used. The kernels are the byte-pinned
    irma.core.elastic_dw ones the MF7/MT2 writer uses.
    """
    from irma.core.crystal import compute_bragg_edges_general
    from irma.core.elastic_dw import resolve_species_dw, make_edge_delta

    bragg_data, nedge, species_corr, bragg_dir_terms = compute_bragg_edges_general(
        crystal, emax=float(emax_eV))
    bragg = [(float(bragg_data[j, 0]), float(bragg_data[j, 1]))
             for j in range(nedge)]
    crystal_info = {
        "atom_types": [{"awr": float(awr[si]), "b_coh": float(b_coh_fm[si]),
                        "dwpix": [float(f0[si])]} for si in range(len(awr))],
        "species_corr": species_corr,
    }
    if directional is not None:
        crystal_info.update(directional, bragg_dir_terms=bragg_dir_terms)
    tempr = [float(T_K)]
    species_dw = resolve_species_dw(crystal_info, tempr, 1)
    # species_corr is always set, so species_dw is never None and the scalar
    # isotropic fallback (dwpix_iso) is not used. scale=1.0 keeps the native
    # per-atom normalization (the 1/N in the xsectfact).
    edge_delta = make_edge_delta(bragg, None, scale=1.0,
                                 species_dw=species_dw, tempr=tempr)
    E_edge_meV, f_bragg = [], []
    for j in range(nedge):
        d = float(edge_delta(j, 0))            # barn*eV (DW-attenuated)
        if d > 0.0:
            E_edge_meV.append(bragg[j][0] / EV_PER_MEV)   # eV -> meV
            f_bragg.append(d / EV_PER_MEV)                # barn*eV -> barn*meV
    return _finalize_bragg_peaks(E_edge_meV, f_bragg)


def _incoherent_channels(weight, sigma_inc, f0_lambda, awr, kT_meV):
    """Per-atom incoherent Debye-Waller channels shared by the ElasticModel
    builders: ``((weight_i * sigma_inc_i, W'_i))`` over species with
    ``sigma_inc_i > 0``, where ``weight_i = mult_i / N`` is the per-atom
    multiplicity weight and ``W'_i = f0_i / (awr_i * kT)`` [1/meV]."""
    return tuple(
        (float(weight[i] * sigma_inc[i]), float(f0_lambda[i] / (awr[i] * kT_meV)))
        for i in range(len(awr)) if sigma_inc[i] > 0.0)


def _incoherent_summary(channels):
    """Display fields for the channel list: total sigma_b and the W' of the
    largest-sigma channel."""
    if not channels:
        return 0.0, 0.0
    return float(sum(sb for sb, _ in channels)), max(channels, key=lambda c: c[0])[1]


def from_engine_elastic_state(elastic_state, *, b_coh_fm, sigma_inc_b, awr,
                              T_K=None, elastic_kind="both", emax_eV=5.0,
                              label=None, incoherent_elastic_mode="isotropic"):
    """Build an :class:`ElasticModel` from a noncubic-engine ``elastic_state``.

    Parameters
    ----------
    elastic_state : dict
        ``result['elastic_state']`` from ``run_noncubic_sab_inprocess``:
        ``thermal_displacement_matrices_ang2`` (n_atoms,3,3), ``primitive_*``
        geometry, ``temperature_k``.
    b_coh_fm, sigma_inc_b, awr : float or sequence (per primitive atom)
        Coherent scattering length [fm], incoherent bound xs [barn], and mass
        ratio A=M/m_n. Scalars are broadcast to every primitive atom; atoms of
        the same symbol must share these (they define a species).
    elastic_kind : {"both", "coherent", "incoherent"}
        ``"both"`` (default) -> the physical mixed elastic line: the pure
        coherent Bragg peaks AND a separate incoherent Debye-Waller line. The
        single-channel modes isolate one: ``"coherent"`` -> Bragg peaks only,
        ``"incoherent"`` -> DW line only. (Unlike the ENDF MF7/MT2 SEF path,
        there is NO ``(sigma_coh+sigma_inc)/sigma_coh`` fold -- the incoherent
        piece is carried explicitly, so both channels add without double count.)
    emax_eV : float
        Bragg-edge enumeration cutoff [eV]. The 5 eV default is the ENDF
        MF7/MT2 tape convention (Q = 2*sqrt(E/C_E) ~ 98 1/A); spectra callers
        should pass the instrument reach (:func:`instrument_reach_emax_eV`) --
        edges beyond the largest evaluated Q contribute exactly zero, and the
        enumeration cost grows steeply with the cutoff.
    incoherent_elastic_mode : {"isotropic", "directional"}
        ``"isotropic"`` (default) keeps the trace/3 scalar Debye-Waller
        exponent per species. ``"directional"`` evaluates the incoherent line
        with the orientation-averaged ``<exp(-Q^2 uhat.U_d.uhat)>`` per atom
        (irma.core.incoherent_dw) -- the same physics option as the NCrystal
        exporter's ``incoherent_elastic_mode``. The coherent Bragg peaks are
        unaffected (they are already directional per reflection).

    Returns the same ``ElasticModel`` type ``from_endf_mf7mt2`` returns, so
    every downstream method works unchanged. Numerically the result matches
    ``from_endf_mf7mt2`` on an elastic_mode=2 (MEF) tape up to NJOY
    edge-thinning; an SEF (elastic_mode=1) tape additionally embeds the
    ``(sigma_coh+sigma_inc)/sigma_coh`` or ``1/f_DC`` structure-factor fold,
    which this per-atom builder deliberately does not apply.
    """
    from irma.core.crystal import (
        CrystalStructure, AtomSite, _site_tensors_uniform, lattice_to_cell_params,
        standard_frame_rotation,
    )
    from irma.core.phonopy_io import thermal_displacements_to_f_matrix

    U = np.asarray(elastic_state["thermal_displacement_matrices_ang2"], float)
    symbols = list(elastic_state["primitive_symbols"])
    frac = np.asarray(elastic_state["primitive_scaled_positions"], float)
    lattice = np.asarray(elastic_state["primitive_lattice_ang"], float)
    n_atoms = len(symbols)
    if T_K is None:
        T_K = float(elastic_state["temperature_k"])

    b_coh_fm = np.broadcast_to(np.asarray(b_coh_fm, float), (n_atoms,))
    sigma_inc_b = np.broadcast_to(np.asarray(sigma_inc_b, float), (n_atoms,))
    awr_atom = np.broadcast_to(np.asarray(awr, float), (n_atoms,))

    species_symbols, groups = _group_atoms_by_symbol(symbols)

    # per-species DW: F = A kT U / (hbar^2/2m_n) averaged over each species' sites
    kT_eV = BK * float(T_K)
    F_atom = thermal_displacements_to_f_matrix(U, awr_atom, kT_eV)   # (n_atoms,3,3)
    # into the frame of the Bragg comb built from the cell parameters
    M = standard_frame_rotation(lattice)
    if M is not None:
        F_atom = M.T @ F_atom @ M
    F_species = np.stack([F_atom[g].mean(axis=0) for g in groups])    # (nsp,3,3)
    f0_species = np.trace(F_species, axis1=1, axis2=2) / 3.0          # dimensionless
    awr_species = np.array([awr_atom[g[0]] for g in groups], float)
    bcoh_species = np.array([b_coh_fm[g[0]] for g in groups], float)
    sinc_species = np.array([sigma_inc_b[g[0]] for g in groups], float)

    want_coh = elastic_kind in ("both", "coherent")
    want_inc = elastic_kind in ("both", "incoherent")

    E_edge_meV = f_bragg = Q_bragg = np.array([])
    if want_coh:
        # Species in group order; the per-site tensors follow the same
        # species-then-position order (which fixes crystal.py's site_terms
        # order). dir_tensors_uniform keeps the species-averaged fast path
        # when every group's tensors are identical (up to bit noise).
        sites = [AtomSite(b_coh_fm=float(b_coh_fm[g[0]]),
                          positions=[tuple(frac[i]) for i in g]) for g in groups]
        crystal = CrystalStructure(*lattice_to_cell_params(lattice), sites)
        directional = {
            "F_species_per_temp": [F_species],
            "F_sites_per_temp": [F_atom[np.concatenate(groups)]],
            "dir_tensors_uniform": all(
                _site_tensors_uniform(F_atom, list(g)) for g in groups),
        }
        E_edge_meV, f_bragg, Q_bragg = _bragg_peaks(
            crystal, awr_species, bcoh_species, f0_species, T_K, emax_eV,
            directional=directional)

    has_coh = want_coh and E_edge_meV.size > 0
    # Multi-species incoherent: every element keeps its Debye-Waller line,
    # weighted per represented atom (mult_d/N) to match the engine inelastic's
    # per-atom normalization. Summing all species keeps the dominant H line of
    # a hydrogenous sample even when a non-H species is the coherent principal.
    kT_meV = BK * 1000.0 * float(T_K)
    weight = np.array([len(g) for g in groups], float) / n_atoms
    channels = (_incoherent_channels(weight, sinc_species, f0_species, awr_species, kT_meV)
                if want_inc else ())
    has_inc = bool(channels)
    SB, Wp = _incoherent_summary(channels)

    # Directional incoherent channels: one per atom with weight 1/N. The
    # isotropic channels above stay populated for the SB/W' display fields.
    channels_dir = ()
    if incoherent_elastic_mode == "directional" and has_inc:
        from irma.core.incoherent_dw import u_eigenvalues
        channels_dir = tuple(
            (float(sigma_inc_b[i]) / n_atoms, tuple(u_eigenvalues(U[i])))
            for i in range(n_atoms) if sigma_inc_b[i] > 0.0)

    return ElasticModel(
        T_K=float(T_K), Q_bragg=Q_bragg, f_bragg=f_bragg, E_edge_meV=E_edge_meV,
        sigma_b=SB, Wprime_invmeV=float(Wp),
        has_coherent=has_coh, has_incoherent=has_inc,
        incoherent_channels=channels, incoherent_channels_dir=channels_dir,
        label=label or f"engine elastic_state ({elastic_kind})")


# -----------------------------------------------------------------------------
# Tape-free elastic for the DOS-based path (mode 0): isotropic DW from contin
# -----------------------------------------------------------------------------
# Without eigenvectors the Debye-Waller is isotropic: one lambda_s per species
# (the ``kernels.contin`` f0), the quantity from_engine_elastic_state derives as
# trace(F_s)/3; in both, 2W = alpha * lambda_s with alpha = C_E Q^2/(awr kT).
# The user supplies the crystal structure for the Bragg peaks.
def from_dos_elastic(crystal, *, awr, sigma_inc_b, f0_lambda, multiplicity, T_K,
                     elastic_kind="both", emax_eV=5.0, label=None):
    """Build an :class:`ElasticModel` from per-species isotropic Debye-Waller
    coefficients (the mode-0 / DOS elastic line).

    PER-ATOM (per represented atom), matching the mode-0 inelastic and
    inelastic_mode 1/2: the coherent Bragg peaks keep their native per-atom
    normalization (the ``1/N`` in ``compute_bragg_edges_general``'s xsectfact), and
    the incoherent line is the per-atom average over every species,
    ``sum_d (mult_d/N) (sigma_inc_d/4pi) exp(-2 W_d(Q))`` with
    ``N = sum(multiplicity)`` -- summing all species keeps the H line
    (:func:`from_engine_elastic_state` uses the same multi-species channel
    sum). So mode-0 {coherent, incoherent, inelastic} share one per-atom
    scale, directly comparable to mode 1/2.

    Parameters
    ----------
    crystal : irma.core.crystal.CrystalStructure or None
        Lattice + per-species coherent scattering length and fractional sites,
        in the order of the per-species arrays below. ``None`` builds the
        incoherent line only (no Bragg peaks).
    awr, sigma_inc_b, f0_lambda, multiplicity : sequence (per species)
        Mass ratio A=M/m_n, bound incoherent xs [barn], the dimensionless
        LEAPR Debye-Waller coefficient ``lambda_s`` (= ``contin`` f0), and the
        number of atoms of the species in the represented cell (with a crystal,
        ``len(site.positions)``).
    T_K : float
    elastic_kind : {"both", "coherent", "incoherent"}
        ``"both"`` -> coherent Bragg peaks AND the summed incoherent DW line; the
        single-channel modes isolate one.
    emax_eV : Bragg-edge enumeration cutoff [eV]. The 5 eV default is the ENDF
        MF7/MT2 tape convention and reaches Q = 2*sqrt(E/C_E) ~ 98 1/A; spectra
        callers should pass the instrument reach
        (:func:`instrument_reach_emax_eV`) -- edges beyond the largest evaluated
        Q contribute exactly zero to the spectrum.

    Returns the SAME ``ElasticModel`` object the tape/engine builders return
    (with ``incoherent_channels`` populated for the multi-species line).
    """
    T_K = float(T_K)
    awr = np.asarray(awr, float)
    sigma_inc_b = np.asarray(sigma_inc_b, float)
    f0 = np.asarray(f0_lambda, float)
    mult = np.asarray(multiplicity, float)

    E_edge_meV = f_bragg = Q_bragg = np.array([])
    if crystal is not None and elastic_kind in ("both", "coherent"):
        bcoh_species = np.array([s.b_coh_fm for s in crystal.sites], float)
        E_edge_meV, f_bragg, Q_bragg = _bragg_peaks(
            crystal, awr, bcoh_species, f0, T_K, emax_eV)

    kT_meV = BK * 1000.0 * T_K
    channels = (_incoherent_channels(mult / float(mult.sum()), sigma_inc_b, f0, awr, kT_meV)
                if elastic_kind in ("both", "incoherent") else ())
    SB, Wp = _incoherent_summary(channels)

    return ElasticModel(
        T_K=T_K, Q_bragg=Q_bragg, f_bragg=f_bragg, E_edge_meV=E_edge_meV,
        sigma_b=SB, Wprime_invmeV=float(Wp),
        has_coherent=E_edge_meV.size > 0, has_incoherent=bool(channels),
        incoherent_channels=channels,
        label=label or f"DOS elastic ({elastic_kind})")


# -----------------------------------------------------------------------------
# Bank-integrated elastic line (the robust quantity for a 1-D spectrum)
# -----------------------------------------------------------------------------
# A single nominal Q_el per detector either hits or misses a Bragg peak, so the
# point differential is fragile for coherent scatterers.  A real finite bank
# integrates dsigma/dOmega over its angular acceptance, which collects every
# reflection whose tau falls in the bank's elastic-Q window.  Integrating the
# Bragg peaks over [2th_min, 2th_max] gives the clean closed form
#     int_bank dsigma_coh/dOmega dOmega = (1/E_el) * sum_{tau in window} f_i
# with E_el = C_E * k_el^2 (= Ef for indirect, Ei for direct).  Dividing by the
# bank solid angle dOmega = 2*pi*(cos2th_min - cos2th_max) yields barn/sr,
# consistent with the inelastic I(E) units used by the forward model.
def _k_elastic(E_fixed):
    """Neutron wavevector k [1/A] at energy ``E_fixed`` [meV]."""
    return np.sqrt(E_fixed / C_E)


def _Q_elastic(E_fixed, two_theta_deg):
    """Elastic momentum transfer Q = 2 k sin(theta) [1/A] (theta = 2theta/2)."""
    k = _k_elastic(E_fixed)
    return 2.0 * k * np.sin(np.deg2rad(np.asarray(two_theta_deg, float)) / 2.0)


def bank_elastic_area(model: ElasticModel, E_fixed, two_theta_deg,
                      dtheta_deg=5.0, n_theta=721, per_sr=True):
    """Elastic-line area for one detector bank [barn/sr] (or barn if per_sr=False).

    E_fixed   : Ef (indirect) or Ei (direct), meV  -- sets the elastic wavevector
    two_theta_deg : bank centre scattering angle
    dtheta_deg : half-width of the bank's 2theta acceptance (>0 integrates the
                 Bragg peaks robustly; 0 falls back to the point differential)
    Returns the integrated elastic differential cross section suitable to pass as
    ``elastic_area`` to ``instrument_spectrum``.
    """
    if dtheta_deg <= 0.0:
        # point differential at the bank centre (smooth incoherent + bare peaks)
        Q0 = _Q_elastic(E_fixed, two_theta_deg)
        return float(model.elastic_dsigma_dOmega(Q0, q_res=0.0)[0])

    tt = np.linspace(max(two_theta_deg - dtheta_deg, 1e-3),
                     min(two_theta_deg + dtheta_deg, 180.0 - 1e-3), n_theta)
    E_el = E_fixed
    q_lo = float(_Q_elastic(E_fixed, tt.min()))
    q_hi = float(_Q_elastic(E_fixed, tt.max()))

    # coherent: closed form -- sum f_i over reflections in the Q window / E_el
    coh_xs = 0.0
    if model.has_coherent and model.Q_bragg.size:
        m = (model.Q_bragg >= q_lo) & (model.Q_bragg <= q_hi)
        coh_xs = float(np.sum(model.f_bragg[m]) / E_el)   # barn

    # incoherent: integrate the smooth differential over solid angle
    inc_xs = 0.0
    if model.has_incoherent:
        Q = _Q_elastic(E_fixed, tt)
        dsig = model.incoherent_dsigma_dOmega(Q)          # barn/sr
        tt_rad = np.deg2rad(tt)
        inc_xs = float(np.trapezoid(dsig * 2.0 * np.pi * np.sin(tt_rad), tt_rad))

    dOmega = 2.0 * np.pi * (np.cos(np.deg2rad(tt.min()))
                            - np.cos(np.deg2rad(tt.max())))  # sr
    total_xs = coh_xs + inc_xs                              # barn
    return total_xs / dOmega if per_sr else total_xs


# -----------------------------------------------------------------------------
# Self-test: the Bragg peaks integrate back to sigma_coh(E)=S(E)/E
# -----------------------------------------------------------------------------
def selftest(model: ElasticModel, E_meV=200.0, q_res=0.0):
    """Verify  2*pi * int_0^{2k} dsigma/dOmega (Q*C_E/E) dQ == sigma_coh(E)."""
    k = np.sqrt(E_meV / C_E)
    Qmax = 2.0 * k
    if q_res <= 0.0:
        # use the analytic peak sum directly
        m = model.Q_bragg <= Qmax
        lhs = 2.0 * np.pi * np.sum(
            (model.f_bragg[m] / model.Q_bragg[m] / (2 * np.pi * C_E))
            * (model.Q_bragg[m] * C_E / E_meV))
    else:
        Q = np.linspace(1e-6, Qmax, 200001)
        integ = model.coherent_dsigma_dOmega(Q, q_res=q_res) * (Q * C_E / E_meV)
        lhs = 2.0 * np.pi * np.trapezoid(integ, Q)
    rhs = float(np.atleast_1d(model.sigma_coherent(E_meV))[0])
    return float(lhs), rhs
