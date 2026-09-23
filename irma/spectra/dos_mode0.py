"""DOS-based ("mode 0") powder S(Q,E) for the irma.spectra forward model.

The incoherent-approximation phonon expansion straight from a phonon DOS -- no
phonopy eigenvectors required. This is the lightweight, DOS-only end of IRMA's
fidelity range (complementary to the phonopy-backed inelastic_mode 1/2): correct
and efficient for incoherent / hydrogen-rich / disordered materials and for any
case where you only have a DOS (MD/VACF, experiment, a quick calc). It does NOT
capture coherent *inelastic* (dispersion).

MULTI-ATOM: each species scatters by ITS OWN partial DOS, mass (-> alpha) and
Debye-Waller -- you do NOT collapse to one effective spectrum. The result is the
cross-section/multiplicity-weighted PER-ATOM average

    d2sigma/dOmega/dE'(Q,E) = (1/N) sum_d m_d (sigma_d/4pi) e^{-2W_d(Q)} [expansion of rho_d]

with alpha_d = C_E Q^2 / (awr_d kT) and N = sum_d m_d the atoms per cell. The
1/N makes the absolute scale PER REPRESENTED ATOM, matching inelastic_mode 1/2
(the engine normalizes per represented atom too) -- so mode-0 and mode-1/2
spectra are directly comparable. A single species (one entry) is the classic
LEAPR single-material path -- the right fallback when only a total DOS is known
(excellent for H-rich solids, where H dominates sigma).

CONVENTION: the LEAPR ``contin()``
kernel fills its array with the ASYMMETRIC downscatter law S_asym_down (= S_sym *
exp(+beta/2)); the physical S(Q,E) is then ``sigma_d/(4 pi kT) * S_asym_down``,
i.e. ``sqe._law_to_sqe``, with no further exp(+beta/2) factor.
"""
from __future__ import annotations

import numpy as np

from irma.core.constants import BK                 # Boltzmann, eV/K
from irma.core.kernels import contin, start
from irma.core.noncubic_helpers import derive_required_multiphonon_order
from irma.spectra.sqe import _law_to_sqe, C_E, KB  # C_E meV*A^2 ; KB meV/K


def _validate_dos(omega_ev, rho, symbol):
    """Validate one species' DOS arrays and return them as float arrays."""
    omega = np.asarray(omega_ev, float)
    rho = np.asarray(rho, float)
    if omega.ndim != 1 or omega.size < 2 or rho.shape != omega.shape:
        raise ValueError(f"species {symbol!r}: dos omega/rho must be matching 1-D "
                         f"arrays with >= 2 points")
    d = np.diff(omega)
    delta1 = float(d[0])
    if not delta1 > 0.0:
        raise ValueError(f"species {symbol!r}: dos omega grid must increase with a "
                         f"positive uniform spacing (got delta={delta1})")
    if not np.allclose(d, delta1, rtol=1e-4, atol=0.0):
        raise ValueError(f"species {symbol!r}: dos omega grid must be UNIFORMLY "
                         f"spaced (resample first)")
    if not np.all(rho >= 0.0) or not np.any(rho[1:] > 0.0):
        raise ValueError(f"species {symbol!r}: dos rho must be non-negative with at "
                         f"least one positive point above omega=0")
    return omega, rho, delta1


def derive_mode0_phonon_order(*, species, temperature_k, q_max_ang_inv):
    """Required mode-0 phonon-expansion order (the engine auto-order analogue).

    The ``contin`` ladder's order-``n`` weight is Poisson with mean
    ``lambda*alpha = f0 * C_E*Q^2/(awr*kT)`` (``f0`` = the DOS Debye-Waller
    lambda from :func:`irma.core.kernels.start`), so the worst case over all
    species at ``Q_max`` is converged by ``n ~ lam + 6*sqrt(lam) + 2`` -- the
    exact rule modes 1/2 use (:func:`derive_required_multiphonon_order` with
    the isotropic ``U_eq = f0*C_E/(awr*kT)`` standing in for the largest
    thermal-displacement eigenvalue, since ``Q^2 U_eq == lambda*alpha``).
    ``start()`` is O(npt), so this costs ~nothing next to one ladder order.

    Returns ``(effective_order, required_order, lam_alpha_max)`` where
    ``effective_order = min(required_order, 2000)`` (the engine safety cap;
    ``required > 2000`` means even the capped ladder truncates -- warn).
    """
    T_K = float(temperature_k)
    tev = BK * T_K
    kT = KB * T_K
    u_eq = []
    for sp in species:
        omega, rho, delta1 = _validate_dos(sp["omega_ev"], sp["rho"],
                                           sp.get("symbol", "?"))
        awr = float(sp["awr"])
        rho = rho.copy()
        rho[0] = 0.0                                # start() reconstructs p[0]
        _, f0, _, _ = start(rho, omega.size, delta1, tev, 1.0)
        # isotropic thermal-displacement analogue: 2W = Q^2 U_eq = f0*alpha
        u_eq.append(float(f0) * C_E / (awr * kT))
    mats = np.array([u * np.eye(3) for u in u_eq])
    # requested_order=2 makes the returned effective == min(required, cap)
    effective, required, lam_alpha_max, _ = derive_required_multiphonon_order(
        float(q_max_ang_inv), mats, 2)
    return effective, required, lam_alpha_max


def compute_mode0_sqe(*, species, temperature_k, q_ang_inv, e_mev, nphon=100):
    """Total powder ``S(Q,E)`` from per-species phonon DOS (mode 0).

    Parameters
    ----------
    species : list of dict, one per scattering species, each with
        ``symbol`` (label), ``omega_ev`` + ``rho`` (the partial phonon DOS on a
        UNIFORM omega grid, eV; rho[0] is reconstructed internally), ``awr``
        (mass ratio M_d/m_n -> alpha_d and the Debye-Waller), ``sigma_bound_b``
        (the TOTAL bound cross section sigma_coh+sigma_inc for the incoherent-
        approximation inelastic weight) and ``multiplicity`` (number of atoms of
        this species represented; default 1).
    temperature_k : float
    q_ang_inv, e_mev : the (1-D) momentum- and energy-transfer grids S(Q,E) is
        built on (E is the energy-LOSS side, >= 0).
    nphon : phonon-expansion order: an int (the kernel runs exactly this many
        convolution orders, no convergence break) or ``"auto"`` to size it
        from the DOS via :func:`derive_mode0_phonon_order` -- the converged
        Poisson(f0*alpha_max) order at the largest requested Q, capped at
        2000. ``"auto"`` both shrinks the ladder when 100 is overkill
        (thermal Q ranges typically converge by ~10-20 orders; the ladder
        cost is ~quadratic in the order) and grows it for high-alpha cases
        (H at high Q) where a fixed 100 truncates.

    The DOS carries the full vibrational weight (tbeta = 1): this path models
    solids only, with no diffusive or free-gas translational channel.

    Returns a dict with ``q_ang_inv``, ``e_mev``, ``sqe_barn_per_meV`` (nq, nE)
    = the PER-ATOM physical d2sigma/dOmega/dE' (the cross-section/multiplicity-
    weighted cell sum divided by the atoms per cell, matching mode 1/2), the
    per-species Debye-Waller and effective-temperature factors, the atom-averaged
    bound cross section, and ``nphon_effective`` / ``nphon_required`` (the order the ladder actually ran
    / the derived convergence requirement; ``required > effective`` only when
    the 2000 safety cap truncated an ``"auto"`` request).
    """
    if not species:
        raise ValueError("compute_mode0_sqe: at least one species is required")
    T_K = float(temperature_k)
    if not T_K > 0.0:
        raise ValueError(f"temperature_k must be > 0, got {T_K}")
    tev = BK * T_K                                  # kT in eV (kernel units)
    kT = KB * T_K                                   # kT in meV
    Q = np.asarray(q_ang_inv, float)
    E = np.asarray(e_mev, float)
    beta = E / kT                                   # dimensionless, >= 0
    nQ, nE = Q.size, E.size

    if isinstance(nphon, str):
        if nphon != "auto":
            raise ValueError(f"nphon must be an int or 'auto', got {nphon!r}")
        nphon_eff, nphon_req, _lam = derive_mode0_phonon_order(
            species=species, temperature_k=T_K,
            q_max_ang_inv=float(Q.max()))
    else:
        nphon_eff = int(nphon)                      # explicit order: honored verbatim
        if nphon_eff < 1:
            raise ValueError(f"nphon must be >= 1 (or 'auto'), got {nphon!r}")
        nphon_req = nphon_eff

    S_cell = np.zeros((nQ, nE))
    sigma_b_cell = 0.0
    n_atoms = 0.0
    per_species = []
    for sp in species:
        symbol = sp.get("symbol", "?")
        omega, rho, delta1 = _validate_dos(sp["omega_ev"], sp["rho"], symbol)
        awr = float(sp["awr"])
        sigma_d = float(sp["sigma_bound_b"])
        mult = float(sp.get("multiplicity", 1))
        rho = rho.copy()
        rho[0] = 0.0                                # start() reconstructs p[0]
        np1 = omega.size

        # per-species alpha grid (mass enters here)
        alpha_d = C_E * Q ** 2 / (awr * kT)
        ssm = np.zeros((nE, nQ))                    # [nbeta, nalpha], filled in place
        f0, tbar, _deltab = contin(ssm, alpha_d, beta, nQ, nE, 0, 1.0, tev,
                                   rho, np1, delta1, 1.0, nphon_eff)
        # asym_downscatter (the P0-validated convention): S = sigma_d/(4pi kT) * ssm
        S_d = _law_to_sqe(ssm, T_K, sigma_d)
        S_cell += mult * S_d
        sigma_b_cell += mult * sigma_d
        n_atoms += mult
        per_species.append({"symbol": symbol, "awr": awr, "sigma_bound_b": sigma_d,
                            "multiplicity": mult, "dw_lambda": float(f0),
                            "teff_ratio": float(tbar)})

    # PER-ATOM (per represented atom): divide the per-cell sum by the atom count,
    # so the absolute scale matches inelastic_mode 1/2 (which the engine also
    # normalizes per represented atom).
    S_total = S_cell / n_atoms
    sigma_b_total = sigma_b_cell / n_atoms          # atom-weighted average bound xs
    return {
        "q_ang_inv": Q,
        "e_mev": E,
        "sqe_barn_per_meV": S_total,                # (nQ, nE) physical d2sig/dOmega/dE'
        "per_species": per_species,
        "sigma_b_total": float(sigma_b_total),      # atom-averaged represented bound xs
        "temperature_k": T_K,
        "nphon_effective": int(nphon_eff),          # the order the ladder ran
        "nphon_required": int(nphon_req),           # derived requirement ('auto')
    }


class GainGridTooLargeError(ValueError):
    """The direct energy-gain FFT grid would exceed the memory budget.

    Raised by :func:`compute_mode0_gain_direct` for extreme (light mass, very
    high Q, very low T) inputs whose converged phonon ladder needs an
    impractically large signed grid -- a regime where the energy-gain side is
    physically negligible anyway. The forward model catches this and falls back
    to the detailed-balance mirror (with a NOTE)."""


def compute_mode0_gain_direct(*, species, temperature_k, q_ang_inv, e_gain_mev,
                              nphon="auto", max_nfft=1 << 24):
    """DIRECT energy-gain S(Q, E<0): explicit Bose factors, no detailed balance.

    Computes the incoherent-approximation phonon expansion on a SIGNED energy
    grid with the occupation factors written out per process -- ``n(omega)+1``
    for phonon creation (neutron energy loss) and ``n(omega)`` for annihilation
    (energy gain) -- so a p-phonon event mixes creation and annihilation in all
    orderings through signed-grid convolutions. The energy-gain side returned
    here is COMPUTED, never mirrored from the loss side.

    The ladder is summed in Fourier space. With ``t1`` the unit-mass signed
    one-phonon distribution and ``lam`` its Debye-Waller integral,

        S_inel(alpha, beta) = IFFT[ exp(alpha*lam*(FFT(t1) - 1)) - exp(-alpha*lam) ]

    sums EVERY order at once (the discrete-grid analogue of
    ``exp(-2W) sum_p (2W)^p/p! T_p``) and is used when ``nphon='auto'`` (the
    converged-order convention of :func:`compute_mode0_sqe`). An explicit
    integer ``nphon`` instead sums the ladder to EXACTLY that order via the
    truncated Fourier partial sum ``exp(-alpha*lam) sum_{p=1}^{nphon}
    (alpha*lam)^p/p! FFT(t1)^p`` -- so the gain side carries the same phonon
    order as the (finite-order) loss side, never more.

    For the harmonic model in equilibrium this equals the detailed-balance
    mirror of the loss side identically; the agreement is pinned in CI as a
    cross-validation of BOTH paths.

    Parameters mirror :func:`compute_mode0_sqe`; ``e_gain_mev`` is the
    energy-GAIN grid (negative, ascending, e.g. ``-E_loss[::-1]``). ``max_nfft``
    caps the signed FFT length; an input whose converged ladder needs a longer
    grid raises :class:`GainGridTooLargeError` (the gain side is negligible
    there and the caller falls back to the mirror). Returns a dict with
    ``sqe_barn_per_meV`` (nQ, nGain) in the same PER-ATOM physical units as
    :func:`compute_mode0_sqe` and the per-species Debye-Waller ``dw_lambda``.
    """
    if not species:
        raise ValueError("compute_mode0_gain_direct: at least one species is required")
    T_K = float(temperature_k)
    if not T_K > 0.0:
        raise ValueError(f"temperature_k must be > 0, got {T_K}")
    auto_order = isinstance(nphon, str)
    if auto_order and nphon != "auto":
        raise ValueError(f"nphon must be an int or 'auto', got {nphon!r}")
    nphon_explicit = None if auto_order else int(nphon)
    if nphon_explicit is not None and nphon_explicit < 1:
        raise ValueError(f"nphon must be >= 1, got {nphon_explicit}")
    kT = KB * T_K                                       # meV
    Q = np.asarray(q_ang_inv, float)
    Eg = np.asarray(e_gain_mev, float)
    if Q.ndim != 1 or Eg.ndim != 1 or Q.size < 1 or Eg.size < 1:
        raise ValueError("q_ang_inv and e_gain_mev must be 1-D non-empty grids")
    if np.any(Eg > 0.0) or np.any(np.diff(Eg) <= 0.0):
        raise ValueError("e_gain_mev must be ascending and <= 0 (the energy-gain side)")
    beta_gain = Eg / kT                                 # <= 0
    nQ = Q.size

    S_cell = np.zeros((nQ, Eg.size))
    n_atoms = 0.0
    per_species = []
    for sp in species:
        symbol = sp.get("symbol", "?")
        omega, rho, delta1 = _validate_dos(sp["omega_ev"], sp["rho"], symbol)
        awr = float(sp["awr"])
        sigma_d = float(sp["sigma_bound_b"])
        mult = float(sp.get("multiplicity", 1))
        if not (awr > 0.0 and sigma_d >= 0.0 and mult > 0.0):
            raise ValueError(f"species {symbol!r}: need awr>0, sigma_bound_b>=0, "
                             f"multiplicity>0")
        rho = rho.copy()
        rho[0] = 0.0
        w_meV = omega * 1000.0
        db = (delta1 * 1000.0) / kT                     # beta step = DOS step
        # normalize the continuous DOS to unit area in beta (the LEAPR tbeta=1
        # convention contin applies internally)
        rho = rho / np.trapezoid(rho, w_meV / kT)

        alpha_d = C_E * Q ** 2 / (awr * kT)
        # signed beta reach: the requested gain range, plus the recoil shift +
        # spread of the ladder at the largest alpha. The ladder mean order is
        # the Poisson mean alpha*lam; the support spans ~that many one-phonon
        # hops (each bounded by the DOS reach b1max). An EXPLICIT nphon caps the
        # summed orders, so the grid never needs to hold more than nphon hops.
        b1max = float(w_meV.max() / kT)
        # lam is not known before t1 is built; bound it by the worst case
        # integral with coth(beta/2) <= 2/beta + 1 to size the grid
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_bound = float(np.trapezoid(
                np.where(w_meV > 0, rho * (2.0 * kT / np.maximum(w_meV, 1e-12) + 1.0), 0.0),
                w_meV / kT))
        a_max = float(alpha_d.max())
        mean_hops = a_max * lam_bound
        stat_orders = mean_hops + 10.0 * np.sqrt(mean_hops + 1.0) + 10.0
        eff_orders = (stat_orders if nphon_explicit is None
                      else min(stat_orders, float(nphon_explicit)))
        reach = abs(float(beta_gain.min())) + b1max * (eff_orders + 1.0)
        n_half = int(np.ceil(reach / db)) + 1
        n_fft = 1 << int(np.ceil(np.log2(2 * n_half + 1)))
        if n_fft > int(max_nfft):
            raise GainGridTooLargeError(
                f"species {symbol!r}: the direct energy-gain ladder needs a "
                f"signed FFT grid of {n_fft} points (> max_nfft={int(max_nfft)}) "
                f"at T={T_K:g} K, Q_max={float(Q.max()):g} 1/A, awr={awr:g} -- "
                "a regime where the gain side is physically negligible; use the "
                "detailed-balance mirror there.")
        n_half = (n_fft - 1) // 2                       # recenter on the padded grid
        beta = (np.arange(n_fft) - n_half) * db         # signed, beta[n_half] = 0

        rho_b = np.interp(np.abs(beta) * kT, w_meV, rho, left=0.0, right=0.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            occ = 1.0 / np.expm1(np.abs(beta))          # Bose n(|beta|)
        f1 = np.zeros(n_fft)
        pos = beta > 0
        neg = beta < 0
        f1[pos] = rho_b[pos] / beta[pos] * (occ[pos] + 1.0)     # creation
        f1[neg] = rho_b[neg] / np.abs(beta[neg]) * occ[neg]     # annihilation
        # beta -> 0: rho ~ c*beta^2 so both one-sided limits equal c*kT-ish;
        # take the average of the adjacent values (both finite)
        f1[n_half] = 0.5 * (f1[n_half + 1] + f1[n_half - 1])

        lam = float(np.trapezoid(f1, beta))             # Debye-Waller integral
        t1 = f1 * db / lam                              # unit-mass signed t1
        t1_hat = np.fft.fft(np.roll(t1, -n_half))       # beta=0 at index 0

        # ladder per Q (vectorized over the energy axis): closed form (auto) or
        # the truncated Poisson partial sum to nphon (explicit order match).
        S_d = np.empty((nQ, Eg.size))
        for iq in range(nQ):
            al = alpha_d[iq] * lam
            if nphon_explicit is None:
                s_hat = np.exp(al * (t1_hat - 1.0)) - np.exp(-al)
            else:
                # exp(-al) * sum_{p=1}^{N} (al)^p/p! * t1_hat^p  (no p=0 elastic)
                term = al * t1_hat                      # p = 1
                acc = term.copy()
                for p in range(2, nphon_explicit + 1):
                    term = term * (al / p) * t1_hat
                    acc += term
                s_hat = np.exp(-al) * acc
            s = np.fft.ifft(s_hat).real
            s = np.maximum(np.roll(s, n_half) / db, 0.0)  # mass -> density, clip FFT noise
            S_d[iq] = np.interp(beta_gain, beta, s, left=0.0, right=0.0)
        # physical per-atom downscatter convention: sigma/(4 pi kT) * S_asym
        S_cell += mult * (sigma_d / (4.0 * np.pi * kT)) * S_d
        n_atoms += mult
        per_species.append({"symbol": symbol, "awr": awr,
                            "sigma_bound_b": sigma_d, "multiplicity": mult,
                            "dw_lambda": lam})

    return {
        "q_ang_inv": Q,
        "e_gain_mev": Eg,
        "sqe_barn_per_meV": S_cell / n_atoms,
        "per_species": per_species,
        "temperature_k": T_K,
    }
