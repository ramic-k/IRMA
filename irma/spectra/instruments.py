"""Instrument-geometry layer: VISION / generic indirect / direct geometry.

Wraps the low-level forward model in :mod:`irma.spectra.sqe` and the rigorous
elastic line in :mod:`irma.spectra.elastic` behind instrument presets that
mirror OCLIMAX's INSTR options:

  * ``INSTR=0`` VISION          -> inverse geometry, fixed Ef, the two VISION banks
  * ``INSTR=1`` generic indirect-> fixed Ef, a user list/range of detector angles
  * ``INSTR=2`` direct geometry -> fixed Ei, a user list/range of detector angles

A detector "bank" is a set of scattering angles.  For each angle we sample the
powder S(Q,E) along that angle's kinematic Q(E) trajectory, convolve the energy
resolution, and (optionally) add the bank's elastic line from MF7/MT2.  The
angle results are then combined (mean or sum) into one 1-D spectrum.

Energy convention follows :mod:`irma.spectra.sqe`: E>0 is neutron energy loss
(downscatter); the energy-gain side is reconstructed by detailed balance.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence, Optional

import numpy as np

from irma.spectra import sqe as si
from irma.spectra import elastic as el


@dataclasses.dataclass
class Instrument:
    """A spectrometer configuration for the forward model.

    name            : label for plots
    geometry        : 'indirect' (fixed Ef) or 'direct' (fixed Ei)
    E_fixed         : Ef (indirect) or Ei (direct) in meV
    angles_deg      : detector scattering angles (2theta) making up the bank
    sigma_coeffs    : Gaussian energy-resolution sigma polynomial (meV)
    bank_halfwidth_deg : angular acceptance (+/-) used for the elastic Bragg
                      integration of EACH angle (collects reflections in the
                      angle's elastic-Q window); set 0 for a point differential
    combine         : 'mean' (per-detector average) or 'sum' over angles
    resolution_model : 'poly' (sigma from sigma_coeffs) or 'chopper' (auto
                      chopper resolution from chopper_spec = {instrument,
                      package, frequency}); 'chopper' is direct-geometry only
    chopper_spec    : dict instrument,package,frequency for the 'chopper' model
    """
    name: str
    geometry: str
    E_fixed: float
    angles_deg: Sequence[float]
    sigma_coeffs: tuple = si.VISION_SIGMA_COEFFS
    bank_halfwidth_deg: float = 5.0
    combine: str = "mean"
    resolution_model: str = "poly"
    chopper_spec: Optional[dict] = None

    def __post_init__(self):
        # The auto chopper model is a direct-geometry construct (it propagates the
        # incident-energy resolution through the chopper). Reject an inconsistent
        # geometry at CONSTRUCTION -- the same contract config.validate() enforces
        # -- so a hand-built Instrument can't silently mis-resolve an indirect bank.
        if self.resolution_model == "chopper" and self.geometry != "direct":
            raise ValueError(
                "resolution_model='chopper' is a direct-geometry model; it "
                f"requires geometry='direct' (got {self.geometry!r})")
        if self.combine not in ("mean", "sum"):
            # Anything other than 'mean' silently fell through to np.sum below;
            # reject an unknown reducer at construction instead.
            raise ValueError(
                f"combine must be 'mean' or 'sum' (got {self.combine!r})")

    # --- resolution width source -----------------------------------------
    def width_source(self):
        """Return the resolution width argument for ``resolution_convolve``.

        'poly' -> the sigma_coeffs polynomial (legacy, unchanged). 'chopper'
        -> a callable E->sigma(meV) bound to Ei=E_fixed, re-evaluated on
        whatever E_out each call site uses.
        """
        if self.resolution_model == "chopper":
            if not self.chopper_spec:
                raise ValueError("resolution_model='chopper' requires chopper_spec "
                                 "(instrument, package, frequency)")
            from irma.spectra.chopper_resolution import chopper_sigma_of_E
            s = self.chopper_spec
            return lambda E: chopper_sigma_of_E(E, Ei=self.E_fixed, **s)
        return self.sigma_coeffs

    # --- kinematics -------------------------------------------------------
    def Q_of_E(self, two_theta_deg):
        """Return a callable E(meV) -> Q(1/A) for one detector angle."""
        if self.geometry == "indirect":
            return lambda E: si.Q_indirect(E, self.E_fixed, two_theta_deg)
        elif self.geometry == "direct":
            return lambda E: si.Q_direct(E, self.E_fixed, two_theta_deg)
        raise ValueError(f"unknown geometry {self.geometry!r}")

    def kf_ki(self):
        """kf/ki factor callable for this geometry (count-rate spectra)."""
        if self.geometry == "indirect":
            return si.kf_ki_indirect(self.E_fixed)
        return si.kf_ki_direct(self.E_fixed)


# ---- presets ----------------------------------------------------------------
def VISION(Ef=si.VISION_EF_MEV, bank_halfwidth_deg=5.0, sigma_coeffs=None,
           combine="mean"):
    """VISION inverse-geometry preset (forward 45 deg + backward 135 deg banks).

    ``sigma_coeffs``/``combine`` override the preset resolution polynomial and
    bank combination — by default the published VISION sigma polynomial and the
    bank mean.
    """
    return Instrument(
        name="VISION", geometry="indirect", E_fixed=Ef,
        angles_deg=list(si.VISION_BANKS.values()),
        sigma_coeffs=(si.VISION_SIGMA_COEFFS if sigma_coeffs is None
                      else tuple(sigma_coeffs)),
        bank_halfwidth_deg=bank_halfwidth_deg, combine=combine)


def indirect(Ef, angles_deg, sigma_coeffs=si.VISION_SIGMA_COEFFS,
             bank_halfwidth_deg=5.0, combine="mean", name=None):
    """Generic indirect-geometry instrument (OCLIMAX INSTR=1).

    Bank combination (``combine``) is a FLAT mean/sum over the listed detector
    angles — each angle's inelastic spectrum is sampled at the bank center with
    equal weight (no solid-angle factor), while within a bank the elastic
    channel is integrated over the bank's angular acceptance. Match your
    instrument's reduction: a flat mean corresponds to per-bank-normalized
    summing (the OCLIMAX convention).
    """
    return Instrument(name=name or f"indirect Ef={Ef:g}meV", geometry="indirect",
                      E_fixed=Ef, angles_deg=list(angles_deg),
                      sigma_coeffs=sigma_coeffs,
                      bank_halfwidth_deg=bank_halfwidth_deg, combine=combine)


def direct(Ei, angles_deg, sigma_coeffs=None, bank_halfwidth_deg=5.0,
           combine="mean", name=None, resolution_model="poly",
           chopper_spec=None):
    """Generic direct-geometry instrument (OCLIMAX INSTR=2).

    Resolution: ``resolution_model='poly'`` uses ``sigma_coeffs`` (if None, a
    constant 2% of Ei placeholder). ``'chopper'`` auto-computes the resolution
    for a named instrument from ``chopper_spec`` (instrument, package,
    frequency) -- the recommended choice for a real direct-geometry instrument
    (ARCS, SEQUOIA, ...).
    """
    if sigma_coeffs is None:
        sigma_coeffs = (0.02 * Ei, 0.0, 0.0)
    return Instrument(name=name or f"direct Ei={Ei:g}meV", geometry="direct",
                      E_fixed=Ei, angles_deg=list(angles_deg),
                      sigma_coeffs=sigma_coeffs,
                      bank_halfwidth_deg=bank_halfwidth_deg, combine=combine,
                      resolution_model=resolution_model,
                      chopper_spec=chopper_spec)


# ---- simulation -------------------------------------------------------------
def simulate(p: si.PowderSQE, instrument: Instrument, E_out,
             elastic_model: Optional[el.ElasticModel] = None,
             include_gain=True, kinematic_factor=False, per_angle=False,
             shape="gaussian"):
    """Forward-model a 1-D spectrum for ``instrument`` from powder S(Q,E) ``p``.

    p              : PowderSQE source (any provenance)
    instrument     : Instrument config
    E_out          : output energy-transfer grid (meV); may include negatives
    elastic_model  : if given, add the rigorous MF7/MT2 elastic line per angle
    include_gain   : add the detailed-balance energy-gain side
    kinematic_factor : multiply by kf/ki for a count-rate spectrum (default off,
                     matching OCLIMAX's S(Q,omega) convention)
    per_angle      : also return the individual per-angle spectra
    shape          : resolution line shape, 'gaussian' or 'lorentzian'

    Returns dict: 'E','Q'(per angle),'I_inelastic','I_elastic','I_total','label',
    and (if per_angle) 'angles','I_inelastic_per_angle','I_elastic_per_angle'.
    """
    E_out = np.asarray(E_out, float)
    kfac = instrument.kf_ki() if kinematic_factor else None

    inel_stack, el_stack, Q_stack = [], [], []
    for tt in instrument.angles_deg:
        Qof = instrument.Q_of_E(tt)
        area = None
        if elastic_model is not None:
            area = el.bank_elastic_area(
                elastic_model, instrument.E_fixed, tt,
                dtheta_deg=instrument.bank_halfwidth_deg)
        out = si.instrument_spectrum(
            p, Qof, E_out, instrument.width_source(),
            include_gain=include_gain, elastic_area=area,
            kinematic_factor=kfac, shape=shape)
        inel_stack.append(out["I_inelastic"])
        el_stack.append(out["I_elastic"])
        Q_stack.append(out["Q"])

    inel = np.array(inel_stack)
    elc = np.array(el_stack)
    reducer = np.mean if instrument.combine == "mean" else np.sum
    I_inel = reducer(inel, axis=0)
    I_el = reducer(elc, axis=0)

    res = {
        "E": E_out,
        "Q": np.array(Q_stack),
        "I_inelastic": I_inel,
        "I_elastic": I_el,
        "I_total": I_inel + I_el,
        "label": f"{p.label} @ {instrument.name}",
    }
    if per_angle:
        res.update(angles=list(instrument.angles_deg),
                   I_inelastic_per_angle=inel,
                   I_elastic_per_angle=elc)
    return res


def simulate_q_cuts(p: si.PowderSQE, q_values, E_out, sigma_coeffs,
                    elastic_model=None, include_gain=True, q_res=0.05,
                    shape="gaussian", q_band=None, kinematic_factor=None):
    """Constant-|Q| cuts of the powder S(Q,E): I(E) at each fixed Q.

    A real detector follows a Q(E) locus; a constant-Q cut is the VERTICAL
    slice of S(Q,E) at a single |Q| (a theory cut, useful for reading off which
    modes live at a given Q). Each Q is sampled with the same resolution
    convolution as a bank, plus an optional elastic peak whose area is the
    elastic differential at that Q (``q_res`` broadens the Bragg peaks so a cut
    near an edge still collects it).

    ``q_band`` (half-width, 1/A): if given (> 0), each cut is AVERAGED over
    ``|Q| in [Q0 - q_band, Q0 + q_band]`` -- a finite detector Q-bin -- instead
    of an infinitely-thin slice at exactly Q0. The band is sampled on the S(Q,E)
    grid step ``q_res`` (>=3 points, capped at 21) and mean-averaged, keeping the
    cut on the same intensity scale as a thin slice. Returns per-Q arrays
    (n_q, nE).

    ``kinematic_factor`` (callable Etr->kf/ki, or None) is applied exactly as in
    the bank path :func:`simulate`, so a component cut stays on the same
    count-rate scale as the banks when kf/ki weighting is on.
    """
    E_out = np.asarray(E_out, float)
    q_values = [float(q) for q in q_values]
    band = float(q_band) if q_band else 0.0
    inel, elc = [], []
    for qv in q_values:
        if band > 0.0:
            n = int(min(21, max(3, round(2.0 * band / max(q_res, 1e-6)) + 1)))
            q_samples = np.clip(np.linspace(qv - band, qv + band, n), 1e-6, None)
            # Keep only band samples inside the powder Q-support: an outside
            # sample interpolates to exactly 0 (fill_value) and would silently
            # dilute the band mean. If the whole band is outside, keep the raw
            # samples -- the cut is genuinely unsupported and reads ~0.
            inside = q_samples[(q_samples >= float(p.q.min()))
                               & (q_samples <= float(p.q.max()))]
            if inside.size:
                q_samples = inside
        else:
            q_samples = [qv]
        sub_in, sub_el = [], []
        for q in q_samples:
            area = None
            if elastic_model is not None:
                area = float(elastic_model.elastic_dsigma_dOmega(float(q), q_res=q_res)[0])
            out = si.instrument_spectrum(
                p, (lambda E, q=float(q): np.full(np.shape(E), q, float)), E_out,
                sigma_coeffs, include_gain=include_gain, elastic_area=area,
                kinematic_factor=kinematic_factor, shape=shape)
            sub_in.append(out["I_inelastic"])
            sub_el.append(out["I_elastic"])
        inel.append(np.mean(sub_in, axis=0))
        elc.append(np.mean(sub_el, axis=0))
    return {
        "E": E_out,
        "q_values": q_values,
        "I_inelastic_per_q": np.array(inel) if inel else np.empty((0, E_out.size)),
        "I_elastic_per_q": np.array(elc) if elc else np.empty((0, E_out.size)),
    }
