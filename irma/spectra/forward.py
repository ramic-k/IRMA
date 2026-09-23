"""Forward-spectrum orchestration for ``irma.spectra``.

``compute_spectrum`` computes ``S(Q,E)`` FRESH from a phonon model by
calling the in-process noncubic SAB engine on an instrument-tailored locus grid
(NOT a dense 2-D grid), reads the engine's physical ``sqe_*`` map directly via
``_pick_sqe_key`` (no SAB inversion, no tape), then projects it onto the
instrument's kinematic locus with ``instruments.simulate``.

Cost note: the locus grid keeps the Q-support to the bank-locus envelope (tens of
shells), but the multiphonon background at each Q still needs the full energy
work-grid self-convolution -- so the win vs a dense validation grid is ~2-10x
(fewer Q shells), not orders of magnitude.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _resolve_jobs(jobs):
    """Resolve the worker count for the engine's parallel block loops.

    The noncubic engine parallelizes the coherent / incoherent / multiphonon
    direction-block sums across ``jobs`` fork workers; the forward model must
    NOT silently default to serial. ``jobs=None`` (the default) means AUTO: the
    ``IRMA_JOBS`` / ``IRMA_NCPU`` env override if set, else every CPU core. An
    explicit positive int is honored as-is.
    """
    if jobs is not None and int(jobs) > 0:
        return int(jobs)
    for env in ("IRMA_JOBS", "IRMA_NCPU"):
        v = os.environ.get(env)
        if v and v.strip().isdigit() and int(v) > 0:
            return int(v)
    return max(1, os.cpu_count() or 1)


# Engine compute contexts are expensive (phonopy load, full-mesh eigensolves,
# star-projection tensors) and temperature-independent: repeated spectra calls
# on the same model/grids (GUI parameter sweeps, a cut followed by a map, the
# CLI run+map pair) reuse the previous one instead of rebuilding.
# Size-one policy: contexts hold O(100 MB) at production meshes, so a
# long-lived GUI session keeps only the most recent.
_ENGINE_CONTEXT_CACHE: dict = {}


def _get_engine_context(*, phonopy_yaml, force_constants, force_sets,
                        born_path, mesh, Q_support, E_support, num_directions,
                        multiphonon_num_directions, n_jobs,
                        site_scattering_lengths_angstrom,
                        site_incoherent_cross_sections_barn,
                        scattering_lengths_json,
                        incoherent_cross_sections_json):
    """Digest-keyed engine context for the spectra bridge.

    Delegates to :func:`irma.core.standalone_sab.get_or_build_context` so the
    spectra bridge and the ENDF standalone path share one cache-key
    discipline (identical model + mesh + grids + sampling -> the same
    context arrays).
    """
    from irma.core.standalone_sab import get_or_build_context, _grid_digest
    grid_key = ("physical-qe", _grid_digest(np.asarray(Q_support, float)),
                _grid_digest(np.asarray(E_support, float)))
    return get_or_build_context(
        phonopy_yaml=str(phonopy_yaml),
        force_constants=force_constants, force_sets=force_sets,
        born=born_path, mesh_dim=[int(m) for m in mesh], grid_key=grid_key,
        q_grid_ang_inv=np.asarray(Q_support, float),
        e_grid_mev=np.asarray(E_support, float),
        num_directions=int(num_directions),
        multiphonon_num_directions=int(multiphonon_num_directions),
        num_jobs=int(n_jobs),
        scattering_lengths_json=scattering_lengths_json,
        incoherent_cross_sections_json=incoherent_cross_sections_json,
        site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
        site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
        context_cache=_ENGINE_CONTEXT_CACHE)


def _pick_sqe_key(output_arrays, multiphonon_max_order, inelastic_mode):
    """Select the physical ``sqe_*_barn_per_meV`` key for (mode, effective order).

    Mirrors ``irma.core.standalone_sab._pick_sab_key`` EXACTLY -- the same
    ``(inelastic_mode, multiphonon_max_order)`` branch -- with the
    ``sab_asym_downscatter_`` prefix replaced by ``sqe_`` and the
    ``_barn_per_meV`` suffix appended. The ``sqe_*`` maps are the physical
    ``d2sigma/dOmega/dE'`` (= ``PowderSQE.S``); the ``sab_*`` maps carry the
    extra ``4*pi*kT/sigma_b`` thermal-scattering-law factor, so reading
    ``sqe_*`` is what the forward model wants.

    ``output_arrays`` is used only to validate the key is present, turning a
    desync between this mapping and the engine's population guards into a clear
    error instead of a silent zero / bare KeyError downstream.
    """
    if inelastic_mode == 2:
        base = ("one_phonon_total_plus_incoherent_approx_multiphonon"
                if multiphonon_max_order >= 2 else "one_phonon_total")
    elif inelastic_mode == 1:
        base = ("incoherent_approx_n1_term_plus_incoherent_approx_multiphonon"
                if multiphonon_max_order >= 2 else "incoherent_approx_n1_term")
    else:
        raise ValueError(
            f"_pick_sqe_key: inelastic_mode must be 1 or 2, got {inelastic_mode}")
    key = f"sqe_{base}_barn_per_meV"
    if output_arrays is not None and key not in output_arrays:
        raise KeyError(
            f"_pick_sqe_key selected '{key}' for inelastic_mode={inelastic_mode}, "
            f"multiphonon_max_order={multiphonon_max_order}, but it is absent from "
            "the engine output arrays -- the sqe-key mapping and the engine's "
            "array-population guards have desynchronized.")
    return key


@dataclasses.dataclass
class SpectrumResult:
    """Result of a forward INS-spectrum calculation.

    The PER-ANGLE arrays (one row per detector bank) are the primary output --
    each bank sits at a different scattering angle and records its own spectrum.
    The combined ``I_*`` fields are the bank-reduced (mean/sum) convenience curve.
    """
    E: np.ndarray              # output energy-transfer axis (meV)
    Q: np.ndarray              # per-angle Q(E) loci (n_angles, nE)
    I_inelastic: np.ndarray    # bank-combined inelastic spectrum (barn/sr/meV, an energy density)
    I_elastic: np.ndarray      # bank-combined elastic line (barn/sr/meV; integrates over E to the barn/sr area)
    I_total: np.ndarray
    geometry: str
    metadata: dict
    angles_deg: list = dataclasses.field(default_factory=list)
    I_inelastic_per_angle: np.ndarray = None    # (n_angles, nE)
    I_elastic_per_angle: np.ndarray = None       # (n_angles, nE)
    # optional constant-|Q| cuts (vertical slices of S(Q,E))
    q_cut_values: list = dataclasses.field(default_factory=list)
    I_inelastic_per_q: np.ndarray = None         # (n_q, nE)
    I_elastic_per_q: np.ndarray = None           # (n_q, nE)

    @property
    def I_total_per_angle(self):
        """Per-bank total (inelastic + elastic), (n_angles, nE), or None for a
        cuts-only result that produced no per-angle spectra."""
        if self.I_inelastic_per_angle is None:
            return None
        return self.I_inelastic_per_angle + self.I_elastic_per_angle

    @property
    def I_total_per_q(self):
        """Per-Q-cut total (inelastic + elastic), (n_q, nE)."""
        if self.I_inelastic_per_q is None:
            return None
        return self.I_inelastic_per_q + self.I_elastic_per_q


def build_locus_support(geometry, e_fixed_meV, angles_deg, dE, e_max, dQ,
                        e_min=0.0, q_pad=0.5, q_floor=0.05, include_gain=True):
    """Build the engine's (Q_support, E_support) for an instrument locus.

    E_support is the uniform LOSS-side energy grid the engine computes S(Q,E)
    on; the energy-gain side is reconstructed downstream by detailed balance
    (mirroring the loss side), so it reaches only -(max loss energy). When a
    deeper gain side is requested (``include_gain`` and ``e_min < -e_max``), the
    loss grid is extended to ``|e_min|`` so the gain wing in [e_min, -e_max] has
    loss data to mirror instead of being silently zeroed by the fill_value=0
    interpolator. Q_support is a ``dQ``-spaced grid spanning the UNION envelope
    of every bank locus ``Q(E)`` over the full E range.
    """
    from irma.spectra.sqe import Q_indirect, Q_direct
    e_loss_max = float(e_max)
    if include_gain and float(e_min) < 0.0:
        e_loss_max = max(e_loss_max, -float(e_min))               # cover the gain wing
    E_support = np.arange(0.0, e_loss_max + 0.5 * dE, dE)         # engine loss grid
    # The Q envelope must cover the full SIGNED output range [e_min, e_max], not
    # just the loss side: when e_min < 0 the energy-GAIN locus Q(E<0) reaches
    # different (often higher) |Q| than the loss side, and the output spectrum is
    # sampled there too. Spanning only [0, e_max] would let the fill_value=0
    # interpolator silently zero the energy-gain wing.
    E_env = np.arange(min(0.0, float(e_min)), float(e_max) + 0.5 * dE, dE)
    if geometry in ("vision", "indirect"):
        def Qof(tt):
            """|Q|(E) locus at scattering angle ``tt`` (indirect geometry)."""
            return Q_indirect(E_env, e_fixed_meV, tt)
    elif geometry == "direct":
        def Qof(tt):
            """|Q|(E) locus at scattering angle ``tt`` (direct geometry)."""
            return Q_direct(E_env, e_fixed_meV, tt)
    else:
        raise ValueError(f"build_locus_support: unknown geometry {geometry!r}")
    Qall = np.concatenate([np.asarray(Qof(tt), float) for tt in angles_deg])
    Qall = Qall[np.isfinite(Qall) & (Qall > 0)]
    q_lo = float(Qall.min()); q_hi = float(Qall.max())
    Q_support = np.arange(max(q_floor, q_lo - q_pad), q_hi + q_pad + dQ, dQ)
    return Q_support, E_support


def _report_mode0_order(m0, auto_order, q_grid, progress):
    """Report the mode-0 phonon order (and warn when the 2000 cap truncates).

    The mode-0 analogue of the engine's auto-order INFO/WARNING lines: 'auto'
    sizes the contin ladder to the converged Poisson(f0*alpha_max) order from
    the DOS Debye-Waller lambda (derive_mode0_phonon_order), so the chosen
    order must be visible at runtime, not only in the result metadata.
    """
    eff = int(m0["nphon_effective"])
    req = int(m0["nphon_required"])
    if auto_order:
        progress(f"mode-0 multiphonon: auto-sized phonon-expansion order to "
                 f"{eff} (converges the Poisson(f0*alpha) ladder at "
                 f"Q_max={float(np.max(q_grid)):.1f} 1/Angstrom)")
    else:
        progress(f"mode-0 multiphonon: phonon-expansion order {eff} (explicit)")
    if req > eff:
        progress(f"WARNING: mode-0 required phonon order ~{req} exceeds the "
                 f"safety cap ({eff}); the highest-Q rows fall back to the "
                 "short-collision-time tail short of full convergence.")


def _resolve_gain_side(*, gain_side, include_gain, dos_species, progress):
    """Resolve the requested energy-gain evaluation.

    ``"direct"`` computes the gain side with explicit Bose occupation factors
    (annihilation weight ``n(omega)``), never a detailed-balance mirror. Both
    the DOS path (mode 0, ``compute_mode0_gain_direct``) and the eigenvector
    engine (modes 1/2, ``emit_gain_side``) support it; the mode-0 path may still
    degrade to the mirror for an extreme grid (handled in ``_mode0_direct_gain``),
    and the engine path degrades only if it surfaced no gain arrays (handled at
    the call site). ``"detailed_balance"`` selects the mirror explicitly.

    ``dos_species`` is accepted for signature stability and has no effect
    on the decision (the engine computes the gain side directly).
    """
    if gain_side not in ("direct", "detailed_balance"):
        raise ValueError(
            f"gain_side must be 'direct' or 'detailed_balance', got {gain_side!r}")
    if not include_gain or gain_side == "detailed_balance":
        return "detailed_balance"
    return "direct"


def _engine_direct_gain(*, out, loss_key, gain_side_used, progress):
    """Extract the engine's directly-computed gain side (modes 1/2).

    Returns ``(gain_kwargs, gain_used)``: the ``{E_gain, S_gain}`` kwargs for
    :func:`irma.spectra.sqe.from_noncubic_arrays` plus the resolved gain mode.
    Passes through unchanged when the gain side was not requested; downgrades to
    ``"detailed_balance"`` with a NOTE if the engine surfaced no gain arrays
    (defensive -- ``emit_gain_side`` should always produce them). The gain key
    is the selected loss key with ``_gain`` inserted, on the engine's
    ``e_gain_mev`` grid (strictly negative, the mirror of the positive loss
    grid) -- so the layout matches what ``signed_sqe`` expects.
    """
    if gain_side_used != "direct":
        return {}, gain_side_used
    gain_key = loss_key.replace("_barn_per_meV", "_gain_barn_per_meV")
    if gain_key in out and "e_gain_mev" in out:
        progress("energy-gain side: DIRECT engine evaluation (explicit Bose "
                 "annihilation factors at -hw + signed multiphonon ladder; no "
                 "detailed-balance mirror)")
        return ({"E_gain": np.asarray(out["e_gain_mev"], float),
                 "S_gain": np.asarray(out[gain_key], float)}, "direct")
    progress("NOTE: gain_side='direct' but the engine surfaced no gain arrays "
             f"for {gain_key!r}; falling back to the detailed-balance mirror.")
    return {}, "detailed_balance"


def _mode0_direct_gain(*, dos_species, temperature_k, q, E_loss, nphon, progress):
    """Direct-Bose gain side for the mode-0 powder -> (gain_kwargs, gain_used).

    Evaluates :func:`irma.spectra.dos_mode0.compute_mode0_gain_direct` on the
    mirror grid ``-E_loss[E_loss>0][::-1]`` so the signed assembly in
    ``signed_sqe`` meshes exactly with the loss side, summing the SAME phonon
    order ``nphon`` the loss side used (so the gain side is never more complete
    than the loss side). Returns ``({E_gain, S_gain}, "direct")`` on success;
    on an extreme (light-mass / very-high-Q / very-low-T) input whose ladder
    overflows the FFT-grid budget -- where the gain side is physically
    negligible -- it falls back with ``({}, "detailed_balance")`` and a NOTE.
    """
    Epos = np.asarray(E_loss, float)
    Epos = Epos[Epos > 0.0]
    if Epos.size == 0:
        return {}, "direct"
    from irma.spectra.dos_mode0 import (compute_mode0_gain_direct,
                                        GainGridTooLargeError)
    try:
        g = compute_mode0_gain_direct(
            species=dos_species, temperature_k=float(temperature_k),
            q_ang_inv=np.asarray(q, float), e_gain_mev=-Epos[::-1], nphon=nphon)
    except GainGridTooLargeError as exc:
        progress(f"NOTE: direct energy-gain evaluation falls back to the "
                 f"detailed-balance mirror -- {exc}")
        return {}, "detailed_balance"
    order = ("all orders (closed form)" if isinstance(nphon, str)
             else f"order {int(nphon)}")
    progress(f"energy-gain side: DIRECT evaluation (explicit Bose factors, "
             f"{order}; no detailed-balance mirror)")
    return {"E_gain": g["e_gain_mev"], "S_gain": g["sqe_barn_per_meV"]}, "direct"


def _build_mode0_elastic_model(*, dos_species, m0, dos_crystal, elastic_kind,
                               temperature_k, emax_eV, geometry, progress):
    """Tape-free mode-0 elastic line from the DOS-derived isotropic DW f0.

    Routes by ``elastic_kind`` and the presence of ``dos_crystal`` exactly like
    the 1-D spectrum path: ``'incoherent'`` (or ``'both'`` degraded when no
    crystal is given) builds the lattice-free Debye-Waller line; a crystal
    (lattice + per-species ``b_coh_fm``/``positions``) enables the coherent
    Bragg peaks. Returns ``None`` (with a NOTE) when the requested channel
    cannot be built -- the result is then inelastic-only. Shared by
    ``compute_spectrum`` and ``compute_sqe_map`` so the map carries the SAME
    elastic line as the 1-D spectra.
    """
    awr_l = [float(sp["awr"]) for sp in dos_species]
    sinc_l = [float(sp.get("sigma_inc_b", 0.0)) for sp in dos_species]
    f0_l = [float(ps["dw_lambda"]) for ps in m0["per_species"]]
    if elastic_kind == "incoherent" or (elastic_kind == "both"
                                        and dos_crystal is None):
        # lattice-free: 'incoherent' by request, or 'both' degraded to
        # its available channel when no crystal was given
        from irma.spectra.elastic import from_dos_incoherent_only
        mult_l = [int(sp.get("multiplicity", 1)) for sp in dos_species]
        elastic_model = from_dos_incoherent_only(
            awr=awr_l, sigma_inc_b=sinc_l, f0_lambda=f0_l,
            multiplicity=mult_l, T_K=float(temperature_k),
            label=f"IRMA {geometry} mode-0 elastic (incoherent)")
        progress(f"mode-0 elastic (incoherent): "
                 f"sigma_b={elastic_model.sigma_b:.4g} b (lattice-free)")
        if elastic_kind == "both":
            progress("NOTE: elastic_kind='both' without material.lattice "
                     "builds the incoherent line only; add the lattice "
                     "(+ per-species positions) for the coherent Bragg "
                     "peaks.")
        return elastic_model
    if dos_crystal is not None:
        from irma.spectra.elastic import from_dos_and_lattice
        from irma.core.crystal import CrystalStructure, AtomSite
        sites = []
        for sp in dos_species:
            if sp.get("positions") is None or sp.get("b_coh_fm") is None:
                raise ValueError(
                    "the mode-0 coherent elastic line needs b_coh_fm + "
                    f"positions for every species; "
                    f"{sp.get('symbol', '?')!r} is missing one")
            sites.append(AtomSite(b_coh_fm=float(sp["b_coh_fm"]),
                                  positions=[tuple(float(x) for x in p)
                                             for p in sp["positions"]]))
        a_, b_, c_, al_, be_, ga_ = (float(x) for x in dos_crystal)
        crystal = CrystalStructure(a_, b_, c_, al_, be_, ga_, sites)
        elastic_model = from_dos_and_lattice(
            crystal, awr=awr_l, sigma_inc_b=sinc_l, f0_lambda=f0_l,
            T_K=float(temperature_k), elastic_kind=elastic_kind,
            emax_eV=emax_eV,
            label=f"IRMA {geometry} mode-0 elastic ({elastic_kind})")
        progress(f"mode-0 elastic ({elastic_kind}): "
                 f"{elastic_model.Q_bragg.size} Bragg edges, "
                 f"sigma_b={elastic_model.sigma_b:.4g} b")
        return elastic_model
    # elastic_kind == 'coherent' with no crystal
    progress("NOTE: the mode-0 coherent elastic line needs "
             "material.lattice (+ per-species positions); the result "
             "is inelastic-only. Use elastic_kind='incoherent' "
             "(or 'both') for a lattice-free elastic line.")
    return None


def _build_engine_elastic_model(*, elastic_state, elastic_scatterers,
                                elastic_kind, temperature_k, emax_eV,
                                geometry, progress,
                                incoherent_elastic_mode="isotropic"):
    """Tape-free elastic line from the engine's surfaced ``elastic_state``.

    ``elastic_scatterers`` maps each primitive symbol to
    ``{b_coh_fm, sigma_inc_b, awr}``; the Debye-Waller state comes from the
    engine's thermal-displacement matrices, so the line shares the DW physics
    with the inelastic kernel by construction. Returns ``None`` (with a
    WARNING) when the engine surfaced no elastic state. Shared by
    ``compute_spectrum`` and ``compute_sqe_map``.
    """
    es = elastic_state
    if es is None:
        progress("WARNING: elastic=True but the engine surfaced no "
                 "elastic_state (no thermal-displacement matrices); "
                 "elastic line skipped.")
        return None
    if not elastic_scatterers:
        raise ValueError(
            "elastic=True requires "
            "elastic_scatterers = {symbol: {b_coh_fm, sigma_inc_b, awr}}")
    from irma.spectra.elastic import from_engine_elastic_state
    syms = es["primitive_symbols"]
    missing = sorted(set(syms) - set(elastic_scatterers))
    if missing:
        raise ValueError(
            f"elastic_scatterers is missing entries for {missing} "
            f"(primitive symbols: {sorted(set(syms))})")
    b_coh_fm = [float(elastic_scatterers[s]["b_coh_fm"]) for s in syms]
    sigma_inc_b = [float(elastic_scatterers[s].get("sigma_inc_b", 0.0))
                   for s in syms]
    awr_el = [float(elastic_scatterers[s]["awr"]) for s in syms]
    elastic_model = from_engine_elastic_state(
        es, b_coh_fm=b_coh_fm, sigma_inc_b=sigma_inc_b, awr=awr_el,
        T_K=float(temperature_k), elastic_kind=elastic_kind,
        emax_eV=emax_eV,
        incoherent_elastic_mode=incoherent_elastic_mode,
        label=f"IRMA {geometry} elastic ({elastic_kind})")
    progress(f"elastic line ({elastic_kind}): "
             f"{elastic_model.Q_bragg.size} Bragg edges, "
             f"sigma_b={elastic_model.sigma_b:.4g} b")
    return elastic_model


def compute_spectrum(*, geometry, phonopy_yaml, temperature_k, mesh,
                     sab_mass_ratio, sab_sigma_barn,
                     angles_deg=None, e_fixed_meV=None, q_cuts=None, cut_dq=None,
                     produce_angle_spectra=True,
                     dE=0.5, dQ=0.05, e_min=0.0, e_max=250.0,
                     bank_halfwidth_deg=5.0, sigma_coeffs=None,
                     resolution_shape="gaussian",
                     resolution_model="poly", chopper_spec=None,
                     combine="mean", inelastic_mode=2,
                     dos_species=None, dos_crystal=None,
                     num_directions=10000, multiphonon_num_directions=1000,
                     multiphonon_max_order=100, auto_multiphonon_order=True,
                     min_phonon_energy_mev=0.0,
                     jobs=None,
                     force_constants=None, force_sets=None, born_path=None,
                     site_scattering_lengths_angstrom=None,
                     site_incoherent_cross_sections_barn=None,
                     scattering_lengths_json=None,
                     incoherent_cross_sections_json=None,
                     elastic_model=None, elastic=False,
                     elastic_kind="both", elastic_scatterers=None,
                     incoherent_elastic_mode="isotropic",
                     include_gain=True, gain_side="direct",
                     kinematic_factor=False, q_pad=0.5, workdir=None,
                     label=None, progress=print):
    """Compute a fresh-from-phonons forward INS spectrum for one instrument.

    Reuses the IRMA noncubic SAB engine (``run_noncubic_sab_inprocess``) on the
    instrument-locus grid; reads the physical ``sqe_*`` map directly; projects
    via ``instruments.simulate``. By default the engine also emits the
    energy-gain side directly (``gain_side="direct"``, all modes); see the
    "Energy-gain side" paragraph below.

    ``inelastic_mode`` defaults to 2 (exact coherent one-phonon), the same
    default :class:`~irma.spectra.config.PhysicsConfig`, the ``irma spectra``
    CLI, the GUI and :func:`compute_sqe_map` carry -- a Python caller and a
    CLI user running the same calculation must get the same physics.

    Mode 0 (DOS-based): pass ``dos_species`` (a list of per-species dicts as
    accepted by :func:`irma.spectra.dos_mode0.compute_mode0_sqe`) to build
    ``S(Q,E)`` from phonon DOS by the incoherent-approximation phonon expansion
    -- no ``phonopy_yaml``/``mesh``/engine. The instrument projection, resolution,
    cuts and 2-D map are identical downstream. For the mode-0 elastic line set
    ``elastic=True`` and pass ``dos_crystal=(a,b,c,alpha,beta,gamma)`` plus each
    species' ``b_coh_fm`` + ``positions`` (fractional sites) on its
    ``dos_species`` entry: the coherent Bragg peaks + incoherent DW line are built
    from the DOS-derived isotropic Debye-Waller f0 (the spectra analogue of the
    ENDF iel=10 coherent-elastic option). An explicit ``elastic_model`` still
    overrides.

    Elastic line: pass an explicit ``elastic_model`` (e.g. from an ENDF tape),
    OR set ``elastic=True`` to build it tape-free from the engine's surfaced
    ``elastic_state`` -- ``elastic_scatterers`` then maps each primitive
    symbol to ``{b_coh_fm, sigma_inc_b, awr}`` and ``elastic_kind`` selects
    ``"both"`` (default: coherent Bragg peaks + incoherent DW line), or one of
    ``"coherent"`` / ``"incoherent"`` to isolate a channel. The line shares the
    Debye-Waller state with
    the inelastic kernel by construction (same phonon calculation).

    Energy-gain side: ``gain_side="direct"`` (default, all modes) computes E<0
    with explicit Bose occupation factors -- phonon annihilation weights
    ``n(omega)``, all orders, no detailed-balance mirror. Mode 0 evaluates the
    gain ladder itself; modes 1/2 read the engine's direct gain arrays (see
    ``_engine_direct_gain``). ``"detailed_balance"`` selects the mirror as an
    explicit option; it is also the defensive fallback (with a NOTE) if an
    engine result surfaces no gain sibling. Direct and mirror agree to
    round-off for the equilibrium harmonic model (pinned in CI).
    """
    from irma.spectra import sqe as _sqe, instruments as _ins

    # A chopper model produces a Gaussian sigma; a Lorentzian shape contradicts
    # it. config.validate() rejects this, but a direct caller of compute_spectrum
    # bypasses that, so guard at the entry point too.
    if resolution_model == "chopper" and resolution_shape == "lorentzian":
        raise ValueError(
            "resolution_model='chopper' computes a Gaussian sigma and is "
            "incompatible with resolution_shape='lorentzian'; use "
            "resolution_shape='gaussian' (or resolution_model='poly').")

    # resolve the instrument preset + the kinematics inputs
    if geometry == "vision":
        # honor explicit user overrides; the preset fills the published VISION
        # sigma polynomial and bank-mean combination otherwise
        instr = _ins.VISION(Ef=(e_fixed_meV or _sqe.VISION_EF_MEV),
                            bank_halfwidth_deg=bank_halfwidth_deg,
                            sigma_coeffs=sigma_coeffs, combine=combine)
    elif geometry == "indirect":
        instr = _ins.indirect(e_fixed_meV, angles_deg,
                              sigma_coeffs=sigma_coeffs or _sqe.VISION_SIGMA_COEFFS,
                              bank_halfwidth_deg=bank_halfwidth_deg, combine=combine)
    elif geometry == "direct":
        instr = _ins.direct(e_fixed_meV, angles_deg, sigma_coeffs=sigma_coeffs,
                            bank_halfwidth_deg=bank_halfwidth_deg, combine=combine,
                            resolution_model=resolution_model,
                            chopper_spec=chopper_spec)
    else:
        raise ValueError(f"compute_spectrum: unknown geometry {geometry!r}")
    e_fixed = instr.E_fixed
    angles = list(instr.angles_deg)

    Q_support, E_support = build_locus_support(
        geometry, e_fixed, angles, dE, e_max, dQ, e_min, q_pad,
        include_gain=include_gain)

    # extend the Q-support to cover any requested constant-Q cuts, else the
    # interpolator (fill_value=0) would zero a cut that falls outside the bank
    # loci envelope. The pad must cover the full cut BAND (cut_dq half-width),
    # not just q_pad: band samples outside the support interpolate to exactly
    # 0 and silently dilute the band mean.
    if q_cuts:
        qc = [float(x) for x in q_cuts]
        pad = max(q_pad, float(cut_dq) if cut_dq else 0.0)
        lo = max(0.05, min(Q_support.min(), min(qc) - pad))
        hi = max(Q_support.max(), max(qc) + pad)
        Q_support = np.arange(lo, hi + dQ, dQ)

    # Reach-aware Bragg-enumeration cutoff for the tape-free elastic line: the
    # elastic line is only ever evaluated at the banks' elastic-Q windows
    # (<= 2 k_el), on the simulated Q-support and at the requested constant-Q
    # cut bands (q_res-broadened; q_res=dQ in the simulate_q_cuts call below);
    # edges beyond that reach contribute exactly zero, so the builders stop
    # enumerating there instead of at the ENDF tape writer's 5 eV default (the
    # MF7/MT2 path in the core driver keeps its own full-range call).
    from irma.spectra.elastic import instrument_reach_emax_eV
    elastic_emax_eV = instrument_reach_emax_eV(
        q_max_invA=float(Q_support.max()), e_fixed_meV=e_fixed,
        q_cuts=q_cuts, cut_dq=cut_dq, q_res_invA=dQ)

    # The energy-gain side is only ever sampled when the output grid extends
    # below 0; computing it for a loss-only grid (the default e_min=0) is pure
    # waste, so gate the direct evaluation on e_min < 0.
    gain_side_used = _resolve_gain_side(
        gain_side=gain_side, include_gain=(include_gain and float(e_min) < 0.0),
        dos_species=dos_species, progress=progress)

    if dos_species is not None:
        # mode-0: DOS-based incoherent-approximation S(Q,E) -- no phonopy engine.
        # Each species scatters by its OWN partial DOS, mass and Debye-Waller; the
        # total is the cross-section/multiplicity-weighted sum (compute_mode0_sqe).
        from irma.spectra.dos_mode0 import compute_mode0_sqe
        progress(f"compute_spectrum[{geometry}]: mode-0 DOS "
                 f"({len(dos_species)} species) nQ={Q_support.size} "
                 f"nE={E_support.size} T={temperature_k} K ...")
        m0 = compute_mode0_sqe(
            species=dos_species, temperature_k=float(temperature_k),
            q_ang_inv=Q_support, e_mev=E_support,
            nphon=("auto" if auto_multiphonon_order
                   else int(multiphonon_max_order)))
        q = np.asarray(m0["q_ang_inv"], float)
        E = np.asarray(m0["e_mev"], float)
        S = np.asarray(m0["sqe_barn_per_meV"], float)
        key = "mode0_dos"
        eff_order = int(m0["nphon_effective"])
        _report_mode0_order(m0, auto_multiphonon_order, Q_support, progress)
        result = {"metadata": {"mode": 0, "n_species": len(dos_species),
                               "per_species": m0["per_species"],
                               "gdos_e_mev": m0["gdos_e_mev"], "gdos": m0["gdos"],
                               "sigma_b_total": m0["sigma_b_total"]},
                  "elastic_state": None}
        gain_kw = {}
        if gain_side_used == "direct":
            gain_kw, gain_side_used = _mode0_direct_gain(
                dos_species=dos_species, temperature_k=temperature_k, q=q,
                E_loss=E,
                nphon=("auto" if auto_multiphonon_order
                       else int(multiphonon_max_order)),
                progress=progress)
        powder = _sqe.from_noncubic_arrays(
            q, E, S, T_K=float(temperature_k),
            sigma_b=float(m0["sigma_b_total"]),
            label=label or f"IRMA {geometry} mode-0 (DOS)", **gain_kw)

        # mode-0 elastic line: coherent Bragg peaks + incoherent DW from the
        # DOS-derived isotropic Debye-Waller coefficient f0 (contin). The
        # coherent peaks need a user-supplied crystal (dos_crystal =
        # (a,b,c,alpha,beta,gamma) [A,deg] + per-species b_coh_fm/positions);
        # the incoherent-only line runs lattice-free. Built here because f0 is
        # only known after the inelastic mode-0 pass.
        if elastic_model is None and elastic:
            elastic_model = _build_mode0_elastic_model(
                dos_species=dos_species, m0=m0, dos_crystal=dos_crystal,
                elastic_kind=elastic_kind, temperature_k=temperature_k,
                emax_eV=elastic_emax_eV, geometry=geometry, progress=progress)
    else:
        from irma.core.noncubic_inelastic import run_noncubic_sab_inprocess
        # Auto-created workdirs are removed after the engine call (the engine
        # writes nothing there with write_output_files=False; every mode-1/2
        # call used to leak one empty tempdir -- review S9). A caller-supplied
        # workdir is left alone.
        wd_auto = workdir is None
        wd = Path(workdir) if workdir is not None else Path(
            tempfile.mkdtemp(prefix="irma_spectra_"))
        n_jobs = _resolve_jobs(jobs)
        progress(f"compute_spectrum[{geometry}]: engine mode={inelastic_mode} "
                 f"mesh={tuple(int(m) for m in mesh)} nQ={Q_support.size} "
                 f"nE={E_support.size} T={temperature_k} K jobs={n_jobs} ...")
        try:
            context = _get_engine_context(
                phonopy_yaml=phonopy_yaml, force_constants=force_constants,
                force_sets=force_sets, born_path=born_path, mesh=mesh,
                Q_support=Q_support, E_support=E_support,
                num_directions=num_directions,
                multiphonon_num_directions=multiphonon_num_directions,
                n_jobs=n_jobs,
                site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
                site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
                scattering_lengths_json=scattering_lengths_json,
                incoherent_cross_sections_json=incoherent_cross_sections_json)
            result = run_noncubic_sab_inprocess(
                inelastic_mode=int(inelastic_mode),
                phonopy_yaml=str(phonopy_yaml),
                force_constants=force_constants, force_sets=force_sets, born_path=born_path,
                temperature_k=float(temperature_k),
                mesh=tuple(int(m) for m in mesh),
                q_grid_ang_inv=Q_support, e_grid_mev=E_support,
                output_prefix=wd / "spectra_unused",
                sab_mass_ratio=float(sab_mass_ratio),
                num_directions=int(num_directions),
                multiphonon_num_directions=int(multiphonon_num_directions),
                jobs=n_jobs,
                multiphonon_max_order=int(multiphonon_max_order),
                auto_multiphonon_order=bool(auto_multiphonon_order),
                min_phonon_energy_mev=float(min_phonon_energy_mev),
                sab_sigma_barn=float(sab_sigma_barn),
                site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
                site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
                scattering_lengths_json=scattering_lengths_json,
                incoherent_cross_sections_json=incoherent_cross_sections_json,
                context=context, write_output_files=False,
                emit_gain_side=(gain_side_used == "direct"))
        finally:
            if wd_auto:
                shutil.rmtree(wd, ignore_errors=True)

        out = result["output_arrays"]
        eff_order = int(result.get("metadata", {}).get(
            "multiphonon_max_order", multiphonon_max_order))
        key = _pick_sqe_key(out, eff_order, int(inelastic_mode))
        q = np.asarray(out["q_ang_inv"], float)
        E = np.asarray(out["e_mev"], float)
        S = np.asarray(out[key], float)
        gain_kw, gain_side_used = _engine_direct_gain(
            out=out, loss_key=key, gain_side_used=gain_side_used, progress=progress)

        # coverage guard: every locus point must fall inside the engine's Q grid,
        # else the interpolator (fill_value=0) silently zeros the high-E wing.
        if q.min() < Q_support.min() - 1e-9 or q.max() > Q_support.max() + 1e-9:
            progress(f"WARNING: engine q_ang_inv [{q.min():.3f},{q.max():.3f}] exceeds "
                     f"Q_support [{Q_support.min():.3f},{Q_support.max():.3f}]")
        # high-E truncation guard: the chosen map must be non-zero up to e_max.
        Sqe = S if S.shape == (q.size, E.size) else S.T
        e_axis_nonzero = np.flatnonzero(np.any(np.abs(Sqe) > 0, axis=0))
        if e_axis_nonzero.size:
            e_top = float(E[e_axis_nonzero[-1]])
            if e_top < 0.95 * float(e_max):
                progress(f"WARNING: S(Q,E) is identically zero above E={e_top:.1f} meV "
                         f"while e_max={e_max:.1f} meV -- multiphonon order likely too "
                         "small for the high-Q locus; raise max_phonon_order or keep "
                         "auto_multiphonon_order on.")

        powder = _sqe.from_noncubic_arrays(
            q, E, S, T_K=float(temperature_k), sigma_b=float(sab_sigma_barn),
            label=label or f"IRMA {geometry} mode-{inelastic_mode}", **gain_kw)

    # tape-free elastic line from the engine's surfaced DW + geometry.
    # Mode 0 builds its own elastic above (DOS-derived), so skip this here.
    if elastic_model is None and elastic and dos_species is None:
        elastic_model = _build_engine_elastic_model(
            elastic_state=result.get("elastic_state"),
            elastic_scatterers=elastic_scatterers, elastic_kind=elastic_kind,
            temperature_k=temperature_k, emax_eV=elastic_emax_eV,
            geometry=geometry, progress=progress,
            incoherent_elastic_mode=incoherent_elastic_mode)

    E_out = np.arange(float(e_min), float(e_max) + 0.5 * dE, dE)
    if produce_angle_spectra:
        sim = _ins.simulate(powder, instr, E_out, elastic_model=elastic_model,
                            include_gain=include_gain, kinematic_factor=kinematic_factor,
                            per_angle=True, shape=resolution_shape)
    else:
        # constant-Q-only output: no detector-angle spectra are produced.
        z = np.zeros(E_out.size)
        sim = {"E": E_out, "Q": np.empty((0, E_out.size)),
               "I_inelastic": z, "I_elastic": z, "I_total": z,
               "angles": [], "I_inelastic_per_angle": np.empty((0, E_out.size)),
               "I_elastic_per_angle": np.empty((0, E_out.size))}

    # optional constant-|Q| cuts (vertical slices of S(Q,E))
    q_cut_values, Iin_q, Iel_q = [], None, None
    if q_cuts:
        qc = _ins.simulate_q_cuts(
            powder, q_cuts, E_out, instr.width_source(),
            elastic_model=elastic_model, include_gain=include_gain, q_res=dQ,
            shape=resolution_shape, q_band=cut_dq,
            kinematic_factor=(instr.kf_ki() if kinematic_factor else None))
        q_cut_values = qc["q_values"]
        Iin_q, Iel_q = qc["I_inelastic_per_q"], qc["I_elastic_per_q"]
        band = f" (+/-{cut_dq} band)" if cut_dq else ""
        progress(f"constant-Q cuts at Q={q_cut_values} 1/A{band}")

    return SpectrumResult(
        E=sim["E"], Q=sim["Q"], I_inelastic=sim["I_inelastic"],
        I_elastic=sim["I_elastic"], I_total=sim["I_total"], geometry=geometry,
        angles_deg=list(sim["angles"]),
        I_inelastic_per_angle=np.asarray(sim["I_inelastic_per_angle"], float),
        I_elastic_per_angle=np.asarray(sim["I_elastic_per_angle"], float),
        q_cut_values=q_cut_values, I_inelastic_per_q=Iin_q, I_elastic_per_q=Iel_q,
        metadata={"sqe_key": key, "effective_multiphonon_order": eff_order,
                  "n_q_support": int(q.size), "n_e": int(E.size),
                  "inelastic_mode": int(inelastic_mode),
                  "resolution_shape": resolution_shape,
                  "resolution_model": resolution_model,
                  "angles_deg": list(sim["angles"]),
                  "elastic": elastic_model is not None,
                  "elastic_kind": (elastic_kind if elastic_model is not None else None),
                  "n_bragg_edges": (int(elastic_model.Q_bragg.size)
                                    if elastic_model is not None else 0),
                  "gain_side": gain_side, "gain_side_used": gain_side_used,
                  "engine_metadata": result.get("metadata", {})})


# -----------------------------------------------------------------------------
# 2-D S(Q,E) powder map + instrument kinematic envelope
# -----------------------------------------------------------------------------
def kinematic_envelope(geometry, e_fixed_meV, two_theta_min_deg, two_theta_max_deg, E):
    """Accessible-Q band (q_lo(E), q_hi(E)) for a geometry + detector angle range.

    The kinematically allowed |Q| at energy transfer E spans from the smallest to
    the largest detector angle: q_lo = Q(E, 2th_min), q_hi = Q(E, 2th_max), with
    the elastic wavevector from Ef (vision/indirect) or Ei (direct). This is the
    envelope drawn over the Euphonic-style direct-geometry maps.
    """
    from irma.spectra.sqe import Q_indirect, Q_direct
    E = np.asarray(E, float)
    Qof = Q_indirect if geometry in ("vision", "indirect") else Q_direct
    return (np.asarray(Qof(E, e_fixed_meV, two_theta_min_deg), float),
            np.asarray(Qof(E, e_fixed_meV, two_theta_max_deg), float))


def kinematic_mask(Q, E, env_E, q_lo, q_hi):
    """Accessibility mask ``(nQ, nE)`` for the instrument kinematic band.

    ``True`` where ``|Q|`` is reachable at energy transfer ``E`` -- i.e.
    ``q_lo(E) <= |Q| <= q_hi(E)`` with the :func:`kinematic_envelope` edges
    aligned to ``E``. Forbidden energies carry NaN edges, which compare False, so
    they map to inaccessible. Use it to BLANK ``S(Q,E)`` outside the accessible
    region (the Euphonic-style ``--angle-range`` mask -- the characteristic
    "arch") instead of only overlaying the envelope curves.
    """
    Q = np.asarray(Q, float)
    E = np.asarray(E, float)
    lo = np.asarray(q_lo, float)
    hi = np.asarray(q_hi, float)
    env_E = np.asarray(env_E, float)
    if lo.shape != E.shape or not np.array_equal(env_E, E):
        lo = np.interp(E, env_E, lo)
        hi = np.interp(E, env_E, hi)
    with np.errstate(invalid="ignore"):           # NaN edges -> False (inaccessible)
        return (Q[:, None] >= lo[None, :]) & (Q[:, None] <= hi[None, :])


def save_sqe_map(path, Q, E, S, envelope=None, masked=False):
    """Write a 2-D S(Q,E) map to ``.npz`` or long-form ``.csv`` (by extension).

    ``masked=True`` (with an ``envelope = (env_E, q_lo, q_hi)``) blanks S OUTSIDE
    the kinematically-accessible band first -- the Euphonic-style arch -- so the
    export matches the masked plot. The CSV is long-form ``Q_invA,E_meV,S`` rows
    (the masked variant omits the blanked cells); the npz stores Q, E, S (NaN
    where masked) plus the envelope. Returns the path written.
    """
    Q = np.asarray(Q, float)
    E = np.asarray(E, float)
    S = np.asarray(S, float)
    if masked and envelope is not None:
        env_E, q_lo, q_hi = envelope
        S = np.where(kinematic_mask(Q, E, env_E, q_lo, q_hi), S, np.nan)
    path = str(path)
    if path.endswith(".npz"):
        arrs = dict(Q=Q, E=E, S=S)
        if envelope is not None:
            env_E, q_lo, q_hi = envelope
            arrs.update(envelope_E=np.asarray(env_E, float),
                        envelope_q_lo=np.asarray(q_lo, float),
                        envelope_q_hi=np.asarray(q_hi, float))
        np.savez_compressed(path, **arrs)
    else:                                          # long-form CSV (.csv default)
        QQ, EE = np.meshgrid(Q, E, indexing="ij")
        flat = np.column_stack([QQ.ravel(), EE.ravel(), S.ravel()])
        if masked:                                 # drop the blanked (inaccessible) cells
            flat = flat[np.isfinite(flat[:, 2])]
        np.savetxt(path, flat, delimiter=",", header="Q_invA,E_meV,S", comments="")
    return path


@dataclasses.dataclass
class SQEMap:
    """Dense 2-D powder S(Q,E) map (heatmap-ready)."""
    Q: np.ndarray              # (nQ,) momentum transfer [1/A]
    E: np.ndarray              # (nE,) energy transfer [meV] (signed if e_min<0)
    S: np.ndarray              # (nQ, nE) intensity [barn/sr/meV]
    geometry: str
    envelope: tuple            # (E, q_lo, q_hi) accessible band, or None
    metadata: dict


def compute_sqe_map(*, geometry, phonopy_yaml, temperature_k, mesh, sab_mass_ratio,
                    sab_sigma_barn, e_fixed_meV=None, angle_range_deg=None,
                    q_min=0.0, q_max=12.0, dQ_map=0.05, e_min=0.0, e_max=250.0, dE=0.5,
                    inelastic_mode=2, dos_species=None, num_directions=10000,
                    multiphonon_num_directions=1000, multiphonon_max_order=100,
                    auto_multiphonon_order=True, min_phonon_energy_mev=0.0, jobs=None,
                    force_constants=None, force_sets=None, born_path=None,
                    sigma_coeffs=None, resolution_shape="gaussian",
                    resolution_model="poly", chopper_spec=None,
                    broaden=True, include_gain=True, gain_side="direct",
                    elastic_model=None, elastic=False, elastic_kind="both",
                    elastic_scatterers=None, incoherent_elastic_mode="isotropic",
                    dos_crystal=None,
                    scattering_lengths_json=None, incoherent_cross_sections_json=None,
                    site_scattering_lengths_angstrom=None,
                    site_incoherent_cross_sections_barn=None,
                    workdir=None, label=None, progress=print):
    """Compute a dense 2-D S(Q,E) powder map over a uniform Q x E grid.

    Unlike ``compute_spectrum`` (which samples S(Q,E) along instrument loci on
    the cheap locus grid), this runs the engine on a DENSE uniform Q grid so the
    whole S(Q,E) surface can be shown as a heatmap -- more cost, the dense grid
    P2 avoids. When the axis extends below 0 (``e_min < 0``) the energy-gain
    side is computed directly by default (``gain_side="direct"``, all modes:
    mode 0 evaluates the gain ladder, modes 1/2 read the engine's direct gain
    arrays); ``"detailed_balance"`` selects the mirror explicitly and is the
    defensive fallback when an engine result surfaces no gain sibling. The
    columns are optionally resolution-broadened, and the instrument kinematic
    envelope is returned when an ``angle_range_deg=(2th_min, 2th_max)`` +
    ``e_fixed_meV`` are given (for the Euphonic-style direct-geometry overlay).

    Elastic line: same options as ``compute_spectrum`` -- an explicit
    ``elastic_model`` (e.g. from an ENDF tape), or ``elastic=True`` to build it
    tape-free (engine ``elastic_state`` + ``elastic_scatterers`` for modes 1/2;
    the DOS-derived f0 + optional ``dos_crystal`` for mode 0). The per-Q elastic
    area ``elastic_dsigma_dOmega(Q, q_res=dQ_map)`` [barn/sr] is deposited as an
    E=0 line on the map's energy axis (split across the two bins bracketing 0 so
    its energy integral is exact) BEFORE the resolution pass, so ``broaden=True``
    gives the line the instrument's sigma(E=0) width like every other feature.
    When E=0 is an ENDPOINT of the axis (e.g. the default ``e_min=0``) only the
    on-axis half of the line is representable -- the map then carries half the
    elastic area (with a NOTE); extend the axis past 0 for the full line. The
    metadata reports ``elastic`` (a model was active) and ``elastic_deposited``
    (the line actually landed on this energy axis).
    """
    from irma.spectra import sqe as _sqe

    # Fail fast (before the engine) on the chopper+Lorentzian contradiction: a
    # chopper model yields a Gaussian sigma. config.validate() rejects it, but a
    # direct caller of compute_sqe_map bypasses that.
    if broaden and resolution_model == "chopper" and resolution_shape == "lorentzian":
        raise ValueError(
            "resolution_model='chopper' computes a Gaussian sigma and is "
            "incompatible with resolution_shape='lorentzian'; use "
            "resolution_shape='gaussian' (or resolution_model='poly').")

    Q_grid = np.arange(float(q_min), float(q_max) + 0.5 * dQ_map, dQ_map)
    # Loss grid reaches |e_min| when a deeper gain side is requested, so the gain
    # wing in [e_min, -e_max] has loss data to mirror (else it is zeroed).
    e_loss_max = float(e_max)
    if include_gain and float(e_min) < 0.0:
        e_loss_max = max(e_loss_max, -float(e_min))
    E_support = np.arange(0.0, e_loss_max + 0.5 * dE, dE)         # loss side
    # The energy-gain side is only ever sampled when the output grid extends
    # below 0; computing it for a loss-only grid (the default e_min=0) is pure
    # waste, so gate the direct evaluation on e_min < 0.
    gain_side_used = _resolve_gain_side(
        gain_side=gain_side, include_gain=(include_gain and float(e_min) < 0.0),
        dos_species=dos_species, progress=progress)
    if dos_species is not None:
        # mode-0: dense map straight from the DOS expansion (no engine).
        from irma.spectra.dos_mode0 import compute_mode0_sqe
        progress(f"compute_sqe_map[{geometry}]: mode-0 DOS "
                 f"({len(dos_species)} species) nQ={Q_grid.size} "
                 f"nE={E_support.size} T={temperature_k} K ...")
        m0 = compute_mode0_sqe(
            species=dos_species, temperature_k=float(temperature_k),
            q_ang_inv=Q_grid, e_mev=E_support,
            nphon=("auto" if auto_multiphonon_order
                   else int(multiphonon_max_order)))
        skey = "mode0_dos"
        eff_order = int(m0["nphon_effective"])
        _report_mode0_order(m0, auto_multiphonon_order, Q_grid, progress)
        res = {"metadata": {"mode": 0, "n_species": len(dos_species),
                            "per_species": m0["per_species"],
                            "sigma_b_total": m0["sigma_b_total"]}}
        gain_kw = {}
        if gain_side_used == "direct":
            gain_kw, gain_side_used = _mode0_direct_gain(
                dos_species=dos_species, temperature_k=temperature_k,
                q=np.asarray(m0["q_ang_inv"], float),
                E_loss=np.asarray(m0["e_mev"], float),
                nphon=("auto" if auto_multiphonon_order
                       else int(multiphonon_max_order)),
                progress=progress)
        powder = _sqe.from_noncubic_arrays(
            np.asarray(m0["q_ang_inv"], float), np.asarray(m0["e_mev"], float),
            np.asarray(m0["sqe_barn_per_meV"], float), T_K=float(temperature_k),
            sigma_b=float(m0["sigma_b_total"]), label=label or f"IRMA {geometry} map",
            **gain_kw)
    else:
        from irma.core.noncubic_inelastic import run_noncubic_sab_inprocess
        n_jobs = _resolve_jobs(jobs)
        # Auto-created workdirs are removed after the engine call (see the
        # compute_spectrum sibling; review S9).
        wd_auto = workdir is None
        wd = Path(workdir) if workdir is not None else Path(
            tempfile.mkdtemp(prefix="irma_map_"))
        progress(f"compute_sqe_map[{geometry}]: mode={inelastic_mode} "
                 f"mesh={tuple(int(m) for m in mesh)} nQ={Q_grid.size} "
                 f"nE={E_support.size} T={temperature_k} K jobs={n_jobs} ...")
        try:
            context = _get_engine_context(
                phonopy_yaml=phonopy_yaml, force_constants=force_constants,
                force_sets=force_sets, born_path=born_path, mesh=mesh,
                Q_support=Q_grid, E_support=E_support,
                num_directions=num_directions,
                multiphonon_num_directions=multiphonon_num_directions,
                n_jobs=n_jobs,
                site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
                site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
                scattering_lengths_json=scattering_lengths_json,
                incoherent_cross_sections_json=incoherent_cross_sections_json)
            res = run_noncubic_sab_inprocess(
                inelastic_mode=int(inelastic_mode), phonopy_yaml=str(phonopy_yaml),
                force_constants=force_constants, force_sets=force_sets, born_path=born_path,
                temperature_k=float(temperature_k), mesh=tuple(int(m) for m in mesh),
                q_grid_ang_inv=Q_grid, e_grid_mev=E_support,
                output_prefix=wd / "map_unused", sab_mass_ratio=float(sab_mass_ratio),
                num_directions=int(num_directions),
                multiphonon_num_directions=int(multiphonon_num_directions), jobs=n_jobs,
                multiphonon_max_order=int(multiphonon_max_order),
                auto_multiphonon_order=bool(auto_multiphonon_order),
                min_phonon_energy_mev=float(min_phonon_energy_mev),
                sab_sigma_barn=float(sab_sigma_barn),
                site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
                site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
                scattering_lengths_json=scattering_lengths_json,
                incoherent_cross_sections_json=incoherent_cross_sections_json,
                context=context, write_output_files=False,
                emit_gain_side=(gain_side_used == "direct"))
        finally:
            if wd_auto:
                shutil.rmtree(wd, ignore_errors=True)
        out = res["output_arrays"]
        eff_order = int(res.get("metadata", {}).get(
            "multiphonon_max_order", multiphonon_max_order))
        skey = _pick_sqe_key(out, eff_order, int(inelastic_mode))
        gain_kw, gain_side_used = _engine_direct_gain(
            out=out, loss_key=skey, gain_side_used=gain_side_used, progress=progress)
        powder = _sqe.from_noncubic_arrays(
            np.asarray(out["q_ang_inv"], float), np.asarray(out["e_mev"], float),
            np.asarray(out[skey], float), T_K=float(temperature_k),
            sigma_b=float(sab_sigma_barn), label=label or f"IRMA {geometry} map",
            **gain_kw)

    # tape-free elastic line: the same construction as compute_spectrum, with
    # the Bragg enumeration stopped at the dense map's own Q reach. Routing the
    # map's Q_max through the q_cuts tail gives the Bragg peaks the full 40-sigma
    # Gaussian underflow headroom at the grid edge, so the q_res-broadened peaks
    # on Q_grid are bit-identical to a full enumeration.
    if elastic_model is None and elastic:
        from irma.spectra.elastic import instrument_reach_emax_eV
        elastic_emax_eV = instrument_reach_emax_eV(
            q_max_invA=float(Q_grid.max()),
            q_cuts=[float(Q_grid.max())], q_res_invA=dQ_map)
        if dos_species is not None:
            elastic_model = _build_mode0_elastic_model(
                dos_species=dos_species, m0=m0, dos_crystal=dos_crystal,
                elastic_kind=elastic_kind, temperature_k=temperature_k,
                emax_eV=elastic_emax_eV, geometry=geometry, progress=progress)
        else:
            elastic_model = _build_engine_elastic_model(
                elastic_state=res.get("elastic_state"),
                elastic_scatterers=elastic_scatterers, elastic_kind=elastic_kind,
                temperature_k=temperature_k, emax_eV=elastic_emax_eV,
                geometry=geometry, progress=progress,
                incoherent_elastic_mode=incoherent_elastic_mode)

    qg, Es, Ss = _sqe.signed_sqe(powder, include_gain=include_gain)
    interp = _sqe.sqe_interpolator(qg, Es, Ss)
    E_out = np.arange(float(e_min), float(e_max) + 0.5 * dE, dE)
    QQ, EE = np.meshgrid(Q_grid, E_out, indexing="ij")
    S_map = interp(np.column_stack([QQ.ravel(), EE.ravel()])).reshape(QQ.shape)
    elastic_deposited = False
    if elastic_model is not None:
        # Deposit the elastic line BEFORE the resolution pass so sigma(E=0)
        # shapes it like every other feature (broaden=False keeps the spike).
        # The per-Q area [barn/sr] is split across the two E bins bracketing 0
        # in linear proportion -- uniform grids like arange(-100, 290, 1.5)
        # have no exact-zero bin -- which keeps the energy integral exactly
        # area(Q) and the line centroid at E=0.
        if E_out[0] <= 0.0 <= E_out[-1]:
            area = elastic_model.elastic_dsigma_dOmega(Q_grid, q_res=dQ_map)
            if E_out.size == 1:
                S_map[:, 0] += area / dE
            else:
                j = min(max(int(np.searchsorted(E_out, 0.0, side="right")) - 1, 0),
                        E_out.size - 2)
                w_hi = (0.0 - E_out[j]) / (E_out[j + 1] - E_out[j])
                S_map[:, j] += area * (1.0 - w_hi) / dE
                S_map[:, j + 1] += area * w_hi / dE
            elastic_deposited = True
            if E_out[0] == 0.0 or E_out[-1] == 0.0:
                # The line CENTER sits on the axis edge: once broadened, only the
                # on-axis half of the line is representable, so the map's elastic
                # area is ~half of what the 1-D spectrum reports (the 1-D path
                # renormalizes its truncated half-line back to the full area).
                # This is a deliberate convention, NOT silent corruption -- to
                # make the map and 1-D/cut elastic areas agree, give the axis room
                # on both sides of 0 (e_min<0) so the full line is resolved.
                progress("NOTE: E=0 is an endpoint of the energy axis -- after "
                         "broadening the map carries only the on-axis half of the "
                         "elastic line, so its area is ~half the 1-D/cut value "
                         "(which renormalizes the truncated half-line to full "
                         "area). Set e_min<0 for the map and 1-D elastic areas to "
                         "agree.")
        else:
            progress(f"NOTE: E=0 is outside the requested energy axis "
                     f"[{E_out[0]:g}, {E_out[-1]:g}] meV; the elastic line "
                     "does not appear on the map.")
    if broaden:
        if resolution_model == "chopper":
            # Mirror Instrument.width_source()'s guard: a chopper width needs both
            # a spec and a fixed incident energy. Without it, float(None) / **None
            # would raise an opaque TypeError deep in the broadening loop.
            if not chopper_spec or e_fixed_meV is None:
                raise ValueError(
                    "resolution_model='chopper' requires chopper_spec "
                    "(instrument, package, frequency) and e_fixed_meV")
            from irma.spectra.chopper_resolution import chopper_sigma_of_E
            width = lambda E: chopper_sigma_of_E(E, Ei=float(e_fixed_meV), **chopper_spec)
        else:
            if sigma_coeffs is not None:
                width = sigma_coeffs
            elif geometry == "direct" and e_fixed_meV is not None:
                # match instruments.direct(): a direct instrument's default poly
                # width is ~2% of Ei, NOT the VISION indirect-geometry polynomial
                width = (0.02 * float(e_fixed_meV), 0.0, 0.0)
            else:
                width = _sqe.VISION_SIGMA_COEFFS
        # one kernel for every Q row: the (nE, nE) build dominates the
        # per-row convolution cost
        R = _sqe.resolution_kernel(E_out, width, shape=resolution_shape)
        for iq in range(Q_grid.size):
            S_map[iq] = _sqe.apply_resolution_kernel(R, E_out, S_map[iq])

    envelope = None
    if angle_range_deg is not None and e_fixed_meV is not None:
        q_lo, q_hi = kinematic_envelope(
            geometry, e_fixed_meV, float(angle_range_deg[0]),
            float(angle_range_deg[1]), E_out)
        envelope = (E_out, q_lo, q_hi)
    return SQEMap(
        Q=Q_grid, E=E_out, S=S_map, geometry=geometry, envelope=envelope,
        metadata={"sqe_key": skey, "effective_multiphonon_order": eff_order,
                  "inelastic_mode": int(inelastic_mode), "broadened": bool(broaden),
                  "resolution_shape": resolution_shape,
                  "resolution_model": resolution_model,
                  "elastic": elastic_model is not None,
                  "elastic_deposited": elastic_deposited,
                  "elastic_kind": (elastic_kind if elastic_model is not None else None),
                  "n_bragg_edges": (int(elastic_model.Q_bragg.size)
                                    if elastic_model is not None else 0),
                  "gain_side": gain_side, "gain_side_used": gain_side_used,
                  "engine_metadata": res.get("metadata", {})})
