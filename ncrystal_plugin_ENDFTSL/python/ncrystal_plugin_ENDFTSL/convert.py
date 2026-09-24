"""Assemble an ENDFTSLPack from a parsed evaluation (ports irma.ncrystal.convert math)."""
from __future__ import annotations
import bisect
import math
from dataclasses import dataclass
from .reader import TSLEvaluation, read_tsl
from . import physics
from .pack import ENDFTSLPack


def _refine_log_linear_beta(beta, sab, beta_int, tol=1e-3):
    """Insert beta points in the intervals the tape interpolates log-linearly
    (ENDF INT=4), so NCrystal's linear-in-beta interpolation of the pack follows
    the tape's law to a relative error ``tol``.

    Linear interpolation of an exponential over a sub-interval with log-ratio x
    errs by about x^2/8, so n = ceil(max_alpha |ln(S_j+1 / S_j)| / sqrt(8 tol))
    sub-intervals meet ``tol``. Lin-lin intervals, and intervals with a zero
    cell (treated as linear), are left as they are. ``sab`` is [alpha][beta];
    returns the new beta grid and table.
    """
    step = math.sqrt(8.0 * tol)
    new_beta = [beta[0]]
    new_sab = [[row[0]] for row in sab]
    for j in range(len(beta) - 1):
        lo = [row[j] for row in sab]
        hi = [row[j + 1] for row in sab]
        n = 1
        if beta_int[j] == 4 and all(a > 0.0 and b > 0.0 for a, b in zip(lo, hi)):
            n = max(1, math.ceil(max(abs(math.log(b / a))
                                     for a, b in zip(lo, hi)) / step))
        for k in range(1, n):
            t = k / n
            new_beta.append(beta[j] + t * (beta[j + 1] - beta[j]))
            for row, a, b in zip(new_sab, lo, hi):
                row.append(a * (b / a) ** t)
        new_beta.append(beta[j + 1])
        for row, b in zip(new_sab, hi):
            row.append(b)
    return new_beta, new_sab


def build_pack(ev: TSLEvaluation, T: float, material_id: str,
               element_mass_amu: float, *, inelastic_scale: float = 1.0,
               coherent_scale: float = 1.0) -> ENDFTSLPack:
    law = physics.physical_inelastic(ev, T)
    awr = law.awr
    alpha_nc = [a * awr for a in law.alpha_phys]            # α_ncrystal = α_phys · AWR
    # NCrystal interpolates S linearly in beta; the tape's INT=4 intervals are
    # refined so that follows the tape's log-linear law
    beta, sab = _refine_log_linear_beta(law.beta_phys, law.sab_scaled_sym,
                                        ev.beta_int)
    # scaled-sym, beta-major (loop β outer, α inner) — matches NC SCALED_SYM_SAB / irma pack
    sab_values = [sab[ai][bi] for bi in range(len(beta)) for ai in range(len(alpha_nc))]
    # NCrystal cross sections are per atom, but the C++ plugin sums every pack's
    # channels at weight 1.0, so unscaled multi-species packs would give
    # cross sections per formula unit (about N_atoms times too high). The
    # converter restores the per-atom convention by scaling each pack:
    #   - inelastic + incoherent elastic: * inelastic_scale (= the atom fraction f),
    #     so sum_i f_i*sigma_i is the per-atom average; these laws are per species.
    #   - coherent Bragg edges: * coherent_scale, which build_packs sets to the
    #     atom fraction f for every coherent-bearing tape.
    # Monatomic (both scales = 1) is unchanged. The physical species bound is kept
    # in metadata.
    coh = physics.coherent_edges(ev, T)
    coh_cumS = [s * coherent_scale for s in coh[1]] if coh else []
    inc = physics.incoherent_msd(ev, T)
    inc_xs = (inc[1] * inelastic_scale) if inc else None
    meta = {"source_za": f"{ev.za:.0f}", "lthr": str(ev.lthr), "lat": str(ev.lat),
            "lasym": str(ev.lasym), "lln": str(ev.lln),
            "inelastic_scale": f"{inelastic_scale:.10g}",
            "coherent_scale": f"{coherent_scale:.10g}",
            "physical_bound_xs_barn": f"{law.bound_xs_barn:.10g}",
            "converter": "ncrystal_plugin_ENDFTSL/0.0.1"}
    return ENDFTSLPack(
        material_id=material_id, temperature_K=float(T),
        bound_xs_barn=law.bound_xs_barn * inelastic_scale,
        element_mass_amu=float(element_mass_amu),
        sab_representation="scaled_sym_sab",
        alpha_grid=alpha_nc, beta_grid=beta, sab_values=sab_values,
        coh_edges_ev=list(coh[0]) if coh else [], coh_cumS=coh_cumS,
        elastic_msd_a2=inc[0] if inc else None,
        elastic_incoherent_xs_barn=inc_xs,
        elastic_scale=1.0, metadata=meta)


@dataclass
class SpeciesSpec:
    """One principal scatterer of a (poly)atomic material: its own ENDF/TSL tape."""
    tape: str
    symbol: str
    mass: float
    fraction: float | None = None  # atom fraction; weights this tape's coherent Bragg edges


def build_packs(specs, T: float, material_id: str):
    """Build one ENDFTSLPack per principal scatterer (e.g. BeO -> Be + O packs).

    Every channel of a tape is scaled by its species' atom fraction f, as
    transport codes do: each tape's cross section is per atom of its species,
    so the material's per-atom cross section is sum_i f_i*sigma_i. That holds
    for the coherent Bragg edges too, whether they are replicated on several
    tapes (standard ENDF/B-VIII.1, IRMA MEF) or carried by one tape (a hydride
    whose metal alone scatters coherently, IRMA SEF with its 1/f_DC edges).
    ``specs`` is a list of SpeciesSpec, each needing its atom ``fraction``.
    """
    fracs = [sp.fraction for sp in specs]
    if any(f is None for f in fracs):
        missing = [sp.symbol for sp in specs if sp.fraction is None]
        raise ValueError(f"species {missing} need atom fractions")
    if any(not (0.0 < float(f) <= 1.0) for f in fracs):
        raise ValueError(f"atom fractions must each lie in (0, 1], got {fracs}")
    if abs(sum(float(f) for f in fracs) - 1.0) > 1e-6:
        raise ValueError(f"atom fractions must sum to 1, got {sum(float(f) for f in fracs)}")

    evs = [read_tsl(sp.tape) for sp in specs]
    coh_data = [physics.coherent_edges(ev, T) for ev in evs]
    has_coh = [c is not None for c in coh_data]
    n_coh = sum(has_coh)

    if n_coh >= 2:
        # Fraction-weighting replicated edges is correct only if the coherent
        # tapes carry the same per-atom whole-crystal edges and their atom
        # fractions total 1 (so sum_i f_i*sigma_coh == sigma_coh); otherwise the
        # weight-1.0 C++ sum gives a partial Bragg cross section. Check both.
        coh_frac_sum = sum(float(sp.fraction)
                           for sp, hc in zip(specs, has_coh) if hc)
        if abs(coh_frac_sum - 1.0) > 1e-6:
            raise ValueError(
                f"coherent elastic is carried by {n_coh} tapes whose atom fractions sum "
                f"to {coh_frac_sum:.6g} != 1; a per-atom whole-crystal Bragg-edge "
                "structure cannot be reconstructed by fraction-weighting a partial set "
                "(standard ENDF replicates the per-atom edges on every principal tape).")
        # Compare the coherent cross section sigma_coh(E) = cumS(<=E)/E, not the
        # raw edge arrays: independently written tapes can represent the same
        # whole-crystal edges at slightly different edge energies (e.g. SiO2-alpha
        # Si vs O differ by up to 0.7% in edge energy). sigma_coh is insensitive
        # to that, while a per-species partitioned edge set differs by O(1). The
        # median over the sampled energies ignores the few samples that straddle
        # a shifted edge.
        def _sig(edges, cumS, E):
            i = bisect.bisect_right(edges, E) - 1
            return cumS[i] / E if i >= 0 else 0.0
        ref_e, ref_s = next(c for c in coh_data if c is not None)
        lo = ref_e[1] if len(ref_e) > 1 else ref_e[0]
        hi = ref_e[-1]
        sampleE = ([lo * (hi / lo) ** (k / 15.0) for k in range(16)]
                   if hi > lo > 0 else [hi])
        for c in coh_data:
            if c is None:
                continue
            e, s = c
            devs = sorted(abs(_sig(e, s, E) - _sig(ref_e, ref_s, E))
                          / _sig(ref_e, ref_s, E)
                          for E in sampleE if _sig(ref_e, ref_s, E) > 0)
            median = devs[len(devs) // 2] if devs else 0.0
            if median > 0.10:
                raise ValueError(
                    f"coherent-bearing tapes carry different Bragg edges (median "
                    f"sigma_coh deviation {median:.2f}); fraction-weighting assumes the "
                    "per-atom whole-crystal edge structure is replicated across principal "
                    "tapes. A "
                    "per-species partitioned coherent layout is not supported.")

    packs = []
    for sp, ev, hc in zip(specs, evs, has_coh):
        f = float(sp.fraction)
        packs.append(build_pack(
            ev, T, f"{material_id}__{sp.symbol}", sp.mass,
            inelastic_scale=f, coherent_scale=f if hc else 1.0))
    return packs
