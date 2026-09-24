#!/usr/bin/env python3
"""Why low-temperature TSL tapes need ilog=1 (ENDF LLN=1, log storage).

This is a self-contained numerical demonstration of the storage cutoff that
silently drops high-energy (e.g. optic) phonon structure from cryogenic
S(alpha,beta) tapes -- and how log storage fixes it. It runs in a fraction of a
second and needs only numpy (no engine, no phonopy).

Background
----------
ENDF MF7/MT4 with LASYM=0 stores the *symmetric* law, related to the physical
downscatter (phonon-creation) law by the detailed-balance factor

    S_sym(alpha, beta) = S_phys(alpha, beta) * exp(-beta/2),   beta = E / kT.

On read-back you recover S_phys = S_sym * exp(+beta/2).

With linear storage (ilog=0) every S_sym value below the Card 4 smin cutoff
(default 1e-75, as in NJOY) is written as 0. Because beta scales as 1/T, at low
temperature S_sym falls below smin -- so S_phys = 0 * exp(+beta/2) = 0 on
read-back, and the high-energy structure is gone. Log storage (ilog=1 -> LLN=1)
writes ln(S_sym) instead, so the value survives the round trip. Lowering smin
also keeps it with ilog=0, down to THERMR's own floor of about 1e-98.

Run:  python lln_low_temperature_demo.py
"""
import numpy as np

KB_meV_per_K = 8.617333e-2          # Boltzmann constant
SMIN = 1.0e-75                      # Card 4 smin default


def endf_linear_store(s_sym):
    """Linear ENDF storage (ilog=0): values below smin are written as 0."""
    return np.where(s_sym < SMIN, 0.0, s_sym)


def endf_log_store(s_sym):
    """Log ENDF storage (ilog=1 / LLN=1): store ln(S); 0 -> -999 sentinel."""
    return np.where(s_sym > 0.0, np.log(np.maximum(s_sym, 1e-300)), -999.0)


def endf_log_read(ln_s):
    """Read an LLN=1 tape: exp(ln S); the -999 sentinel -> 0."""
    with np.errstate(under="ignore"):
        return np.where(ln_s <= -900.0, 0.0, np.exp(ln_s))


def roundtrip(s_phys, beta, ilog):
    """Write S_phys as symmetric, store per ilog, read back the physical law."""
    s_sym = s_phys * np.exp(-beta / 2.0)            # symmetric law (what's stored)
    if ilog == 0:
        stored = endf_linear_store(s_sym)
        s_sym_read = stored
    else:
        stored = endf_log_store(s_sym)
        s_sym_read = endf_log_read(stored)
    return s_sym_read * np.exp(+beta / 2.0)          # recovered physical law


def main():
    # A representative graphite optic phonon: physical S ~ 0.1 at 185 meV.
    E_optic = 185.0     # meV (the C-C stretch region)
    S_optic = 0.10      # physical scattering there (order-of-magnitude)

    print(f"Representative optic feature: physical S = {S_optic} at "
          f"E = {E_optic} meV\n")
    print(f"{'T (K)':>7} | {'beta':>8} | {'S_sym stored':>13} | "
          f"{'ilog=0 read':>12} | {'ilog=1 read':>12}")
    print("-" * 64)
    for T in (296.0, 100.0, 30.0, 5.0):
        beta = E_optic / (KB_meV_per_K * T)
        s_sym = S_optic * np.exp(-beta / 2.0)
        r0 = float(roundtrip(np.array([S_optic]), np.array([beta]), ilog=0)[0])
        r1 = float(roundtrip(np.array([S_optic]), np.array([beta]), ilog=1)[0])
        tag0 = "LOST" if r0 == 0.0 else "ok"
        print(f"{T:7.0f} | {beta:8.1f} | {s_sym:13.2e} | "
              f"{r0:9.4f} {tag0:>3} | {r1:9.4f} ok")

    print("\nilog=0 (linear) silently zeroes the optic feature once beta is")
    print("large enough that S_sym falls below the Card 4 smin cutoff (1e-75;")
    print("below about 6 K for this feature). ilog=1 (LLN log storage) preserves")
    print("it at every temperature; lowering smin also keeps it with ilog=0.")
    print("\n=> For cryogenic TSL tapes, set ilog=1 on Card 4:  mat za isabt 1 smin")


if __name__ == "__main__":
    main()
