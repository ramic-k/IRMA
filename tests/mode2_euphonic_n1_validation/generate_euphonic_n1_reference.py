#!/usr/bin/env python3
"""Generate the frozen Euphonic one-phonon reference for the mode-2 validation.

This is the documented, reproducible origin of ``<material>/euphonic_n1_reference.npz``:
an INDEPENDENT calculation of the powder-averaged coherent one-phonon
S(alpha,beta) from the SAME vendored phonon model (phonopy.yaml +
FORCE_CONSTANTS) that the IRMA mode-2 deck uses.

Method
------
* Debye-Waller factor from a Monkhorst-Pack grid (default 40x40x40 — matched
  to the IRMA deck's mesh) at the deck temperature.
* For every alpha point of the IRMA deck grid, the corresponding
  |Q| = sqrt(alpha * awr * kT / (hbar^2/2m_n)) sphere is powder-averaged with
  ``euphonic.powder.sample_sphere_structure_factor`` using golden-spiral
  sampling (default 10000 directions — matched to the IRMA deck's ndir).
* Euphonic returns the one-phonon COHERENT dynamic structure factor with
  Bose occupation for phonon creation (= neutron downscatter, beta>0) and the
  Debye-Waller factor applied — the same physical object as IRMA's mode-2
  n=1 coherent term (IRMA adds the tiny incoherent n=1; for graphite/Be
  sigma_inc/sigma_coh < 4e-4, negligible).
* Conversion to the ENDF/IRMA asymmetric downscatter law on the deck's
  (alpha,beta) grid:
      S_unitless(Q,E)[1/meV] = y_euphonic[mb/sr/meV] * 4*pi / sigma_coh[mb]
      SS(alpha,|beta|)       = kT[meV] * S_unitless(Q,E)[1/meV]
  with sigma_coh = 4*pi*b_coh^2 per atom (euphonic structure factors are
  normalized per average atom of the cell; the absolute scale is verified
  against IRMA's integrated spectra in the validation report).
* The spectrum is binned into Voronoi cells around the deck's beta grid
  points, mirroring IRMA's in-process binning of one-phonon delta
  contributions.

Everything needed to regenerate the reference (grid, awr, b_coh, T) is parsed
from the committed IRMA deck template, so the two calculations cannot drift
apart silently. All settings are stored as JSON metadata inside the .npz.

Usage:
    python generate_euphonic_n1_reference.py graphite [--beta-max 16.0]

The Debye-Waller mesh and powder-direction count default to the deck
template's Card 6f mesh and Card 6g ndir; --mesh/--npts exist only as
explicit overrides and print a notice when they diverge from the deck.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the enclosing checkout's irma over any ambient/editable install, so
# the frozen reference is produced by the tree being validated, not by
# whatever irma the environment carries. Same pattern as the validate_*.py
# comparators.
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import irma.core.constants as _irma_constants
from irma.core.constants import BK, HBAR2_OVER_2MN_MEV_A2

KB_MEV_PER_K = BK * 1.0e3


def parse_deck_template(material_dir):
    """Pull grid + atom data out of the committed IRMA deck template."""
    path = os.path.join(material_dir, "irma_mode2_n1.input.template")
    text = open(path).read()
    lines = [ln.split("/")[0] for ln in text.splitlines()]

    # Card 6d: Z A awr b_coh sigma_inc npos  (first line with 6 numbers where
    # the first two are small integers) — search after the lattice card.
    awr = b_coh = sigma_inc = None
    for ln in lines:
        toks = ln.split()
        if len(toks) == 6:
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                continue
            if vals[0] == int(vals[0]) and vals[1] == int(vals[1]) \
                    and 0 < vals[0] < 100 and vals[1] >= vals[0]:
                _, _, awr, b_coh, sigma_inc, _ = vals
                break
    if awr is None:
        raise ValueError("could not locate Card 6d in deck template")

    # Card 7 "nalpha nbeta lat" then the alpha block, beta block, temperature.
    # (Card 3 "1 1 1 /" also matches the pattern; Card 7 is the LAST match.)
    m = list(re.finditer(r"^\s*(\d+)\s+(\d+)\s+1\s*/", text, re.M))[-1]
    nalpha, nbeta = int(m.group(1)), int(m.group(2))
    if nalpha < 10 or nbeta < 10:
        raise ValueError(f"implausible Card 7 grid {nalpha}x{nbeta}")
    tail = text[m.end():]
    nums = []
    for ln in tail.splitlines():
        nums += ln.split("/")[0].split()
        if len(nums) >= nalpha + nbeta + 1:
            break
    nums = [float(t) for t in nums[: nalpha + nbeta + 1]]
    alpha = np.array(nums[:nalpha])
    beta = np.array(nums[nalpha: nalpha + nbeta])
    temperature_k = nums[nalpha + nbeta]

    # Card 6f-2 (mesh_nx ny nz ncpu use_born) follows the quoted
    # phonopy.yaml path card; Card 6g (ndir mpdir [auto]) follows
    # the 6f block. Parsing them here keeps the Euphonic reference's
    # Debye-Waller mesh and powder-direction count tied to the deck, so
    # the two calculations cannot drift apart silently.
    m6f = re.search(
        r"^\s*'[^']*'\s*/\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+(\d+)\s*/",
        text, re.M)
    if m6f is None:
        raise ValueError("could not locate Card 6f-2 (mesh) in deck template")
    mesh = (int(m6f.group(1)), int(m6f.group(2)), int(m6f.group(3)))
    if len(set(mesh)) != 1:
        raise ValueError(f"anisotropic mesh {mesh} not supported by --mesh")
    tail6 = text[m6f.end():]
    if m6f.group(4) == "1":            # use_born=1: skip the BORN path card
        tail6 = tail6.split("/", 1)[1]
    m6g = re.search(r"^\s*(\d+)\s+(\d+)", tail6, re.M)
    if m6g is None:
        raise ValueError("could not locate Card 6g (ndir mpdir) in template")
    ndir = int(m6g.group(1))
    return alpha, beta, temperature_k, awr, b_coh, sigma_inc, mesh[0], ndir


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("material", help="material subdirectory, e.g. graphite")
    ap.add_argument("--npts", type=int, default=None,
                    help="powder-sampling directions per |Q| (default: the "
                         "deck template's Card 6g ndir)")
    ap.add_argument("--mesh", type=int, default=None,
                    help="MP grid per axis for the Debye-Waller factor "
                         "(default: the deck template's Card 6f mesh)")
    ap.add_argument("--beta-max", type=float, default=16.0,
                    help="beta range to tabulate (one-phonon support ends "
                         "at omega_max/kT; default 16 covers the cuts)")
    args = ap.parse_args(argv[1:])

    mat_dir = os.path.join(HERE, args.material)
    (alpha, beta, T, awr, b_coh, sigma_inc,
     deck_mesh, deck_ndir) = parse_deck_template(mat_dir)
    if args.mesh is None:
        args.mesh = deck_mesh
    elif args.mesh != deck_mesh:
        print(f"NOTE: --mesh {args.mesh} OVERRIDES the deck's Card 6f mesh "
              f"({deck_mesh}^3); the reference will not match a tape made "
              f"from the unmodified template.")
    if args.npts is None:
        args.npts = deck_ndir
    elif args.npts != deck_ndir:
        print(f"NOTE: --npts {args.npts} OVERRIDES the deck's Card 6g ndir "
              f"({deck_ndir}); the reference will not match a tape made "
              f"from the unmodified template.")
    kT_mev = KB_MEV_PER_K * T
    sigma_coh_mb = 4.0 * np.pi * b_coh**2 * 10.0   # fm^2 -> mb (1 mb = 0.1 fm^2)
    q_of_alpha = np.sqrt(alpha * awr * kT_mev / HBAR2_OVER_2MN_MEV_A2)

    print(f"material={args.material}  T={T} K  awr={awr}  b_coh={b_coh} fm "
          f"(sigma_coh={sigma_coh_mb/1000:.4f} b)  "
          f"grid {len(alpha)}x{len(beta)}  Q range "
          f"[{q_of_alpha[0]:.3f}, {q_of_alpha[-1]:.2f}] 1/A")

    from euphonic import ForceConstants, ureg
    from euphonic.util import mp_grid
    from euphonic.powder import sample_sphere_structure_factor

    fc = ForceConstants.from_phonopy(path=mat_dir, summary_name="phonopy.yaml")

    print(f"computing Debye-Waller on {args.mesh}^3 MP grid ...")
    t0 = time.time()
    modes = fc.calculate_qpoint_phonon_modes(
        mp_grid([args.mesh] * 3), reduce_qpts=True)
    dw = modes.calculate_debye_waller(T * ureg("K"))
    print(f"  done in {time.time()-t0:.0f} s")

    # beta-grid Voronoi bins (mirrors IRMA's binning of 1-phonon deltas).
    bsel = beta <= args.beta_max
    bgrid = beta[bsel]
    edges = np.concatenate((
        [max(bgrid[0] - 0.5 * (bgrid[1] - bgrid[0]), 0.0)],
        0.5 * (bgrid[1:] + bgrid[:-1]),
        [bgrid[-1] + 0.5 * (bgrid[-1] - bgrid[-2])],
    ))
    energy_bins = (edges * kT_mev) * ureg("meV")
    widths_mev = np.diff(edges) * kT_mev

    ss = np.zeros((len(beta), len(alpha)))
    t0 = time.time()
    for j, q in enumerate(q_of_alpha):
        spec = sample_sphere_structure_factor(
            fc, q * ureg("1/angstrom"), dw=dw,
            temperature=T * ureg("K"), sampling="golden",
            npts=args.npts, energy_bins=energy_bins)
        y = spec.y_data
        # Normalize to a spectral density in mb/sr/meV regardless of whether
        # euphonic returned a per-bin or per-meV quantity.
        try:
            y_mev = y.to("millibarn / steradian / millielectron_volt").magnitude
        except Exception:
            y_mev = y.to("millibarn / steradian").magnitude / widths_mev
        s_unitless_per_mev = y_mev * 4.0 * np.pi / sigma_coh_mb
        ss[bsel, j] = kT_mev * s_unitless_per_mev
        if j % 20 == 0:
            el = time.time() - t0
            print(f"  alpha {j+1}/{len(alpha)} (Q={q:.3f} 1/A)  "
                  f"[{el:.0f} s elapsed]")

    meta = dict(
        material=args.material, temperature_K=T, awr=awr, b_coh_fm=b_coh,
        sigma_inc_barn=sigma_inc, sigma_coh_mb=sigma_coh_mb,
        npts=args.npts, sampling="golden", dw_mesh=[args.mesh] * 3,
        beta_max_tabulated=args.beta_max,
        convention=("asymmetric downscatter SS(alpha,|beta|) = kT[meV] * "
                    "S(Q,E)[1/meV]; S = y_euphonic*4pi/sigma_coh; "
                    "beta-grid Voronoi binning; coherent one-phonon only"),
        generator="generate_euphonic_n1_reference.py",
    )
    try:
        import euphonic
        meta["euphonic_version"] = euphonic.__version__
    except Exception:
        pass
    # Producer provenance: the exact code and inputs behind
    # the frozen reference must be reconstructible from its metadata.
    meta["irma_import_path"] = os.path.abspath(_irma_constants.__file__)
    try:
        import subprocess
        repo = os.path.abspath(os.path.join(HERE, "..", ".."))
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        meta["irma_git_sha"] = sha + ("-dirty" if dirty else "")
    except Exception:
        meta["irma_git_sha"] = "unknown"
    import hashlib
    for tag, rel in (("phonopy_yaml", "phonopy.yaml"),):
        fp = os.path.join(mat_dir, rel)
        if os.path.exists(fp):
            with open(fp, "rb") as fh:
                meta[f"{tag}_sha256"] = hashlib.sha256(fh.read()).hexdigest()

    out = os.path.join(mat_dir, "euphonic_n1_reference.npz")
    np.savez_compressed(out, alpha=alpha, beta=beta, q_of_alpha=q_of_alpha,
                        ss=ss, metadata=json.dumps(meta))
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
