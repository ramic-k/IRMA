"""``irma spectra`` command-line interface.

Three preset/flag subcommands (``vision`` / ``indirect`` / ``direct``) plus a
config runner (``run <cfg> [--set k=v]``). The flag form is **pure sugar**: it
builds the identical :class:`~irma.spectra.config.SpectraConfig` the config form
loads and calls the identical ``run_spectra`` -- exactly one compute path. The
geometry guard is structural: ``--ef`` exists only on vision/indirect, ``--ei``
only on direct, so argparse rejects the wrong one automatically.

``main(argv)`` returns a process exit code (it never calls ``sys.exit``), so it
is unit-testable; ``irma.cli`` and ``python -m irma.spectra`` wrap it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from irma.spectra.config import (
    SpectraConfig, SpectraConfigError, check_input_files, run_spectra,
)


# -----------------------------------------------------------------------------
# value parsers
# -----------------------------------------------------------------------------
def _max_order(s):
    """``--max-phonon-order`` value: an integer or ``'auto'``.

    Raise a self-describing ArgumentTypeError -- argparse would otherwise
    build the message from this function's __name__ and leak the internal
    name ("invalid _max_order value: ...") without saying 'auto' is accepted."""
    if s == "auto":
        return s
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be an integer or 'auto' (got {s!r})") from None


def parse_angles(s):
    """``"30,60,90"`` -> [30,60,90]; ``"start:stop:step"`` -> inclusive range."""
    s = s.strip()
    if ":" in s:
        try:
            parts = [float(x) for x in s.split(":")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--angles range must be numeric start:stop:step (got {s!r})")
        if len(parts) != 3 or parts[2] == 0:
            raise argparse.ArgumentTypeError(
                f"--angles range must be start:stop:step (got {s!r})")
        start, stop, step = parts
        return [float(a) for a in np.arange(start, stop + 0.5 * step, step)]
    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--angles must be a comma list of numbers (got {s!r})")


def parse_coeffs(s):
    """Parse a comma-separated list of polynomial coefficients."""
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a comma-separated list of numbers (got {s!r})")


_SCATTERER_KEYS = ("dos", "unit", "mult", "pos")


def parse_scatterer(s):
    """``"SYM,sigma_bound_b,awr[,b_coh_fm[,sigma_inc_b]]"`` -> dict.

    Trailing ``key=value`` tokens add the mode-0 (DOS) fields (order-free,
    backward compatible -- existing comma lines have no ``=``):
    ``dos=<file>``, ``unit=<meV|eV|cm-1|THz>``, ``mult=<int>``, and
    ``pos=x1:y1:z1;x2:y2:z2`` (fractional sites for the coherent-elastic peaks).
    Example: ``"H,80.27,0.999,-3.74,80.26,dos=h_dos.txt,mult=2,pos=0:0:0;0.5:0.5:0.5"``.
    """
    pos_parts, kv = [], {}
    for tok in (p.strip() for p in s.split(",")):
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip().lower()
            if k not in _SCATTERER_KEYS:        # a typo'd key must fail loudly,
                raise argparse.ArgumentTypeError(  # not silently drop the field
                    f"--scatterer: unknown key {k!r} (accepted: "
                    f"{', '.join(_SCATTERER_KEYS)})")
            kv[k] = v.strip()
        else:
            pos_parts.append(tok)        # KEEP empty placeholders -- an absent
            #                              b_coh_fm must not shift sigma_inc_b up
    while pos_parts and pos_parts[-1] == "":    # but drop trailing-comma empties
        pos_parts.pop()
    if len(pos_parts) < 3:
        raise argparse.ArgumentTypeError(
            f"--scatterer needs at least SYMBOL,sigma_bound_b,awr (got {s!r})")

    def _num(val, field, cast=float):
        """Cast one scatterer field, naming it in the error on failure."""
        # name the offending FIELD instead of letting argparse leak the generic
        # "invalid parse_scatterer value: ..." for an unparseable number.
        try:
            return cast(val)
        except (TypeError, ValueError):
            kind = "an integer" if cast is int else "a number"
            raise argparse.ArgumentTypeError(
                f"--scatterer {field} must be {kind} (got {val!r}) in {s!r}")

    d = {"symbol": pos_parts[0],
         "sigma_bound_b": _num(pos_parts[1], "sigma_bound_b"),
         "awr": _num(pos_parts[2], "awr")}
    if len(pos_parts) >= 4 and pos_parts[3]:
        d["b_coh_fm"] = _num(pos_parts[3], "b_coh_fm")
    if len(pos_parts) >= 5 and pos_parts[4]:
        d["sigma_inc_b"] = _num(pos_parts[4], "sigma_inc_b")
    if kv.get("dos"):
        d["dos_file"] = kv["dos"]
    if kv.get("unit"):
        d["dos_unit"] = kv["unit"]
    if kv.get("mult"):
        d["multiplicity"] = _num(kv["mult"], "mult", int)
    if kv.get("pos"):
        try:
            d["positions"] = [[float(c) for c in site.split(":")]
                              for site in kv["pos"].split(";") if site.strip()]
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(
                f"--scatterer pos must be x:y:z numeric sites (got {kv['pos']!r})")
    return d


def format_scatterer(s):
    """Inverse of :func:`parse_scatterer`: a ``Scatterer`` -> the one-line form
    (positional scattering data + ``key=value`` mode-0 fields when set)."""
    pos = [s.symbol, s.sigma_bound_b, s.awr]
    if s.b_coh_fm is not None or s.sigma_inc_b is not None:
        pos.append(s.b_coh_fm if s.b_coh_fm is not None else "")
    if s.sigma_inc_b is not None:
        pos.append(s.sigma_inc_b)
    out = ",".join("" if x == "" else str(x) for x in pos)
    if getattr(s, "dos_file", None):
        out += f",dos={s.dos_file}"
        if getattr(s, "dos_unit", "meV") and s.dos_unit != "meV":
            out += f",unit={s.dos_unit}"
    if getattr(s, "multiplicity", 1) and s.multiplicity != 1:
        out += f",mult={s.multiplicity}"
    if getattr(s, "positions", None):
        sites = ";".join(":".join(str(c) for c in p) for p in s.positions)
        out += f",pos={sites}"
    return out


# -----------------------------------------------------------------------------
# argument parser
# -----------------------------------------------------------------------------
def _add_common(p):
    """Attach the options shared by every spectra subcommand to ``p``."""
    p.add_argument("--phonopy-yaml",
                   help="phonopy model file; required for inelastic modes 1/2 "
                        "and for mode 0 with --dos-source phonopy")
    p.add_argument("--force-constants",
                   help="explicit force-constants file (default: discovered "
                        "next to the phonopy.yaml)")
    p.add_argument("--force-sets", help="explicit FORCE_SETS file")
    p.add_argument("--born",
                   help="BORN file (non-analytical-term correction / LO-TO "
                        "splitting)")
    p.add_argument("--mesh", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   default=[40, 40, 40],
                   help="phonon q-mesh for the eigenvector/DOS sampling "
                        "(default: 40 40 40)")
    p.add_argument("--temperature", type=float, default=296.0,
                   help="sample temperature in K (default: 296)")
    p.add_argument("--inelastic-mode", type=int, choices=[0, 1, 2], default=2,
                   help="0 = DOS + isotropic Debye-Waller (no eigenvectors), "
                        "1 = incoherent approximation, 2 = exact coherent "
                        "one-phonon + incoherent multiphonon (default: 2)")
    p.add_argument("--dos-source", choices=["file", "phonopy"], default="file",
                   help="mode 0 only -- where each element's phonon DOS comes "
                        "from: per-scatterer dos= files (default) or derived "
                        "from the phonopy model")
    p.add_argument("--lattice", type=parse_coeffs, metavar="a,b,c,al,be,ga",
                   help="mode 0 only -- unit cell [Angstrom, degrees] for the "
                        "coherent-elastic Bragg peaks (with per-scatterer pos= "
                        "sites)")
    p.add_argument("--scatterer", action="append", type=parse_scatterer, default=[],
                   help="SYMBOL,sigma_bound_b,awr[,b_coh_fm[,sigma_inc_b]] "
                        "(repeatable). Mode-0 extras ride as key=value tokens: "
                        "dos=FILE, unit=meV|eV|cm-1|THz, mult=N, "
                        "pos=x:y:z;x:y:z;...")
    p.add_argument("--de", type=float, default=0.5,
                   help="energy-transfer grid step in meV (default: 0.5)")
    p.add_argument("--e-min", type=float, default=0.0,
                   help="energy-transfer grid start in meV; negative includes "
                        "the energy-gain side (default: 0)")
    p.add_argument("--e-max", type=float, default=250.0,
                   help="energy-transfer grid end in meV (default: 250)")
    p.add_argument("--dq", type=float, default=0.05,
                   help="powder S(Q,E) Q-support spacing in 1/A (default: 0.05)")
    p.add_argument("--q-max", type=float,
                   help="2-D map Q-axis maximum in 1/A (1-D runs derive Q from "
                        "the instrument and ignore it)")
    p.add_argument("--max-phonon-order", type=_max_order, default="auto",
                   help="multiphonon expansion order; 'auto' sizes it to "
                        "convergence in every mode (mode 0 derives it from "
                        "the DOS Debye-Waller lambda) (default: auto)")
    p.add_argument("--min-phonon-energy", type=float, default=0.0, metavar="MEV",
                   help="modes 1/2 -- remove every phonon mode with energy at or "
                        "below this value (meV) from all terms; 0 = the automatic "
                        "floors only (default: 0)")
    p.add_argument("--directions", type=int, default=10000,
                   help="modes 1/2 -- coherent powder-average directions "
                        "(default: 10000)")
    p.add_argument("--mp-directions", type=int, default=1000,
                   help="modes 1/2 -- multiphonon powder-average directions "
                        "(default: 1000)")
    p.add_argument("--jobs", type=int, help="worker processes (default: all cores)")
    p.add_argument("--elastic", choices=["on", "off"], default="on",
                   help="include the elastic line (default: on)")
    p.add_argument("--gain-side", choices=["direct", "detailed_balance"],
                   default="direct",
                   help="energy-gain (E<0) evaluation: 'direct' computes it "
                        "with explicit Bose factors in every mode (modes 1/2 "
                        "read the engine's directly computed gain arrays), "
                        "'detailed_balance' mirrors the loss side "
                        "(default: direct)")
    p.add_argument("--elastic-kind", choices=["both", "coherent", "incoherent"],
                   default="both", help="both (Bragg + Debye-Waller) | coherent | incoherent")
    p.add_argument("--elastic-from-tape",
                   help="build the elastic line from this ENDF MF7/MT2 tape "
                        "instead of tape-free from the phonon model")
    p.add_argument("--bank-halfwidth", type=float, default=5.0,
                   help="detector-bank angular half-width in degrees for the "
                        "bank-integrated elastic line (default: 5)")
    p.add_argument("--sigma-coeffs", type=parse_coeffs,
                   help="resolution width poly c0,c1,c2 (meV): sigma for gaussian, HWHM for lorentzian")
    p.add_argument("--resolution-shape", choices=["gaussian", "lorentzian"],
                   default="gaussian",
                   help="resolution line shape; gaussian is the OCLIMAX-equivalent, "
                        "lorentzian is the heavier-tailed option")
    p.add_argument("--resolution-model", choices=["poly", "chopper"],
                   default="poly",
                   help="width source: poly (sigma c0,c1,c2) | chopper (auto, any "
                        "PyChop instrument); chopper is direct-geometry only")
    # automatic chopper resolution (resolution-model=chopper)
    p.add_argument("--chopper-instrument",
                   help="instrument for --resolution-model chopper: ARCS, SEQUOIA, "
                        "MAPS, MARI, MERLIN, HYSPEC (Fermi) or CNCS, LET (disk)")
    p.add_argument("--chopper-package",
                   help="chopper package / resolution mode (e.g. ARCS-700-1.5-AST, "
                        "High-Resolution, Standard)")
    p.add_argument("--chopper-frequency", type=float,
                   help="chopper / resolution-disk frequency (Hz)")
    p.add_argument("--combine", choices=["mean", "sum"], default="mean",
                   help="combine detector banks by mean or sum (default: mean)")
    p.add_argument("--q-cuts", type=parse_coeffs,
                   help="constant-|Q| cuts: comma list of Q values [1/A]")
    p.add_argument("--components", action="store_true",
                   help="also write each cut's inelastic + elastic breakdown "
                        "(default: total only)")
    p.add_argument("-o", "--output", required=True,
                   help="output spectrum file: .csv (default), .npz, or .json")


def build_parser():
    """Build the ``irma spectra`` argument parser with all subcommands."""
    p = argparse.ArgumentParser(
        prog="irma spectra",
        description="Neutron-scattering forward spectra (VISION / indirect / direct).")
    sub = p.add_subparsers(dest="command", required=True)

    vis = sub.add_parser("vision", help="VISION preset (indirect, Ef=3.5, 45/135 banks)")
    _add_common(vis)
    vis.add_argument("--ef", type=float, default=None, help="override Ef (meV)")

    ind = sub.add_parser("indirect", help="generic indirect geometry")
    _add_common(ind)
    ind.add_argument("--ef", type=float, required=True, help="fixed final energy Ef (meV)")
    ind.add_argument("--angles", type=parse_angles, required=True,
                     help="comma list or start:stop:step (deg)")

    dir_ = sub.add_parser("direct", help="generic direct geometry")
    _add_common(dir_)
    dir_.add_argument("--ei", type=float, required=True, help="fixed incident energy Ei (meV)")
    dir_.add_argument("--angles", type=parse_angles, required=True,
                      help="comma list or start:stop:step (deg)")
    dir_.add_argument("--kinematic-factor", action="store_true",
                      help="multiply by kf/ki (count-rate spectrum)")

    runp = sub.add_parser("run", help="run a SpectraConfig file (YAML/TOML/JSON)")
    runp.add_argument("config")
    runp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      dest="overrides", help="dotted override, e.g. material.temperature_K=500")
    runp.add_argument("-o", "--output", required=True)

    mapp = sub.add_parser("map", help="dense 2-D S(Q,E) powder map (+ kinematic envelope)")
    mapp.add_argument("config")
    mapp.add_argument("--q-min", type=float, default=0.0)
    mapp.add_argument("--q-max", type=float, default=None,
                      help="map Q maximum [1/A]; default = config grid.q_max_invA, "
                           "else auto-cover the kinematic envelope")
    mapp.add_argument("--dq-map", type=float, default=None,
                      help="map Q step [1/A]; default = config grid.dq_max_invA")
    mapp.add_argument("--angle-range", nargs=2, type=float, metavar=("MIN", "MAX"),
                      help="detector 2theta range for the kinematic envelope (deg); "
                           "default = config instrument.map_coverage_deg")
    mapp.add_argument("--no-broaden", action="store_true",
                      help="skip the energy-resolution broadening")
    mapp.add_argument("--mask", action=argparse.BooleanOptionalAction, default=None,
                      help="blank S(Q,E) outside the detector kinematic envelope "
                           "(the accessible arch); default = config instrument.map_mask")
    mapp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      dest="overrides")
    mapp.add_argument("-o", "--output", required=True)
    return p


def resolve_map_output_path(path):
    """The exact final path ``write_map`` opens for ``-o path``.

    One resolver shared by the preflight and the writer (review SP-2a): the
    old flow preflighted the RAW argument while the writer opened a mutated
    one, so ``-o out.json`` was checked at ``out.json`` but written at
    ``out.json.npz`` -- an unwritable target there failed only AFTER the
    (potentially hours-long) compute."""
    path = str(path)
    root, ext = os.path.splitext(path)
    low = ext.lower()
    if low in (".npz", ".csv"):
        return root + low
    return path + ".npz"


def write_map(sqemap, path, masked=False):
    """Write an SQEMap to ``.npz`` or long-form ``.csv`` (by extension,
    case-insensitively); a path with another/no extension gets ``.npz``
    appended. ``masked=True`` blanks S outside the kinematic envelope
    (Euphonic-style arch). Returns the path actually written."""
    from irma.spectra.forward import save_sqe_map
    path = str(path)
    resolved = resolve_map_output_path(path)
    if resolved != path:
        _, ext = os.path.splitext(path)
        # The npz/csv format dispatch is by exact lowercase suffix; honor the
        # user's evident intent ('-o MAP.NPZ' must not get CSV text) by
        # normalizing -- and say so. An unknown suffix gets .npz appended.
        print(f"WARNING: map output {path!r} resolved to {resolved!r} "
              f"(map formats are .npz/.csv; extension {ext!r})",
              file=sys.stderr)
    return save_sqe_map(resolved, sqemap.Q, sqemap.E, sqemap.S,
                        envelope=sqemap.envelope, masked=masked)


# -----------------------------------------------------------------------------
# flag form -> SpectraConfig
# -----------------------------------------------------------------------------
def config_from_args(ns):
    """Build the SpectraConfig a vision/indirect/direct flag invocation describes."""
    material = {"phonopy_yaml": ns.phonopy_yaml, "mesh": list(ns.mesh),
                "temperature_K": ns.temperature, "scatterers": list(ns.scatterer)}
    for k, attr in (("born", "born"), ("force_constants", "force_constants"),
                    ("force_sets", "force_sets")):
        if getattr(ns, attr):
            material[k] = getattr(ns, attr)
    if ns.lattice is not None:
        material["lattice"] = list(ns.lattice)
    physics = {
        "inelastic_mode": ns.inelastic_mode, "max_phonon_order": ns.max_phonon_order,
        "min_phonon_energy_meV": float(getattr(ns, "min_phonon_energy", 0.0) or 0.0),
        "n_directions": ns.directions, "multiphonon_directions": ns.mp_directions,
        "jobs": ns.jobs, "elastic": ns.elastic == "on", "elastic_kind": ns.elastic_kind,
        "elastic_from_tape": ns.elastic_from_tape,
        "gain_side": ns.gain_side,
        "kinematic_kf_ki": bool(getattr(ns, "kinematic_factor", False)),
    }
    if ns.inelastic_mode == 0:
        physics["dos_source"] = ns.dos_source
    grid = {"e_min_meV": ns.e_min, "e_max_meV": ns.e_max, "de_meV": ns.de,
            "dq_max_invA": ns.dq}
    if ns.q_max is not None:
        grid["q_max_invA"] = ns.q_max
    instrument = {"geometry": ns.command, "bank_halfwidth_deg": ns.bank_halfwidth,
                  "resolution_shape": ns.resolution_shape,
                  "resolution_model": ns.resolution_model, "combine": ns.combine}
    if ns.resolution_model == "chopper":
        # Fail here, by flag name, when a chopper flag was omitted -- the raw
        # None values would otherwise crash from_dict (float(None) TypeError)
        # or reach validate() as the literal string 'None'.
        missing = [flag for flag, val in
                   (("--chopper-instrument", ns.chopper_instrument),
                    ("--chopper-package", ns.chopper_package),
                    ("--chopper-frequency", ns.chopper_frequency))
                   if val is None]
        if missing:
            raise SpectraConfigError(
                f"--resolution-model chopper requires {', '.join(missing)} "
                "(the chopper instrument, package and frequency together "
                "define the resolution)")
        instrument["chopper_spec"] = {
            "instrument": ns.chopper_instrument, "package": ns.chopper_package,
            "frequency": ns.chopper_frequency}
    if ns.command == "direct":
        instrument["e_fixed_meV"] = ns.ei
    elif ns.ef is not None:
        instrument["e_fixed_meV"] = ns.ef
    if getattr(ns, "angles", None) is not None:
        instrument["angles_deg"] = ns.angles
    if getattr(ns, "q_cuts", None) is not None:
        instrument["q_cuts"] = ns.q_cuts
    if getattr(ns, "components", False):
        instrument["export_components"] = True
    if ns.sigma_coeffs is not None:
        instrument["sigma_coeffs"] = ns.sigma_coeffs
    return SpectraConfig.from_dict(
        {"material": material, "physics": physics, "grid": grid, "instrument": instrument})


def _coerce(v):
    """Coerce a --set string value to bool/int/float/None/str.

    on/off/yes/no coerce to booleans: the CLI's own flag vocabulary is
    on|off, and an un-coerced 'off' is a truthy string that would silently
    leave a boolean physics switch enabled."""
    low = v.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def apply_overrides(cfg_dict, overrides):
    """Apply ``--set a.b.c=value`` dotted overrides to a parsed config dict."""
    for ov in overrides:
        if "=" not in ov:
            raise SpectraConfigError(f"--set expects KEY=VALUE, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg_dict
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise SpectraConfigError(f"--set {key}: {p} is not a section")
        node[parts[-1]] = _coerce(val)
    return cfg_dict


# -----------------------------------------------------------------------------
# output writers
# -----------------------------------------------------------------------------
def resolve_spectrum_output_path(path):
    """The exact final path ``write_spectrum`` opens for ``-o path``.

    Known suffixes are lowercase-normalized; anything else (including no
    extension) is written verbatim as CSV text. Shared by the preflight and
    the writer (review SP-2a/b: the old mode-blind candidate set also
    preflighted ``path + '.npz'`` for a CUTS run, so an unrelated
    ``out.npz`` directory falsely rejected a run whose real target was
    fine)."""
    path = str(path)
    root, ext = os.path.splitext(path)
    low = ext.lower()
    if low in (".npz", ".json", ".csv"):
        return root + low
    return path


def write_spectrum(result, path, components=True):
    """Write a SpectrumResult to .csv (default) / .npz / .json by extension
    (case-insensitive). Returns the path actually written.

    Output is per detector bank / per Q-cut (one column block per bank/cut).

    ``components``: when True, each cut gets its full TOTAL + INELASTIC + ELASTIC
    breakdown. When False (the config/GUI default), only the TOTAL of each cut is
    written -- the lean default; pass components=True / tick the breakdown option
    to keep the inelastic and elastic channels too.
    """
    path = str(path)
    root, ext = os.path.splitext(path)
    low = ext.lower()
    resolved = resolve_spectrum_output_path(path)
    if resolved != path:
        # Format dispatch is by lowercase suffix (and np.savez would append a
        # second '.npz' to an uppercase one); honor the user's evident intent
        # ('-o OUT.NPZ' must not get CSV text) by normalizing -- and say so.
        print(f"WARNING: output extension {ext!r} normalized to {low!r}; "
              f"writing {resolved}", file=sys.stderr)
        path = resolved
    elif low not in ("", ".npz", ".json", ".csv"):
        print(f"WARNING: unrecognized output extension {ext!r} on {path}; "
              "writing CSV text -- use .csv, .npz, or .json", file=sys.stderr)
    E = result.E
    # PER-ANGLE block -- present unless this was a constant-Q-ONLY run (cut_by='q'),
    # which produces no detector-angle spectra.
    angles = list(result.angles_deg) if result.angles_deg else []
    _pa = result.I_inelastic_per_angle
    has_ang = bool(angles) and _pa is not None and np.atleast_2d(_pa).shape[0] > 0
    if has_ang:
        Ipa = np.atleast_2d(result.I_inelastic_per_angle)
        Epa = np.atleast_2d(result.I_elastic_per_angle)
        Tpa = Ipa + Epa
    # PER-Q block -- constant-|Q| cuts.
    qcuts = list(result.q_cut_values) if result.q_cut_values else []
    has_q = bool(qcuts)
    if has_q:
        Iq = np.atleast_2d(result.I_inelastic_per_q)
        Eq = np.atleast_2d(result.I_elastic_per_q)
        Tq = Iq + Eq
    if not has_ang and not has_q:        # degenerate -> combined curve as one bank
        has_ang, angles = True, [0.0]
        Ipa = result.I_inelastic[None, :]
        Epa = result.I_elastic[None, :]
        Tpa = Ipa + Epa
    na = len(angles) if has_ang else 0
    meta = {"geometry": result.geometry, **{k: v for k, v in result.metadata.items()
                                            if k != "engine_metadata"}}
    if low == ".npz":
        arrs = dict(E_meV=E)
        if has_ang:
            arrs.update(angles_deg=np.asarray(angles, float),
                        I_total_per_angle=Tpa, Q_per_angle=np.asarray(result.Q))
            if components:
                arrs.update(I_inelastic_per_angle=Ipa, I_elastic_per_angle=Epa)
        if has_q:
            arrs.update(q_cuts=np.asarray(qcuts, float), I_total_per_q=Tq)
            if components:
                arrs.update(I_inelastic_per_q=Iq, I_elastic_per_q=Eq)
        np.savez_compressed(path, **arrs)
    elif low == ".json":
        out = {"E_meV": E.tolist(), "metadata": meta}
        if has_ang:
            out.update(angles_deg=[float(a) for a in angles],
                       I_total_per_angle=Tpa.tolist())
            if components:
                out.update(I_inelastic_per_angle=Ipa.tolist(),
                           I_elastic_per_angle=Epa.tolist())
        if has_q:
            out.update(q_cuts=[float(q) for q in qcuts], I_total_per_q=Tq.tolist())
            if components:
                out.update(I_inelastic_per_q=Iq.tolist(), I_elastic_per_q=Eq.tolist())
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
    else:  # .csv default
        def _cols(tag):
            """Column names for one spectrum tag (with/without components)."""
            return ([f"total@{tag}", f"inelastic@{tag}", f"elastic@{tag}"]
                    if components else [f"total@{tag}"])
        cols = ["E_meV"]
        for a in angles:
            cols += _cols(f"{a:g}deg")
        for qv in qcuts:
            cols += _cols(f"Q={qv:g}")
        with open(path, "w") as fh:
            fh.write(f"# IRMA spectrum: geometry={result.geometry} "
                     f"mode={meta.get('inelastic_mode')} "
                     f"angles_deg={angles} q_cuts={qcuts} components={components} "
                     f"elastic={meta.get('elastic')} edges={meta.get('n_bragg_edges')}\n")
            fh.write(",".join(cols) + "\n")
            for i in range(E.size):
                row = [f"{E[i]:.6g}"]
                for k in range(na):
                    row += ([f"{Tpa[k, i]:.8g}", f"{Ipa[k, i]:.8g}", f"{Epa[k, i]:.8g}"]
                            if components else [f"{Tpa[k, i]:.8g}"])
                for k in range(len(qcuts)):
                    row += ([f"{Tq[k, i]:.8g}", f"{Iq[k, i]:.8g}", f"{Eq[k, i]:.8g}"]
                            if components else [f"{Tq[k, i]:.8g}"])
                fh.write(",".join(row) + "\n")
    return path


def _provenance(cfg, output):
    """Provenance lines (inputs, hashes, versions) printed before a run."""
    m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument
    if m.phonopy_yaml:
        try:
            with open(m.phonopy_yaml, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()[:8]
        except OSError:
            digest = "????????"
        model = f"yaml={m.phonopy_yaml}#{digest} mesh={m.mesh}"
    else:   # mode 0 from per-scatterer DOS files: there is no phonopy model
        model = ("dos=[" + ",".join(s.dos_file or "?" for s in m.scatterers) + "]")
    return (f"irma spectra: geometry={ins.geometry} {model} "
            f"mode={p.inelastic_mode} T={m.temperature_K}K "
            f"E=[{g.e_min_meV},{g.e_max_meV}]/{g.de_meV} dQ={g.dq_max_invA} "
            f"elastic={'on' if p.elastic else 'off'} -> {output}")


# -----------------------------------------------------------------------------
# entry point
# -----------------------------------------------------------------------------
def _load_cfg(ns):
    """Read + parse + override + validate the ``run``/``map`` config file.

    Every user-input failure (missing file, malformed YAML/JSON/TOML, schema or
    semantic error, a referenced input file that does not exist) surfaces as
    :class:`SpectraConfigError`, so ``main`` can print one clean message and
    exit 2 instead of leaking a traceback.
    """
    import irma.spectra.config as _cfgmod
    from pathlib import Path
    path = Path(ns.config)
    try:
        text = path.read_text()
    except OSError as exc:
        raise SpectraConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        cfg_dict = _cfgmod._parse(text, path.suffix)
    except SpectraConfigError:
        raise
    except Exception as exc:    # yaml.YAMLError / JSONDecodeError / TOML errors
        raise SpectraConfigError(
            f"config file {path} could not be parsed: {exc}") from exc
    if not isinstance(cfg_dict, dict):
        raise SpectraConfigError(
            f"config file {path} must contain a mapping of sections "
            f"(material/physics/grid/instrument), got {type(cfg_dict).__name__}")
    cfg_dict = apply_overrides(cfg_dict, ns.overrides)
    cfg = SpectraConfig.from_dict(cfg_dict)
    from irma.spectra.config import validate
    validate(cfg)
    check_input_files(cfg)       # fail by FIELD NAME, not a mid-run bare errno
    return cfg


def _run_map_command(ns):
    """Run the 2-D S(Q,E) map subcommand from parsed CLI options."""
    # Stream convention (matches the deck CLI): provenance/progress/success
    # lines go to stdout; only diagnostics (errors, warnings) go to stderr.
    from irma.spectra.config import run_map
    canonical = resolve_map_output_path(ns.output)
    rc = _check_output_path(canonical)
    if rc is not None:
        return rc
    try:
        cfg = _load_cfg(ns)
    except (SpectraConfigError, ValueError, TypeError) as exc:
        print(f"\nSpectra config error:\n  {exc}", file=sys.stderr)
        return 2
    print(_provenance(cfg, canonical))
    try:
        sm = run_map(cfg, q_min=ns.q_min, q_max=ns.q_max, dQ_map=ns.dq_map,
                     angle_range=(tuple(ns.angle_range) if ns.angle_range else None),
                     broaden=not ns.no_broaden, progress=print)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"\nIRMA map failed: {exc}", file=sys.stderr)
        return 3
    masked = ns.mask if ns.mask is not None else cfg.instrument.map_mask
    try:
        written = write_map(sm, canonical, masked=masked)
    except OSError as exc:
        print(f"\nIRMA map failed writing output: {exc}", file=sys.stderr)
        return 3
    print(f"Wrote {written}")
    return 0


def _check_output_path(canonical):
    """Pre-validate the CANONICAL output path (the one the writer will open,
    from the mode-aware resolver) before the multi-minute compute, so a
    typo'd ``-o`` path, a directory target, a read-only parent, or a
    read-only existing target fails fast instead of only after the run
    (reviews S10 + SP-2a/b: preflighting anything other than the writer's
    exact target either misses a collision or falsely rejects a valid run).
    Returns an exit code on failure, else None."""
    from irma.cli import validate_output_path        # one shared preflight
    err = validate_output_path(str(canonical))
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    return None


def main(argv):
    """``irma spectra`` entry point: parse, dispatch, return an exit code."""
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "map":
        return _run_map_command(ns)

    # Stream convention (matches the deck CLI): provenance/progress/success
    # lines go to stdout; only diagnostics (errors, warnings) go to stderr.
    try:
        if ns.command == "run":
            cfg = _load_cfg(ns)         # _load_cfg preflights the input files
        else:
            cfg = config_from_args(ns)
            check_input_files(cfg)   # fail by FIELD NAME, not a mid-run errno
    except (SpectraConfigError, ValueError, TypeError) as exc:
        print(f"\nSpectra config error:\n  {exc}", file=sys.stderr)
        return 2

    # Mode-aware canonical output path: the config decides map vs cuts, so
    # the exact writer target is only known here -- preflight THAT path.
    if cfg.instrument.output_mode == "map":
        output = resolve_map_output_path(ns.output)
    else:
        output = resolve_spectrum_output_path(ns.output)
    rc = _check_output_path(output)
    if rc is not None:
        return rc
    print(_provenance(cfg, output))
    # A config that declares output_mode='map' produces the dense 2-D S(Q,E) map,
    # not 1-D cuts. (The 'map' subcommand is the explicit-flag entry point; here
    # the config drives every parameter.) Without this, a map-only config -- which
    # validation deliberately exempts from the angles_deg requirement -- would
    # fall through to the cuts path and crash on the missing loci.
    if cfg.instrument.output_mode == "map":
        from irma.spectra.config import run_map
        try:
            sm = run_map(cfg, progress=print)
        except (RuntimeError, FileNotFoundError, ValueError, ImportError) as exc:
            print(f"\nIRMA map failed: {exc}", file=sys.stderr)
            return 3
        try:
            written = write_map(sm, output, masked=cfg.instrument.map_mask)
        except OSError as exc:
            # Post-compute output failures get the same clean boundary as the
            # compute itself (review CLI-1b): the preflight covers the common
            # cases up front, this covers disk-full/removal races after it.
            print(f"\nIRMA map failed writing output: {exc}", file=sys.stderr)
            return 3
        print(f"Wrote {written}")
        return 0
    try:
        result = run_spectra(cfg, progress=print)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"\nIRMA spectra failed: {exc}", file=sys.stderr)
        return 3
    try:
        written = write_spectrum(
            result, output,
            components=bool(cfg.instrument.export_components))
    except OSError as exc:
        print(f"\nIRMA spectra failed writing output: {exc}", file=sys.stderr)
        return 3
    print(f"Wrote {written}")
    return 0
