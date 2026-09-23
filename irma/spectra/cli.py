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
from pathlib import Path

import numpy as np

from irma.spectra.config import (
    SpectraConfig, SpectraConfigError, _parse, check_input_files, run_map,
    run_spectra, validate,
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


# -----------------------------------------------------------------------------
# argument parser
# -----------------------------------------------------------------------------
def _add_common(p):
    """Attach the options shared by the vision/indirect/direct subcommands.

    Every flag defaults to None, which keeps the SpectraConfig default quoted
    in its help; ``--set`` reaches any other config key.
    """
    p.add_argument("--phonopy-yaml",
                   help="phonopy model file; required for inelastic modes 1/2 "
                        "and for mode 0 with --dos-source phonopy")
    p.add_argument("--born",
                   help="BORN file (non-analytical-term correction / LO-TO "
                        "splitting)")
    p.add_argument("--mesh", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   help="phonon q-mesh for the eigenvector/DOS sampling "
                        "(default: 40 40 40)")
    p.add_argument("--temperature", type=float,
                   help="sample temperature in K (default: 296)")
    p.add_argument("--inelastic-mode", type=int, choices=[0, 1, 2],
                   help="0 = DOS + isotropic Debye-Waller (no eigenvectors), "
                        "1 = incoherent approximation, 2 = exact coherent "
                        "one-phonon + incoherent multiphonon (default: 2)")
    p.add_argument("--dos-source", choices=["file", "phonopy"],
                   help="mode 0 only -- where each element's phonon DOS comes "
                        "from: per-scatterer dos= files (default) or derived "
                        "from the phonopy model")
    p.add_argument("--lattice", type=parse_coeffs, metavar="a,b,c,al,be,ga",
                   help="mode 0 only -- unit cell [Angstrom, degrees] for the "
                        "coherent-elastic Bragg peaks (with per-scatterer pos= "
                        "sites)")
    p.add_argument("--scatterer", action="append", type=parse_scatterer, default=[],
                   help="SYMBOL,sigma_bound_b,awr[,b_coh_fm[,sigma_inc_b]] "
                        "(repeatable; modes 1/2 need b_coh_fm and sigma_inc_b except for C). "
                        "Mode-0 extras ride as key=value tokens: "
                        "dos=FILE, unit=meV|eV|cm-1|THz, mult=N, "
                        "pos=x:y:z;x:y:z;...")
    p.add_argument("--de", type=float,
                   help="energy-transfer grid step in meV (default: 0.5)")
    p.add_argument("--e-min", type=float,
                   help="energy-transfer grid start in meV; negative includes "
                        "the energy-gain side (default: 0)")
    p.add_argument("--e-max", type=float,
                   help="energy-transfer grid end in meV (default: 250)")
    p.add_argument("--dq", type=float,
                   help="powder S(Q,E) Q-support spacing in 1/A (default: 0.05)")
    p.add_argument("--max-phonon-order", type=_max_order,
                   help="multiphonon expansion order; 'auto' sizes it to "
                        "convergence in every mode (mode 0 derives it from "
                        "the DOS Debye-Waller lambda) (default: auto)")
    p.add_argument("--min-phonon-energy", type=float, metavar="MEV",
                   help="modes 1/2 -- remove every phonon mode with energy at or "
                        "below this value (meV) from all terms; 0 = the automatic "
                        "floors only (default: 0)")
    p.add_argument("--directions", type=int,
                   help="modes 1/2 -- one-phonon (coherent and incoherent) "
                        "powder-average directions "
                        "(default: 10000)")
    p.add_argument("--mp-directions", type=int,
                   help="modes 1/2 -- multiphonon powder-average directions "
                        "(default: 1000)")
    p.add_argument("--jobs", type=int, help="worker processes (default: all cores)")
    p.add_argument("--elastic", choices=["on", "off"],
                   help="include the elastic line (default: on)")
    p.add_argument("--gain-side", choices=["direct", "detailed_balance"],
                   help="energy-gain (E<0) evaluation: 'direct' computes it "
                        "with explicit Bose factors in every mode (modes 1/2 "
                        "read the engine's directly computed gain arrays), "
                        "'detailed_balance' mirrors the loss side "
                        "(default: direct)")
    p.add_argument("--elastic-kind", choices=["both", "coherent", "incoherent"],
                   help="both (Bragg + Debye-Waller) | coherent | incoherent "
                        "(default: both)")
    p.add_argument("--sigma-coeffs", type=parse_coeffs,
                   help="resolution width poly c0,c1,c2 (meV): sigma for gaussian, HWHM for lorentzian")
    p.add_argument("--resolution-shape", choices=["gaussian", "lorentzian"],
                   help="resolution line shape; gaussian (default) is the "
                        "OCLIMAX-equivalent, lorentzian is the heavier-tailed option")
    p.add_argument("--resolution-model", choices=["poly", "chopper"],
                   help="width source: poly (sigma c0,c1,c2; default) | chopper "
                        "(auto, any PyChop instrument); chopper is direct-geometry only")
    # automatic chopper resolution (resolution-model=chopper)
    p.add_argument("--chopper-instrument",
                   help="instrument for --resolution-model chopper: ARCS, SEQUOIA, "
                        "MAPS, MARI, MERLIN, HYSPEC (Fermi) or CNCS, LET (disk)")
    p.add_argument("--chopper-package",
                   help="chopper package / resolution mode (e.g. ARCS-700-1.5-AST, "
                        "High-Resolution, Standard)")
    p.add_argument("--chopper-frequency", type=float,
                   help="chopper / resolution-disk frequency (Hz)")
    p.add_argument("--q-cuts", type=parse_coeffs,
                   help="constant-|Q| cuts: comma list of Q values [1/A]")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   dest="overrides",
                   help="dotted config override, e.g. instrument.combine=sum")
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
    """The path ``main`` writes a map to: ``.npz``/``.csv`` (lowercased), else
    ``.npz`` appended."""
    root, ext = os.path.splitext(str(path))
    return root + ext.lower() if ext.lower() in (".npz", ".csv") else str(path) + ".npz"


def write_map(sqemap, path, masked=False):
    """Write an SQEMap to ``.npz`` or long-form ``.csv``; ``masked=True`` blanks
    S outside the kinematic envelope. Returns the path written."""
    from irma.spectra.forward import save_sqe_map
    return save_sqe_map(path, sqemap.Q, sqemap.E, sqemap.S,
                        envelope=sqemap.envelope, masked=masked)


# -----------------------------------------------------------------------------
# flag form -> SpectraConfig
# -----------------------------------------------------------------------------
def config_from_args(ns):
    """Build the SpectraConfig a vision/indirect/direct flag invocation describes."""
    def given(**kw):
        """The keyword arguments whose flag was set."""
        return {k: v for k, v in kw.items() if v is not None}

    material = given(
        phonopy_yaml=ns.phonopy_yaml, born=ns.born, temperature_K=ns.temperature,
        mesh=None if ns.mesh is None else list(ns.mesh),
        lattice=None if ns.lattice is None else list(ns.lattice),
        scatterers=list(ns.scatterer))
    physics = given(
        inelastic_mode=ns.inelastic_mode, dos_source=ns.dos_source,
        max_phonon_order=ns.max_phonon_order,
        min_phonon_energy_meV=ns.min_phonon_energy, n_directions=ns.directions,
        multiphonon_directions=ns.mp_directions, jobs=ns.jobs,
        elastic=None if ns.elastic is None else ns.elastic == "on",
        elastic_kind=ns.elastic_kind, gain_side=ns.gain_side,
        kinematic_kf_ki=getattr(ns, "kinematic_factor", None))
    grid = given(e_min_meV=ns.e_min, e_max_meV=ns.e_max, de_meV=ns.de,
                 dq_max_invA=ns.dq)
    instrument = given(
        geometry=ns.command, e_fixed_meV=ns.ei if ns.command == "direct" else ns.ef,
        angles_deg=getattr(ns, "angles", None), q_cuts=ns.q_cuts,
        sigma_coeffs=ns.sigma_coeffs, resolution_shape=ns.resolution_shape,
        resolution_model=ns.resolution_model)
    if ns.resolution_model == "chopper":
        instrument["chopper_spec"] = {
            "instrument": ns.chopper_instrument, "package": ns.chopper_package,
            "frequency": ns.chopper_frequency}
    return SpectraConfig.from_dict(apply_overrides(
        {"material": material, "physics": physics, "grid": grid,
         "instrument": instrument}, ns.overrides))


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
    """The path ``main`` writes a spectrum to: a known suffix is lowercased,
    anything else is written verbatim as CSV text."""
    root, ext = os.path.splitext(str(path))
    return root + ext.lower() if ext.lower() in (".npz", ".json", ".csv") else str(path)


def write_spectrum(result, path, components=True):
    """Write a SpectrumResult to .csv (default) / .npz / .json by extension.
    Returns the path written.

    One column block per detector bank and per constant-Q cut. ``components``
    adds each block's inelastic and elastic parts to its total.
    """
    path = str(path)
    low = os.path.splitext(path)[1].lower()
    E = result.E
    angles = list(result.angles_deg) if result.angles_deg else []
    qcuts = list(result.q_cut_values) if result.q_cut_values else []
    blocks = []                    # (kind, values, column tags, inelastic, elastic)
    if angles:
        blocks.append(("angle", angles, [f"{a:g}deg" for a in angles],
                       np.atleast_2d(result.I_inelastic_per_angle),
                       np.atleast_2d(result.I_elastic_per_angle)))
    if qcuts:
        blocks.append(("q", qcuts, [f"Q={q:g}" for q in qcuts],
                       np.atleast_2d(result.I_inelastic_per_q),
                       np.atleast_2d(result.I_elastic_per_q)))
    meta = {"geometry": result.geometry, **{k: v for k, v in result.metadata.items()
                                            if k != "engine_metadata"}}
    if low in (".npz", ".json"):
        arrs = {}
        for kind, values, _tags, inel, elas in blocks:
            arrs["angles_deg" if kind == "angle" else "q_cuts"] = np.asarray(values, float)
            arrs[f"I_total_per_{kind}"] = inel + elas
            if kind == "angle" and low == ".npz":
                arrs["Q_per_angle"] = np.asarray(result.Q)
            if components:
                arrs[f"I_inelastic_per_{kind}"] = inel
                arrs[f"I_elastic_per_{kind}"] = elas
        if low == ".npz":
            np.savez_compressed(path, E_meV=E, **arrs)
        else:
            out = {"E_meV": E.tolist(), "metadata": meta,
                   **{k: v.tolist() for k, v in arrs.items()}}
            with open(path, "w") as fh:
                json.dump(out, fh, indent=2)
        return path
    parts = ("total", "inelastic", "elastic") if components else ("total",)
    cols, series = ["E_meV"], []
    for _kind, _values, tags, inel, elas in blocks:
        for k, tag in enumerate(tags):
            cols += [f"{part}@{tag}" for part in parts]
            series += [inel[k] + elas[k], inel[k], elas[k]][:len(parts)]
    with open(path, "w") as fh:
        fh.write(f"# IRMA spectrum: geometry={result.geometry} "
                 f"mode={meta.get('inelastic_mode')} "
                 f"angles_deg={angles} q_cuts={qcuts} components={components} "
                 f"elastic={meta.get('elastic')} edges={meta.get('n_bragg_edges')}\n")
        fh.write(",".join(cols) + "\n")
        for i in range(E.size):
            fh.write(",".join([f"{E[i]:.6g}"] + [f"{s[i]:.8g}" for s in series]) + "\n")
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
    """Read, override and validate the ``run``/``map`` config file."""
    path = Path(ns.config)
    d = apply_overrides(_parse(path.read_text(), path.suffix), ns.overrides)
    return check_input_files(validate(SpectraConfig.from_dict(d)))


def main(argv):
    """``irma spectra`` entry point: parse, dispatch, return an exit code.

    Provenance, progress and success lines go to stdout; errors and warnings
    go to stderr.
    """
    from irma.cli import validate_output_path
    ns = build_parser().parse_args(argv)
    try:
        cfg = (_load_cfg(ns) if ns.command in ("run", "map")
               else check_input_files(validate(config_from_args(ns))))
    except (OSError, ValueError, TypeError) as exc:
        print(f"\nSpectra config error:\n  {exc}", file=sys.stderr)
        return 2
    # the config decides map vs cuts, so the writer's exact path is known here
    is_map = ns.command == "map" or cfg.instrument.output_mode == "map"
    output = (resolve_map_output_path if is_map else resolve_spectrum_output_path)(ns.output)
    err = validate_output_path(output)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 2
    print(_provenance(cfg, output))
    try:
        if is_map:
            kw = {}
            if ns.command == "map":
                kw = dict(q_min=ns.q_min, q_max=ns.q_max, dQ_map=ns.dq_map,
                          angle_range=tuple(ns.angle_range) if ns.angle_range else None,
                          broaden=not ns.no_broaden)
            masked = cfg.instrument.map_mask
            if getattr(ns, "mask", None) is not None:
                masked = ns.mask
            written = write_map(run_map(cfg, progress=print, **kw), output, masked=masked)
        else:
            written = write_spectrum(run_spectra(cfg, progress=print), output,
                                     components=cfg.instrument.export_components)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        print(f"\nIRMA {'map' if is_map else 'spectra'} failed: {exc}", file=sys.stderr)
        return 3
    print(f"Wrote {written}")
    return 0
