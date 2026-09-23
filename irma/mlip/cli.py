"""`irma mlip` — the MLIP phonon front end CLI.

    irma mlip build <structure> -o <outdir> [...]   structure -> model bundle
    irma mlip emit <bundle> --to endf,spectra,ncrystal [...]
    irma mlip validate <bundle>

Exit codes: 0 success, 2 usage/validation error, 3 unconverged relaxation,
4 missing potential dependency.

`irma mlip validate` is the documented first action on a RECEIVED bundle,
so it treats the bundle as UNTRUSTED: phonopy's YAML parser executes
`!!python/` tags at parse time, and validate_bundle scans and rejects
such files before any phonopy parse (see irma.mlip.bundle and
irma.core.phonopy_io.reject_unsafe_phonopy_yaml).

The public potentials are exactly irma.mlip.calculators.POTENTIALS. The
'emt' development backend is accepted only with the hidden
--allow-dev-backend flag (test/dev use; never a production surface).

All heavy imports are function-level (core-clean module).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

QDEN = 35.0            # mesh rule: int(QDEN/a_i)+1 per axis (INSPIRED's rule)
MESH_FLOOR = 8


def _err(msg) -> int:
    print(f"Error: {msg}", file=sys.stderr)
    return 2


def _allowed_potentials(allow_dev=False):
    from irma.mlip.calculators import POTENTIALS
    return POTENTIALS + ("emt",) if allow_dev else POTENTIALS


def _parse_pairs(values, what, cast=str):
    """['C=31', 'O=32'] -> {'C': 31, 'O': 32} with clear errors."""
    out = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"{what} expects SYMBOL=VALUE, got {item!r}")
        sym, _, val = item.partition("=")
        sym, val = sym.strip(), val.strip()
        if not sym or not val:
            raise ValueError(f"{what} expects SYMBOL=VALUE, got {item!r}")
        try:
            out[sym] = cast(val)
        except ValueError:
            raise ValueError(f"{what} {sym}: cannot parse value {val!r}")
    return out


def _parse_species(values):
    """['C:b_coh_fm=6.646,sigma_inc_b=0.001'] -> {'C': {...float fields}}."""
    out = {}
    for item in values or []:
        if ":" not in item:
            raise ValueError(
                f"--species expects SYMBOL:field=value[,field=value...], "
                f"got {item!r}")
        sym, _, body = item.partition(":")
        fields = {}
        for pair in body.split(","):
            if "=" not in pair:
                raise ValueError(f"--species {sym}: bad field {pair!r}")
            key, _, val = pair.partition("=")
            try:
                fields[key.strip()] = float(val)
            except ValueError:
                raise ValueError(
                    f"--species {sym}: {key.strip()} needs a number, "
                    f"got {val!r}")
        out[sym.strip()] = fields
    return out


def _parse_supercell(text):
    parts = str(text).split()
    if len(parts) == 1:
        try:
            return int(parts[0]) if parts[0].isdigit() else float(parts[0])
        except ValueError:
            raise ValueError(f"--supercell expects Lmin or 'nx ny nz', "
                             f"got {text!r}")
    if len(parts) == 3:
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            raise ValueError(f"--supercell dims must be integers, "
                             f"got {text!r}")
    raise ValueError(f"--supercell expects Lmin or 'nx ny nz', got {text!r}")


def _parse_mesh(text):
    parts = str(text).split()
    try:
        mesh = [int(x) for x in parts]
    except ValueError:
        raise ValueError(f"--mesh expects three positive integers, got {text!r}")
    if len(mesh) != 3 or any(n < 1 for n in mesh):
        raise ValueError(f"--mesh expects three positive integers, got {text!r}")
    return mesh


def _default_mesh(cellpar_abc):
    return [max(MESH_FLOOR, int(QDEN / float(a)) + 1) for a in cellpar_abc]


def _add_emit_options(p):
    p.add_argument("--temperature", type=float, default=296.0,
                   help="temperature (K) for emitted inputs [296]")
    p.add_argument("--mat", action="append", metavar="SYM=MAT",
                   help="ENDF MAT number per principal scatterer "
                        "(required for endf; repeatable)")
    p.add_argument("--nuclide", action="append", metavar="SYM=LABEL",
                   help="select an isotope, e.g. H=2-H, taking its ENDF "
                        "identity AND its scattering constants (repeatable; "
                        "default: the natural element, printed)")
    p.add_argument("--species", action="append",
                   metavar="SYM:field=value,...",
                   help="scattering-constant overrides (fields: awr, "
                        "b_coh_fm, sigma_inc_b, sigma_bound_b; repeatable)")
    p.add_argument("--material-id", default=None,
                   help="ncrystal export material id [mlip_model]")
    p.add_argument("--allow-unstable", action="store_true",
                   help="proceed with DOS-driven emission despite recorded "
                        "imaginary modes")
    p.add_argument("--inelastic-mode", type=int, choices=(0, 1, 2),
                   default=None,
                   help="physics level of the emitted ENDF decks [2]. "
                        "0 = classic isotropic path from the bundle's "
                        "species-projected DOS (principal spectrum on the "
                        "classic cards, Card 6e partial spectra for the "
                        "other species); 1/2 = phonopy-backed directional "
                        "decks. Not applicable to disordered bundles")
    # default None, not "mef": the build records its options in the bundle
    # manifest, and the emitters apply the mef default
    p.add_argument("--elastic-format", choices=("mef", "sef"),
                   help="elastic output convention of the emitted ENDF "
                        "decks: mef writes both elastic components for "
                        "every species (default), sef assigns the complete "
                        "coherent component to the designated-coherent atom")


def _build_parser():
    top = argparse.ArgumentParser(
        prog="irma mlip",
        description="MLIP phonon front end: structure + pretrained "
                    "potential -> phonon-model bundle -> IRMA inputs")
    sub = top.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="structure -> phonon-model bundle")
    b.add_argument("structure", help="structure file (any ASE format)")
    b.add_argument("-o", "--outdir", required=True,
                   help="bundle output directory")
    from irma.mlip.calculators import POTENTIALS as _pots
    b.add_argument("--potential", default="mattersim",
                   help=" | ".join(_pots) + " [mattersim]")
    b.add_argument("--model", default=None,
                   help="specific checkpoint name or path; pet-mad takes "
                        "NAME@VERSION (e.g. pet-mad-s@1.5.0), dpa3 takes "
                        "MODEL::HEAD (default head "
                        "MP_traj_v024_alldata_mixu), nequip takes a model-"
                        "zoo id (e.g. mir-group/NequIP-OAM-L:0.1) or a "
                        "compiled .nequip.pt2 path, grace takes a "
                        "foundation name (e.g. GRACE-2L-OAM; ASL academic "
                        "license); mace also accepts foundation names such "
                        "as medium-omat-0 (ASL -- note printed and "
                        "recorded)")
    b.add_argument("--threads", type=int, default=None,
                   help="native threads for the serial force path "
                        "[all cores]")
    b.add_argument("--jobs", type=int, default=1,
                   help="parallel displacement workers [1]")
    b.add_argument("--worker-threads", type=int, default=1,
                   help="native threads per parallel worker; keep "
                        "jobs x worker-threads <= cores [1]")
    b.add_argument("--supercell", default=None,
                   help="Lmin (Angstrom) or explicit 'nx ny nz' "
                        "[12; 1 1 1 under --disordered]")
    b.add_argument("--delta", type=float, default=0.03,
                   help="displacement amplitude, Angstrom [0.03]")
    b.add_argument("--fmax", type=float, default=0.01,
                   help="relaxation force convergence, eV/A [0.01]")
    b.add_argument("--nmax", type=int, default=100,
                   help="max relaxation steps [100]")
    b.add_argument("--relax-cell", action="store_true",
                   help="relax the cell too (FrechetCellFilter)")
    b.add_argument("--jitter-cycles", type=int, default=0,
                   help="on an unconverged relaxation, kick the highest-"
                        "force atoms (~0.05 A) and re-relax up to N extra "
                        "cycles, keeping the lowest-residual frame seen; "
                        "escapes MLIP force-noise stalls (glasses) [0]")
    b.add_argument("--snap-symmetry", nargs="?", type=float,
                   const=1e-2, default=None, metavar="TOL",
                   help="after relaxation, snap positions onto the exact "
                        "orbits of the spacegroup detected at TOL "
                        "[1e-2 A when given bare] -- fixes float32 "
                        "Wyckoff drift that inflates the displacement "
                        "count and breaks symmetry-reduced BORN files")
    b.add_argument("--force", action="store_true",
                   help="proceed past an unconverged relaxation (recorded)")
    b.add_argument("--born", default=None, metavar="PATH",
                   help="BORN file; NAC is embedded in the bundle "
                        "(never read from the working directory)")
    b.add_argument("--mesh", default=None, metavar="'NX NY NZ'",
                   help="quick-look/emit mesh [qden rule; 1 1 1 disordered]")
    b.add_argument("--disordered", action="store_true",
                   help="disordered/amorphous material: box = its own "
                        "supercell, Gamma mesh, DOS-driven emission")
    b.add_argument("--dos-smearing", type=float, default=None,
                   metavar="MEV", help="DOS Gaussian smearing width in meV "
                                       "[tetrahedron; auto 1 meV smearing "
                                       "when flat bands would be dropped]")
    b.add_argument("--overwrite", action="store_true",
                   help="replace an existing bundle/outputs")
    b.add_argument("--allow-dev-backend", action="store_true",
                   help=argparse.SUPPRESS)   # dev/test backends
    b.add_argument("--emit", default=None, metavar="LIST",
                   help="chain emission after the build: comma list of "
                        "endf,spectra,ncrystal")
    _add_emit_options(b)

    e = sub.add_parser("emit", help="bundle -> ready-to-edit IRMA inputs")
    e.add_argument("bundle", help="bundle directory from `irma mlip build`")
    e.add_argument("--to", required=True, metavar="LIST",
                   help="comma list of endf,spectra,ncrystal")
    e.add_argument("--out-dir", default=None,
                   help="output directory [the bundle directory]")
    e.add_argument("--min-phonon-energy", type=float, default=0.0, metavar="MEV",
                   help="remove every phonon mode with energy at or below this "
                        "value (meV) from all terms, in every emitted file; "
                        "0 = the automatic floors only (default: 0)")
    e.add_argument("--overwrite", action="store_true",
                   help="replace existing emitted files")
    _add_emit_options(e)

    v = sub.add_parser("validate", help="check a bundle end to end")
    v.add_argument("bundle", help="bundle directory")

    en = sub.add_parser(
        "env", help="dedicated per-potential environments (for potentials "
                    "whose pip pins conflict with this install)")
    esub = en.add_subparsers(dest="env_command", required=True)
    ec = esub.add_parser("create", help="provision + register an env")
    ec.add_argument("potential", help="potential to build the env for")
    ec.add_argument("--dry-run", action="store_true",
                    help="print the environment-build commands and exit")
    esub.add_parser("list", help="show registered interpreters")
    er = esub.add_parser("remove",
                         help="unregister + delete an auto-provisioned env")
    er.add_argument("potential")
    return top


def _do_emit(bundle, targets, args) -> int:
    from irma.mlip.emit import (
        emit_endf_decks, emit_ncrystal_yaml, emit_spectra_yaml)
    unknown = sorted(set(targets) - {"endf", "spectra", "ncrystal"})
    if unknown:
        return _err(f"unknown emit target(s) {unknown}; choose from "
                    f"endf, spectra, ncrystal")
    if bundle.manifest.get("calculator", {}).get("dev_backend"):
        print("  WARNING: this bundle was built with a DEVELOPMENT backend; "
              "its physics is not production-grade")
    mats = _parse_pairs(args.mat, "--mat", int)
    nuclides = _parse_pairs(args.nuclide, "--nuclide")
    overrides = _parse_species(args.species)
    out_dir = getattr(args, "out_dir", None) or bundle.path
    min_e = float(getattr(args, "min_phonon_energy", 0.0))

    produced = []
    if "endf" in targets:
        produced += emit_endf_decks(
            bundle, temperature_k=args.temperature, mats=mats,
            nuclides=nuclides, overrides=overrides, out_dir=out_dir,
            overwrite=args.overwrite, allow_unstable=args.allow_unstable,
            inelastic_mode=args.inelastic_mode,
            elastic_format=args.elastic_format or "mef",
            min_phonon_energy_mev=min_e)
    if "spectra" in targets:
        produced.append(emit_spectra_yaml(
            bundle, temperature_k=args.temperature,
            nuclides=nuclides, overrides=overrides,
            out_path=os.path.join(out_dir, "spectra.yaml"),
            overwrite=args.overwrite, allow_unstable=args.allow_unstable,
            min_phonon_energy_mev=min_e))
    if "ncrystal" in targets:
        produced.append(emit_ncrystal_yaml(
            bundle, temperature_k=args.temperature,
            material_id=args.material_id, nuclides=nuclides,
            overrides=overrides,
            out_path=os.path.join(out_dir, "ncrystal.yaml"),
            overwrite=args.overwrite, min_phonon_energy_mev=min_e))

    import shlex
    print("\nNext steps (review each file before production):")
    for path in produced:
        name = os.path.basename(path)
        q = shlex.quote
        if name.startswith("endf_") and name.endswith(".input"):
            out = name.replace(".input", ".endf")
            print(f"  irma {q(path)} {q(os.path.join(out_dir, out))}")
        elif name == "spectra.yaml":
            print(f"  irma spectra run {q(path)} -o spectrum.csv")
        elif name == "ncrystal.yaml":
            print(f"  irma ncrystal -o {q(out_dir)} {q(path)}")
    return 0


def _cmd_build(args) -> int:
    from irma.mlip.calculators import (
        CalculatorSpec, MlipDependencyError, canonicalize_spec,
        make_calculator)

    allowed = _allowed_potentials(args.allow_dev_backend)
    if args.potential not in allowed:
        return _err(f"unknown potential {args.potential!r}; choose from "
                    f"{', '.join(p for p in allowed if p != 'emt')}")
    if not os.path.isfile(args.structure):
        return _err(f"structure file not found: {args.structure}")

    # ALL numeric/option validation runs BEFORE any expensive work (checkpoint
    # load, relaxation), and the output target is preflighted first of all so
    # a doomed run fails in milliseconds, not minutes (review findings 1, 4).
    if args.threads is not None and args.threads < 1:
        return _err(f"--threads must be >= 1, got {args.threads}")
    if args.jobs < 1:
        return _err(f"--jobs must be >= 1, got {args.jobs}")
    if args.jitter_cycles < 0:
        return _err(f"--jitter-cycles must be >= 0, got {args.jitter_cycles}")
    if args.worker_threads < 1:
        return _err(f"--worker-threads must be >= 1, "
                    f"got {args.worker_threads}")
    for flag, value in (("--delta", args.delta), ("--fmax", args.fmax),
                        ("--snap-symmetry", args.snap_symmetry),
                        ("--dos-smearing", args.dos_smearing)):
        if value is not None and not (math.isfinite(value) and value > 0):
            return _err(f"{flag} must be a finite number > 0, got {value}")
    if args.nmax < 0:
        return _err(f"--nmax must be >= 0, got {args.nmax}")
    supercell_arg = _parse_supercell(args.supercell) \
        if args.supercell is not None else None
    mesh_arg = _parse_mesh(args.mesh) if args.mesh is not None else None
    from irma.mlip.bundle import preflight_bundle_outdir
    preflight_bundle_outdir(args.outdir, overwrite=args.overwrite)

    try:
        from ase.io import read as ase_read
    except ImportError:
        print("Error: the MLIP front end needs the irma[mlip] extra "
              "(pip install 'irma[mlip]')", file=sys.stderr)
        return 4
    try:
        atoms = ase_read(args.structure)
    except Exception as exc:
        return _err(f"could not read {args.structure}: {exc} "
                    f"(any ASE-readable format works)")

    # disorder hint: informational only, never a behavior switch, and never
    # allowed to abort a build
    natoms = len(atoms)
    if not args.disordered and natoms >= 100:
        try:
            from irma.mlip.relax import _spacegroup
            if _spacegroup(atoms, 1e-3).endswith("(1)"):
                print(f"  note: {natoms} atoms in P1 -- this looks like a "
                      f"disordered model; consider --disordered")
        except Exception:
            pass

    # the parent process (relaxation + serial force path) always gets the
    # full thread width; parallel workers re-clamp themselves to 1 (review
    # finding 3)
    threads = args.threads if args.threads is not None \
        else (os.cpu_count() or 1)
    # clamp native pools BEFORE anything imports torch (canonicalization
    # may): OpenMP runtimes size their pools at initialization, and a
    # pool born unclamped ignores later env changes -- the ZrO2 probe
    # measured the resulting stray threads spinning at >10 cores of sys
    # time for zero speedup
    from irma.mlip.calculators import _clamp_native_threads
    _clamp_native_threads(threads)
    spec = CalculatorSpec(args.potential, model=args.model, threads=threads)
    try:
        # pin floating model identities (pet-mad @version, dpa3 ::head)
        # BEFORE the spec fans out to workers and the cache fingerprint
        spec = canonicalize_spec(spec)
        calculator, calc_meta = make_calculator(spec)
    except MlipDependencyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        from irma.mlip.envs import ENV_REQUIREMENTS, is_dispatched
        if args.potential in ENV_REQUIREMENTS \
                and not is_dispatched(args.potential):
            print(f"  hint: `irma mlip env create {args.potential}` builds "
                  f"a dedicated environment for it and this install will "
                  f"use it automatically", file=sys.stderr)
        return 4
    from irma.mlip.calculators import POTENTIALS as _PUB
    if args.potential not in _PUB:
        calc_meta["dev_backend"] = True
    if calc_meta.get("license_note"):
        print(f"  NOTE: {calc_meta['license_note']}")
    covered = calc_meta.get("checkpoint_elements")
    if covered:
        from irma.mlip.calculators import missing_elements
        missing = missing_elements(covered, atoms.get_chemical_symbols())
        if missing:
            if hasattr(calculator, "close"):
                calculator.close()
            return _err(
                f"the {args.potential} checkpoint {calc_meta['checkpoint']} "
                f"covers {' '.join(covered)}; the structure contains "
                f"{' '.join(missing)}, which it was not trained on")
        n_int = calc_meta.get("checkpoint_num_interactions")
        print(f"  checkpoint: {calc_meta.get('checkpoint_model_class')}, "
              f"cutoff {calc_meta.get('checkpoint_r_max_A')} A"
              f"{f' x {n_int} interaction layers' if n_int else ''}, "
              f"elements {' '.join(covered)}")

    from irma.mlip.relax import relax
    print(f"  relaxing with {args.potential} "
          f"(fmax={args.fmax} eV/A, nmax={args.nmax}, "
          f"cell={'yes' if args.relax_cell else 'no'})")
    rr = relax(atoms, calculator, fmax=args.fmax, nmax=args.nmax,
               relax_cell=args.relax_cell,
               snap_symmetry=args.snap_symmetry,
               jitter_cycles=args.jitter_cycles)
    print(f"  relaxation: converged={rr.converged} "
          f"fmax={rr.fmax_achieved:.2e} eV/A in {rr.steps_taken} steps")
    if rr.snapped:
        print(f"  symmetry snap: positions moved <= "
              f"{rr.snap_max_shift_A:.2e} A onto the "
              f"{rr.spacegroup_after} orbits")
    if rr.symmetry_changed:
        print(f"  WARNING: spacegroup changed {rr.spacegroup_before} -> "
              f"{rr.spacegroup_after} during relaxation")
    if not rr.converged and not args.force:
        print(f"Error: relaxation did not converge (residual "
              f"{rr.fmax_achieved:.2e} > {args.fmax} eV/A after "
              f"{rr.steps_taken} steps); raise --nmax, loosen --fmax, or "
              f"pass --force to proceed anyway", file=sys.stderr)
        return 3

    # a dispatched relaxation calculator holds a live model copy in its
    # server subprocess; release it before the displacement loop spawns
    # its own workers (each loads another copy)
    if hasattr(calculator, "close"):
        calculator.close()

    if args.born:
        # fail BEFORE the force loop: a BORN file written for the
        # input symmetry stops matching when relaxation drifts the
        # positions off the exact Wyckoff sites (found live: the
        # error otherwise surfaces only after every displacement
        # was computed)
        from irma.mlip.bundle import check_born_rows
        problem = check_born_rows(args.born, rr.atoms)
        if problem:
            return _err(problem)

    from irma.mlip.phonons import compute_force_constants, supercell_matrix
    cellpar = rr.atoms.cell.cellpar()[:3]
    if supercell_arg is not None:      # explicit always wins, even disordered
        supercell = supercell_matrix(cellpar, supercell_arg)
    elif args.disordered:
        supercell = [1, 1, 1]          # the box is its own supercell
    else:
        supercell = supercell_matrix(cellpar, 12.0)
    mesh = mesh_arg if mesh_arg is not None else \
        ([1, 1, 1] if args.disordered else _default_mesh(cellpar))
    print(f"  supercell {tuple(supercell)}, quick-look mesh {tuple(mesh)}")

    pr = compute_force_constants(
        rr.atoms, spec, supercell=supercell, delta=args.delta,
        jobs=args.jobs, worker_threads=args.worker_threads,
        scratch_dir=os.path.join(args.outdir, "scratch"))

    from irma.mlip.bundle import write_bundle
    bundle = write_bundle(
        args.outdir, phonon_result=pr, relax_result=rr,
        calc_meta=calc_meta, args_used=vars(args) | {"argv": "irma mlip"},
        mesh=mesh, input_structure_path=args.structure,
        born_path=args.born, disordered=args.disordered,
        dos_sigma_mev=args.dos_smearing, overwrite=args.overwrite)
    print(f"  bundle written: {bundle.path}")

    if args.emit:
        return _do_emit(bundle, [t.strip() for t in args.emit.split(",")],
                        args)
    import shlex
    print("\nNext steps (template -- fill in the MAT number):")
    print(f"  irma mlip emit {shlex.quote(bundle.path)} "
          f"--to endf,spectra,ncrystal --mat 'SYM=<mat>'")
    print(f"  irma mlip validate {shlex.quote(bundle.path)}")
    return 0


def _cmd_emit(args) -> int:
    from irma.mlip.bundle import load_bundle, validate_bundle
    problems = validate_bundle(args.bundle)
    if problems:
        print(f"Error: {args.bundle} failed bundle validation:",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    bundle = load_bundle(args.bundle)
    return _do_emit(bundle, [t.strip() for t in args.to.split(",")], args)


def _cmd_env(args) -> int:
    from irma.mlip import envs

    if args.env_command == "list":
        table = envs.list_envs()
        if not table:
            print("no environments registered; create one with "
                  "`irma mlip env create <potential>`")
            return 0
        for potential in sorted(table):
            print(f"  {potential:12s} -> {table[potential]}")
        return 0

    if args.env_command == "create":
        if args.potential not in envs.ENV_REQUIREMENTS:
            return _err(f"unknown potential {args.potential!r}; choose "
                        f"from {', '.join(sorted(envs.ENV_REQUIREMENTS))}")
        print(f"  building a dedicated environment for "
              f"{args.potential} ...")
        python = envs.create_env(args.potential, dry_run=args.dry_run)
        if python:
            print(f"  done; builds with --potential {args.potential} now "
                  f"run in it automatically")
        return 0

    # remove
    if envs.remove_env(args.potential):
        print(f"  {args.potential} unregistered")
    else:
        print(f"  nothing registered for {args.potential}")
    return 0


def _cmd_validate(args) -> int:
    from irma.mlip.bundle import load_bundle, validate_bundle
    problems = validate_bundle(args.bundle)
    if problems:
        print(f"{args.bundle}: INVALID")
        for p in problems:
            print(f"  - {p}")
        return 2
    m = load_bundle(args.bundle).manifest
    print(f"{args.bundle}: OK")
    return _render_summary(m)


def _render_summary(m) -> int:
    try:
        _print_summary(m)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        print(f"  (manifest summary unavailable: malformed field {exc!r})")
        return 2
    return 0


def _print_summary(m):
    print(f"  potential: {m['calculator'].get('potential')} "
          f"({m['calculator'].get('checkpoint')})")
    print(f"  relaxed: converged={m['relaxation']['converged']} "
          f"fmax={m['relaxation']['fmax_achieved']:.2e} eV/A")
    print(f"  displacements: {m['displacements']['count']} "
          f"(delta={m['displacements']['delta']} A, "
          f"supercell {tuple(m['displacements']['supercell'])})")
    ph = m["phonons"]
    print(f"  phonons: freq_max={ph['freq_max_meV']:.2f} meV, "
          f"imaginary={ph['n_imaginary']} on mesh {tuple(ph['mesh'])}")
    print(f"  nac_embedded={m['nac_embedded']} "
          f"disordered={m['disordered']}")
    print(f"  fingerprint: {m['fingerprint'][:16]}...")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parser = _build_parser()
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            return int(exc.code or 0)
        try:
            if args.command == "build":
                return _cmd_build(args)
            if args.command == "emit":
                return _cmd_emit(args)
            if args.command == "env":
                return _cmd_env(args)
            return _cmd_validate(args)
        except Exception as exc:
            from irma.mlip.calculators import MlipDependencyError
            if isinstance(exc, MlipDependencyError):
                print(f"Error: {exc}", file=sys.stderr)
                return 4
            if not isinstance(exc, (ValueError, FileExistsError,
                                    FileNotFoundError, RuntimeError,
                                    OSError, KeyError, TypeError)):
                raise                     # genuinely unexpected: traceback

            # every expected failure surfaces as a clean message, never a
            # traceback: usage/validation/execution problems are exit 2
            return _err(str(exc))
    except KeyboardInterrupt:
        print("\ninterrupted (cached forces are kept; rerun to resume)",
              file=sys.stderr)
        return 130
