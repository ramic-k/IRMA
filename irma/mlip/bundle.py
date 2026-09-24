"""The phonon-model bundle: the output of the MLIP front end.

A bundle directory is the complete, self-contained deliverable of
`irma mlip build`:

    <outdir>/
      phonopy.yaml            relaxed cell + embedded force constants
                              (+ embedded NAC when a BORN file was given)
      structure_relaxed.vasp  POSCAR form of the relaxed cell
      manifest.json           provenance incl. artifact sha256s (schema 1)
      dos.dat                 total DOS on the quick-look mesh (meV, 2 col)
      dos.png                 written only when matplotlib is importable
      scratch/                per-displacement force cache (removable)

The force constants are embedded in phonopy.yaml because IRMA's resolver
prefers embedded force constants over every loose file and never falls back
to the cwd, so this one file feeds Card 6f (ENDF modes 1/2), the spectra
config, and the NCrystal exporter, and no separate force-constants file can
be mixed up with it.

NAC: a BORN file is parsed against the phonopy primitive and assigned to
nac_params for the duration of the save (then restored), so the yaml
embeds it. Emitted decks use
use_born=0 -- IRMA consumes embedded NAC directly. No BORN file is copied
into the bundle.

All heavy imports are function-level (core-clean module).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass

MANIFEST_SCHEMA = 1
THZ_TO_MEV = 4.13566553853599
IMAGINARY_FLOOR_MEV = -0.05     # below this a mode counts as imaginary

_ADVICE = ("imaginary modes usually mean the structure is not at a true "
           "minimum of this potential or the supercell truncates the force "
           "constants: tighten --fmax, enlarge --supercell, or try another "
           "potential; small flexural artifacts are common in layered "
           "materials")


@dataclass
class Bundle:
    path: str
    phonopy_yaml: str
    structure: str
    manifest: dict

    @property
    def fingerprint(self) -> str:
        return self.manifest.get("fingerprint", "")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_name(name, what) -> str:
    """Manifest file entries must be plain names inside the bundle dir."""
    if (not name or os.path.isabs(name) or os.sep in name
            or (os.altsep and os.altsep in name) or ".." in name):
        raise ValueError(f"unsafe {what} in bundle manifest: {name!r}")
    return name


def dos_grid_mev(freq_max_mev):
    """The uniform DOS grid [meV] from 0 to ``freq_max_mev``: at least 200
    points and a pitch of at most 0.5 meV. It is also the grid the deck
    cards take (``emit._uniform_rho``), so the emitted spectra are not
    resampled."""
    import numpy as np
    top = float(freq_max_mev)
    return np.linspace(0.0, top, max(200, int(np.ceil(top / 0.5)) + 1))


def _dos_and_census(phonon, mesh):
    """Quick-look mesh: total DOS (meV) + weight-aware imaginary census.

    The DOS is a histogram of the mesh modes (``irma.core.phonopy_io``'s
    ``mode_histogram``, the same rule as every IRMA DOS), per meV per cell:
    it integrates to the number of kept modes per cell. The modes are those
    ``mode_floor_mask`` keeps, so the Gamma translations and imaginary modes
    are left out, and flat bands (isolated molecular modes, every band of a
    Gamma-only mesh) count in full."""
    import numpy as np
    from irma.core.phonopy_io import mode_floor_mask, mode_histogram
    phonon.run_mesh(list(mesh))
    # symmetry-reduced mesh: weight each irreducible q-point's modes so the
    # DOS and the census count modes over the full requested mesh
    freqs = np.asarray(phonon.mesh.frequencies, float) * THZ_TO_MEV
    weights = np.asarray(phonon.mesh.weights, int)
    n_branches = freqs.shape[1]
    keep = mode_floor_mask(freqs.reshape(-1), phonon.mesh.qpoints, n_branches)
    e_mev = dos_grid_mev(freqs.max())
    rho = mode_histogram(freqs.reshape(-1)[keep],
                         np.repeat(weights, n_branches)[keep].astype(float),
                         e_mev) / (weights.sum() * (e_mev[1] - e_mev[0]))
    n_modes = int(weights.sum()) * n_branches
    n_imag = int((weights[:, None] * (freqs < IMAGINARY_FLOOR_MEV)).sum())
    census = {
        "mesh": [int(n) for n in mesh],
        "freq_min_meV": float(freqs.min()),
        "freq_max_meV": float(freqs.max()),
        "n_modes": n_modes,
        "n_modes_irreducible": int(freqs.size),
        "n_imaginary": n_imag,
    }
    if n_imag:
        census["advice"] = _ADVICE
    return e_mev, rho, census


def _write_dos(outdir, e_mev, rho):
    import numpy as np
    path = os.path.join(outdir, "dos.dat")
    np.savetxt(path, np.column_stack((e_mev, rho)),
               header="energy_meV  dos_per_meV (total, quick-look mesh)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return path, None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(e_mev, rho)
    ax.set_xlabel("energy (meV)")
    ax.set_ylabel("DOS (1/meV)")
    ax.set_title("MLIP phonon model: total DOS (quick-look)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(outdir, "dos.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return path, png


def _parse_born(phonon, born_path):
    """Parse BORN against the phonopy primitive -> nac_params dict."""
    from phonopy.file_IO import parse_BORN
    nac = parse_BORN(phonon.primitive, filename=str(born_path))
    if nac is None:
        raise ValueError(f"could not parse BORN file {born_path}")
    if not nac.get("factor"):
        # BORN files may omit the conversion factor; install phonopy's own
        # VASP-units value rather than a hand-rounded 14.4
        from phonopy.physical_units import get_physical_units
        units = get_physical_units()
        nac["factor"] = units.Hartree * units.Bohr
    return nac


def check_born_rows(born_path, relaxed_atoms) -> str | None:
    """Parse the BORN file against the relaxed structure, as phonopy will,
    before the displacement loop.

    phonopy's BORN format carries one tensor row per symmetry-independent
    atom, so a file written for the input symmetry stops matching when
    relaxation drifts the positions off the exact Wyckoff sites. Reporting
    that here turns an error after every displacement into an immediate one.
    Returns None when the file parses, else the error message.
    """
    from phonopy.file_IO import parse_BORN
    from phonopy.structure.atoms import PhonopyAtoms

    pa = PhonopyAtoms(symbols=relaxed_atoms.get_chemical_symbols(),
                      cell=relaxed_atoms.get_cell().array,
                      scaled_positions=relaxed_atoms.get_scaled_positions(),
                      masses=relaxed_atoms.get_masses())
    try:
        parse_BORN(pa, filename=born_path)
    except OSError as exc:
        return f"cannot read BORN file {born_path}: {exc}"
    except Exception as exc:
        return (f"BORN file {born_path} does not fit the RELAXED structure "
                f"({exc}). If relaxation drifted the positions off the ideal "
                f"Wyckoff sites, rebuild with --snap-symmetry.")
    return None


def preflight_bundle_outdir(outdir, overwrite=False):
    """Fail fast on a target that write_bundle would reject after the
    (potentially long) relax + displacement compute: a path that is a file,
    or a non-empty directory without overwrite. Scratch-only content is
    fine (that is the resume case). Call before any expensive work."""
    outdir = os.path.abspath(str(outdir))
    if os.path.isfile(outdir):
        raise FileExistsError(f"{outdir} is a file, not a directory")
    if os.path.isdir(outdir):
        existing = [n for n in os.listdir(outdir) if n != "scratch"]
        if existing and not overwrite:
            raise FileExistsError(
                f"{outdir} is not empty (found {sorted(existing)[:5]}...); "
                f"pass --overwrite to replace a previous bundle")
    return outdir


def write_bundle(outdir, *, phonon_result, relax_result, calc_meta,
                 args_used, mesh, input_structure_path=None, born_path=None,
                 disordered=False, overwrite=False,
                 progress=print) -> Bundle:
    """Serialize the model + provenance; see module docstring for layout.

    The serialized model is always phonon_result.phonon, so the metrics,
    the fingerprint and the saved yaml describe the same object.
    The manifest is written last (its presence defines a bundle).
    """
    from ase.io import write as ase_write
    from irma import __version__ as irma_version
    from irma.core.phonopy_io import phonopy_yaml_embeds_nac

    phonon = phonon_result.phonon
    outdir = preflight_bundle_outdir(outdir, overwrite)
    os.makedirs(outdir, exist_ok=True)
    for name in ("manifest.json", "dos.png"):       # a stale dos.png too
        if os.path.exists(os.path.join(outdir, name)):
            os.remove(os.path.join(outdir, name))

    yaml_path = os.path.join(outdir, "phonopy.yaml")
    prior_nac = phonon.nac_params
    try:
        if born_path is not None:
            phonon.nac_params = _parse_born(phonon, born_path)
        phonon.save(yaml_path, settings={"force_constants": True})
        # the DOS and census describe the saved model (NAC applied)
        e_mev, rho, census = _dos_and_census(phonon, mesh)
    finally:
        phonon.nac_params = prior_nac      # NAC never leaks into a later bundle
    nac_embedded = phonopy_yaml_embeds_nac(yaml_path)

    structure_path = os.path.join(outdir, "structure_relaxed.vasp")
    ase_write(structure_path, relax_result.atoms, direct=True, format="vasp")

    dos_path, png_path = _write_dos(outdir, e_mev, rho)
    if census["n_imaginary"]:
        progress(f"  WARNING: {census['n_imaginary']} imaginary mode(s) on "
                 f"the {tuple(mesh)} mesh (min {census['freq_min_meV']:.2f} "
                 f"meV). {_ADVICE}.")

    import ase
    import numpy as np
    import phonopy as _phonopy

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "kind": "irma-mlip-phonon-bundle",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "irma_version": irma_version,
        "disordered": bool(disordered),
        "input": {
            "args": dict(args_used),
            "structure_path": (os.path.abspath(input_structure_path)
                               if input_structure_path else None),
            "structure_sha256": (_sha256(input_structure_path)
                                 if input_structure_path
                                 and os.path.isfile(input_structure_path)
                                 else None),
        },
        "calculator": dict(calc_meta),
        "relaxation": {
            "snapped": relax_result.snapped,
            "snap_max_shift_A": relax_result.snap_max_shift_A,
            "jitter_cycles_used": getattr(relax_result,
                                          "jitter_cycles_used", 0),
            "converged": relax_result.converged,
            "fmax_target": relax_result.fmax_target,
            "fmax_initial": relax_result.fmax_initial,
            "fmax_achieved": relax_result.fmax_achieved,
            "fmax_atoms": relax_result.fmax_atoms,
            "steps": relax_result.steps_taken,
            "nmax": relax_result.nmax,
            "cell_relaxed": relax_result.cell_relaxed,
            "spacegroup_before": relax_result.spacegroup_before,
            "spacegroup_after": relax_result.spacegroup_after,
            "symmetry_changed": relax_result.symmetry_changed,
            "symprec": relax_result.symprec,
        },
        "displacements": {
            "count": phonon_result.n_displacements,
            "from_cache": phonon_result.n_from_cache,
            "delta": phonon_result.delta,
            "jobs": phonon_result.jobs,
            "supercell": list(phonon_result.supercell),
            "wall_s": phonon_result.wall_s,
        },
        "fc_symmetrization": {
            "asr_drift_before": phonon_result.asr_drift_before,
            "correction_max_abs": phonon_result.symmetrization_delta,
        },
        "phonons": census,
        "nac_embedded": bool(nac_embedded),
        "born": {
            "path": os.path.abspath(born_path) if born_path else None,
            "sha256": (_sha256(born_path)
                       if born_path and os.path.isfile(born_path) else None),
        },
        "fingerprint": phonon_result.fingerprint,
        "versions": {
            "ase": ase.__version__,
            "phonopy": _phonopy.__version__,
            "numpy": np.__version__,
        },
        "files": {
            "phonopy_yaml": os.path.basename(yaml_path),
            "structure": os.path.basename(structure_path),
            "dos": os.path.basename(dos_path),
            "dos_png": os.path.basename(png_path) if png_path else None,
        },
        "sha256": {
            "phonopy_yaml": _sha256(yaml_path),
            "structure": _sha256(structure_path),
            "dos": _sha256(dos_path),
            "dos_png": _sha256(png_path) if png_path else None,
        },
    }
    manifest_tmp = os.path.join(outdir, "manifest.json.tmp")
    with open(manifest_tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(manifest_tmp, os.path.join(outdir, "manifest.json"))

    return Bundle(path=outdir,
                  phonopy_yaml=os.path.join(outdir, "phonopy.yaml"),
                  structure=os.path.join(outdir, "structure_relaxed.vasp"),
                  manifest=manifest)


def load_bundle(path) -> Bundle:
    """Open an existing bundle directory (no model loading; cheap).

    Manifest file entries are constrained to plain names inside the bundle
    directory (no separators, '..', or absolute paths).
    """
    path = os.path.abspath(str(path))
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"{path} is not a bundle: no manifest.json (expected a directory "
            f"produced by `irma mlip build`)")
    manifest = json.load(open(manifest_path))
    if not isinstance(manifest, dict) \
            or manifest.get("kind") != "irma-mlip-phonon-bundle":
        raise ValueError(f"{manifest_path} is not an irma-mlip bundle manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{manifest_path}: missing/invalid 'files' section")
    yaml_name = _safe_name(files.get("phonopy_yaml"), "phonopy_yaml")
    struct_name = _safe_name(files.get("structure"), "structure")
    return Bundle(
        path=path,
        phonopy_yaml=os.path.join(path, yaml_name),
        structure=os.path.join(path, struct_name),
        manifest=manifest)


def validate_bundle(path) -> list:
    """Check a bundle; returns a list of problems (empty = valid).

    Artifact presence and sha256, then a phonopy reload under IRMA's pinned
    primitive policy with the embedded-FC and NAC states checked against
    the manifest. The bundle is treated as untrusted: phonopy's YAML parser
    executes ``!!python/`` tags, so the phonopy.yaml is scanned
    (reject_unsafe_phonopy_yaml) before any phonopy parse runs.
    """
    try:
        bundle = load_bundle(path)
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError,
            json.JSONDecodeError) as exc:
        return [str(exc)]
    m = bundle.manifest
    if m.get("schema") != MANIFEST_SCHEMA:
        return [f"manifest schema {m.get('schema')!r} != {MANIFEST_SCHEMA}"]
    problems = []
    try:
        for label, name in m["files"].items():
            if name is None:
                continue
            target = os.path.join(bundle.path, _safe_name(name, label))
            if not os.path.isfile(target):
                problems.append(f"missing file: {target}")
            elif _sha256(target) != m["sha256"].get(label):
                problems.append(f"{label}: sha256 mismatch (file changed "
                                f"after the bundle was written)")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        problems.append(f"malformed manifest: {exc}")
    if problems:
        return problems

    from irma.core.phonopy_io import (
        phonopy_yaml_embeds_force_constants, pinned_primitive_matrix_kwargs,
        reject_unsafe_phonopy_yaml)
    try:
        reject_unsafe_phonopy_yaml(bundle.phonopy_yaml)
    except ValueError as exc:
        return [str(exc)]
    except OSError as exc:
        return [f"could not scan {bundle.phonopy_yaml}: {exc}"]
    if not phonopy_yaml_embeds_force_constants(bundle.phonopy_yaml):
        problems.append("phonopy.yaml does not embed force constants")
    try:
        import phonopy

        from irma.core.phonopy_io import isolated_phonopy_cwd

        # the reload must see only the bundle: phonopy.load picks up a
        # stray ./BORN from the working directory
        with isolated_phonopy_cwd():
            ph = phonopy.load(
                bundle.phonopy_yaml, log_level=0,
                **pinned_primitive_matrix_kwargs(bundle.phonopy_yaml))
        if bool(m.get("nac_embedded")) != (ph.nac_params is not None):
            problems.append("manifest nac_embedded disagrees with the "
                            "reloaded model")
    except Exception as exc:
        problems.append(f"phonopy reload failed: {exc}")
    return problems
