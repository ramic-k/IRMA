"""Reference-pin provenance for a baked pack.

IRMA is the reference: a pack records which IRMA build and
which run parameters produced it, so the plugin CI can regenerate and
byte/tolerance-compare against the mode-2 tape and catch any drift. These land as
``meta.*`` lines (schema v2) that the C++ loader ignores.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def irma_git_sha() -> str:
    """Short git SHA of the IRMA checkout that owns this module; ``unknown`` if
    git is unavailable or the source is not in a work tree."""
    repo_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return "unknown"
    sha = out.stdout.strip()
    if out.returncode != 0 or not sha:
        return "unknown"
    # Mark a dirty tree so a pack built from uncommitted edits is never mistaken
    # for a reproducible one.
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            sha += "-dirty"
    except Exception:
        pass
    return sha


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(
    *,
    phonopy_yaml: str | Path,
    mesh: tuple[int, int, int] | list[int],
    temperature_K: float,
    num_directions: int,
    multiphonon_num_directions: int,
    multiphonon_max_order: int | str,
    inelastic_mode: int = 2,
    min_phonon_energy_meV: float = 0.0,
    born: str | Path | None = None,
    force_constants: str | Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assemble the provenance metadata dict folded into a pack's ``metadata``."""
    from irma import __version__
    meta: dict[str, str] = {
        "irma_version": __version__,
        "irma_git_sha": irma_git_sha(),
        "irma_inelastic_mode": str(int(inelastic_mode)),
        "mesh": "x".join(str(int(m)) for m in mesh),
        "temperature_K": f"{float(temperature_K):.17g}",
        "num_directions": str(int(num_directions)),
        "multiphonon_num_directions": str(int(multiphonon_num_directions)),
        "multiphonon_max_order": str(multiphonon_max_order),
        # 0 = automatic floors only; a positive value means every term of the
        # pack was built from a truncated vibrational model
        "min_phonon_energy_meV": f"{float(min_phonon_energy_meV):.17g}",
    }
    meta["phonopy_yaml_sha256"] = file_sha256(phonopy_yaml)
    meta["phonopy_yaml_name"] = Path(phonopy_yaml).name
    # The BORN/NAC file changes NAC frequencies+eigenvectors and thus S(alpha,beta),
    # but is a SEPARATE input from the (hashed) phonopy_yaml. Hash it too so a CI
    # drift-check on a polar material catches a swapped/changed BORN. Only stamped
    # when present, so non-NAC packs (graphite, Be) are byte-unchanged.
    if born is not None:
        meta["born_sha256"] = file_sha256(born)
        meta["born_name"] = Path(born).name
    # the separate force-constants (or force-sets) file the model was built
    # from; None when the phonopy.yaml embeds them (it is hashed above)
    if force_constants is not None:
        meta["force_constants_sha256"] = file_sha256(force_constants)
        meta["force_constants_name"] = Path(force_constants).name
    if extra:
        meta.update({str(k): str(v) for k, v in extra.items()})
    return meta
