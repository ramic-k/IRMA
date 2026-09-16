"""Reference-pin provenance for a baked pack.

IRMA-as-reference is non-negotiable: a pack must record exactly which IRMA build and
which run parameters produced it, so the SP3 plugin CI can regenerate and
byte/tolerance-compare against the mode-2 tape and catch any drift. These land as
``meta.*`` lines (schema v2) that the C++ loader ignores.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def irma_version() -> str:
    """Best-effort IRMA version string; never raises.

    Prefers the live source (``irma.__version__``, derived from the CHANGELOG)
    over installed dist metadata: an editable/dev checkout can carry stale or
    *duplicate* ``irma-*.dist-info`` dirs (seen: 0.9.0 + 0.17.0 alongside a live
    0.18.0 source), which makes ``importlib.metadata.version`` non-deterministic
    and would stamp a pack with a version its producing code never had.
    """
    try:
        import irma

        v = str(getattr(irma, "__version__", "") or "")
        if v and v != "unknown":
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("irma")
    except Exception:
        return "unknown"


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
    """SHA-256 of a file's bytes; ``unknown`` if it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return "unknown"


def collect_provenance(
    *,
    phonopy_yaml: str | Path | None,
    mesh: tuple[int, int, int] | list[int],
    temperature_K: float,
    num_directions: int,
    multiphonon_num_directions: int,
    multiphonon_max_order: int | str,
    inelastic_mode: int = 2,
    min_phonon_energy_meV: float = 0.0,
    born: str | Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assemble the provenance metadata dict folded into a pack's ``metadata``."""
    meta: dict[str, str] = {
        "irma_version": irma_version(),
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
    if phonopy_yaml is not None:
        meta["phonopy_yaml_sha256"] = file_sha256(phonopy_yaml)
        meta["phonopy_yaml_name"] = Path(phonopy_yaml).name
    # The BORN/NAC file changes NAC frequencies+eigenvectors and thus S(alpha,beta),
    # but is a SEPARATE input from the (hashed) phonopy_yaml. Hash it too so a CI
    # drift-check on a polar material catches a swapped/changed BORN. Only stamped
    # when present, so non-NAC packs (graphite, Be) are byte-unchanged.
    if born is not None:
        meta["born_sha256"] = file_sha256(born)
        meta["born_name"] = Path(born).name
    if extra:
        meta.update({str(k): str(v) for k, v in extra.items()})
    return meta
