"""Calculator factory for the MLIP front end.

A CalculatorSpec is a small picklable value object; make_calculator() turns it
into a live ASE calculator plus a provenance dict. The spec/factory split
exists for the parallel displacement loop: worker processes receive the spec
(never a live calculator) and build their own instance in a top-level
initializer, so nothing unpicklable crosses the process boundary.

CPU only, by project decision. dtype policy follows each potential's
reference usage in INSPIRED: MatterSim and SevenNet run float32, MACE and
MACE-OFF run float64, ORB uses its "float32-high" precision mode, PET-MAD
and DPA-3 use their model defaults (float32 and float64 respectively).
ORB is pinned to the conservative-forces variant, and PET-MAD is never
built with upet's opt-in non_conservative mode: non-conservative graph
potentials return forces that are not the gradient of an energy, which
breaks the harmonic force-constant assumption.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace

POTENTIALS = ("mattersim", "orb", "sevennet", "mace", "mace-off",
              "pet-mad", "dpa3", "nequip", "grace")

# "emt" is a hidden test backend (ASE's built-in EMT): it lets the
# displacement loop and its spawn pool be exercised cross-process in CI with
# no torch and no checkpoint downloads. Not part of the public POTENTIALS.
_VALID = POTENTIALS + ("emt",)

# potential -> (pip package, top-level import used for the availability check)
_PACKAGES = {
    "mattersim": ("mattersim", "mattersim"),
    "orb": ("orb-models", "orb_models"),
    "sevennet": ("sevenn", "sevenn"),
    "mace": ("mace-torch", "mace"),
    "mace-off": ("mace-torch", "mace"),
    "pet-mad": ("upet", "upet"),
    "dpa3": ("deepmd-kit", "deepmd"),
    "nequip": ("nequip", "nequip"),
    "grace": ("tensorpotential", "tensorpotential"),
    "emt": ("ase", "ase"),
}

_DEFAULT_MODELS = {
    "mattersim": "MatterSim-v1.0.0-5M.pth",
    "orb": "orb_v3_conservative_inf_omat",
    "sevennet": "7net-mf-ompa",
    "mace": "medium-mpa-0",
    "mace-off": "medium",
    "pet-mad": "pet-mad-s",
    "dpa3": "DPA-3.1-3M",
    "nequip": "mir-group/NequIP-OAM-L:0.1",
    "grace": "GRACE-2L-OAM",
}

# DPA checkpoints are multitask: one backbone, one fitting net per training
# dataset ("head"). The head is part of the physical model identity, so it
# rides in the model string as MODEL::HEAD and lands in the fingerprint.
# MP_traj_v024_alldata_mixu is the MPtrj PBE(+U) head, the same reference
# family as the other universal potentials here.
DPA_DEFAULT_HEAD = "MP_traj_v024_alldata_mixu"
_DPA_HF_REPO = "deepmodelingcommunity/DPA"
_DPA_HF_FILES = {"DPA-3.1-3M": "DPA-3.1-3M.pt"}
# Supply-chain pin for the DPA download (security finding): torch
# checkpoints are pickle-based, so fetching a repo's MUTABLE default
# branch means a retagged upstream is code execution on the next cache
# miss. The revision pins the exact HF commit this alias was vetted at,
# and the sha256 is verified on every resolve (cached file included, so a
# tampered cache entry is caught too). Values recorded from
# huggingface.co/deepmodelingcommunity/DPA at pinning time (2026-07-30;
# repo last modified 2025-08-06).
_DPA_HF_REVISION = "cac67b9c29b05d5dcd81c17e7be49bd433c887e8"
_DPA_HF_SHA256 = {
    "DPA-3.1-3M.pt":
        "86dd3a804d78ca5d203ebf98747e8f16dff9713ba8950097ceb760b161e19907",
}

# Best-effort pins for upet checkpoints (security finding): the lab-cosmo/
# upet repo has NO release tags — version-named files accumulate on the
# mutable main branch — so the revision cannot be pinned without breaking
# every version released after this code shipped. Files whose sha256 was
# recorded at pinning time (2026-07-30) are verified after download; a
# version not listed here downloads unverified, and its content sha256 is
# recorded in the bundle provenance (calculator.checkpoint_sha256) so the
# build stays auditable.
_PET_KNOWN_SHA256 = {
    "pet-mad-s-v1.0.2.ckpt":
        "c7173d5990fce821ffbe75be30771d352a8af2e1a880257cd737877c10d72f30",
    "pet-mad-s-v1.1.0.ckpt":
        "8d5e1588c70f7a4998940e7b535ab84aeb42c0f2253c889d4af23a1a8c6fd553",
    "pet-mad-s-v1.5.0.ckpt":
        "9f55849d1e757ceefdd2d7ebd2374b543d7fdb8972d0b96fb9bc890eff3b6029",
    "pet-mad-xs-v1.5.0.ckpt":
        "50a16b14bbb67add3e8bc1c37edb94c1f4b6334b01c027998ff256c14fc14a3b",
}


def _verify_checkpoint_sha256(path: str, expected: str, origin: str) -> str:
    """Reject (and remove) a downloaded checkpoint whose hash is wrong.

    The file is deleted on mismatch so a rerun cannot silently pick the
    bad copy off the cache fast path.
    """
    actual = _checkpoint_sha256(path)
    if actual != expected:
        try:
            os.remove(path)
        except OSError:
            pass
        raise RuntimeError(
            f"checkpoint integrity check FAILED for {origin}: sha256 "
            f"{actual} != pinned {expected}; the upstream file changed "
            f"after this IRMA release pinned it (or the download was "
            f"tampered with) — refusing to load it. The file has been "
            f"removed from the cache.")
    return path


def _mlip_cache_dir() -> str:
    return os.environ.get(
        "IRMA_MLIP_CACHE", os.path.expanduser("~/.cache/irma-mlip"))


# The newer mace-foundations checkpoint generation (MACE-OMAT-0, the
# MACE-MATPES pair, the multi-head MACE-MH line) is released under the
# Academic Software License, unlike the MIT-licensed MP/MPA line that
# includes the default medium-mpa-0. The build stamps a note into the
# manifest and the CLI prints it, so the restriction travels with the
# results instead of living only in a download click-through.
_MACE_ASL_HINTS = ("omat", "matpes", "-mh-")


def _mace_license_note(model) -> str | None:
    name = os.path.basename(str(model)).lower()
    if any(h in name for h in _MACE_ASL_HINTS) or name.startswith("mh-"):
        return ("this MACE foundation checkpoint is distributed under the "
                "Academic Software License: research-only, non-commercial; "
                "commercial use needs a separate license from the MACE "
                "authors")
    return None


class MlipDependencyError(ImportError):
    """A potential package needed by the requested MLIP is not installed."""


# Set to True by force_server.py after it loads this file standalone in a
# foreign environment: the server must NEVER re-enter dispatch (infinite
# recursion) and must never import the irma package (provisioned envs
# only carry ase + the potential).
_DISPATCH_DISABLED = False


def _dispatch_interpreter(potential: str) -> str | None:
    """The foreign interpreter to dispatch to, or None for in-process."""
    if _DISPATCH_DISABLED or potential == "emt":
        return None
    from irma.mlip import envs
    if envs.is_dispatched(potential):
        return envs.registered_interpreter(potential)
    return None


def _clamp_native_threads(threads: int):
    """Force the native thread pools to the requested width.

    torch.set_num_threads alone does not bound OMP/BLAS pools that libraries
    spin up independently, and irma/__init__ only setdefault()s these
    variables, so an inherited OMP_NUM_THREADS=8 would silently oversubscribe
    every pool worker (the measured >11x slowdown documented in
    irma.core.noncubic_workers). Environment variables are forced BEFORE the
    potential package imports; threadpoolctl additionally clamps pools that
    were already created.
    """
    value = str(int(threads))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        os.environ[var] = value
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"   # grace/TensorFlow
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=int(threads))
    except ImportError:
        pass


@dataclass(frozen=True)
class CalculatorSpec:
    """Picklable recipe for building an MLIP calculator.

    potential: one of POTENTIALS.
    model: checkpoint name or path; None selects the potential's default.
    threads: native torch threads for force calls made by THIS process
        (the parallel displacement loop passes 1 per worker; the serial
        path passes the user's --threads).
    """
    potential: str
    model: str | None = None
    threads: int = 1

    def __post_init__(self):
        if self.potential not in _VALID:
            raise ValueError(
                f"unknown potential {self.potential!r}; choose from "
                f"{', '.join(POTENTIALS)}")
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")


def _require(potential: str):
    pip_name, module = _PACKAGES[potential]
    try:
        __import__(module)
    except ImportError as exc:
        raise MlipDependencyError(
            f"potential {potential!r} needs the {pip_name!r} package "
            f"(pip install {pip_name}), alongside the irma[mlip] extra; "
            f"underlying error: {exc}") from exc


def _package_version(potential: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    pip_name, module = _PACKAGES[potential]
    for name in (pip_name, module):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


def _checkpoint_sha256(model) -> str | None:
    if model and isinstance(model, (str, os.PathLike)) and os.path.isfile(model):
        h = hashlib.sha256()
        with open(model, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    return None


def _split_pet_model(model) -> tuple[str, str | None]:
    """'alias[@version]' -> (alias, version|None). Paths are not split.

    The literal version 'latest' means UNPINNED, exactly like an absent
    version: it must never survive into a spec, manifest, or fingerprint,
    where it would silently alias different upstream releases (review
    finding: a post-release rerun would reuse the older release's force
    cache under the same identity).
    """
    text = str(model)
    if os.path.isfile(text):
        return text, None
    base, sep, version = text.partition("@")
    if not base or (sep and not version):
        raise ValueError(
            f"malformed pet-mad model {text!r}; expected NAME or "
            f"NAME@VERSION (e.g. pet-mad-s@1.5.0) or a checkpoint path")
    if version.lower() == "latest":
        version = ""
    return base, (version or None)


def _resolve_pet_version(alias: str, version: str | None) -> str:
    """Pin a upet alias to a concrete released version string.

    'latest' is resolved HERE, once, so the pinned version can enter the
    spec/manifest/fingerprint; workers rebuilt from a pinned spec never
    repeat the network lookup.
    """
    from upet.calculator import UPET_AVAILABLE_MODELS, upet_resolve_model
    if alias.lower() not in UPET_AVAILABLE_MODELS:
        raise ValueError(
            f"unknown pet-mad model {alias!r}; choose from "
            f"{', '.join(UPET_AVAILABLE_MODELS)} or pass a checkpoint path")
    base, size = alias.rsplit("-", 1)
    try:
        _, resolved = upet_resolve_model(base, requested_size=size,
                                         requested_version=version)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve the {alias!r} model version from "
            f"HuggingFace ({exc}); pass an explicit --model "
            f"'{alias}@<version>' or a local checkpoint path to work "
            f"offline") from exc
    return str(resolved)


def _split_dpa_model(model) -> tuple[str, str | None]:
    """'base[::head]' -> (base, head).

    A string naming an existing file is taken whole (paths may legally
    contain '::'); otherwise the split is at the LAST '::' so only the
    head, which never contains '::', is peeled off. A bare known alias
    gets DPA_DEFAULT_HEAD; a bare filesystem path gets head=None (a
    frozen single-task model needs no head).
    """
    text = str(model)
    if os.path.isfile(text):
        return text, None
    base, sep, head = text.rpartition("::")
    if not sep:
        return text, (DPA_DEFAULT_HEAD if text in _DPA_HF_FILES else None)
    if not base or not head:
        raise ValueError(
            f"malformed dpa3 model {text!r}; expected MODEL or MODEL::HEAD "
            f"(e.g. DPA-3.1-3M::{DPA_DEFAULT_HEAD})")
    return base, head


def _resolve_dpa_checkpoint(base: str) -> str:
    """Return a local checkpoint file for a dpa3 model string.

    A known alias is downloaded once into the irma-mlip cache as a real
    file (deepmd detects its backend from the file extension, which the
    huggingface blob store strips); an existing cached file is preferred so
    reruns and pool workers stay offline and deterministic. The download
    is pinned to a vetted HF revision and the file's sha256 is verified
    against the pinned value on EVERY resolve — cached copies included —
    so neither a mutated upstream branch nor a tampered cache entry can
    feed deepmd's pickle-based loader (security finding). A user-supplied
    local path is the user's own trust decision and is not checked.
    """
    if os.path.isfile(base):
        return base
    filename = _DPA_HF_FILES.get(base)
    if filename is None:
        raise ValueError(
            f"unknown dpa3 model {base!r}; choose from "
            f"{', '.join(sorted(_DPA_HF_FILES))} or pass a local "
            f".pt/.pth checkpoint path")
    expected = _DPA_HF_SHA256[filename]
    cached = os.path.join(_mlip_cache_dir(), filename)
    if os.path.isfile(cached):
        return _verify_checkpoint_sha256(
            cached, expected, f"cached {filename}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MlipDependencyError(
            "downloading the DPA checkpoint needs the 'huggingface_hub' "
            "package (pip install huggingface_hub), or pass a local "
            "checkpoint path") from exc
    path = hf_hub_download(_DPA_HF_REPO, filename,
                           revision=_DPA_HF_REVISION,
                           local_dir=_mlip_cache_dir())
    return _verify_checkpoint_sha256(
        path, expected, f"{_DPA_HF_REPO}/{filename}@{_DPA_HF_REVISION}")


def _pet_checkpoint_filename(alias: str, version: str) -> str:
    # upet's HF layout: models/{model}-{size}-v{version}.ckpt, where our
    # alias is already "{model}-{size}"
    return f"{alias}-v{version}.ckpt"


def _download_pet_checkpoint(alias: str, version: str) -> str:
    """Fetch a released pet-mad checkpoint into the irma-mlip cache.

    An already-cached file is returned without touching the network, so
    pinned reruns work offline.

    Trust note (security finding): the upet repo carries no release tags,
    so this fetch necessarily resolves the MUTABLE main branch — the
    version pin lives only in the FILENAME. Versions whose sha256 was
    recorded in _PET_KNOWN_SHA256 are verified (cached copies included);
    an unlisted version downloads unverified, and its content sha256 is
    recorded in the bundle provenance (calculator.checkpoint_sha256) for
    after-the-fact audit. A local checkpoint path bypasses the download
    entirely and is the fully user-controlled route.
    """
    filename = _pet_checkpoint_filename(alias, version)
    expected = _PET_KNOWN_SHA256.get(filename)
    cached = os.path.join(_mlip_cache_dir(), "models", filename)
    if os.path.isfile(cached):
        if expected:
            _verify_checkpoint_sha256(cached, expected, f"cached {filename}")
        return cached
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("lab-cosmo/upet", filename, subfolder="models",
                           local_dir=_mlip_cache_dir())
    if expected:
        _verify_checkpoint_sha256(path, expected,
                                  f"lab-cosmo/upet/models/{filename}")
    return path


_NEQUIP_ZOO_PREFIX = "nequip.net:"


def _normalize_nequip_zoo_id(model) -> str:
    """'[nequip.net:][group/]Name:version' -> 'group/Name:version'."""
    import re
    text = str(model)
    if text.startswith(_NEQUIP_ZOO_PREFIX):
        text = text[len(_NEQUIP_ZOO_PREFIX):]
    if "/" not in text:
        text = "mir-group/" + text
    hint = ""
    if os.sep in str(model):
        hint = (f"; {model!r} looks like a file path, but no such file "
                f"exists")
    # the exact grammar the nequip model zoo accepts: one group, one
    # name, one version, no empty parts and no stray separators
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
                        r":[A-Za-z0-9_.\-]+", text):
        raise ValueError(
            f"malformed nequip model {model!r}; expected "
            f"[nequip.net:][group/]NAME:VERSION (e.g. "
            f"mir-group/NequIP-OAM-L:0.1) or a compiled "
            f".nequip.pth/.nequip.pt2 path{hint}")
    return text


def _nequip_artifact_path(zoo_id: str, ext: str) -> str:
    # the readable part is lossy ('a-b/c:1' and 'a/b-c:1' would collide),
    # so a short hash of the exact id keeps distinct models distinct.
    # '-st1' marks the single-thread compile profile: AOTInductor bakes
    # the compile-time thread count into the kernels, and an artifact
    # compiled multi-threaded ran 5x SLOWER per call (measured on ZrO2,
    # 25.0 s vs 5.1 s) -- profile-less artifacts from older builds must
    # never be reused
    safe = zoo_id.replace("/", "-").replace(":", "-")
    tag = hashlib.sha256(zoo_id.encode()).hexdigest()[:8]
    return os.path.join(_mlip_cache_dir(), "models",
                        f"{safe}-{tag}-ase-st1{ext}")


def _nequip_mode() -> tuple[str, str]:
    """(compile mode, artifact extension) for the RUNNING torch.

    torch >= 2.10 dropped TorchScript entirely, so .nequip.pth artifacts
    compiled under an older torch are unloadable there and must not be
    picked up from the cache.
    """
    import re

    import torch
    nums = []
    for part in torch.__version__.split(".")[:2]:
        m = re.match(r"\d+", part)
        nums.append(int(m.group()) if m else 0)
    if tuple(nums) >= (2, 10):
        return "aotinductor", ".nequip.pt2"
    return "torchscript", ".nequip.pth"


# Bootstrap for the nequip-compile subprocess: one torch thread, because
# the thread count is baked into the compiled kernels.
_NEQUIP_COMPILE_BOOTSTRAP = """\
import torch
torch.set_num_threads(1)
from nequip.scripts.compile import main
main()
"""


def _compile_nequip_model(zoo_id: str) -> str:
    """nequip-compile a model-zoo entry into the irma-mlip cache.

    The compiled artifact is what every worker loads (offline, via
    NequIPCalculator.from_compiled_model); torch >= 2.10 dropped
    TorchScript support, so the mode follows the running torch.
    """
    import subprocess
    import sys as _sys

    mode, ext = _nequip_mode()
    out = _nequip_artifact_path(zoo_id, ext)
    if os.path.isfile(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # per-process tmp name (concurrent builds must not share one), the
    # required extension kept, and always cleaned up; publication via
    # atomic os.replace so racers waste work but never corrupt
    tmp = out[:-len(ext)] + f".compiling-{os.getpid()}" + ext
    cmd = [_sys.executable, "-c", _NEQUIP_COMPILE_BOOTSTRAP,
           f"{_NEQUIP_ZOO_PREFIX}{zoo_id}", tmp,
           "--mode", mode, "--device", "cpu", "--target", "ase"]
    # single-thread compile environment: the thread count is baked into
    # the AOT kernels (see _nequip_artifact_path)
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = "1"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env)
        if result.returncode != 0 or not os.path.isfile(tmp):
            tail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"nequip-compile of {zoo_id!r} failed (exit "
                f"{result.returncode}):\n{tail[-2000:]}")
        os.replace(tmp, out)
    finally:
        if os.path.isfile(tmp):
            os.unlink(tmp)
    return out


# --- MACE-family checkpoints -------------------------------------------------

MACE_FAMILY = ("mace", "mace-off")
MACE_OFF_NOTE = (
    "the MACE-OFF checkpoints are distributed under the Academic Software "
    "License: research-only, non-commercial; commercial use needs a "
    "separate license")


def inspect_mace_model(loaded, origin: str) -> dict:
    """Describe a deserialized MACE module (class, cutoff, elements, heads).

    Raises ValueError when the object is not a full serialized MACE model:
    a state dict or a training checkpoint lacks the attributes MACE's own
    calculator needs, and a file extension proves nothing about the
    contents.
    """
    if not (hasattr(loaded, "atomic_numbers") and hasattr(loaded, "r_max")
            and callable(getattr(loaded, "parameters", None))):
        raise ValueError(
            f"{origin} is not a serialized MACE model (a state dict or a "
            f"training checkpoint, perhaps); MACE writes the full model as "
            f"<name>.model at the end of training, pass that file")
    heads = getattr(loaded, "heads", None)
    n_int = getattr(loaded, "num_interactions", None)
    return {
        "model_class": type(loaded).__name__,
        "r_max_A": float(loaded.r_max),
        "num_interactions": int(n_int) if n_int is not None else None,
        "atomic_numbers": [int(z) for z in loaded.atomic_numbers],
        "heads": [str(h) for h in heads] if heads is not None else None,
    }


def _mace_calculator_from_file(path: str):
    """(MACECalculator, description) for a checkpoint FILE, read once: the
    recorded sha256 is that of the bytes deserialized. Genuine multi-head
    checkpoints are refused: IRMA has no head selection, and MACE would
    otherwise pick a head silently.

    Trust note: a MACE model file is a pickle, and loading it runs code
    from the file. The user chose the file; nothing here makes an
    untrusted file safe.
    """
    import io

    import torch
    from mace.calculators import MACECalculator
    with open(path, "rb") as fh:
        data = fh.read()
    digest = hashlib.sha256(data).hexdigest()
    loaded = torch.load(io.BytesIO(data), map_location="cpu",
                        weights_only=False)
    info = inspect_mace_model(loaded, f"checkpoint {path}")
    if info["heads"] and len(info["heads"]) > 1:
        raise ValueError(
            f"checkpoint {path} is a multi-head MACE model (heads: "
            f"{', '.join(info['heads'])}); IRMA does not select a head, "
            f"so it cannot tell which fitting net would produce the "
            f"forces. Export a single-head model for the head you want.")
    calc = MACECalculator(models=loaded, default_dtype="float64",
                          device="cpu")
    info["sha256"] = digest
    return calc, info


def missing_elements(covered_symbols, structure_symbols) -> list[str]:
    """Element symbols present in the structure but not in the
    checkpoint's table, sorted; empty when the checkpoint covers all."""
    covered = set(covered_symbols)
    return sorted({str(s) for s in structure_symbols} - covered)


def _finish_mace(calc):
    """Provenance of a constructed MACE-family calculator."""
    from ase.data import chemical_symbols
    first = calc.models[0]
    n_int = getattr(first, "num_interactions", None)
    return calc, {
        "checkpoint_model_class": type(first).__name__,
        "checkpoint_r_max_A": float(calc.r_max),
        "checkpoint_num_interactions": (int(n_int) if n_int is not None
                                        else None),
        "checkpoint_elements": [chemical_symbols[int(z)] for z in calc.z_table.zs],
        "checkpoint_heads": (list(calc.available_heads)
                             if getattr(calc, "available_heads", None)
                             else None),
        "checkpoint_head": getattr(calc, "head", None),
    }


def canonicalize_spec(spec: CalculatorSpec) -> CalculatorSpec:
    """Pin floating model identities in a spec, before it fans out.

    pet-mad aliases resolve to a LOCAL CHECKPOINT PATH (upet's standard
    'name-size-vX.Y.Z.ckpt' naming, so the version stays legible): both
    the version listing and the checkpoint download in upet's alias path
    are network calls that every spawned worker would otherwise repeat,
    and a mid-run upstream release could split the build across versions.
    dpa3 aliases get their '::head' made explicit and a MACE-family
    checkpoint FILE its absolute path. The returned spec is what
    relaxation, every pool worker, and the cache fingerprint all see.
    Other potentials pass through unchanged.
    """
    if spec.potential in MACE_FAMILY:
        if spec.model and os.path.isfile(str(spec.model)):
            path = os.path.abspath(str(spec.model))
            return spec if path == spec.model else replace(spec, model=path)
        return spec
    if spec.potential == "pet-mad":
        model = spec.model or _DEFAULT_MODELS["pet-mad"]
        if os.path.isfile(str(model)):
            return spec
        alias, version = _split_pet_model(model)
        if version is not None:
            cached = os.path.join(_mlip_cache_dir(), "models",
                                  _pet_checkpoint_filename(alias, version))
            if os.path.isfile(cached):        # pinned rerun: offline, hash checked
                return replace(spec, model=_download_pet_checkpoint(alias, version))
        interp = _dispatch_interpreter("pet-mad")
        if interp:
            # resolution needs the upet package; run it in the registered
            # env (the checkpoint lands in the shared irma-mlip cache)
            from irma.mlip import envs
            return replace(spec,
                           model=envs.remote_canonicalize(spec, interp))
        _require("pet-mad")
        version = _resolve_pet_version(alias, version)
        return replace(spec, model=_download_pet_checkpoint(alias, version))
    if spec.potential == "dpa3":
        model = spec.model or _DEFAULT_MODELS["dpa3"]
        base, head = _split_dpa_model(model)
        if not os.path.isfile(base) and base not in _DPA_HF_FILES:
            raise ValueError(
                f"unknown dpa3 model {base!r}; choose from "
                f"{', '.join(sorted(_DPA_HF_FILES))} or pass a local "
                f".pt/.pth checkpoint path")
        return replace(spec, model=f"{base}::{head}" if head else base)
    if spec.potential == "nequip":
        model = spec.model or _DEFAULT_MODELS["nequip"]
        if os.path.isfile(str(model)):
            return spec
        zoo_id = _normalize_nequip_zoo_id(model)
        # dispatch decision comes BEFORE any torch-dependent lookup: the
        # artifact format follows the EXECUTING environment's torch, and
        # this environment may not have torch at all (review finding)
        interp = _dispatch_interpreter("nequip")
        if interp:
            from irma.mlip import envs
            return replace(spec,
                           model=envs.remote_canonicalize(spec, interp))
        _require("nequip")
        return replace(spec, model=_compile_nequip_model(zoo_id))
    return spec


def resolved_checkpoint_identity(spec: CalculatorSpec) -> str:
    """The string that stands for the checkpoint in cache fingerprints.

    Local files are content-hashed. pet-mad aliases resolve to
    'alias@version' (a released version names immutable weights); dpa3
    aliases resolve to the downloaded file's content hash plus the head,
    because the head selects a different fitting net from the same file.
    """
    model = spec.model or _DEFAULT_MODELS.get(spec.potential, "builtin")
    if spec.potential == "pet-mad" and not os.path.isfile(str(model)):
        alias, version = _split_pet_model(model)
        if version is None:
            _require("pet-mad")
            version = _resolve_pet_version(alias, None)
        return f"{alias}@{version}"
    if spec.potential == "dpa3":
        base, head = _split_dpa_model(model)
        path = _resolve_dpa_checkpoint(base)
        return f"{_checkpoint_sha256(path)}|head={head}"
    return _checkpoint_sha256(spec.model) or str(model)


def fingerprint_identity(spec: CalculatorSpec) -> tuple[str, str]:
    """(checkpoint identity, package version) for the force-cache
    fingerprint, from the environment that runs the potential."""
    interp = _dispatch_interpreter(spec.potential)
    if interp:
        from irma.mlip import envs
        return envs.remote_identity(spec, interp)
    return resolved_checkpoint_identity(spec), _package_version(spec.potential)


# SevenNet checkpoints trained on several fidelities (SevenNetCalculator
# refuses them without a modal)
_SEVENNET_MULTI_FIDELITY = ("7net-mf-ompa", "7net-omni")


def make_calculator(spec: CalculatorSpec):
    """Build (ase_calculator, meta) from a spec. CPU only; lazy imports.

    When the potential has a registered environment (irma mlip env
    create / IRMA_MLIP_PYTHON_<POTENTIAL>) that is not the running
    interpreter, the returned calculator transparently proxies every
    force call to a force_server subprocess in that environment; nothing
    potential-specific is imported here. Otherwise raises
    MlipDependencyError naming the missing pip package when the backend
    is not installed.
    """
    interp = _dispatch_interpreter(spec.potential)
    if interp:
        from irma.mlip import envs
        return envs.remote_calculator(spec, interp)

    # Threads are set before the backend import (tensorpotential starts
    # TensorFlow, and a torch op starts the inter-op pool, at import time).
    # A missing torch is reported by _require below; set_num_interop_threads
    # raises once the pool exists (a second calculator in this process).
    _clamp_native_threads(spec.threads)
    if spec.potential not in ("grace", "emt"):
        try:
            import torch
            torch.set_num_threads(spec.threads)
            torch.set_num_interop_threads(1)
        except (ImportError, RuntimeError):
            pass
    _require(spec.potential)

    if spec.potential == "emt":
        # Development/test backend (deliberately NOT in POTENTIALS): lets the
        # displacement loop and its spawn pool run cross-process without torch
        # or checkpoint downloads. The CLI must validate against POTENTIALS,
        # never against this wider set.
        from ase.calculators.emt import EMT
        return EMT(), {
            "potential": "emt", "package": "ase",
            "package_version": _package_version("emt"),
            "checkpoint": "builtin", "checkpoint_sha256": None,
            "device": "cpu", "dtype": "float64", "threads": spec.threads,
        }

    model = spec.model or _DEFAULT_MODELS[spec.potential]
    dtype = "float32"
    license_note = None
    checkpoint_sha256 = None
    extra_meta = None

    if spec.potential == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator
        torch.set_default_dtype(torch.float32)
        calc = MatterSimCalculator(load_path=model, device="cpu")
    elif spec.potential == "orb":
        # a filesystem path is a weights_path for the default conservative
        # builder rather than a builder name
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        if os.path.isfile(str(model)):
            builder = pretrained.orb_v3_conservative_inf_omat
            built = builder(weights_path=str(model), device="cpu",
                            precision="float32-high")
        else:
            builder = getattr(pretrained, str(model), None)
            if builder is None:
                raise ValueError(
                    f"unknown ORB pretrained builder {model!r}; expected a "
                    f"function name from orb_models.forcefield.pretrained "
                    f"or a checkpoint file path")
            if "conservative" not in str(model):
                # Finite-displacement force constants need forces that are
                # the gradient of the model's energy; the ORB "direct"
                # force heads are not, and their phonons are unreliable.
                raise ValueError(
                    f"ORB pretrained builder {model!r} is not a "
                    "conservative-force variant. IRMA only accepts ORB "
                    "builders with 'conservative' in their name, or a "
                    "checkpoint file path (loaded through the conservative "
                    "builder); direct-force heads give unreliable phonons.")
            built = builder(device="cpu", precision="float32-high")
        orbff, adapter = built
        calc = ORBCalculator(orbff, atoms_adapter=adapter, device="cpu")
        dtype = "float32-high"
    elif spec.potential == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        torch.set_default_dtype(torch.float32)
        if str(model) in _SEVENNET_MULTI_FIDELITY:
            # these checkpoints need a fidelity; IRMA uses the MPtrj+sAlex one
            calc = SevenNetCalculator(model=model, modal="mpa", device="cpu")
            extra_meta = {"modal": "mpa"}
        else:
            calc = SevenNetCalculator(model=model, device="cpu")
    elif spec.potential in MACE_FAMILY:
        is_file = os.path.isfile(str(model))
        if is_file:
            calc, info = _mace_calculator_from_file(str(model))
            checkpoint_sha256 = info["sha256"]
        else:
            from mace.calculators import mace_mp, mace_off
            loader = mace_mp if spec.potential == "mace" else mace_off
            calc = loader(model=model, default_dtype="float64", device="cpu")
        calc, extra_meta = _finish_mace(calc)
        dtype = "float64"
        # the named MACE-OFF checkpoints are ASL; a user file carries its own
        license_note = (_mace_license_note(model) if spec.potential == "mace"
                        else None if is_file else MACE_OFF_NOTE)
    elif spec.potential == "pet-mad":
        # non_conservative=False is passed EXPLICITLY on every construction:
        # upet's direct-force mode returns forces that are not gradients of
        # the energy, which breaks the finite-displacement force-constant
        # assumption, so this must not float on an upstream default.
        from upet.calculator import UPETCalculator
        if os.path.isfile(str(model)):
            calc = UPETCalculator(checkpoint_path=str(model), device="cpu",
                                  non_conservative=False)
        else:
            alias, version = _split_pet_model(model)
            version = version or _resolve_pet_version(alias, None)
            calc = UPETCalculator(model=alias, version=version, device="cpu",
                                  non_conservative=False)
            model = f"{alias}@{version}"
    elif spec.potential == "dpa3":
        base, head = _split_dpa_model(model)
        path = _resolve_dpa_checkpoint(base)
        # the missing-mpich and torch-ABI failures can fire at import or
        # at construction
        try:
            from deepmd.calculator import DP
            calc = DP(model=path, head=head)
        except ImportError as exc:
            raise MlipDependencyError(
                f"deepmd-kit could not load ({exc}); pip install mpich if "
                f"it is named") from exc
        except RuntimeError as exc:
            if "ABI" in str(exc) or "deepmd_op_pt" in str(exc):
                raise MlipDependencyError(
                    "deepmd-kit's C++ op library does not match this torch: "
                    + str(exc).splitlines()[0]) from exc
            raise
        dtype = "float64"
        checkpoint_sha256 = _checkpoint_sha256(path)
        model = f"{path}::{head}" if head else str(path)
    elif spec.potential == "nequip":
        try:                                   # >= 0.19 layout
            from nequip.integrations.ase import NequIPCalculator
        except ImportError:                    # pre-0.19 (old path)
            from nequip.ase import NequIPCalculator
        if not os.path.isfile(str(model)):
            model = _compile_nequip_model(_normalize_nequip_zoo_id(model))
        calc = NequIPCalculator.from_compiled_model(str(model),
                                                    device="cpu")
        dtype = "model-default"
    elif spec.potential == "grace":
        # every GRACE foundation model (and tensorpotential itself) is
        # ASL-licensed; the note travels with the manifest like the MACE
        # foundation checkpoints' one
        from tensorpotential.calculator import grace_fm
        if os.path.isfile(str(model)):
            raise ValueError(
                "grace takes a foundation-model name (e.g. GRACE-2L-OAM),"
                " not a checkpoint path; local models are not supported")
        try:
            calc = grace_fm(str(model))
        except AssertionError as exc:
            # tensorpotential asserts on unknown registry names; a bad
            # --model is a usage error (exit 2), not a traceback
            raise ValueError(
                f"unknown GRACE foundation model {model!r} "
                f"({exc})") from exc
        dtype = "model-default"
        license_note = (
            "GRACE foundation models and the tensorpotential package are "
            "distributed under the Academic Software License: "
            "research-only, non-commercial; commercial use needs a "
            "separate license from ICAMS")

    if checkpoint_sha256 is None:    # dpa3 and MACE files hashed above
        checkpoint_sha256 = _checkpoint_sha256(spec.model)
    meta = {
        "potential": spec.potential,
        "package": _PACKAGES[spec.potential][0],
        "package_version": _package_version(spec.potential),
        "checkpoint": str(model),
        "checkpoint_sha256": checkpoint_sha256,
        "device": "cpu",
        "dtype": dtype,
        "threads": spec.threads,
    }
    if extra_meta:
        meta.update(extra_meta)
    if license_note:
        meta["license_note"] = license_note
    return calc, meta
