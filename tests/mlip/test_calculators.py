"""Calculator factory tests. No potential package is required: the missing-
dependency contract is tested by poisoning sys.modules, and the branch logic
by injecting stub module trees. Real-potential instantiation lives behind the
manual mlip_real marker (never in CI)."""
import os
import pickle
import sys
import types

import pytest

from irma.mlip.calculators import (
    POTENTIALS, CalculatorSpec, MlipDependencyError, make_calculator)


def test_spec_validates_potential_and_threads():
    with pytest.raises(ValueError, match="unknown potential"):
        CalculatorSpec("quantum-espresso")
    with pytest.raises(ValueError, match="threads"):
        CalculatorSpec("mattersim", threads=0)


def test_spec_is_picklable():
    spec = CalculatorSpec("mace-off", model="small", threads=3)
    assert pickle.loads(pickle.dumps(spec)) == spec


@pytest.mark.parametrize("potential,pip_name,module", [
    ("mattersim", "mattersim", "mattersim"),
    ("orb", "orb-models", "orb_models"),
    ("sevennet", "sevenn", "sevenn"),
    ("mace", "mace-torch", "mace"),
    ("mace-off", "mace-torch", "mace"),
    ("pet-mad", "upet", "upet"),
    ("dpa3", "deepmd-kit", "deepmd"),
    ("nequip", "nequip", "nequip"),
    ("grace", "tensorpotential", "tensorpotential"),
])
def test_missing_dependency_names_pip_package(monkeypatch, potential,
                                              pip_name, module):
    # A None entry in sys.modules makes `import module` raise ImportError,
    # regardless of whether the real package is installed.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(MlipDependencyError, match=pip_name):
        make_calculator(CalculatorSpec(potential))
    # the CI condition: torch itself absent must ALSO surface as the
    # named dependency error, never a raw ModuleNotFoundError('torch')
    # (regression: the clamp-before-import reorder broke this)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(MlipDependencyError, match=pip_name):
        make_calculator(CalculatorSpec(potential))


def _install_stub(monkeypatch, path, **attrs):
    """Register a stub module (and its parents) in sys.modules."""
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = sys.modules.get(name)
        if mod is None or not isinstance(mod, types.ModuleType) \
                or i == len(parts):
            mod = types.ModuleType(name)
            monkeypatch.setitem(sys.modules, name, mod)
        if i > 1:
            setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], mod)
    for key, value in attrs.items():
        setattr(sys.modules[path], key, value)
    return sys.modules[path]


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.float32 = "float32"
        self.float64 = "float64"
        self.num_threads = None
        self.interop_threads = None
        self.default_dtype = None

    def set_num_threads(self, n):
        self.num_threads = n

    def set_num_interop_threads(self, n):
        self.interop_threads = n

    def set_default_dtype(self, d):
        self.default_dtype = d


def test_mattersim_branch_with_stubs(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    class StubCalc:
        def __init__(self, load_path=None, device=None):
            made.update(load_path=load_path, device=device)

    _install_stub(monkeypatch, "mattersim")
    _install_stub(monkeypatch, "mattersim.forcefield")
    _install_stub(monkeypatch, "mattersim.forcefield.potential",
                  MatterSimCalculator=StubCalc)

    ckpt = tmp_path / "custom.pth"
    ckpt.write_bytes(b"weights")
    calc, meta = make_calculator(
        CalculatorSpec("mattersim", model=str(ckpt), threads=2))

    assert isinstance(calc, StubCalc)
    assert made == {"load_path": str(ckpt), "device": "cpu"}
    assert fake_torch.num_threads == 2
    assert fake_torch.interop_threads == 1     # clamped BEFORE the import
    assert fake_torch.default_dtype == "float32"
    assert meta["potential"] == "mattersim"
    assert meta["device"] == "cpu"
    assert meta["dtype"] == "float32"
    assert meta["checkpoint"] == str(ckpt)
    assert meta["checkpoint_sha256"] is not None and len(meta["checkpoint_sha256"]) == 64


class _FakeDtype:
    def __init__(self, name, floating=True):
        self.name = name
        self.is_floating_point = floating

    def __str__(self):
        return "torch." + self.name


class _FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype


class _FakeMaceModel:
    """What torch.load returns for a MACE .model file: the attributes
    MACE's calculator reads, plus parameters/buffers for the dtype."""

    def __init__(self, atomic_numbers=(1, 6), r_max=6.5, dtypes=("float32",),
                 heads=("Default",), num_interactions=2):
        self.atomic_numbers = list(atomic_numbers)
        self.r_max = r_max
        self.num_interactions = num_interactions
        if heads is not None:
            self.heads = list(heads)
        self._dtypes = list(dtypes)

    def parameters(self):
        return [_FakeTensor(_FakeDtype(d)) for d in self._dtypes]

    def buffers(self):
        return [_FakeTensor(_FakeDtype("int64", floating=False))]


class _FakeZTable:
    def __init__(self, zs):
        self.zs = list(zs)


class _FakeMaceCalculator:
    """Stands in for mace.calculators.MACECalculator: records how it was
    built and counts backend calls (the guard tests need the count)."""
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, models=None, model_paths=None, default_dtype="",
                 device="cpu"):
        model = models[0] if isinstance(models, list) else models
        self.models = [model]
        self.z_table = _FakeZTable(model.atomic_numbers)
        self.r_max = float(model.r_max)
        self.available_heads = list(getattr(model, "heads", ["Default"]))
        self.head = self.available_heads[0]
        self.default_dtype = default_dtype
        self.device = device
        self.calls = 0
        self.closed = False
        self.results = {}

    def calculate(self, atoms=None, properties=None, system_changes=None):
        import numpy as np
        self.calls += 1
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}

    def close(self):
        self.closed = True


def _stub_mace(monkeypatch, loaded, made):
    """Fake torch + mace so a checkpoint FILE loads through the seam:
    torch.load returns `loaded` and records the bytes it was given;
    MACECalculator is the fake above; the named-model entry points build
    the same fake from a fresh model."""
    fake_torch = _FakeTorch()

    def load(f, map_location=None, weights_only=None):
        made.setdefault("loads", []).append(
            (f.read(), map_location, weights_only))
        return loaded
    fake_torch.load = load
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_stub(monkeypatch, "mace")

    def named(model=None, default_dtype=None, device=None):
        made.setdefault("named", []).append((model, default_dtype, device))
        return _FakeMaceCalculator(models=_FakeMaceModel(
            atomic_numbers=range(1, 90), r_max=6.0, dtypes=("float64",),
            heads=("default",)), default_dtype=default_dtype, device=device)
    _install_stub(monkeypatch, "mace.calculators",
                  MACECalculator=_FakeMaceCalculator, mace_mp=named,
                  mace_off=named)
    return fake_torch


@pytest.mark.parametrize("potential", ["mace", "mace-off"])
def test_mace_named_models_are_guarded_and_described(monkeypatch,
                                                     potential):
    pytest.importorskip("ase")
    made = {}
    _stub_mace(monkeypatch, _FakeMaceModel(), made)
    calc, meta = make_calculator(CalculatorSpec(potential))
    default = {"mace": "medium-mpa-0", "mace-off": "medium"}[potential]
    assert made["named"] == [(default, "float64", "cpu")]
    assert "loads" not in made              # nothing read from disk
    assert meta["dtype"] == "float64"
    assert meta["checkpoint"] == default
    assert meta["checkpoint_sha256"] is None   # named model, not a file
    assert isinstance(calc, _FakeMaceCalculator)
    assert meta["checkpoint_elements"][:3] == ["H", "He", "Li"]
    assert len(meta["checkpoint_elements"]) == 89
    assert meta["checkpoint_r_max_A"] == 6.0
    assert meta["checkpoint_num_interactions"] == 2
    assert meta["checkpoint_heads"] == ["default"]
    assert meta["checkpoint_head"] == "default"
    if potential == "mace-off":
        assert "Academic Software License" in meta["license_note"]
    else:
        assert "license_note" not in meta   # MPA line is MIT: no note


def test_mace_foundation_checkpoints_carry_the_asl_note(monkeypatch,
                                                        tmp_path):
    pytest.importorskip("ase")
    _stub_mace(monkeypatch, _FakeMaceModel(), {})
    omat_file = tmp_path / "mace-omat-0.model"
    omat_file.write_bytes(b"weights")
    for name in ("medium-omat-0", "MACE-matpes-pbe-0", str(omat_file)):
        _, meta = make_calculator(CalculatorSpec("mace", model=name))
        assert "Academic Software License" in meta.get("license_note", ""), \
            name
    for name in ("medium-mpa-0", "medium", "small-0b"):
        _, meta = make_calculator(CalculatorSpec("mace", model=name))
        assert "license_note" not in meta, name


@pytest.mark.parametrize("potential", ["mace", "mace-off"])
def test_mace_checkpoint_file_is_loaded_once_from_its_bytes(
        monkeypatch, tmp_path, potential):
    pytest.importorskip("ase")
    from irma.mlip.calculators import _checkpoint_sha256, canonicalize_spec
    made = {}
    loaded = _FakeMaceModel()
    _stub_mace(monkeypatch, loaded, made)
    # tmp_path, not NamedTemporaryFile: Windows cannot reopen a file
    # that is still held open by its creator
    model_file = tmp_path / "weights.model"
    model_file.write_bytes(b"weights")
    digest = _checkpoint_sha256(str(model_file))

    spec = canonicalize_spec(CalculatorSpec(potential, model=str(model_file)))
    calc, meta = make_calculator(spec)

    # exactly one deserialization, of the bytes that were hashed, and
    # the calculator is built from that same object (no second load, no
    # model_paths round trip through the file name)
    assert made["loads"] == [(b"weights", "cpu", False)]
    assert "named" not in made
    assert calc.models[0] is loaded
    assert calc.default_dtype == "float64"
    assert meta["checkpoint"] == str(model_file)
    assert meta["checkpoint_sha256"] == digest
    assert meta["checkpoint_model_class"] == "_FakeMaceModel"
    assert meta["checkpoint_r_max_A"] == 6.5
    assert meta["checkpoint_num_interactions"] == 2
    assert meta["checkpoint_elements"] == ["H", "C"]
    assert meta["checkpoint_heads"] == ["Default"]
    assert meta["checkpoint_head"] == "Default"
    # a user checkpoint FILE carries its own license: no ASL note (the
    # named MACE-OFF checkpoints are ASL, see the named-model test)
    assert "license_note" not in meta


def test_mace_checkpoint_without_heads_is_single_head(monkeypatch, tmp_path):
    pytest.importorskip("ase")
    model_file = tmp_path / "w.model"
    model_file.write_bytes(b"w")
    # legacy checkpoints without a heads attribute are single-head
    _stub_mace(monkeypatch, _FakeMaceModel(heads=None), {})
    _, meta = make_calculator(CalculatorSpec("mace", model=str(model_file)))
    assert meta["checkpoint_heads"] == ["Default"]


def test_mace_multihead_and_non_mace_files_are_refused(monkeypatch,
                                                       tmp_path):
    pytest.importorskip("ase")
    model_file = tmp_path / "w.model"
    model_file.write_bytes(b"w")
    _stub_mace(monkeypatch, _FakeMaceModel(heads=("mp", "omat")), {})
    with pytest.raises(ValueError, match="multi-head"):
        make_calculator(CalculatorSpec("mace", model=str(model_file)))
    # a state dict (a plain mapping) is not a serialized model
    _stub_mace(monkeypatch, {"weights": 1}, {})
    with pytest.raises(ValueError, match="not a serialized MACE model"):
        make_calculator(CalculatorSpec("mace-off", model=str(model_file)))


def test_missing_elements():
    from irma.mlip.calculators import missing_elements
    assert missing_elements(["H", "C"], ["C", "H", "H"]) == []
    assert missing_elements(["H", "C"], ["C", "O", "N", "O"]) == ["N", "O"]


def test_canonicalize_pins_mace_checkpoint_files(monkeypatch, tmp_path):
    from irma.mlip.calculators import (
        _checkpoint_sha256, canonicalize_spec, resolved_checkpoint_identity)
    model_file = tmp_path / "w.model"
    model_file.write_bytes(b"weights")
    digest = _checkpoint_sha256(str(model_file))
    monkeypatch.chdir(tmp_path)

    for potential in ("mace", "mace-off"):
        pinned = canonicalize_spec(CalculatorSpec(potential, model="w.model"))
        assert pinned.model == str(model_file)          # absolute path
        assert canonicalize_spec(pinned) is pinned      # idempotent
        # the fingerprint identity is the file's content hash
        assert resolved_checkpoint_identity(pinned) == digest
        # named models and missing files pass through unchanged
        for model in (None, "medium", str(tmp_path / "missing.model")):
            spec = CalculatorSpec(potential, model=model)
            assert canonicalize_spec(spec) is spec


def test_native_thread_env_is_clamped(monkeypatch):
    pytest.importorskip("ase")          # the emt backend needs ase
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    make_calculator(CalculatorSpec("emt", threads=1))
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"


def test_sevennet_branch_with_stubs(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    def stub(model=None, modal=None, device=None):
        made.update(model=model, modal=modal, device=device)
        return "7net-calc"

    _install_stub(monkeypatch, "sevenn")
    _install_stub(monkeypatch, "sevenn.calculator", SevenNetCalculator=stub)

    calc, meta = make_calculator(CalculatorSpec("sevennet"))
    assert calc == "7net-calc"
    assert made == {"model": "7net-mf-ompa", "modal": "mpa", "device": "cpu"}
    assert fake_torch.default_dtype == "float32"

    made.clear()
    calc, _ = make_calculator(CalculatorSpec("sevennet", model="7net-0"))
    assert made == {"model": "7net-0", "modal": None, "device": "cpu"} or \
           made == {"model": "7net-0", "device": "cpu"}


def _stub_orb(monkeypatch, calculator_cls, **builders):
    _install_stub(monkeypatch, "orb_models")
    _install_stub(monkeypatch, "orb_models.forcefield")
    _install_stub(monkeypatch, "orb_models.forcefield.pretrained", **builders)
    _install_stub(monkeypatch, "orb_models.forcefield.inference")
    _install_stub(monkeypatch, "orb_models.forcefield.inference.calculator",
                  ORBCalculator=calculator_cls)


def test_orb_current_api_tuple_and_adapter(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    class StubORB:
        def __init__(self, model, atoms_adapter=None, device=None):
            made.update(model=model, adapter=atoms_adapter, device=device)

    _stub_orb(monkeypatch, StubORB, orb_v3_conservative_inf_omat=lambda **kw:
              ("the-model", "the-adapter"))
    calc, _ = make_calculator(CalculatorSpec("orb"))
    assert made == {"model": "the-model", "adapter": "the-adapter",
                    "device": "cpu"}


def test_orb_filesystem_checkpoint_uses_weights_path(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    def builder(weights_path=None, device=None, precision=None):
        made.update(weights_path=weights_path, device=device,
                    precision=precision)
        return "path-model", "adapter"

    class StubORB:
        def __init__(self, model, atoms_adapter=None, device=None):
            made.update(model=model)

    _stub_orb(monkeypatch, StubORB, orb_v3_conservative_inf_omat=builder)
    ckpt = tmp_path / "orb.ckpt"
    ckpt.write_bytes(b"weights")
    make_calculator(CalculatorSpec("orb", model=str(ckpt)))
    assert made["weights_path"] == str(ckpt)
    assert made["model"] == "path-model"


def test_orb_unknown_builder_name_is_a_clear_error(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _stub_orb(monkeypatch, type("S", (), {}),
              orb_v3_conservative_inf_omat=lambda **kw: ("m", "a"))
    # a bad --model name is a USAGE error (CLI exit 2), not a missing
    # dependency (exit 4) -- review reclassification
    with pytest.raises(ValueError, match="pretrained"):
        make_calculator(CalculatorSpec("orb", model="not_a_builder"))


def test_orb_direct_force_builder_is_refused(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _stub_orb(monkeypatch, type("S", (), {}),
              orb_v3_conservative_inf_omat=lambda **kw: ("cons", "a"),
              orb_v3_direct_inf_omat=lambda **kw: ("direct", "a"))
    # the builder EXISTS in orb_models, but it is a direct-force head: the
    # conservative-force rule must refuse it by name (docs/mlip.md promises
    # "no option can select a direct-force model")
    with pytest.raises(ValueError, match="conservative"):
        make_calculator(CalculatorSpec("orb", model="orb_v3_direct_inf_omat"))


def _stub_upet(monkeypatch, made, resolved_version="9.9.9"):
    """Stub upet.calculator with a recording UPETCalculator.

    non_conservative is recorded so tests can assert it is passed as
    exactly False on every construction (conservative forces are a hard
    requirement, and the invariant must not float on upstream defaults).
    """
    class StubUPET:
        def __init__(self, model=None, version=None, checkpoint_path=None,
                     device=None, non_conservative=True):
            # the stub default is the DANGEROUS value: only an explicit
            # False from the factory makes the assertions pass
            made.update(model=model, version=version,
                        checkpoint_path=checkpoint_path, device=device,
                        non_conservative=non_conservative)

    def stub_resolve(model, requested_size=None, requested_version=None):
        made.update(resolved=(model, requested_size, requested_version))
        return requested_size, requested_version or resolved_version

    _install_stub(monkeypatch, "upet")
    _install_stub(monkeypatch, "upet.calculator",
                  UPETCalculator=StubUPET,
                  upet_resolve_model=stub_resolve,
                  UPET_AVAILABLE_MODELS=["pet-mad-s", "pet-mad-xs",
                                         "pet-omat-s"])


def test_pet_mad_alias_pins_resolved_version(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_upet(monkeypatch, made)

    calc, meta = make_calculator(CalculatorSpec("pet-mad"))
    assert made["model"] == "pet-mad-s" and made["version"] == "9.9.9"
    assert made["device"] == "cpu" and made["checkpoint_path"] is None
    assert made["non_conservative"] is False       # explicit, every time
    assert meta["checkpoint"] == "pet-mad-s@9.9.9"
    assert meta["dtype"] == "float32"
    assert meta["checkpoint_sha256"] is None


def test_pet_mad_explicit_version_skips_resolution(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_upet(monkeypatch, made)

    _, meta = make_calculator(CalculatorSpec("pet-mad",
                                             model="pet-omat-s@1.2.3"))
    assert made["model"] == "pet-omat-s" and made["version"] == "1.2.3"
    assert "resolved" not in made            # no network resolution path
    assert meta["checkpoint"] == "pet-omat-s@1.2.3"


def test_pet_mad_checkpoint_file_uses_checkpoint_path(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_upet(monkeypatch, made)

    ckpt = tmp_path / "pet-mad-s-v1.5.0.ckpt"
    ckpt.write_bytes(b"weights")
    _, meta = make_calculator(CalculatorSpec("pet-mad", model=str(ckpt)))
    assert made["checkpoint_path"] == str(ckpt) and made["model"] is None
    assert made["non_conservative"] is False
    assert meta["checkpoint_sha256"] is not None

    # a checkpoint path containing '@' is a path, never an alias@version
    made.clear()
    odd = tmp_path / "pet@2.ckpt"
    odd.write_bytes(b"weights")
    make_calculator(CalculatorSpec("pet-mad", model=str(odd)))
    assert made["checkpoint_path"] == str(odd)


def test_pet_mad_unknown_alias_is_a_clear_error(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _stub_upet(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown pet-mad model"):
        make_calculator(CalculatorSpec("pet-mad", model="pet-nonsense-s"))
    with pytest.raises(ValueError, match="malformed pet-mad model"):
        make_calculator(CalculatorSpec("pet-mad", model="pet-mad-s@"))


def _stub_deepmd(monkeypatch, made):
    class StubDP:
        def __init__(self, model=None, head=None):
            made.update(model=model, head=head)

    _install_stub(monkeypatch, "deepmd")
    _install_stub(monkeypatch, "deepmd.calculator", DP=StubDP)


def test_dpa3_local_file_with_head(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_deepmd(monkeypatch, made)

    ckpt = tmp_path / "frozen.pth"
    ckpt.write_bytes(b"dpa-weights")
    _, meta = make_calculator(
        CalculatorSpec("dpa3", model=f"{ckpt}::Omat24"))
    assert made == {"model": str(ckpt), "head": "Omat24"}
    assert meta["checkpoint"] == f"{ckpt}::Omat24"
    assert meta["dtype"] == "float64"
    assert meta["checkpoint_sha256"] is not None   # hash of the real file


def test_dpa3_local_file_without_head_passes_none(monkeypatch, tmp_path):
    # a frozen single-task model has no branches; the MPtrj default must
    # only apply to the known multitask aliases
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_deepmd(monkeypatch, made)

    ckpt = tmp_path / "frozen.pth"
    ckpt.write_bytes(b"dpa-weights")
    _, meta = make_calculator(CalculatorSpec("dpa3", model=str(ckpt)))
    assert made["head"] is None
    assert meta["checkpoint"] == str(ckpt)


def test_dpa3_alias_downloads_once_and_gets_default_head(monkeypatch,
                                                         tmp_path):
    import hashlib

    from irma.mlip import calculators

    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made, downloads = {}, []
    _stub_deepmd(monkeypatch, made)
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))
    # the integrity pin must match what the stub serves
    monkeypatch.setattr(
        calculators, "_DPA_HF_SHA256",
        {"DPA-3.1-3M.pt": hashlib.sha256(b"downloaded").hexdigest()})

    def stub_download(repo, filename, revision=None, local_dir=None):
        downloads.append((repo, filename, revision, local_dir))
        path = tmp_path / filename
        path.write_bytes(b"downloaded")
        return str(path)

    _install_stub(monkeypatch, "huggingface_hub",
                  hf_hub_download=stub_download)

    _, meta = make_calculator(CalculatorSpec("dpa3"))
    assert made["model"] == str(tmp_path / "DPA-3.1-3M.pt")
    assert made["head"] == "MP_traj_v024_alldata_mixu"
    # the fetch is PINNED to the vetted revision, never a mutable branch
    assert downloads == [("deepmodelingcommunity/DPA", "DPA-3.1-3M.pt",
                          calculators._DPA_HF_REVISION, str(tmp_path))]

    # second build finds the cached file and never touches the network
    make_calculator(CalculatorSpec("dpa3"))
    assert len(downloads) == 1


def test_dpa3_download_integrity_mismatch_is_rejected(monkeypatch, tmp_path):
    # a checkpoint whose sha256 does not match the pin must never reach
    # deepmd's pickle-based loader, and must not survive in the cache
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_deepmd(monkeypatch, made)
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))

    def stub_download(repo, filename, revision=None, local_dir=None):
        path = tmp_path / filename
        path.write_bytes(b"retagged upstream payload")
        return str(path)

    _install_stub(monkeypatch, "huggingface_hub",
                  hf_hub_download=stub_download)

    with pytest.raises(RuntimeError, match="integrity check FAILED"):
        make_calculator(CalculatorSpec("dpa3"))
    assert made == {}                              # DP was never built
    assert not (tmp_path / "DPA-3.1-3M.pt").exists()   # bad copy removed

    # a tampered CACHED copy is caught the same way (cache-hit path)
    (tmp_path / "DPA-3.1-3M.pt").write_bytes(b"tampered cache")
    with pytest.raises(RuntimeError, match="integrity check FAILED"):
        make_calculator(CalculatorSpec("dpa3"))
    assert not (tmp_path / "DPA-3.1-3M.pt").exists()


def test_pet_known_version_hash_is_verified(monkeypatch, tmp_path):
    import hashlib

    from irma.mlip import calculators
    from irma.mlip.calculators import canonicalize_spec

    made = {}
    _stub_upet(monkeypatch, made)
    _stub_pet_download(monkeypatch, made, tmp_path)   # serves b"pet-weights"

    # matching pin: download verifies and canonicalization succeeds
    monkeypatch.setattr(
        calculators, "_PET_KNOWN_SHA256",
        {"pet-mad-s-v9.9.9.ckpt":
         hashlib.sha256(b"pet-weights").hexdigest()})
    pinned = canonicalize_spec(CalculatorSpec("pet-mad"))
    assert os.path.isfile(pinned.model)

    # wrong pin: rejected, and the bad file does not survive in the cache
    os.remove(pinned.model)
    monkeypatch.setattr(calculators, "_PET_KNOWN_SHA256",
                        {"pet-mad-s-v9.9.9.ckpt": "0" * 64})
    with pytest.raises(RuntimeError, match="integrity check FAILED"):
        canonicalize_spec(CalculatorSpec("pet-mad"))
    assert not os.path.isfile(pinned.model)

    # an UNKNOWN version (no recorded pin) still downloads: its content
    # hash is recorded in the bundle provenance instead (see meta)
    monkeypatch.setattr(calculators, "_PET_KNOWN_SHA256", {})
    pinned = canonicalize_spec(CalculatorSpec("pet-mad"))
    assert os.path.isfile(pinned.model)


def test_dpa3_unknown_alias_and_malformed_head_are_clear_errors(monkeypatch):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _stub_deepmd(monkeypatch, {})
    with pytest.raises(ValueError, match="unknown dpa3 model"):
        make_calculator(CalculatorSpec("dpa3", model="DPA-99"))
    with pytest.raises(ValueError, match="malformed dpa3 model"):
        make_calculator(CalculatorSpec("dpa3", model="DPA-3.1-3M::"))


@pytest.mark.skipif(sys.platform == "win32", reason="':' is illegal in NTFS filenames, so a '::'-bearing local checkpoint path cannot exist on Windows")
def test_dpa3_checkpoint_path_containing_separator(monkeypatch, tmp_path):
    # '::' is legal in POSIX filenames; an existing file must be taken
    # whole, and a real prefix path may still carry an explicit head
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}
    _stub_deepmd(monkeypatch, made)

    odd = tmp_path / "run::v2.pt"
    odd.write_bytes(b"w")
    make_calculator(CalculatorSpec("dpa3", model=str(odd)))
    assert made == {"model": str(odd), "head": None}

    made.clear()
    plain = tmp_path / "m.pt"
    plain.write_bytes(b"w")
    make_calculator(CalculatorSpec("dpa3", model=f"{plain}::Omat24"))
    assert made == {"model": str(plain), "head": "Omat24"}


def test_dpa3_import_time_failure_is_a_dependency_error(monkeypatch,
                                                        tmp_path):
    # deepmd.calculator imports DeepPot at module load, so the mpich/ABI
    # failures can fire at IMPORT, not just at DP construction
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _install_stub(monkeypatch, "deepmd")
    monkeypatch.setitem(sys.modules, "deepmd.calculator", None)  # ImportError
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"w")
    with pytest.raises(MlipDependencyError, match="deepmd-kit could not"):
        make_calculator(CalculatorSpec("dpa3", model=str(ckpt)))


def test_dpa3_abi_mismatch_is_a_dependency_error(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class ExplodingDP:
        def __init__(self, model=None, head=None):
            raise RuntimeError(
                "This deepmd-kit package was compiled with CXX11_ABI_FLAG=0,"
                " but PyTorch runtime was compiled with CXX11_ABI_FLAG=1.")

    _install_stub(monkeypatch, "deepmd")
    _install_stub(monkeypatch, "deepmd.calculator", DP=ExplodingDP)
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"w")
    with pytest.raises(MlipDependencyError, match="does not match this torch"):
        make_calculator(CalculatorSpec("dpa3", model=str(ckpt)))


def _stub_pet_download(monkeypatch, made, tmp_path):
    """Stub the HF checkpoint download and point the cache at tmp_path."""
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))

    def stub_download(repo, filename, subfolder=None, local_dir=None):
        made.setdefault("downloads", []).append((repo, filename, subfolder))
        path = tmp_path / (subfolder or "") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pet-weights")
        return str(path)

    _install_stub(monkeypatch, "huggingface_hub",
                  hf_hub_download=stub_download)


def test_canonicalize_spec_pins_to_a_local_checkpoint(monkeypatch, tmp_path):
    from irma.mlip.calculators import canonicalize_spec
    made = {}
    _stub_upet(monkeypatch, made)
    _stub_pet_download(monkeypatch, made, tmp_path)

    pinned = canonicalize_spec(CalculatorSpec("pet-mad"))
    expected = str(tmp_path / "models" / "pet-mad-s-v9.9.9.ckpt")
    assert pinned.model == expected          # a real local file, not an alias
    assert os.path.isfile(pinned.model)
    assert made["downloads"] == [("lab-cosmo/upet", "pet-mad-s-v9.9.9.ckpt",
                                  "models")]
    assert "resolved" in made                # version resolved exactly once

    # idempotent, and the pinned path never re-resolves or re-downloads
    made.pop("resolved")
    assert canonicalize_spec(pinned) is pinned
    assert "resolved" not in made and len(made["downloads"]) == 1

    # a pinned @version with the checkpoint already cached is fully offline
    made.clear()
    offline = canonicalize_spec(CalculatorSpec("pet-mad",
                                               model="pet-mad-s@9.9.9"))
    assert offline.model == expected
    assert made == {}                        # no resolve, no download

    dpa = canonicalize_spec(CalculatorSpec("dpa3"))
    assert dpa.model == "DPA-3.1-3M::MP_traj_v024_alldata_mixu"
    with pytest.raises(ValueError, match="unknown dpa3 model"):
        canonicalize_spec(CalculatorSpec("dpa3", model="DPA-99"))

    for untouched in (CalculatorSpec("mattersim"),
                      CalculatorSpec("mace", model="medium-mpa-0"),
                      CalculatorSpec("emt")):
        assert canonicalize_spec(untouched) is untouched


def test_pet_mad_latest_is_unpinned_never_an_identity(monkeypatch, tmp_path):
    # review finding: '@latest' must behave exactly like no version -- it
    # must resolve to a numeric version, never survive into the identity
    from irma.mlip.calculators import (
        canonicalize_spec, resolved_checkpoint_identity)
    made = {}
    _stub_upet(monkeypatch, made)
    _stub_pet_download(monkeypatch, made, tmp_path)

    pinned = canonicalize_spec(CalculatorSpec("pet-mad",
                                              model="pet-mad-s@latest"))
    assert os.path.basename(pinned.model) == "pet-mad-s-v9.9.9.ckpt"
    assert made["resolved"][2] is None       # resolver saw it as unpinned

    ident = resolved_checkpoint_identity(
        CalculatorSpec("pet-mad", model="pet-mad-s@latest"))
    assert ident == "pet-mad-s@9.9.9"
    assert "latest" not in ident


def test_resolved_checkpoint_identity(monkeypatch, tmp_path):
    from irma.mlip.calculators import (
        _DEFAULT_MODELS, _checkpoint_sha256, resolved_checkpoint_identity)

    # the HISTORICAL fingerprint formula, verbatim from the pre-review
    # _fingerprint: any deviation for the five legacy potentials would
    # silently invalidate every existing force cache
    def old_formula(spec):
        model = spec.model or _DEFAULT_MODELS.get(spec.potential, "builtin")
        return _checkpoint_sha256(spec.model) or str(model)

    ckpt = tmp_path / "w.pth"
    ckpt.write_bytes(b"weights")
    legacy = ("mattersim", "orb", "sevennet", "mace", "mace-off")
    for potential in legacy:
        for model in (None, "some-named-model",
                      str(tmp_path / "missing.pth"), str(ckpt)):
            spec = CalculatorSpec(potential, model=model)
            assert resolved_checkpoint_identity(spec) == old_formula(spec), \
                (potential, model)

    # pinned pet-mad needs no upet import at all
    assert resolved_checkpoint_identity(
        CalculatorSpec("pet-mad", model="pet-mad-s@1.5.0")) == \
        "pet-mad-s@1.5.0"

    # dpa3 identity = content hash + head (same file, different physics)
    dpa = tmp_path / "m.pt"
    dpa.write_bytes(b"dpa-weights")
    ident = resolved_checkpoint_identity(
        CalculatorSpec("dpa3", model=f"{dpa}::Omat24"))
    assert ident == f"{_checkpoint_sha256(str(dpa))}|head=Omat24"
    ident2 = resolved_checkpoint_identity(
        CalculatorSpec("dpa3", model=f"{dpa}::SPICE2"))
    assert ident2 != ident


def test_nequip_zoo_id_normalization_and_cache(monkeypatch, tmp_path):
    from irma.mlip.calculators import (
        _nequip_artifact_path, _normalize_nequip_zoo_id, canonicalize_spec)
    norm = _normalize_nequip_zoo_id
    assert norm("NequIP-OAM-L:0.1") == "mir-group/NequIP-OAM-L:0.1"
    assert norm("nequip.net:mir-group/NequIP-OAM-XL:0.1") == \
        "mir-group/NequIP-OAM-XL:0.1"
    assert norm("other-group/Model:2.0") == "other-group/Model:2.0"
    for bad in ("NequIP-OAM-L", "NequIP-OAM-L:", "nequip.net:",
                "a/b/c:1", "a/:1", ":1", "a:1/b"):
        with pytest.raises(ValueError, match="malformed nequip model"):
            norm(bad)
    # a mistyped PATH must say so, not be treated as a zoo id
    with pytest.raises(ValueError, match="looks like a file path"):
        norm(str(tmp_path / "missing.nequip.pt2"))

    # lossy readable names must not alias distinct ids (review finding)
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))
    assert _nequip_artifact_path("a-b/c:1", ".nequip.pt2") != \
        _nequip_artifact_path("a/b-c:1", ".nequip.pt2")
    # a compiled artifact path is already canonical
    art = tmp_path / "m.nequip.pt2"
    art.write_bytes(b"compiled")
    pinned = CalculatorSpec("nequip", model=str(art))
    assert canonicalize_spec(pinned) is pinned


def test_nequip_canonicalize_dispatches_before_any_torch_use(monkeypatch,
                                                             tmp_path):
    # review finding: a torch-free host with a registered nequip env must
    # canonicalize remotely, never resolving artifact mode locally
    from irma.mlip import calculators, envs
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "torch", None)      # torch-free host
    monkeypatch.setitem(sys.modules, "nequip", None)
    monkeypatch.setattr(calculators, "_dispatch_interpreter",
                        lambda p: "/foreign/python" if p == "nequip"
                        else None)
    monkeypatch.setattr(envs, "remote_canonicalize",
                        lambda spec, interp: "/shared/artifact.nequip.pt2")
    pinned = calculators.canonicalize_spec(CalculatorSpec("nequip"))
    assert pinned.model == "/shared/artifact.nequip.pt2"


def test_nequip_branch_loads_compiled_artifact(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    class StubNequIP:
        @classmethod
        def from_compiled_model(cls, path, device=None):
            made.update(path=path, device=device)
            return "nequip-calc"

    _install_stub(monkeypatch, "nequip")
    _install_stub(monkeypatch, "nequip.ase", NequIPCalculator=StubNequIP)
    art = tmp_path / "model.nequip.pt2"
    art.write_bytes(b"compiled")
    calc, meta = make_calculator(CalculatorSpec("nequip", model=str(art)))
    assert calc == "nequip-calc"
    assert made == {"path": str(art), "device": "cpu"}
    assert meta["checkpoint"] == str(art)
    assert meta["checkpoint_sha256"] is not None
    assert "license_note" not in meta          # MIT/CC-BY: no note


def test_grace_branch_and_asl_note(monkeypatch, tmp_path):
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    made = {}

    def stub_grace_fm(model):
        made.update(model=model)
        return "grace-calc"

    _install_stub(monkeypatch, "tensorpotential")
    _install_stub(monkeypatch, "tensorpotential.calculator",
                  grace_fm=stub_grace_fm)
    calc, meta = make_calculator(CalculatorSpec("grace"))
    assert calc == "grace-calc" and made == {"model": "GRACE-2L-OAM"}
    assert "Academic Software License" in meta["license_note"]

    ckpt = tmp_path / "local.model"
    ckpt.write_bytes(b"w")
    with pytest.raises(ValueError, match="foundation-model name"):
        make_calculator(CalculatorSpec("grace", model=str(ckpt)))

    # tensorpotential asserts on unknown names; that must surface as a
    # usage ValueError (CLI exit 2), never a bare AssertionError
    def asserting_grace_fm(model):
        raise AssertionError(f"{model} not in registry")

    _install_stub(monkeypatch, "tensorpotential.calculator",
                  grace_fm=asserting_grace_fm)
    with pytest.raises(ValueError, match="unknown GRACE foundation"):
        make_calculator(CalculatorSpec("grace", model="GRACE-99-XXL"))


def test_all_potentials_have_defaults_and_packages():
    from irma.mlip.calculators import _DEFAULT_MODELS, _PACKAGES
    assert set(_DEFAULT_MODELS) == set(POTENTIALS)
    assert set(_PACKAGES) == set(POTENTIALS) | {"emt"}   # emt = test backend


def test_emt_test_backend_builds_without_torch():
    pytest.importorskip("ase")          # emt is ase's toy calculator
    from irma.mlip.calculators import make_calculator
    calc, meta = make_calculator(CalculatorSpec("emt"))
    assert meta["potential"] == "emt" and meta["device"] == "cpu"
    assert type(calc).__name__ == "EMT"


@pytest.mark.mlip_real
@pytest.mark.skipif("os.environ.get('IRMA_MLIP_REAL_TESTS') != '1'",
                    reason="real-potential test; opt in with IRMA_MLIP_REAL_TESTS=1")
def test_real_mattersim_instantiates():
    pytest.importorskip("mattersim")
    calc, meta = make_calculator(CalculatorSpec("mattersim"))
    assert calc is not None and meta["package_version"] != "unknown"


@pytest.mark.mlip_real
@pytest.mark.skipif("os.environ.get('IRMA_MLIP_REAL_TESTS') != '1'",
                    reason="real-potential test; opt in with IRMA_MLIP_REAL_TESTS=1")
def test_real_pet_mad_instantiates_with_pinned_version():
    pytest.importorskip("upet")
    from irma.mlip.calculators import canonicalize_spec
    spec = canonicalize_spec(CalculatorSpec("pet-mad"))
    assert "@" in spec.model                 # pinned, not floating
    calc, meta = make_calculator(spec)
    assert calc is not None and meta["checkpoint"] == spec.model


@pytest.mark.mlip_real
@pytest.mark.skipif("os.environ.get('IRMA_MLIP_REAL_TESTS') != '1'",
                    reason="real-potential test; opt in with IRMA_MLIP_REAL_TESTS=1")
def test_real_dpa3_forces_on_al():
    pytest.importorskip("deepmd")
    from ase.build import bulk
    calc, meta = make_calculator(CalculatorSpec("dpa3"))
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.calc = calc
    forces = atoms.get_forces()
    assert abs(forces).max() < 1e-4          # equilibrium symmetric cell
    assert meta["checkpoint_sha256"] is not None
