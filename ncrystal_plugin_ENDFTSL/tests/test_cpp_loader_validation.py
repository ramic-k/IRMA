"""C++ loader validation gate (review CPP-5): the coherent-elastic block
checks must be NaN-safe and must check element 0.

NC::safe_str2dbl deliberately accepts the literals "nan"/"inf", and the
old validation loop started at i=1 with plain `<`/`>` comparisons — so a
NaN anywhere (and any bad element 0, including a non-positive first edge,
which makes sigma = S(E)/E diverge as E->0) reached EndfCohElasScatter
unchecked. These tests convert the vendored example TAPE into a pack (the
`.endftslpack` itself is git-ignored converter output, so a fresh checkout
never has one), then feed hand-poisoned copies of that pack through the
INSTALLED plugin's C++ loader and assert the load now throws. Runs only
where NCrystal and the compiled plugin are installed (same gating as the
reference tests); skips cleanly otherwise. Every case gets unique file
names so NCrystal factory caches can never alias two cases. This test
suite must NOT import irma.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)

from ncrystal_plugin_ENDFTSL.__main__ import main as convert_main  # noqa: E402

_HERE = Path(__file__).resolve().parent
_TAPE = _HERE.parent / "examples" / "graphite" / "graphite_mef_296K.endf"


def _plugin_available() -> bool:
    try:
        return "ENDFTSL" in [p[0] for p in NC.browsePlugins()]
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _plugin_available(),
    reason="compiled ENDFTSL plugin not installed (build it in the plugin env)")

_NCMAT_TEMPLATE = """NCMAT v5
@DENSITY
  2.26 g_per_cm3
@DYNINFO
  element C
  fraction 1
  type freegas
@CUSTOM_ENDFTSL
  pack {pack_name}
"""


@pytest.fixture(scope="session")
def example_pack(tmp_path_factory) -> Path:
    """The graphite example pack, generated once per session from the
    committed tape — exactly the convert step examples/README.md documents.
    The tests then poison COPIES of this pack, so a fresh checkout (no
    leftover local converter output) exercises the same loader paths."""
    out = tmp_path_factory.mktemp("endftsl_example_pack")
    convert_main([str(_TAPE), "-o", str(out), "--material-id", "graphite",
                  "--symbol", "C", "--mass", "12.0107", "--density", "2.26",
                  "--temperature", "296"])
    pack = out / "graphite.endftslpack"
    assert pack.is_file(), "converter did not produce graphite.endftslpack"
    return pack


def _write_case(src_pack: Path, dst: Path, name: str, field: str | None,
                edit) -> str:
    """Copy the generated example pack into dst under a unique name,
    optionally rewriting one field line (`edit` maps the stored value
    tokens to a same-length list of replacement tokens). Returns the NCMAT
    file name (written alongside, referencing the pack by bare name)."""
    ncmat_name = f"{name}.ncmat"
    pack_name = f"{name}.endftslpack"
    (dst / ncmat_name).write_text(_NCMAT_TEMPLATE.format(pack_name=pack_name))
    pack_lines = src_pack.read_text().splitlines()
    if field is not None:
        for i, line in enumerate(pack_lines):
            if line.startswith(field + " "):
                vals = line.split("=", 1)[1].split()
                new = edit(vals)
                assert len(new) == len(vals)
                pack_lines[i] = f"{field} = " + " ".join(str(v) for v in new)
                break
        else:
            raise AssertionError(f"example pack has no {field!r} line")
    (dst / pack_name).write_text("\n".join(pack_lines) + "\n")
    return ncmat_name


def test_nan_cumulative_s_element0_is_a_load_error(example_pack, tmp_path,
                                                   monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_nan_cums0",
                       "coh_cumulative_s", lambda v: ["nan"] + v[1:])
    with pytest.raises(NC.NCBadInput, match="coh_cumulative_s"):
        NC.createScatter(f"{name};temp=296K")


def test_negative_cumulative_s_element0_is_a_load_error(example_pack, tmp_path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_neg_cums0",
                       "coh_cumulative_s", lambda v: ["-0.5"] + v[1:])
    with pytest.raises(NC.NCBadInput, match="coh_cumulative_s"):
        NC.createScatter(f"{name};temp=296K")


def test_nan_cumulative_s_mid_array_is_a_load_error(example_pack, tmp_path,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_nan_cums_mid",
                       "coh_cumulative_s", lambda v: v[:2] + ["nan"] + v[3:])
    with pytest.raises(NC.NCBadInput, match="coh_cumulative_s"):
        NC.createScatter(f"{name};temp=296K")


def test_nonpositive_first_edge_is_a_load_error(example_pack, tmp_path,
                                                monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_zero_edge0",
                       "coh_edges_ev", lambda v: ["0"] + v[1:])
    with pytest.raises(NC.NCBadInput, match="coh_edges_ev"):
        NC.createScatter(f"{name};temp=296K")


def test_nan_first_edge_is_a_load_error(example_pack, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_nan_edge0",
                       "coh_edges_ev", lambda v: ["nan"] + v[1:])
    with pytest.raises(NC.NCBadInput, match="coh_edges_ev"):
        NC.createScatter(f"{name};temp=296K")


def test_near_miss_pristine_copy_still_loads(example_pack, tmp_path,
                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(example_pack, tmp_path, "mal_near_miss_pristine",
                       None, None)
    sc = NC.createScatter(f"{name};temp=296K")
    xs = float(sc.crossSectionIsotropic(0.025))
    assert math.isfinite(xs) and xs > 0.0
