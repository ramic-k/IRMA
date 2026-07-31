"""C++ loader validation gate (review CPP-6): non-finite per-site neutron
data must be a LOAD error, not a NaN cross section.

NC::safe_str2dbl deliberately accepts the literals "nan"/"inf", NaN passes
every ordinary comparison (the old `< 0.0` test included), and NCrystal's
PowderBragg does not reject NaN plane weights either — so before the fix a
NaN b_coh/sigma_inc silently propagated to a NaN Bragg cross section.
These tests feed hand-poisoned copies of the vendored expected pack
through the INSTALLED plugin's C++ loader and assert the load now throws.

Same gating as the reference tests: runs only where NCrystal and the
compiled ``ncrystal_plugin_IRMA`` .so are installed; skips cleanly
otherwise. Every poisoned copy gets a UNIQUE material/pack file name so
NCrystal's factory caches can never alias two cases.
"""
from __future__ import annotations

import site
from pathlib import Path

import pytest

NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)

_HERE = Path(__file__).resolve().parent
_EXPECTED = _HERE / "expected"


def _plugin_installed() -> bool:
    for root in site.getsitepackages():
        if (Path(root) / "ncrystal_plugin_IRMA" / "plugins"
                / "libNCPlugin_IRMA.so").exists():
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _plugin_installed(),
    reason="ncrystal_plugin_IRMA .so not installed (build it in the plugin env)")


def _write_case(dst: Path, name: str, field: str | None, edit) -> str:
    """Copy the expected NCMAT+pack into dst under unique names, optionally
    rewriting one pack field line (`edit` maps the stored float list to a
    same-length list of replacement TOKENS, e.g. ["nan", ...]). Returns the
    NCMAT file name."""
    ncmat_name = f"{name}.ncmat"
    pack_name = f"{name}__C.irmapack"
    ncmat = (_EXPECTED / "graphite_reference.ncmat").read_text()
    ncmat = ncmat.replace("pack graphite_reference__C.irmapack",
                          f"pack {pack_name}")
    (dst / ncmat_name).write_text(ncmat)
    pack_lines = (_EXPECTED / "graphite_reference__C.irmapack"
                  ).read_text().splitlines()
    if field is not None:
        for i, line in enumerate(pack_lines):
            if line.startswith(field + " "):
                vals = line.split("=", 1)[1].split()
                new = edit(vals)
                assert len(new) == len(vals)
                pack_lines[i] = f"{field} = " + " ".join(str(v) for v in new)
                break
        else:
            raise AssertionError(f"expected pack has no {field!r} line")
    (dst / pack_name).write_text("\n".join(pack_lines) + "\n")
    return ncmat_name


def test_nan_incoherent_xs_is_a_load_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(tmp_path, "mal_nan_incohxs",
                       "elastic_u_incoherent_xs_barn",
                       lambda v: ["nan"] + v[1:])
    with pytest.raises(NC.NCBadInput,
                       match="elastic_u_incoherent_xs_barn"):
        NC.createScatter(f"{name};temp=296K")


def test_nan_coherent_scatlen_is_a_load_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(tmp_path, "mal_nan_bcoh",
                       "elastic_u_coherent_scatlen_sqrtbarn",
                       lambda v: v[:-1] + ["nan"])
    with pytest.raises(NC.NCBadInput,
                       match="elastic_u_coherent_scatlen_sqrtbarn"):
        NC.createScatter(f"{name};temp=296K")


def test_inf_coherent_scatlen_is_a_load_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = _write_case(tmp_path, "mal_inf_bcoh",
                       "elastic_u_coherent_scatlen_sqrtbarn",
                       lambda v: ["inf"] + v[1:])
    with pytest.raises(NC.NCBadInput,
                       match="elastic_u_coherent_scatlen_sqrtbarn"):
        NC.createScatter(f"{name};temp=296K")


def test_near_miss_pristine_copy_still_loads(tmp_path, monkeypatch):
    import math
    monkeypatch.chdir(tmp_path)
    name = _write_case(tmp_path, "mal_near_miss_pristine", None, None)
    sc = NC.createScatter(f"{name};temp=296K")
    xs = float(sc.crossSectionIsotropic(0.025))
    assert math.isfinite(xs) and xs > 0.0
