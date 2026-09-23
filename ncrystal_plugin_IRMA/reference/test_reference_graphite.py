"""Reference gate: the plugin must reproduce IRMA's baked law.

IRMA is the single producer and reference. ``expected/`` holds an IRMA-produced
graphite pack + NCMAT and the reference cross sections IRMA's law yields through
NCrystal, stamped with the IRMA git SHA that baked them. This test loads the
expected outputs through the installed plugin and asserts the cross sections still
match — so a regression in the plugin's pack reading, the C++ scattering model, or
the NCMAT wiring fails CI. When IRMA's law changes on purpose, regenerate the
expected outputs (``regenerate_expected.sh``); the recorded SHA flags a stale
reference.

Runs only where NCrystal AND the in-repo ``ncrystal_plugin_IRMA`` plugin ``.so``
are installed (e.g. the ``ncrysta_coherent_plugin`` env / the reference CI job);
skips cleanly otherwise.
"""
from __future__ import annotations

import json
import os
import shutil
import site
from pathlib import Path

import pytest

NC = pytest.importorskip("NCrystal", exc_type=ModuleNotFoundError)

_HERE = Path(__file__).resolve().parent
_EXPECTED = _HERE / "expected"

# tolerances: same pack + same NCrystal is deterministic; 1% rel (+ a 2e-3 barn
# floor for the near-zero incoherent channel) absorbs NCrystal patch-version
# drift while still catching any real format/reading regression.
_RTOL = 1.0e-2
_ATOL = 2.0e-3


def _plugin_installed() -> bool:
    for root in site.getsitepackages():
        if (Path(root) / "ncrystal_plugin_IRMA" / "plugins"
                / "libNCPlugin_IRMA.so").exists():
            return True
    return False


pytestmark = pytest.mark.skipif(
    not _plugin_installed(),
    reason="ncrystal_plugin_IRMA .so not installed (build it in the plugin env)")


@pytest.fixture
def in_expected_dir(monkeypatch):
    # the expected NCMAT references its pack by BARE name -> NCrystal resolves it
    # against CWD, so run from the expected dir (portable across machines/CI).
    monkeypatch.chdir(_EXPECTED)
    yield


def test_plugin_reproduces_irma_expected(in_expected_dir):
    expected = json.loads((_EXPECTED / "graphite_reference_xs.json").read_text())
    energies_ev = [e / 1000.0 for e in expected["energies_meV"]]
    temp = expected["temperature_K"]

    for comp, ref in expected["cross_sections_barn"].items():
        cfg = f"graphite_reference.ncmat;temp={temp}K" + (
            "" if comp == "total" else f";comp={comp}")
        sc = NC.createScatter(cfg)
        got = [float(sc.crossSectionIsotropic(e)) for e in energies_ev]
        for e_mev, g, r in zip(expected["energies_meV"], got, ref):
            assert g == pytest.approx(r, rel=_RTOL, abs=_ATOL), (
                f"{comp} at {e_mev} meV: plugin {g:.6g} b, expected {r:.6g} b")


def _load_xs(workdir, cfg, energies_ev):
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        sc = NC.createScatter(cfg)
        return [float(sc.crossSectionIsotropic(e)) for e in energies_ev]
    finally:
        os.chdir(cwd)


def _copy_expected_with_edited_field(dst, field, edit):
    # copy the expected ncmat + pack into dst, rewriting one per-site pack field line
    # (`edit` maps the list of stored values to a new list of the same length).
    shutil.copy(_EXPECTED / "graphite_reference.ncmat", dst / "graphite_reference.ncmat")
    pack = (_EXPECTED / "graphite_reference__C.irmapack").read_text().splitlines()
    for i, line in enumerate(pack):
        if line.startswith(field + " "):
            vals = [float(v) for v in line.split("=", 1)[1].split()]
            new = edit(vals)
            pack[i] = f"{field} = " + " ".join(f"{v:.17g}" for v in new)
            break
    else:
        raise AssertionError(f"expected pack has no {field!r} line to edit")
    (dst / "graphite_reference__C.irmapack").write_text("\n".join(pack) + "\n")


def test_plugin_honors_pack_neutron_data_not_atomdb(tmp_path):
    # The discriminating test: graphite's config b_coh/sigma_inc happen to equal
    # NCrystal's carbon atom DB, so the reference gate alone cannot tell whether the
    # plugin reads the PACK or the DB. Perturb the pack's per-site neutron data and
    # assert the cross section follows the PACK -- coherent elastic scales as b_coh^2,
    # incoherent elastic scales linearly with sigma_inc -- which the atom DB (fixed at
    # carbon) could never reproduce. This is the end-to-end proof of #3.
    expected = json.loads((_EXPECTED / "graphite_reference_xs.json").read_text())
    temp = expected["temperature_K"]
    # energies above the first graphite Bragg edge, so coherent elastic is non-zero.
    energies = [0.0053, 0.025, 0.1]

    base_coh = _load_xs(_EXPECTED, f"graphite_reference.ncmat;temp={temp}K;comp=coh_elas",
                        energies)
    base_inc = _load_xs(_EXPECTED, f"graphite_reference.ncmat;temp={temp}K;comp=incoh_elas",
                        energies)

    # (a) double every site's b_coh -> coherent F(hkl) |F|^2 scales by 4 at every plane
    coh_dir = tmp_path / "coh"
    coh_dir.mkdir()
    _copy_expected_with_edited_field(
        coh_dir, "elastic_u_coherent_scatlen_sqrtbarn", lambda vs: [2.0 * v for v in vs])
    pert_coh = _load_xs(coh_dir, f"graphite_reference.ncmat;temp={temp}K;comp=coh_elas",
                        energies)
    seen_coh = False
    for b, p in zip(base_coh, pert_coh):
        if b > 1.0e-4:
            seen_coh = True
            assert p / b == pytest.approx(4.0, rel=2.0e-2)
    assert seen_coh, "no non-zero coherent elastic sample to discriminate on"

    # (b) raise sigma_inc 500x (0.001 -> 0.5 b) -> incoherent elastic scales linearly
    inc_dir = tmp_path / "inc"
    inc_dir.mkdir()
    _copy_expected_with_edited_field(
        inc_dir, "elastic_u_incoherent_xs_barn", lambda vs: [500.0 * v for v in vs])
    pert_inc = _load_xs(inc_dir, f"graphite_reference.ncmat;temp={temp}K;comp=incoh_elas",
                        energies)
    seen_inc = False
    for b, p in zip(base_inc, pert_inc):
        if b > 1.0e-6:
            seen_inc = True
            assert p / b == pytest.approx(500.0, rel=2.0e-2)
    assert seen_inc, "no non-zero incoherent elastic sample to discriminate on"


def test_pack_carries_irma_provenance():
    expected = json.loads((_EXPECTED / "graphite_reference_xs.json").read_text())
    sha = expected["irma_git_sha"]
    assert sha and sha != "unknown"
    pack = (_EXPECTED / "graphite_reference__C.irmapack").read_text()
    assert pack.splitlines()[0] == "IRMAPACK_TEXT_V1"
    pack_sha = next(line.split("=")[1].strip() for line in pack.splitlines()
                    if line.startswith("meta.irma_git_sha"))
    assert pack_sha == sha, "pack and xs-record provenance disagree"


def _read_pack_tensor_sites(pack_path):
    """Per-site (sigma_inc_barn, U_tensor 3x3) + elastic_scale from a pack file."""
    fields = {}
    for line in Path(pack_path).read_text().splitlines()[1:]:
        if "=" in line and not line.startswith("meta."):
            k, v = [s.strip() for s in line.split("=", 1)]
            fields[k] = v
    tensors = [float(x) for x in fields["elastic_u_tensors_a2"].split()]
    sigmas = [float(x) for x in fields["elastic_u_incoherent_xs_barn"].split()]
    scale = float(fields.get("elastic_scale", "1.0"))
    nsites = len(tensors) // 9
    sites = []
    for i in range(nsites):
        U = [tensors[9 * i:9 * i + 3],
             tensors[9 * i + 3:9 * i + 6],
             tensors[9 * i + 6:9 * i + 9]]
        sites.append((sigmas[i], U))
    return sites, scale


def test_directional_incoherent_elastic_matches_python_oracle(tmp_path):
    """The directional mode's C++ mixture-of-exponentials must reproduce the
    closed-form Python reference (irma.core.incoherent_dw) -- the two sides
    share the branch constants but use INDEPENDENT integration methods, so
    this genuinely cross-checks the C++ sampler's cross section."""
    np = pytest.importorskip("numpy", exc_type=ModuleNotFoundError)
    incoherent_dw = pytest.importorskip("irma.core.incoherent_dw", exc_type=ModuleNotFoundError)

    expected = json.loads((_EXPECTED / "graphite_reference_xs.json").read_text())
    temp = expected["temperature_K"]

    # copy the expected material, raise sigma_inc to a visible 0.5 b per site,
    # and switch on the directional mode.
    _copy_expected_with_edited_field(
        tmp_path, "elastic_u_incoherent_xs_barn", lambda vs: [500.0 * v for v in vs])
    pack_path = tmp_path / "graphite_reference__C.irmapack"
    pack_lines = pack_path.read_text().splitlines()
    insert_at = next(i for i, l in enumerate(pack_lines)
                     if l.startswith("elastic_u_incoherent_xs_barn"))
    pack_lines.insert(insert_at + 1, "incoherent_elastic_mode = directional")
    pack_path.write_text("\n".join(pack_lines) + "\n")

    energies_ev = list(np.geomspace(1e-4, 5.0, 27))
    got = _load_xs(tmp_path, f"graphite_reference.ncmat;temp={temp}K;comp=incoh_elas",
                   energies_ev)

    # Python oracle: same pack data, closed-form f(Q) + independent trapezoid.
    sites, scale = _read_pack_tensor_sites(pack_path)
    site_scale = scale / len(sites)
    channels = [(sigma * site_scale, incoherent_dw.u_eigenvalues(U))
                for sigma, U in sites]
    wl = np.array([NC.ekin2wl(e) for e in energies_ev])       # Angstrom
    ksq = (2.0 * np.pi / wl) ** 2                             # 1/Angstrom^2
    ref = incoherent_dw.sigma_elinc_directional(ksq, channels)

    assert np.allclose(got, ref, rtol=1e-4), (
        f"directional incoherent elastic drifted from the Python oracle:\n"
        f"plugin={got}\noracle={list(ref)}")

    # physics sanity: the directional tail must EXCEED the isotropic (trace/3)
    # result at the same (highest) energy (Jensen inequality).
    iso = _load_xs(_EXPECTED, f"graphite_reference.ncmat;temp={temp}K;comp=incoh_elas",
                   [energies_ev[-1]])
    assert got[-1] > iso[0] * 500.0    # same sigma_inc scaling as the edit
