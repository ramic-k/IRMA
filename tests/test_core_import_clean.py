"""The core import path must not load heavy optional packages.

The bare-install CI job proves the core imports succeed with no extras
installed. This test pins the complementary contract in the FULL environment
(where ase/phonopy/scipy ARE installed): importing the core entry points must
not pull them in. A module-level `import ase` sneaking into irma.cli would
pass the full suite and still break `pip install irma` -- this catches it in
every environment, in-process state notwithstanding, by checking a fresh
interpreter.
"""
import subprocess
import sys

HEAVY = ("ase", "torch", "phonopy", "scipy", "yaml", "periodictable",
         "mattersim", "sevenn", "orb_models", "mace")

CHECK = (
    "import sys; import irma, irma.cli, irma.core.nuclear_data; "
    "loaded = [m for m in {mods!r} if m in sys.modules]; "
    "print('LOADED:' + ','.join(loaded))"
)


def test_core_entry_points_do_not_import_heavy_packages():
    out = subprocess.run(
        [sys.executable, "-c", CHECK.format(mods=HEAVY)],
        capture_output=True, text=True, check=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("LOADED:")]
    assert line, f"probe produced no marker line: {out.stdout!r} {out.stderr!r}"
    loaded = [m for m in line[0][len("LOADED:"):].split(",") if m]
    assert not loaded, (
        f"core import path pulled heavy optional packages: {loaded}; "
        "these must stay behind function-level lazy imports")
