"""Regenerate reference/expected/graphite_ref_xs.json (total sigma on a fixed grid).

Run with the compiled plugin installed + endf_parserpy available, e.g.:

    conda run -n irma_and_mcstas_environment python reference/_record_expected.py

Deterministic: same tape + converter + NCrystal -> identical sigma(E), so the
recorded grid is a stable byte/tolerance reference for test_reference_graphite.py.
"""
import json
import tempfile
from pathlib import Path

import NCrystal as NC
from ncrystal_plugin_ENDFTSL.__main__ import main as convert_main

HERE = Path(__file__).parent
TAPE = HERE.parent / "examples" / "graphite" / "graphite_mef_296K.endf"


def main() -> int:
    out = Path(tempfile.mkdtemp())
    convert_main([str(TAPE), "-o", str(out), "--material-id", "graphite",
                  "--symbol", "C", "--mass", "12.0107", "--density", "2.26",
                  "--temperature", "296"])
    sc = NC.createScatter(f"{out / 'graphite.ncmat'};temp=296.0K")
    grid = [1.0e-3 * 1.3 ** i for i in range(25)]   # ~1 meV .. ~250 eV
    ref = {f"{e:.6g}": sc.crossSectionIsotropic(e) for e in grid}
    (HERE / "expected").mkdir(exist_ok=True)
    path = HERE / "expected" / "graphite_ref_xs.json"
    path.write_text(json.dumps(ref, indent=1) + "\n")
    print(f"wrote {path} with {len(ref)} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
