"""Dump Mantid PyChop (GPL, BLACK-BOX REFERENCE) resolution references.

Run with a Mantid-environment python (PyChop ships inside Mantid's scripts
directory). If PyChop is not importable directly, point the MANTID_SCRIPTS
environment variable at that scripts directory:

    MANTID_SCRIPTS=/path/to/mantid/scripts python \
        tests/chopper_reference/dump_pychop_reference.py [out.json]

PyChop is GPL-3.0+; IRMA is BSD-3. This script ONLY calls PyChop's public API
(getResolution / getWidths) to capture reference NUMBERS for validating IRMA's
independent model. It does not copy any PyChop source. The captured JSON is the
ground truth the chopper-reference comparison tests check against.
"""
import sys, json, os

MANTID_SCRIPTS = os.environ.get("MANTID_SCRIPTS", "")
if MANTID_SCRIPTS:
    sys.path.insert(0, MANTID_SCRIPTS)
import numpy as np
from pychop.Instruments import Instrument

# (instrument, chopper-package-or-None, frequency-or-list, [Ei list]) cases.
# Frequencies/Ei chosen in each instrument's real operating range.
CASES = [
    ("ARCS",    "ARCS-700-1.5-AST", 600, [300.0, 100.0]),
    ("ARCS",    "ARCS-700-0.5-AST", 600, [600.0]),
    ("ARCS",    "ARCS-100-1.5-AST", 300, [50.0]),
    ("SEQUOIA", "High-Resolution",  600, [120.0, 60.0]),
    ("SEQUOIA", "High-Flux",        600, [250.0]),
    ("MAPS",    "A",                400, [400.0, 250.0]),
    ("MAPS",    "B",                300, [150.0]),
    ("MAPS",    "S",                400, [300.0]),
    ("MARI",    "A",                400, [400.0, 180.0]),
    ("MARI",    "C",                300, [80.0]),
    ("MARI",    "S",                350, [100.0]),
    ("MERLIN",  "S",                400, [180.0, 80.0]),
    ("MERLIN",  "G",                400, [120.0]),
    ("HYSPEC",  "OnlyOne",          180, [15.0, 35.0]),
    ("HYSPEC",  "OnlyOne",          420, [50.0]),
    # disk-chopper instruments: frequency is the independent-frequency list
    ("CNCS",    None,   [300, 60],  [3.0, 12.0, 25.0]),
    ("CNCS",    None,   [240, 60],  [6.0]),
    ("LET",     None,   [240, 120], [3.7, 8.0, 15.0]),
    ("LET",     None,   [180, 90],  [2.0]),
]


def _setup(name, package, freq):
    ins = Instrument(name)
    if package is not None:
        ins.setChopper(package)
    ins.setFrequency(freq)
    return ins


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "pychop_reference.json")
    records = []
    for name, package, freq, eis in CASES:
        for Ei in eis:
            try:
                ins = _setup(name, package, freq)
                ins.setEi(Ei)
                Et = np.linspace(0.0, 0.95 * Ei, 20)
                dE = np.asarray(ins.getResolution(Etrans=Et, Ei_in=Ei), float)
                rec = {"instrument": name, "package": package, "frequency": freq,
                       "Ei": Ei, "Etrans": Et.tolist(), "dE": dE.tolist()}
                # component breakdown (debug aid only)
                try:
                    w = ins.getWidths(Ei)
                    rec["widths_keys"] = list(w.keys()) if hasattr(w, "keys") else None
                    rec["widths"] = {k: (np.asarray(v).tolist()
                                         if hasattr(v, "__len__") else float(v))
                                     for k, v in (w.items() if hasattr(w, "items") else [])}
                except Exception as e:
                    rec["widths_err"] = repr(e)
                rec["elastic_dE"] = float(dE[0])
                records.append(rec)
                print(f"OK  {name:8s} {str(package):18s} f={freq} Ei={Ei:7.2f}  "
                      f"elastic dE={dE[0]:.5f}  ({np.min(dE):.4f}..{np.max(dE):.4f})")
            except Exception as e:
                print(f"ERR {name:8s} {str(package):18s} f={freq} Ei={Ei}: {e!r}")
    json.dump(records, open(out, "w"), indent=1)
    print(f"\nwrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
