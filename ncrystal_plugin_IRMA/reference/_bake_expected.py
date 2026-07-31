"""Bake the reference expected pack + NCMAT from the current IRMA exporter.

Run in the PRODUCER env (euphonic_env: irma + phonopy). Invoked by
regenerate_expected.sh with the repo root as argv[1]. IRMA is the single
producer/reference: this bakes the graphite mode-2 pack + the material NCMAT that
the plugin then samples.

This lives in a script file (not a stdin heredoc) on purpose: ``conda run``
does not forward stdin to the child process, so a
``conda run ... python - <<PY`` form would silently read an empty program
and bake nothing -- leaving a stale pack in place while the record step
happily re-records cross sections off it.
"""
import os
import sys
from pathlib import Path

os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")  # so the yaml's relative paths resolve

from irma.ncrystal.config import NCrystalExportConfig
from irma.ncrystal.build import build_packs
from irma.ncrystal.pack import write_pack

out = Path("ncrystal_plugin_IRMA/reference/expected")
out.mkdir(parents=True, exist_ok=True)
cfg = NCrystalExportConfig.from_yaml("ncrystal_plugin_IRMA/reference/graphite_reference.yaml")
# bare pack name in the @CUSTOM_IRMA section -> portable reference (resolves via CWD)
packs, ncmat = build_packs(cfg, pack_path_prefix=None, progress=lambda *a: None)
for p in packs:
    write_pack(p, out / f"{p.material_id}.irmapack")
(out / "graphite_reference.ncmat").write_text(ncmat, encoding="utf-8")
print("  baked:", [p.material_id for p in packs])
