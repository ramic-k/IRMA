#!/usr/bin/env bash
# Regenerate the reference expected outputs (run when IRMA's mode-2 law changes on purpose).
#
# IRMA is the single producer/reference. This (1) bakes a graphite pack + NCMAT from
# the current IRMA via the exporter (euphonic_env), then (2) loads it through the
# installed plugin and records the reference cross sections + the IRMA git SHA
# (plugin env). The reference test (test_reference_graphite.py) then pins the plugin to
# these expected outputs. Run from a CLEAN checkout so the recorded SHA is meaningful.
#
# Usage: ncrystal_plugin_IRMA/reference/regenerate_expected.sh
# Envs (override via IRMA_PRODUCER_ENV / IRMA_PLUGIN_ENV):
#   producer = euphonic_env            (has irma + phonopy)
#   plugin   = ncrysta_coherent_plugin (has NCrystal + the built ncrystal_plugin_IRMA)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PRODUCER_ENV="${IRMA_PRODUCER_ENV:-euphonic_env}"
PLUGIN_ENV="${IRMA_PLUGIN_ENV:-ncrysta_coherent_plugin}"

echo "[1/2] baking expected pack + NCMAT from IRMA ($PRODUCER_ENV) ..."
cd "$REPO"
# NB: a script file, not `python - <<PY`. `conda run` does not forward stdin, so
# the stdin-heredoc form silently baked nothing (stale pack left in place).
OMP_NUM_THREADS=1 IRMA_JOBS=1 conda run -n "$PRODUCER_ENV" python "$HERE/_bake_expected.py" "$REPO"

echo "[2/2] recording reference cross sections via the plugin ($PLUGIN_ENV) ..."
conda run -n "$PLUGIN_ENV" bash -lc "PATH=\"\$CONDA_PREFIX/bin:\$PATH\" python '$HERE/_record_expected_xs.py' '$HERE/expected'"
echo "DONE. Review the diff in ncrystal_plugin_IRMA/reference/expected/ and commit."
