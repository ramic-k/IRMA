"""MLIP phonon front end: structure -> relaxed cell + force constants -> bundle.

Every heavy import (ase, phonopy, torch-backed potential packages, yaml) is
function-level and lazy; importing irma.mlip and its submodules must stay
core-clean (see tests/test_core_import_clean.py and the bare-install CI job).
"""
