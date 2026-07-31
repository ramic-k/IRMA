"""Out-of-module guard for the spectra physics gate.

The spectra forward-model physics gate (tests/test_spectra_forward.py) was once
disabled WHOLESALE on CI by a blanket module-level ``pytestmark =
pytest.mark.skipif(...)`` keyed on a hardcoded local path -- so every clean
checkout skipped the entire gate and ran nothing, silently (skips do not fail).

A canary INSIDE that module cannot protect against a relapse: a reintroduced
module-level skip would skip the canary too. This guard therefore lives in a
SEPARATE module (no skip of its own) and fails -- loudly -- if the gate file ever
regains a module-level skip marker. Gate genuinely data-dependent tests with the
per-test ``@requires_ext`` decorator instead of a blanket module skip.
"""
import re
from pathlib import Path

GATE_FILE = Path(__file__).resolve().parent / "test_spectra_forward.py"


def test_spectra_gate_has_no_blanket_module_skip():
    src = GATE_FILE.read_text()
    # a column-0 `pytestmark = ... skip ...` re-disables the whole gate on CI
    offending = re.search(r"(?m)^pytestmark\b.*\bskip", src)
    assert offending is None, (
        "tests/test_spectra_forward.py has a module-level skip marker:\n"
        f"    {offending.group(0)!r}\n"
        "This disables the ENTIRE spectra physics gate on CI (skips are silent). "
        "Remove it and gate only the data-dependent tests with @requires_ext.")


def test_spectra_gate_runs_real_tests_without_external_data():
    """The gate file must contain real, un-gated physics tests -- not only
    @requires_ext ones. If every test became external-only, the gate would again
    run nothing on a clean CI checkout."""
    src = GATE_FILE.read_text()
    test_defs = re.findall(r"(?m)^def (test_\w+)", src)
    # tests carrying @requires_ext on the line above their def are external-only
    gated = set(re.findall(r"@requires_ext\s*\ndef (test_\w+)", src))
    ungated = [t for t in test_defs if t not in gated]
    assert len(ungated) >= 5, (
        f"only {len(ungated)} un-gated spectra tests remain "
        f"({len(gated)} are @requires_ext); the gate must keep real CI-runnable "
        "physics tests so a clean checkout exercises the forward model.")
