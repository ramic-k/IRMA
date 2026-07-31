"""reject_unsafe_phonopy_yaml: the pre-parse guard for untrusted
phonopy.yaml files.

phonopy parses YAML with PyYAML's unsafe loader, so a
`!!python/object/apply:` tag executes AT PARSE TIME. The guard is a
streaming textual scan that must fire BEFORE any phonopy parse, without
itself parsing (or executing) anything.
"""
import gzip

import pytest

from irma.core.phonopy_io import reject_unsafe_phonopy_yaml

LEGIT = (
    "phonopy:\n"
    "  version: 2.21.0\n"
    "unit_cell:\n"
    "  lattice:\n"
    "  - [4.05, 0.0, 0.0]\n"
    "  points:\n"
    "  - symbol: Al\n"
    "    coordinates: [0.0, 0.0, 0.0]\n"
    "force_constants:\n"
    "  format: full\n"
)


def test_legitimate_phonopy_yaml_passes(tmp_path):
    path = tmp_path / "phonopy.yaml"
    path.write_text(LEGIT)
    assert reject_unsafe_phonopy_yaml(str(path)) is None


def test_python_tag_is_rejected_without_execution(tmp_path):
    canary = tmp_path / "pwned"
    path = tmp_path / "phonopy.yaml"
    path.write_text(
        LEGIT + f'extra: !!python/object/apply:os.system ["touch {canary}"]\n')
    with pytest.raises(ValueError, match="!!python/"):
        reject_unsafe_phonopy_yaml(str(path))
    assert not canary.exists()          # scanned, never parsed/executed


def test_verbatim_tag_form_is_rejected(tmp_path):
    path = tmp_path / "phonopy.yaml"
    path.write_text(
        'x: !<tag:yaml.org,2002:python/object/apply:os.getcwd> []\n')
    with pytest.raises(ValueError, match="refusing to parse"):
        reject_unsafe_phonopy_yaml(str(path))


def test_tag_directive_alias_is_rejected(tmp_path):
    # %TAG can alias the python namespace past a plain '!!python/' scan;
    # phonopy never emits directives, so any %TAG is rejected outright
    # the alias below defeats BOTH substring checks ('!x!python/', no
    # 'tag:yaml.org,2002:python'), so only the %TAG rule can catch it
    path = tmp_path / "phonopy.yaml"
    path.write_text("%TAG !x! tag:yaml.org,2002:\n"
                    "---\n"
                    "x: !x!python/object/apply:os.getcwd []\n")
    with pytest.raises(ValueError, match="%TAG"):
        reject_unsafe_phonopy_yaml(str(path))


def test_compressed_files_are_scanned_too(tmp_path):
    bad = tmp_path / "phonopy.yaml.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("x: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(ValueError, match="!!python/"):
        reject_unsafe_phonopy_yaml(str(bad))

    good = tmp_path / "ok.yaml.gz"
    with gzip.open(good, "wt") as fh:
        fh.write(LEGIT)
    assert reject_unsafe_phonopy_yaml(str(good)) is None


def test_real_bundle_scale_content_passes(tmp_path):
    # near-miss bulk: a large embedded-FC-like body with numbers that
    # merely CONTAIN suspicious-looking substrings in comments is fine
    path = tmp_path / "phonopy.yaml"
    rows = "".join(f"  - [0.{i % 7}, 0.0, 0.0]\n" for i in range(5000))
    path.write_text(LEGIT + "more_rows:\n" + rows +
                    "# note: python users see docs\n")
    assert reject_unsafe_phonopy_yaml(str(path)) is None
