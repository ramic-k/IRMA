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
    with pytest.raises(ValueError, match="refusing to parse"):
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


@pytest.mark.parametrize("text, encoding", [
    ("x: !!%70ython/object/apply:os.getcwd []\n", "utf-8"),
    ("x: !<tag:yaml.org%2C2002:python/object/apply:os.getcwd> []\n", "utf-8"),
    ("\ufeff%TAG !x! tag:yaml.org,2002:\n---\nx: !x!python/object/apply:os.getcwd []\n",
     "utf-8"),
    ("x: !!python/object/apply:os.getcwd []\n", "utf-16"),
])
def test_escaped_bom_and_utf16_spellings_are_rejected(tmp_path, text, encoding):
    # libyaml decodes %-escapes in tag URIs, skips a UTF-8 BOM and reads a
    # UTF-16 file from its BOM, so each of these reaches the python tag
    path = tmp_path / "phonopy.yaml"
    path.write_bytes(text.encode(encoding))
    with pytest.raises(ValueError, match="refusing to parse"):
        reject_unsafe_phonopy_yaml(str(path))


def test_compressed_files_are_scanned_too(tmp_path):
    bad = tmp_path / "phonopy.yaml.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("x: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(ValueError, match="refusing to parse"):
        reject_unsafe_phonopy_yaml(str(bad))

    good = tmp_path / "ok.yaml.gz"
    with gzip.open(good, "wt") as fh:
        fh.write(LEGIT)
    assert reject_unsafe_phonopy_yaml(str(good)) is None
