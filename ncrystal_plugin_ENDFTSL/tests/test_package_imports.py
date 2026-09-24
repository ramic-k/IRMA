def test_package_imports_without_irma():
    import sys
    import ncrystal_plugin_ENDFTSL as pkg
    from ncrystal_plugin_ENDFTSL import constants as c
    assert pkg.__version__ == "0.3.0"
    assert abs(c.THERM_EV - 0.0253) < 1e-9
    assert abs(c.T_LAT_K - c.THERM_EV / c.K_B_EV_PER_K) < 1e-6
    assert "irma" not in sys.modules  # never imported transitively


def test_one_version_everywhere():
    """__version__, pyproject.toml and the CMake project carry one version."""
    import pathlib
    import re
    import tomllib
    import ncrystal_plugin_ENDFTSL as pkg
    root = pathlib.Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    cmake = re.search(r"project\(.*VERSION (\S+)", (root / "CMakeLists.txt").read_text())
    assert pkg.__version__ == project["version"] == cmake.group(1)

