def test_package_imports_without_irma():
    import sys
    import ncrystal_plugin_ENDFTSL as pkg
    from ncrystal_plugin_ENDFTSL import constants as c
    assert pkg.__version__ == "0.0.1"
    assert abs(c.THERM_EV - 0.0253) < 1e-9
    assert abs(c.T_LAT_K - c.THERM_EV / c.K_B_EV_PER_K) < 1e-6
    assert "irma" not in sys.modules  # never imported transitively
