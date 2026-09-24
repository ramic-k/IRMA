# ncrystal_plugin_IRMA

The NCrystal plugin that lets NCrystal sample IRMA's mode-2 powder
thermal-scattering law: a per-temperature baked pack (`.irmapack`) carrying the
coherent one-phonon + anisotropic-Debye-Waller inelastic `S(α,β)` and the
anisotropic-DW elastic line. **IRMA is the single producer and reference** — this
plugin only *samples*; the packs and the material `.ncmat` are produced by
`irma.ncrystal` (`python -m irma.ncrystal`, see the [exporter
docs](../docs/ncrystal-plugin.md)). There is no IRMA at NCrystal runtime.

The plugin's Python package only carries the compiled C++ scattering model,
which reads packs natively and is discovered by NCrystal's plugin manager; the
pack format lives in `irma.ncrystal.pack`. The NCMAT activation section is `@CUSTOM_IRMA`.

## Build / install

Use an environment with NCrystal ≥ 4.3, cmake, ninja and scikit-build-core:

```bash
cd ncrystal_plugin_IRMA
rm -rf build _skbuild
PATH="$CONDA_PREFIX/bin:$PATH" python -m pip install --no-build-isolation --force-reinstall --no-cache-dir .
PATH="$CONDA_PREFIX/bin:$PATH" ncrystal-pluginmanager --test IRMA   # self-test
```

## Reference gate (`reference/`)

`reference/expected/` holds an **IRMA-produced** graphite pack + NCMAT and the
reference cross sections it yields through NCrystal, stamped with the IRMA git
SHA that baked them. `reference/test_reference_graphite.py` loads the expected
outputs through the installed plugin and asserts the cross sections still match —
so any drift in the pack reading, the C++ model, or the NCMAT wiring fails CI. It
skips where the plugin `.so` is not installed.

When IRMA's mode-2 law changes **on purpose**, regenerate the expected outputs
from a clean checkout:

```bash
ncrystal_plugin_IRMA/reference/regenerate_expected.sh   # bakes (irma env) + records (plugin env)
```

Run the gate (plugin env):

```bash
PATH="$CONDA_PREFIX/bin:$PATH" python -m pytest ncrystal_plugin_IRMA/reference/test_reference_graphite.py
```
