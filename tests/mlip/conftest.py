"""Hermeticity for the mlip test package.

Every test gets a private irma-mlip cache: the developer's real
~/.cache/irma-mlip (env registry, downloaded checkpoints) must never
leak into stub-based tests -- a registered `mace` environment would
silently turn make_calculator stubs into live subprocess dispatch.
"""
import pytest


@pytest.fixture(autouse=True)
def _private_mlip_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("IRMA_MLIP_CACHE",
                       str(tmp_path_factory.mktemp("mlip-cache")))
    for var in list(__import__("os").environ):
        if var.startswith("IRMA_MLIP_PYTHON_"):
            monkeypatch.delenv(var)
    from irma.mlip import envs
    envs._META_CACHE.clear()
    yield
