"""Lazy in-process access to the validated noncubic inelastic driver."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoncubicInelasticControls:
    """User-facing controls for IRMA's noncubic inelastic MT4 path."""

    num_directions: int
    multiphonon_num_directions: int
    multiphonon_max_order: int
    # Opt-in: when True, the engine sizes the multiphonon order up to the value
    # required to converge the incoherent Poisson(2W) sum for the anisotropic
    # Debye-Waller factor (never below the deck nphon). When False (default), the
    # deck's Card 3 nphon is honored verbatim and the engine only warns if it is
    # below that requirement. Set from the Card 6g optional 3rd field.
    auto_multiphonon_order: bool = False


def _impl() -> Any:
    """Import the engine lazily so this facade stays cheap to import."""
    return importlib.import_module("irma.core.noncubic_engine")


def parse_args(argv=None):
    """See :func:`irma.core.noncubic_engine.parse_args`."""
    return _impl().parse_args(argv)


def compute_from_args(*args, **kwargs):
    """See :func:`irma.core.noncubic_engine.compute_from_args`."""
    return _impl().compute_from_args(*args, **kwargs)


def build_compute_context(*args, **kwargs):
    """See :func:`irma.core.noncubic_inelastic_context.build_compute_context`."""
    from irma.core.noncubic_inelastic_context import build_compute_context as _build

    return _build(*args, **kwargs)


def write_results(*args, **kwargs):
    """See :func:`irma.core.noncubic_engine.write_results`."""
    return _impl().write_results(*args, **kwargs)


def run_noncubic_sab_inprocess(*args, **kwargs):
    """See :func:`irma.core.noncubic_engine.run_noncubic_sab_inprocess`."""
    return _impl().run_noncubic_sab_inprocess(*args, **kwargs)


def main(argv=None):
    """See :func:`irma.core.noncubic_engine.main`."""
    return _impl().main(argv)


__all__ = [
    "build_compute_context",
    "compute_from_args",
    "main",
    "NoncubicInelasticControls",
    "parse_args",
    "run_noncubic_sab_inprocess",
    "write_results",
]
