"""Thin re-export of the IRMA pack reader/writer.

The in-repo IRMA NCrystal plugin does NOT carry its own copy of the pack
format. ``irma.ncrystal.pack`` in the core package is the single producer and
reference for ``.irmapack`` files; this module simply re-exports it so callers can
write ``from ncrystal_plugin_IRMA.pack import IRMAPack, read_pack, write_pack``.
"""

from __future__ import annotations

from irma.ncrystal.pack import *  # noqa: F401,F403

# Explicit re-exports (so static tools and ``from ... import name`` see them):
from irma.ncrystal.pack import (  # noqa: F401
    IRMAPack,
    read_pack,
    write_pack,
    MAGIC,
)
