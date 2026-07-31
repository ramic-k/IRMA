"""In-repo IRMA NCrystal plugin (Python shim).

This package is the Python side of the IRMA NCrystal plugin that lives *inside*
the IRMA repository. It is a thin shim over :mod:`irma.ncrystal` -- the single
producer and reference for ``.irmapack`` files and the ``@CUSTOM_IRMA`` NCMAT
section. The compiled C++ scattering model is bundled alongside this package
(under ``ncrystal_plugin_IRMA/plugins/libNCPlugin_IRMA.so``) and is discovered
by NCrystal's plugin manager; the Python here exists only to re-export the pack
reader/writer so downstream code can do::

    from ncrystal_plugin_IRMA.pack import IRMAPack, read_pack, write_pack

The IRMA core package (which provides ``irma.ncrystal``) is only needed to
*produce* or *inspect* packs from Python. The compiled scattering model reads
packs natively in C++, so NCrystal's plugin discovery must not depend on
``irma`` being importable. We therefore re-export the pack API only if the core
package is present, and import cleanly (without the symbols) if it is not.
"""

from __future__ import annotations

try:  # IRMA core package present (e.g. euphonic_env): expose the pack API.
    from irma.ncrystal.pack import IRMAPack, read_pack, write_pack  # noqa: F401

    __all__ = ["IRMAPack", "read_pack", "write_pack"]
except ImportError:  # IRMA core absent (e.g. the bare plugin runtime env).
    # The C++ plugin .so is still bundled and discoverable; only the optional
    # Python pack producers/readers are unavailable here.
    __all__ = []
