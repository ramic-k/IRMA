# Third-Party Notices

IRMA is distributed under the BSD 3-Clause License (see `LICENSE`).
The notices below cover third-party code IRMA derives from, retained
as their licenses require.

IRMA's classic kernels (irma/core/kernels.py) implement the LEAPR algorithms
of NJOY2016 and are in part derived from its Fortran source (leapr.f90).
NJOY2016 is distributed under the BSD 3-Clause License with the following
notice, retained here as required:

  Copyright (c) 2016, Los Alamos National Security, LLC
  All rights reserved.
  Copyright 2016. Los Alamos National Security, LLC. This software was
  produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos
  National Laboratory (LANL), which is operated by Los Alamos National
  Security, LLC for the U.S. Department of Energy. The U.S. Government has
  rights to use, reproduce, and distribute this software. NEITHER THE
  GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY,
  EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
  SOFTWARE. If software is modified to produce derivative works, such
  modified software should be clearly marked, so as not to confuse it with
  the version available from LANL.

IRMA is such a derivative work and is clearly marked as a reimplementation;
it is not the version available from LANL.

--------------------------------------------------------------------------------

IRMA's generalized coherent-elastic machinery (irma/core/crystal.py) is in
part derived from NCrystal (reciprocal-lattice and Bragg-edge algorithms
mirroring NCLatticeUtils.cc; the wl2ekin constant convention from
NCDefs.hh), per T. Kittelmann et al., Comp. Phys. Comm. 267 (2021) 108082.
NCrystal is distributed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0) with the following notice:

  NCrystal : A library for thermal neutron transport in crystals and other
  materials

  Copyright 2015-2025 NCrystal developers

  This software was mainly developed at the European Spallation Source ERIC
  (ESS) and the Technical University of Denmark (DTU).

The NCrystal-derived portions have been modified for IRMA (translated to
Python and adapted to IRMA's data structures); they are redistributed here
under the terms of the Apache License, Version 2.0, which is compatible
with this distribution.

--------------------------------------------------------------------------------

IRMA's crystalline-extinction models (irma/core/extinction.py) are ported
(translated to Python, NOT imported) from the NCrystal plugin ncplugin-CrysXT
(https://github.com/dddijulio/ncplugin-CrysXT, upstream revision
e68626062ba5593bd5a4c59f0ab46a9038ddd888), in particular the BC2025
Sabine/Becker-Coppens extinction recipes in src/NCbc2025.hh. ncplugin-CrysXT
is distributed under the Apache License, Version 2.0. The ported portions
have been modified for IRMA (translated to Python, restructured around
IRMA's per-plane Bragg data, and the 'lux' recipe intentionally omitted);
they are redistributed here under the terms of the Apache License,
Version 2.0.

The full text of the Apache License, Version 2.0 is included in this
distribution as LICENSES/Apache-2.0.txt (it applies to the NCrystal- and
ncplugin-CrysXT-derived portions identified above; the remainder of IRMA is
BSD-3-Clause, see LICENSE).

--------------------------------------------------------------------------------

IRMA's direct-geometry chopper resolution model
(irma/spectra/chopper_resolution.py) is an independent BSD reimplementation
of the standard analytical time-of-flight resolution formalism, written from
the published literature (S. Ikeda & J.M. Carpenter, NIM A 239 (1985) 536;
M. Marseguerra & G. Pauli, NIM 4 (1959) 140; N. Violini et al., NIM A 736
(2014) 31; C.G. Windsor, "Pulsed Neutron Scattering", 1981). No source code
from Mantid PyChop (GPL-3.0+, https://github.com/mantidproject/mantid) is
included in, ported into, or linked by this distribution.

PyChop is acknowledged in two non-code capacities:

  * Factual instrument parameters (flight paths, aperture and chopper slot
    geometries, Ikeda-Carpenter moderator coefficients, detector angular
    limits) are facts about the instruments, collected from the public
    instrument descriptions and from the instrument data files shipped with
    Mantid PyChop. Facts carry no copyright; the source is credited here
    and in the module docstring.
  * The moderator pulse-width tables and the two disk-chopper calibration
    constants in irma/spectra/chopper_resolution.py, and the validation
    reference tests/chopper_reference/pychop_reference.json (repository only,
    not packaged), are numerical OUTPUT obtained by evaluating PyChop's
    public API as a black box — tabulated samples of its moderator-width
    function on IRMA's own grids, and two calibration scalars. The dumper
    script (tests/chopper_reference/dump_pychop_reference.py) contains no PyChop
    code and requires the user's own Mantid installation to run. Program
    output of this kind is not subject to the program's license.

--------------------------------------------------------------------------------

Optional phonon workflows use phonopy (BSD 3-Clause, A. Togo and
contributors). ENDF-6 serialization uses endf-parserpy (MIT, G. Schnabel).
Neither package's source is included in this distribution.
