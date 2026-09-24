"""Physical constants and element data shared across IRMA core modules.

The numeric values match NJOY2016's physics module exactly (bit-compatible
LEAPR reproduction depends on it).
"""

# ============================================================================
# Physical constants (matching NJOY2016 physics module exactly)
# ============================================================================
BK = 8.617333262e-5       # Boltzmann constant, eV/K
EV = 1.602176634e-12      # erg/eV
CLIGHT = 2.99792458e10    # cm/s
AMU = 931.49410242e6 * EV / (CLIGHT * CLIGHT)  # g/amu
HBAR = 6.582119569e-16 * EV  # Planck/2pi, erg*s
AMASSN = 1.00866491595     # neutron mass in amu

THERM = 0.0253  # thermal energy in eV (for lat=1 scaling)

# WL2EKIN = h²/(2 m_n)  [eV·Å²]   (de Broglie E = h²/(2 m_n λ²)). The CODATA 2014
# value, copied from NCrystal's NCDefs.hh so Bragg thresholds match NCrystal; it
# differs by 9e-9 relative from the CODATA 2018 constants below.
WL2EKIN: float = 0.081804209605330899

# ============================================================================
# CODATA 2018 / SI 2019 constants (phonopy-backed noncubic paths)
# ============================================================================
# References:
#   E. Tiesinga, P. J. Mohr, D. B. Newell, and B. N. Taylor, "CODATA
#   recommended values of the fundamental physical constants: 2018",
#   Rev. Mod. Phys. 93, 025010 (2021); https://physics.nist.gov/constants
#   h, e, and k_B are exact by the 2019 SI redefinition (BIPM, 9th SI
#   brochure, 2019); BK above is that exact k_B/e value to 10 significant
#   figures and is shared by the classic (NJOY-reproducing) path.
# Derived values are computed from the defining constants rather than
# hardcoded, so they cannot drift out of sync.
from math import pi as _pi

PLANCK_J_S = 6.62607015e-34          # h    [J s]   (exact, SI 2019)
ECHARGE_C = 1.602176634e-19          # e    [C]     (exact, SI 2019)
KB_J_PER_K = 1.380649e-23            # k_B  [J/K]   (exact, SI 2019)
HBAR_J_S = PLANCK_J_S / (2.0 * _pi)  # ħ    [J s]
HBAR_EV_S = HBAR_J_S / ECHARGE_C     # ħ    [eV s]  = 6.582119569...e-16
NEUTRON_MASS_KG = 1.67492749804e-27  # m_n  [kg]    (CODATA 2018)
AMU_KG = 1.66053906660e-27           # m_u  [kg]    (CODATA 2018)

# 1 THz in eV: h * 1e12 / e = 4.135667696...e-3 eV
THZ_TO_EV = PLANCK_J_S * 1.0e12 / ECHARGE_C

# ħ²/(2 m_n) in meV·Å² = 2.0721248551... (the free-neutron kinematic
# constant linking alpha to Q²: E[meV] = HBAR2_OVER_2MN_MEV_A2 · Q[1/Å]²)
HBAR2_OVER_2MN_MEV_A2 = (
    HBAR_J_S ** 2 / (2.0 * NEUTRON_MASS_KG) / (ECHARGE_C * 1.0e-3) * 1.0e20
)

# ============================================================================
# Phonon-mode floor policy (mesh-sum numerical guards, modes 1/2)
# ============================================================================
# Per-mode one-phonon amplitudes and thermal displacements diverge as
# kT/ω² for ω → 0. In the exact BZ integral the acoustic divergence is
# integrable, but on a DISCRETE mesh the three Goldstone modes at Γ carry
# finite weight with a numerically meaningless ω (acoustic-sum-rule
# noise, commonly up to ~0.05 meV), so they must be excluded outright.
# Away from Γ every mode is physical and only an overflow guard is
# needed. Hence a two-tier rule:
#   - at Γ q-points: drop modes below GAMMA_ACOUSTIC_FLOOR_MEV
#     (optical Γ modes pass; Goldstone noise cannot);
#   - elsewhere: drop modes below MODE_ENERGY_FLOOR_MEV (1 μeV), far
#     under any acoustic energy at a nonzero mesh q-point, so genuinely
#     ultra-soft physics is kept.
# Shifted Monkhorst-Pack production meshes contain no Γ point and no
# sub-μeV modes, so both guards are inert there.
MODE_ENERGY_FLOOR_MEV = 1.0e-3
GAMMA_ACOUSTIC_FLOOR_MEV = 0.1

_Z_TO_SYMBOL = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',  8: 'O',
    9: 'F', 10: 'Ne',11: 'Na',12: 'Mg',13: 'Al',14: 'Si',15: 'P', 16: 'S',
   17: 'Cl',18: 'Ar',19: 'K', 20: 'Ca',21: 'Sc',22: 'Ti',23: 'V', 24: 'Cr',
   25: 'Mn',26: 'Fe',27: 'Co',28: 'Ni',29: 'Cu',30: 'Zn',31: 'Ga',32: 'Ge',
   33: 'As',34: 'Se',35: 'Br',36: 'Kr',37: 'Rb',38: 'Sr',39: 'Y', 40: 'Zr',
   41: 'Nb',42: 'Mo',43: 'Tc',44: 'Ru',45: 'Rh',46: 'Pd',47: 'Ag',48: 'Cd',
   49: 'In',50: 'Sn',51: 'Sb',52: 'Te',53: 'I', 54: 'Xe',55: 'Cs',56: 'Ba',
   57: 'La',58: 'Ce',59: 'Pr',60: 'Nd',61: 'Pm',62: 'Sm',63: 'Eu',64: 'Gd',
   65: 'Tb',66: 'Dy',67: 'Ho',68: 'Er',69: 'Tm',70: 'Yb',71: 'Lu',72: 'Hf',
   73: 'Ta',74: 'W', 75: 'Re',76: 'Os',77: 'Ir',78: 'Pt',79: 'Au',80: 'Hg',
   81: 'Tl',82: 'Pb',83: 'Bi',84: 'Po',85: 'At',86: 'Rn',87: 'Fr',88: 'Ra',
   89: 'Ac',90: 'Th',91: 'Pa',92: 'U', 93: 'Np',94: 'Pu',95: 'Am',96: 'Cm',
   97: 'Bk',98: 'Cf',99: 'Es',100:'Fm',101:'Md',102:'No',103:'Lr',
}

