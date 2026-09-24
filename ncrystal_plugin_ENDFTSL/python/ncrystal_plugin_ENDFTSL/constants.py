"""Physical constants (hardcoded; the package must not depend on irma)."""
K_B_EV_PER_K = 8.617333262e-5          # Boltzmann [eV/K]
THERM_EV = 0.0253                       # ENDF LAT=1 reference energy [eV]
T_LAT_K = THERM_EV / K_B_EV_PER_K       # ~293.6 K
HBAR2_OVER_2MN_EV_A2 = 2.0721248551069356e-3  # ħ²/2mₙ [eV·Å²] = WL2EKIN/(4π²)
