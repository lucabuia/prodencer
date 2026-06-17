# Example usage of project_harmonics().
#
# We simulated altermagnetic CrSb in Abinit (check input file "CrSb.abi") and
# obtained the charge/spin density, contained in the "CrSbo_DEN.nc". 
# The function prints the atomic electric and magnetic multipoles around the 
# two Cr ions. You can check that most magnetic multipoles cancel each other
# (specifically all the dipoles) but the 32-poles and the octupoles do not.
#
# More details available here: https://doi.org/10.1103/kq6x-7jfc

import prodencer as pd
import numpy as np

space_group_number = 488 # P6_3/mmc, for full list of Hall numbers check https://yseto.net/en/sg/sg1

# Cr ion
center_Cr = np.array([0, 0, 0.5]) # Coordinates of one Cr ion
radius_Cr = 1.05 # In units of Angstroms

# Using Abinit
pd.project_harmonics("CrSbo_DEN.nc", "abinit", center_Cr, radius_Cr, space_group_number, output_components=False, decimals=5)