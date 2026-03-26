# Example usage of project_harmonics().
#
# We simulated altermagnetic KMnF3 in Abinit (check input file "KMnF3.abi") and
# obtained the charge/spin density, contained in the "KMnF3_DEN.nc". 
# The function prints the atomic electric and magnetic multipoles around the 
# two Mn ions. You can check that most magnetic multipoles cancel each other
# (specifically all the dipoles) but the 32-poles and the octupoles do not.
#
# More details available here: https://doi.org/10.1103/kq6x-7jfc

import prodencer as pd
import numpy as np

space_group_number = 425 # I4/mcm, for full list of Hall numbers check https://yseto.net/en/sg/sg1

# Mn ion
center_Mn = np.array([0.2257, 0.7257, 0.5]) # Coordinates of one Mn ion
radius_Mn = 2.11 # In units of Bohr

# Use Abinit
pd.project_harmonics("KMnF3_DEN.nc", "abinit", center_Mn, radius_Mn, space_group_number, output_components=False, decimals=7)