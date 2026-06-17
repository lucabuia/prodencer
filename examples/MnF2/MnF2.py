# Example usage of project_harmonics().
#
# We simulated altermagnetic MnF2 in Abinit (check input file "MnF2.abi") and
# obtained the charge/spin density, contained in the "MnF2o_DEN.nc". 
# The function prints the atomic electric and magnetic multipoles around the 
# two Mn ions. You can check that most magnetic multipoles cancel each other
# (specifically all the dipoles) but the 32-poles and the octupoles do not.
#
# More details available here: https://doi.org/10.1103/kq6x-7jfc

import prodencer as pd
import numpy as np

space_group_number = 419 # P4_2/mnm, for full list of Hall numbers check https://yseto.net/en/sg/sg1

# Central Mn ion
center_Mn = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius_Mn = 1.05 # In units of Angstroms

# It can also be done for the F ions
# center_F = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
# radius_F = 0.7

# You can either use Abinit
pd.project_harmonics("MnF2o_DEN.nc", "abinit", center_Mn, radius_Mn, space_group_number, output_components=False, decimals=5)
# or vasp
# pd.project_harmonics("CHGCAR", "vasp", center_Mn, radius_Mn, space_group_number, output_components=False, decimals=5)