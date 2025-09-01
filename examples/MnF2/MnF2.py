import prodencer as pd
import numpy as np

space_group_number = 419

# Central Mn ion
center_Mn = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius_Mn = 1.12

# It can also be done for the F ions
# center_F = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
# radius_F = 0.7

# You can either use Abinit
# pd.project_harmonics("MnF2_DEN.nc", "abinit", center_Mn, radius_Mn, space_group_number, output_components=False)
# or vasp
pd.project_harmonics("MnF2_CHGCAR", "vasp", center_Mn, radius_Mn, space_group_number, output_components=False)