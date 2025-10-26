import prodencer as pd
import numpy as np

space_group_number = 425

# Mn ion
center_Mn = np.array([0, 0, 0]) # Coordinates of one Mn ion
radius_Mn = 2.11192

# Use Abinit
pd.project_harmonics("KMnF3o_DEN.nc", "abinit", center_Mn, radius_Mn, space_group_number, output_components=False, decimals=7)