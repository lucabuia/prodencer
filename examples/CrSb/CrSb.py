import prodencer as pd
import numpy as np

space_group_number = 488

# Cr ion
center_Cr = np.array([0, 0, 0.5]) # Coordinates of one Cr ion
radius_Cr = 2.10818

# Using Abinit
pd.project_harmonics("Hy/CrSb_10yo_DEN.nc", "abinit", center_Cr, radius_Cr, space_group_number, output_components=False, decimals=5)
# pd.project_harmonics("0_CHGCAR", "vasp", center_Cr, radius_Cr, space_group_number, output_components=False, decimals=5)