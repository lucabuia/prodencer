import prodencer as pd
import numpy as np

space_group_number = 292

# Fe ions
center_Fe1 = np.array([0, 0, 0]) # Coordinates of one Fe ion 4a
center_Fe2 = np.array([0.94636, 0.75000, 0.06622]) # Coordinates of the Fe ion 4c
radius_Fe = 2.11532

Fe2 = pd.wyckoff(center_Fe2, space_group_number)

print("Wyckoff positions for Fe2:")
for i, pos in enumerate(Fe2):
    print(f"{i+1}: {np.round(pos, 6)}")


# Abinit
pd.project_harmonics("CHGCAR_crystal", "vasp", center_Fe2, radius_Fe, space_group_number, output_components=False, decimals=5)