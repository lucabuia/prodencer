import prodenser as pd
import numpy as np
import spglib
from spgrep import get_spacegroup_irreps
from spgrep.representation import get_character

# Import from ABINIT
lattice, grid, charge, mx, my, mz = pd.ABINIT_get_density("MnF2_DEN.nc")

# Import from VASP
# lattice, grid, charge, mx, my, mz = pd.VASP_get_density("MnF2_CHGCAR")


# Central Mn ion
center_Mn = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius_Mn = 2.11192

# One of the F ions
center_F = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
radius_F = 1.40289

# Get all symmetry-equivalent positions
positions_Mn = np.round(pd.wyckoff(center_Mn, 419),5)
positions_F = np.round(pd.wyckoff(center_F, 419),5)


coeffs_list = []  # Empty list to collect coefficients rows
# Loop over all positions with the same Wyckoff symbol and compute the d-multipoles
print("s- and d-multipoles for the Wyckoff position:\n")
for idx, pos in enumerate(positions_Mn):
    print(f"Position {idx + 1}: {pos}")
    coeffs_row = pd.project_sphere(mx, lattice, np.array(pos), radius_Mn)

    print(f"s: {np.round(coeffs_row[0],4)}")
    # print(coeffs_row[1:4])
    print(f"d: {np.round(coeffs_row[4:9],4)}")
    # print(coeffs_row[9:16])
    print(f"g: {np.round(coeffs_row[16:25],4)}")

    coeffs_list.append(coeffs_row)

# Convert list of coeffs to numpy array with shape (N_positions, coefficients_length)
coeffs = np.array(coeffs_list)

# pd.generate_xsf_file(mx, lattice, "mx.xsf")
pd.output_analytical_densities(lattice, positions_Mn, radius_Mn, coeffs, "MnF2_2a_Mx")