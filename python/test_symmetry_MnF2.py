from ABINIT_density_functions import ABINIT_get_density
from VASP_density_functions import VASP_get_density
from harmonics import project_sphere, output_analytical_densities, generate_xsf_file
from symmetry import wyckoff
import numpy as np
import spglib

# ABINIT
lattice, grid, charge, mx, my, mz = ABINIT_get_density("MnF2_DEN.nc")

# # VASP
# lattice, grid, charge, mx, my, mz = VASP_get_density("MnF2_CHGCAR")


# Central Mn ion
center = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius = 2.11192

# # One of the F ions
# center = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
# radius = 1.40289

# Get all symmetry-equivalent positions
positions = np.round(wyckoff(center, 419),5)


coeffs_list = []  # Empty list to collect coefficients rows
# Loop over all positions with the same Wyckoff symbol and compute the d-multipoles
print("s- and d-multipoles for the Wyckoff position:\n")
for idx, pos in enumerate(positions):
    print(f"Position {idx + 1}: {pos}")
    coeffs_row = project_sphere(mx, lattice, np.array(pos), radius)

    print(f"s: {np.round(coeffs_row[0],4)}")
    # print(coeffs_row[1:4])
    print(f"d: {np.round(coeffs_row[4:9],4)}")
    # print(coeffs_row[9:16])
    print(f"g: {np.round(coeffs_row[16:25],4)}")

    coeffs_list.append(coeffs_row)

# Convert list of coeffs to numpy array with shape (N_positions, coefficients_length)
coeffs = np.array(coeffs_list)

# generate_xsf_file(mx, lattice, "mx.xsf")
output_analytical_densities(lattice, positions, radius, coeffs, "MnF2_2a_Mx")