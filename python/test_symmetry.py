from ABINIT_density_functions import ABINIT_get_density
from VASP_density_functions import VASP_get_density
from harmonics import project_sphere
from symmetry import wyckoff
import numpy as np
import spglib

# ABINIT
lattice, grid, charge, mx, my, mz = ABINIT_get_density("GSo_DEN.nc")

# # VASP
# lattice, grid, charge, mx, my, mz = VASP_get_density("CHGCAR")


# Central Mn ion
center = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius = 2.11192

# # One of the F ions
# center = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
# radius = 1.40289

# Get all symmetry-equivalent positions
positions = np.round(wyckoff(center, 419),5)

# Loop over all positions with the same Wyckoff symbol and compute the d-multipoles
print("s- and d-multipoles for the Wyckoff position:\n")
for idx, pos in enumerate(positions):
    print(f"Position {idx + 1}: {pos}")
    s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, *rest = project_sphere(mz, lattice, np.array(pos), radius)
    s, dz2, dxz, dyz, dxy, dx2y2 = map(lambda x: round(x, 6), (s, dz2, dxz, dyz, dxy, dx2y2))
    print(f"  s: {s}, dz2: {dz2}, dxz: {dxz}, dyz: {dyz}, dxy: {dxy}, dx2y2: {dx2y2}\n")