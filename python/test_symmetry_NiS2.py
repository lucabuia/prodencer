from ABINIT_density_functions import ABINIT_get_density
from VASP_density_functions import VASP_get_density
from harmonics import project_sphere, generate_xsf_file
from symmetry import wyckoff
import numpy as np
import spglib

# ABINIT
lattice, grid, charge, mx, my, mz = ABINIT_get_density("NiS2_DEN.nc")


# Central Ni ion
center = np.array([0.0, 0.0, 0.0]) # Coordinates of the central Mn ion
radius = 1.81432

# # One of the S ions
# center = np.array([0.38470, 0.38470, 0.38470]) # Coordinates of one of the F ions
# radius = 1.91527

generate_xsf_file(mz, lattice, "mz.xsf")

# Get all symmetry-equivalent positions
positions = np.round(wyckoff(center, 501),5)

# Initialize accumulators for the sums
total_s     = 0.0
total_dz2   = 0.0
total_dxz   = 0.0
total_dyz   = 0.0
total_dxy   = 0.0
total_dx2y2 = 0.0

# Loop over all positions and compute the d-multipoles
print("Calculating sums of s.Mx and d.Mx multipoles for the Wyckoff position:")
for idx, pos in enumerate(positions):
    s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, *rest = project_sphere(mx, lattice, np.array(pos), radius)
    total_s     += s
    total_dz2   += dz2
    total_dxz   += dxz
    total_dyz   += dyz
    total_dxy   += dxy
    total_dx2y2 += dx2y2

# Round the sums to 6 decimal places
total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2 = map(
    lambda x: round(x, 6), (total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2)
)

# Display the results
print(f"  Total s.Mx:     {total_s}")
print(f"  Total dz2.Mx:   {total_dz2}")
print(f"  Total dxz.Mx:   {total_dxz}")
print(f"  Total dyz.Mx:   {total_dyz}")
print(f"  Total dxy.Mx:   {total_dxy}")
print(f"  Total dx2y2.Mx: {total_dx2y2}\n")


# Initialize accumulators for the sums
total_s =     0.0
total_dz2 =   0.0
total_dxz =   0.0
total_dyz =   0.0
total_dxy =   0.0
total_dx2y2 = 0.0

# Loop over all positions and compute the d-multipoles
print("Calculating sums of s.My and d.My multipoles for the Wyckoff position:")
for idx, pos in enumerate(positions):
    s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, *rest = project_sphere(my, lattice, np.array(pos), radius)
    total_s     += s
    total_dz2   += dz2
    total_dxz   += dxz
    total_dyz   += dyz
    total_dxy   += dxy
    total_dx2y2 += dx2y2

# Round the sums to 6 decimal places
total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2 = map(
    lambda x: round(x, 6), (total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2)
)

# Display the results
print(f"  Total s.My:     {total_s}")
print(f"  Total dz2.My:   {total_dz2}")
print(f"  Total dxz.My:   {total_dxz}")
print(f"  Total dyz.My:   {total_dyz}")
print(f"  Total dxy.My:   {total_dxy}")
print(f"  Total dx2y2.My: {total_dx2y2}\n")

# Initialize accumulators for the sums
total_s     = 0.0
total_dz2   = 0.0
total_dxz   = 0.0
total_dyz   = 0.0
total_dxy   = 0.0
total_dx2y2 = 0.0

# Loop over all positions and compute the d-multipoles
print("Calculating sums of s.Mz and d.Mz multipoles for the Wyckoff position:")
for idx, pos in enumerate(positions):
    s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, *rest = project_sphere(mz, lattice, np.array(pos), radius)
    total_s   += s
    total_dz2 += dz2
    total_dxz += dxz
    total_dyz += dyz
    total_dxy += dxy
    total_dx2y2 += dx2y2

# Round the sums to 6 decimal places
total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2 = map(
    lambda x: round(x, 6), (total_s, total_dz2, total_dxz, total_dyz, total_dxy, total_dx2y2)
)

# Display the results
print(f"  Total s.Mz:     {total_s}")
print(f"  Total dz2.Mz:   {total_dz2}")
print(f"  Total dxz.Mz:   {total_dxz}")
print(f"  Total dyz.Mz:   {total_dyz}")
print(f"  Total dxy.Mz:   {total_dxy}")
print(f"  Total dx2y2.Mz: {total_dx2y2}\n")