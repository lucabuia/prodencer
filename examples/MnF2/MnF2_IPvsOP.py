import prodencer as pd
import numpy as np

# Central Mn ion
center_Mn = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius_Mn = 3

lattice, atomic_positions, grid, charge, mx, my, mz = pd.ABINIT_get_density("MnF2o_DEN.nc")

# Convert center from reduced coordinated to cartesian
center_Mn_cart = np.dot(center_Mn, lattice)

# Lattice parameters and real-space grid
a1, a2, a3 = np.linalg.norm(lattice, axis=0)
ng1, ng2, ng3 = mz.shape
rx, ry, rz = pd.real_space_grid(lattice, ng1, ng2, ng3)
r = np.sqrt(rx**2 + ry**2 + rz**2)

# Sphere - step function definition
sphere0 = np.sqrt((rx - center_Mn_cart[0])**2 + (ry - center_Mn_cart[1])**2 + (rz - center_Mn_cart[2])**2) < radius_Mn
sphere1 = np.sqrt((rx)**2 + (ry)**2 + (rz)**2) < radius_Mn
sphere2 = np.sqrt((rx-a1)**2 + (ry)**2 + (rz)**2) < radius_Mn
sphere3 = np.sqrt((rx)**2 + (ry-a2)**2 + (rz)**2) < radius_Mn
sphere4 = np.sqrt((rx-a1)**2 + (ry-a1)**2 + (rz)**2) < radius_Mn
sphere5 = np.sqrt((rx)**2 + (ry)**2 + (rz-a3)**2) < radius_Mn
sphere6 = np.sqrt((rx-a1)**2 + (ry)**2 + (rz-a3)**2) < radius_Mn
sphere7 = np.sqrt((rx)**2 + (ry-a2)**2 + (rz-a3)**2) < radius_Mn
sphere8 = np.sqrt((rx-a1)**2 + (ry-a1)**2 + (rz-a3)**2) < radius_Mn
mask = sphere0 | sphere1 | sphere2 | sphere3 | sphere4 | sphere5 | sphere6 | sphere7 | sphere8
mz[~mask] = 0

mz_transl, new_c = pd.translate_density(mz, np.array([0, 0, 0]))

pd.generate_xsf_file((mz+mz_transl)/2, lattice, "mz_IP.xsf")
pd.generate_xsf_file((mz-mz_transl)/2, lattice, "mz_OP.xsf")