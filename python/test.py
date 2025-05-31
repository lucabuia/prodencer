# TEST: project the Mz spin density the central Mn ion of MnF2 onto the spherical harmonics. The only activated multipoles should be s, dxy, dz2. Compare VASP-GGA with Abinit-LDA.

from ABINIT_density_functions import ABINIT_get_density
from VASP_density_functions import VASP_get_density
from harmonics import project_sphere
import numpy as np

# lattice, grid, charge, mx, my, mz = ABINIT_get_density("GSo_DEN.nc")
lattice, grid, charge, mx, my, mz = VASP_get_density("CHGCAR")

#print("Lattice Vectors:\n", lattice)
#print("Grid Dimensions:", grid)
#print("Charge Density Shape:", charge.shape)

ng1, ng2, ng3 = charge.shape


a1, a2, a3 = np.linalg.norm(lattice, axis=0)
radius = np.sqrt(a1**2 + a2**2 + a3**2)/4
center = np.array([a1, a2, a3]) / 2

s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = project_sphere(mz, lattice, center, radius)

print(s)
print(dz2, dxz, dyz, dxy, dx2y2)
