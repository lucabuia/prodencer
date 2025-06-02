# TEST: project the Mz spin density the central Mn ion and one of the F ions of MnF2 onto the spherical harmonics. The only activated multipoles should be s, dxy, dz2. Compare VASP-GGA with Abinit-LDA.

from ABINIT_density_functions import ABINIT_get_density
from VASP_density_functions import VASP_get_density
from harmonics import project_sphere
import numpy as np

# ABINIT
# lattice, grid, charge, mx, my, mz = ABINIT_get_density("GSo_DEN.nc")

# VASP
lattice, grid, charge, mx, my, mz = VASP_get_density("CHGCAR")


# Central Mn ion
center = np.array([0.5, 0.5, 0.5]) # Coordinates of the central Mn ion
radius = 2.11192

# One of the F ions
# center = np.array([0.30464, 0.30464, 0]) # Coordinates of one of the F ions
# radius = 1.40289


s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = project_sphere(mz, lattice, center, radius)

print(s)
print(dxy, dyz, dz2, dxz, dx2y2)
print(dz2, dxz, dyz, dxy, dx2y2)
