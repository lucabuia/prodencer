from ABINIT_density_functions import ABINIT_get_density
from harmonics import generate_xsf_file
from symmetry import wyckoff
import numpy as np
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character

# ABINIT
lattice, grid, charge, mx, my, mz = ABINIT_get_density("Eu2Ir2O7_DEN.nc")

symmetry = spglib.get_symmetry_from_database(526) #Space group Hall number (1-530): https://yseto.net/en/sg/sg1
symm = np.array(symmetry['rotations'])
tnons = np.array(symmetry['translations']) # Non-symmorphic translations

position = np.array([1/8, 1/8, 1/8])

site_symm = []
for s in range(symm.shape[0]):
    # Apply the symmetry operation
    new_position = np.dot(symm[s], position) + tnons[s]
    new_position = np.mod(new_position, 1)  # Wrap within [0,1)
    
    # Check if the new position is equivalent to position
    if np.allclose(new_position, position, atol=1e-6):
        site_symm.append(s)


symm = symm[site_symm]
tnos = tnons[site_symm]

proj_c = np.zeros(charge.shape)
proj_mx = np.zeros(mx.shape)
proj_my = np.zeros(my.shape)
proj_mz = np.zeros(mz.shape)
for s in range(symm.shape[0]):
    i, j, k = np.meshgrid(
        np.arange(grid[0]),
        np.arange(grid[1]),
        np.arange(grid[2]),
        indexing='ij'
    )

    v = np.stack((i, j, k), axis=-1)  # shape (Nx, Ny, Nz, 3)
    v_new = np.tensordot(v, symm[s], axes=([3], [1])).astype(float)
    v_new += (tnons[s]) * grid

    i_new = v_new[..., 0] % grid[0]
    j_new = v_new[..., 1] % grid[1]
    k_new = v_new[..., 2] % grid[2]

    i_new = i_new.astype(int)
    j_new = j_new.astype(int)
    k_new = k_new.astype(int)

    proj_c[i, j, k]  += np.real( np.linalg.det(symm[s]) / (symm.shape[0]) * charge[i_new, j_new, k_new])
    proj_mx[i, j, k] += np.real( np.linalg.det(symm[s]) / (symm.shape[0]) * mx[i_new, j_new, k_new])
    proj_my[i, j, k] += np.real( np.linalg.det(symm[s]) / (symm.shape[0]) * my[i_new, j_new, k_new])
    proj_mz[i, j, k] += np.real( np.linalg.det(symm[s]) / (symm.shape[0]) * mz[i_new, j_new, k_new])

generate_xsf_file(proj_mx, lattice, "Monopole_Mx.xsf")
generate_xsf_file(proj_my, lattice, "Monopole_My.xsf")
generate_xsf_file(proj_mz, lattice, "Monopole_Mz.xsf")
generate_xsf_file(proj_c, lattice, "Monopole_c.xsf")