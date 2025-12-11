# Example usage of project_single_irrep() for zone-boundary k-points.
#
# We simulated the 2x2x2 MLL phase of kagome metal CsV3Sb5, then project
# the charge density onto the irreps of P6/mmm at the M point (M1+ irrep)
# and the L point (L2- irrep). Additionally, we project onto the Gm5+ irrep,
# which should be the secondary order parameter according to symmetry analysis.

import prodencer as pd
import numpy as np
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character


# Import density from VASP CHGCAR file
lattice, atomic_positions, grid, charge = pd.VASP_get_density("CHGCAR")

# Import symmetries and irreps from spglib and spgrep
symmetry = spglib.get_symmetry_from_database(485) #Space group Hall number
symm = np.array(symmetry['rotations'])
tnons = np.array(symmetry['translations']) # Non-symmorphic translations
SC_size = [2,2,2] # Supercell size compared to the primitive cell


# Get irreps of little group at M and project onto M1+
kpoint_M = [0.5, 0.0, 0.0]
irreps_M, little_group_M = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint_M)
char_table_M = get_character(irreps_M[0]) #Get characters M_1^+
symm_M = symm[little_group_M]
tnons_M = tnons[little_group_M]

proj_charge_M1p = pd.project_single_irrep(charge, symm_M, tnons_M, char_table_M, SC_size, kpoint_M)
pd.generate_xsf_file(proj_charge_M1p, lattice, "M1p.xsf")


# Get irreps of little group at L and project onto L2-
kpoint_L = [0.0, 0.5, 0.5]
irreps_L, little_group_L = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint_L)
char_table_L = get_character(irreps_L[3]) #Get characters of L_2^-
symm_L = symm[little_group_L]
tnons_L = tnons[little_group_L]

proj_charge_L2m = pd.project_single_irrep(charge, symm_L, tnons_L, char_table_L, SC_size, kpoint_L)
pd.generate_xsf_file(proj_charge_L2m, lattice, "L2m.xsf")


# Get irreps of little group at Gamma and project onto Gamma5+
kpoint_GM = [0.0, 0.0, 0.0]
irreps_GM, little_group_GM = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint_GM)
char_table_GM = get_character(irreps_GM[9]) #Get characters
symm_GM = symm[little_group_GM]
tnons_GM = tnons[little_group_GM]

proj_charge_GM5p = pd.project_single_irrep(charge, symm_GM, tnons_GM, char_table_GM, SC_size, kpoint_GM)
pd.generate_xsf_file(proj_charge_GM5p, lattice, "GM5p.xsf")