import prodencer as pd
import numpy as np
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character


# Import density from VASP CHGCAR file
lattice, grid, charge = pd.VASP_get_density("CHGCAR")

translations_SC = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

symmetry = spglib.get_symmetry_from_database(485) #Space group Hall number (1-530): https://yseto.net/en/sg/sg1
symm = np.array(symmetry['rotations'])
tnons = np.array(symmetry['translations']) # Non-symmorphic translations


kpoint = [0.5, 0.0, 0.0]

irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)

char_table = get_character(irreps[0]) #Get characters
symm = symm[mapping_little_group]

proj_charge = pd.project_SC_irrep(charge, symm, tnons, translations_SC, kpoint, char_table)

pd.generate_xsf_file(proj_charge, lattice, "M1p.xsf")



# kpoint = [0.0, 0.5, 0.5]

# irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)

# char_table = get_character(irreps[3]) #Get characters
# symm = symm[mapping_little_group]

# proj_charge = pd.project_SC_irrep(charge, symm, tnons, translations_SC, kpoint, char_table)

# pd.generate_xsf_file(proj_charge, lattice, "L2m.xsf")


# kpoint = [0.0, 0.0, 0.0]

# irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)

# char_table = get_character(irreps[9]) #Get characters
# symm = symm[mapping_little_group]

# proj_charge = pd.project_SC_irrep(charge, symm, tnons, translations_SC, kpoint, char_table)

# pd.generate_xsf_file(proj_charge, lattice, "GM5p.xsf")