import prodenser as pd
import numpy as np
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character


# Import density from VASP CHGCAR file
lattice, grid, charge = pd.VASP_get_density("TiSe2_CHGCAR")

symmetry = spglib.get_symmetry_from_database(456) #Space group Hall number (1-530): https://yseto.net/en/sg/sg1
symm = np.array(symmetry['rotations'])
tnons = np.array(symmetry['translations']) # Non-symmorphic translations

kpoint = [0.0, 0.0, 0.0]

irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)

char_table = get_character(irreps[1]) #Get characters of GM1m
symm = symm[mapping_little_group]

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

proj_charge = pd.project_SC_irrep(charge, symm, tnons, translations_SC, kpoint, char_table)

pd.generate_xsf_file(proj_charge, lattice, "GM1m.xsf")