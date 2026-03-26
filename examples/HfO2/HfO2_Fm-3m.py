import prodencer as pd
import numpy as np
import spglib

space_group_number = 523 # Fm-3m
radius_O = 1.20 # Angstrom

# Oxygen atom coordinates in reduced coordinates
center_O = np.array([0.25, 0.25, 0.25])

#Use spglib to get the symmetry operations for Fm-3m space group
symmetry = spglib.get_symmetry_from_database(523)
symm = np.array(symmetry['rotations'])[:48]
tnons = np.array(symmetry['translations'])[0::48]

#character table for the 3D GM4- irrep corresponding to order of symmetry operations from spglib.
GM4m = 3*np.array([ 3, -3, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, 1, -1])
phase = np.array([1, 1, 1, 1])

#project charge density to GM4- irrep of Fm-3m and write output in VASP format for visualization
#pd.project_irreps("CHGCAR", "vasp", output="vasp", auto_symmetry=False, spacegroup=space_group_number, threshold=0, manual_symm=symm, manual_tnons=tnons, manual_irreps=[GM4m], manual_phase=phase)

#multipole decomposition of the projected density
pd.project_harmonics("HfO2_Fm-3m_charge_irrep1.vasp", "vasp", center_O, radius_O, spacegroup=space_group_number, output_components=False, decimals=8)
