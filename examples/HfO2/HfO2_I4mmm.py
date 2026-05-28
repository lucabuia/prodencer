import prodencer as pd
import numpy as np

space_group_number = 424 # I4/mmm
radius_O = 1.20 # Angstrom

# Oxygen atom coordinates in reduced coordinates
center_O = np.array([0.25, 0.25, 0.25])

#assign symm and tnon manually
symm = np.zeros((16,3,3), dtype=float)

symm[0] = np.array([[ 1, 0, 0], 
                    [ 0, 1, 0], 
                    [ 0, 0, 1]]) #1
symm[1] = np.array([[-1, 0, 0], 
                    [ 0,-1, 0], 
                    [ 0, 0, 1]]) #2_001
symm[2] = np.array([[ 0,-1, 0],
                    [ 1, 0, 0], 
                    [ 0, 0, 1]]) #4+_001
symm[3] = np.array([[ 0, 1, 0], 
                    [-1, 0, 0], 
                    [ 0, 0, 1]]) #4-_001
symm[4] = np.array([[ 0, 1, 0], 
                    [ 1, 0, 0], 
                    [ 0, 0,-1]]) #2_010
symm[5] = np.array([[ 0,-1, 0], 
                    [-1, 0, 0], 
                    [ 0, 0,-1]]) #2_100
symm[6] = np.array([[ 1, 0, 0], 
                    [ 0,-1, 0], 
                    [ 0, 0,-1]]) #2_110
symm[7] = np.array([[-1, 0, 0], 
                    [ 0, 1, 0], 
                    [ 0, 0,-1]]) #2_1-10
symm[8] = np.array([[-1, 0, 0], 
                    [ 0,-1, 0], 
                    [ 0, 0,-1]]) #-1
symm[9] = np.array([[ 1, 0, 0], 
                    [ 0, 1, 0], 
                    [ 0, 0,-1]]) #m_001
symm[10] = np.array([[ 0, 1, 0], 
                     [-1, 0, 0], 
                     [ 0, 0,-1]]) #-4+_001
symm[11] = np.array([[ 0,-1, 0], 
                     [ 1, 0, 0], 
                     [ 0, 0,-1]]) #-4-_001
symm[12] = np.array([[ 0,-1, 0], 
                     [-1, 0, 0], 
                     [ 0, 0, 1]]) #m_010
symm[13] = np.array([[ 0, 1, 0], 
                     [ 1, 0, 0], 
                     [ 0, 0, 1]]) #m_100
symm[14] = np.array([[-1, 0, 0], 
                     [ 0, 1, 0], 
                     [ 0, 0, 1]]) #m_110
symm[15] = np.array([[ 1, 0, 0], 
                     [ 0,-1, 0], 
                     [ 0, 0, 1]]) #m_1-10

#While the conventional unit cell of I4/mmm has 2 translation components, the supercell provided which is a conventional cell of Fm-4m has 4 translation components.
tnons = np.zeros((4,3))
tnons[0] = np.array([0, 0, 0])
tnons[1] = np.array([0.5, 0.5, 0])
tnons[2] = np.array([0.5, 0, 0.5])
tnons[3] = np.array([0, 0.5, 0.5])

#character table for the GM3- irrep corresponding to order of symmetry operations defined above.
GM3m = np.array([ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1])

#The phase of GM wavevector according to the tranlation components
phase = np.array([1, 1, 1, 1])

#project charge density to GM3- irrep of I4/mmm and write output in VASP format for visualization
pd.project_irreps("CHGCAR", "vasp", output="vasp", auto_symmetry=False, spacegroup=space_group_number, threshold=0, manual_symm=symm, manual_tnons=tnons, manual_irreps=[GM3m], manual_phase=phase)

#multipole decomposition of the projected density
pd.project_harmonics("HfO2_I4mmm_charge_irrep1.vasp", "vasp", center_O, radius_O, auto_symmetry=False, spacegroup=space_group_number, output_components=False, decimals=8)
