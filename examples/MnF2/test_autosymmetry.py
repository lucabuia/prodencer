import prodencer as pd
import numpy as np
import spglib as sg

# lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mx, my, mz = pd.ABINIT_get_density("MnF2o_DEN.nc")
lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mx, my, mz = pd.VASP_get_density("CHGCAR")

structure=(lattice, np.transpose(atomic_positions), atomic_species)
str_object = sg.get_symmetry_dataset(structure, symprec=1e-5)
hall=str_object.hall_number
print(str_object)