import prodencer as pd
import numpy as np

lattice, atomic_positions, atomic_species, grid, charge, mx, my, mz = pd.ABINIT_get_density("CrSbo_DEN.nc")
mz, new_c = pd.translate_density(mz, np.array([0, 0, 0.5]))


mz_transl, new_c = pd.translate_density(mz, np.array([0.5, 0.5, 0.0]))

pd.generate_xsf_file((mz+mz_transl)/2, lattice, "mz_IP.xsf")
pd.generate_xsf_file((mz-mz_transl)/2, lattice, "mz_OP.xsf")