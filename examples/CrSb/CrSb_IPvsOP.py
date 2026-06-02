# Example usage of translate_density_to_point().
#
# We simulated altermagnetic CrSb in Abinit (check input file "CrSb.abi") and
# obtained the charge/spin density, contained in the "CrSbo_DEN.nc". 
# The function translates the spin density from one Cr ion to the other. We 
# then separate the density into in-phase (ferroic) and out-of-phase (anti-
# ferroic) parts

import prodencer as pd
import numpy as np

lattice, atomic_positions, atomic_species, grid, charge, mx, my, mz = pd.ABINIT_get_density("CrSbo_DEN.nc")


mz_transl, new_c = pd.translate_density_to_point(mz, np.array([0, 0, 0]), np.array([0, 0, 0.5]))


pd.generate_xsf_file((mz+mz_transl)/2, lattice, "mz_IP.xsf")
pd.generate_xsf_file((mz-mz_transl)/2, lattice, "mz_OP.xsf")