# Example usage of project_harmonics()
#
# We simualated bcc ferromagnetic Iron in Abinit and obtained the spin density with
# spin-orbit coupling ("Fe_SOC_DEN.nc") and without ("Fe_NoSOC_DEN.nc"). We project
# the spin densities around the Fe ion onto the magnetic multipoles. The results are
# that there is a magnetic dipole mz, and some higher-order non-collinear multipoles
# (inversion-even) in the mx, my channels.

import prodencer as pd

input_file = "Fe_SOC_DEN.nc"
dft_code = "abinit"

# lattice, atomic_positions, atomic_species, grid, charge, mx, my, mz = pd.ABINIT_get_density(input_file)
# pd.generate_xsf_file(mx, lattice, "mx.xsf")
# pd.generate_xsf_file(my, lattice, "my.xsf")
# pd.generate_xsf_file(mz, lattice, "mz.xsf")

# Project the charge and spin density around the Fe ion onto the tesseral harmonics
radius = 1

pd.project_harmonics(input_file, dft_code, [0.5,0.5,0.5], radius, decimals=5, output_components=False)