import prodencer as pd

input_file = "AFM_DEN.nc"
dft_code = "abinit"

lattice, atomic_positions, atomic_species, grid, charge, mx, my, mz = pd.ABINIT_get_density(input_file)
pd.generate_xsf_file(mx, lattice, "mx_afm.xsf")
pd.generate_xsf_file(my, lattice, "my_afm.xsf")
pd.generate_xsf_file(mz, lattice, "mz_afm.xsf")

# Project the charge and spin density around the Fe ion onto the tesseral harmonics
radius = 2

pd.project_harmonics(input_file, dft_code, [0,0,0], radius, decimals=5, output_components=False, spacegroup=304)