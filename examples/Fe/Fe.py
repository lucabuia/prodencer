import prodencer as pd

input_file = "Fe_SOC_DEN.nc"
dft_code = "abinit"

# lattice, atomic_positions, grid, charge, mx, my, mz = pd.ABINIT_get_density(input_file)
# pd.generate_xsf_file(mx, lattice, "mx.xsf")
# pd.generate_xsf_file(my, lattice, "my.xsf")
# pd.generate_xsf_file(mz, lattice, "mz.xsf")

# Project the charge and spin density around the Fe ion onto the tesseral harmonics
radius = 2

pd.project_harmonics(input_file, dft_code, [0.5,0.5,0.5], radius, decimals=5, output_components=True)