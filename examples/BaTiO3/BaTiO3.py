import prodencer as pd

input_file = "GSo_DEN.nc"
dft_code = "abinit"

# lattice, grid, charge = pd.ABINIT_get_density(input_file)
# pd.generate_xsf_file(charge, lattice, "charge.xsf")

# Project the charge onto the irreps of the parent cubic group
sg_cubic = 517
pd.project_irreps(input_file, dft_code, sg_cubic)


# Project the charge aroung the Ti ion onto the tesseral harmonics
sg_tetragonal = 376
Ti_coords = [0.5,0.5,0.55]
radius = 1.5

pd.project_harmonics(input_file, dft_code, Ti_coords, radius, sg_tetragonal, False)