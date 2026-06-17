# Example usage of the function project_irreps().
#
# We simulated tetragonal (ferroelectric) BaTiO3 in Abinit, the density is contained
# in the file "GSo_DEN.nc". We project the density onto the irreps of the parent
# cubic group at the Gamma point. The results are the trivial irrep Gm1+, the primary
# order paramter Gm4- and the secondary order parameter Gm3+. You can check the labels
# using the Bilbao Crystallographic Server or ISOTROPY, and comparing the character
# tables present there with the ones obtained by spgrep and printed in the .pdout file.

import prodencer as pd

input_file = "GSo_DEN.nc"
dft_code = "abinit"

# # Optionally plot the charge density as a .xsf file that you can open in Vesta or XCrysDen
# lattice, atomic_positions, atomic_species, grid, charge = pd.ABINIT_get_density(input_file)
# pd.generate_xsf_file(charge, lattice, "charge.xsf")


# Project the charge onto the irreps of the parent cubic group
sg_cubic = 517 # Pm3-m, for the list of all Hall numbers check https://yseto.net/en/sg/sg1
pd.project_irreps(input_file, dft_code, sg_cubic)


# # Optionally project the charge aroung the Ti ion onto the tesseral harmonics
# sg_tetragonal = 376 # P4mm
# Ti_coords = [0.5,0.5,0.55]
# radius = 2.26 # In units of Angstrom
# pd.project_harmonics(input_file, dft_code, Ti_coords, radius, sg_tetragonal, False)