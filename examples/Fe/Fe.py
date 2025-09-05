import prodencer as pd

input_file = "GSo_DEN.nc"
dft_code = "abinit"

# Project the charge and spin density around the Fe ion onto the tesseral harmonics
radius = 1.24

pd.project_harmonics(input_file, dft_code, [0,0,0], radius)