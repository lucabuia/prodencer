import prodencer as pd
import numpy as np
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt

lattice, atomic_positions, grid, charge, mx, my, mz = pd.ABINIT_get_density("CrSbo_DEN.nc")

# Magnetic (Zeeman) field
lattice_H, atomic_positions_H, grid_H, charge_H, mx_H, my_H, mz_H = pd.ABINIT_get_density("Hy/CrSb_10yo_DEN.nc")

pd.xrd_crystal(charge_H, lattice, plane='001', shift=1, do_plot=True)