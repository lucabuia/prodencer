import numpy as np
import netCDF4 as nc
import os

def ABINIT_get_density(input="GSo_DEN.nc"):
    if not os.path.isfile(input):
        print("ABINIT density file not found")
        exit()

    # Open the NetCDF file
    try:
        dataset = nc.Dataset(input, 'r')
    except Exception as e:
        print(f"Error opening NetCDF file: {e}")
        exit()

    # Read lattice vectors (converted to Angstroms)
    if "primitive_vectors" in dataset.variables:
        lattice = dataset.variables["primitive_vectors"][:]  # Transpose for (3x3) shape
    else:
        print("Primitive vectors not found in the file.")
        exit()

    # Read density data
    if "density" in dataset.variables:
        density = np.squeeze(dataset.variables["density"])
        density = np.transpose(density, (3, 2, 1, 0)) # Reorder so the dimensions are: x, y, z, spin component
    else:
        print("Density data not found in the file.")
        exit()

    # Extract grid dimensions
    ng1, ng2, ng3, components = density.shape

    # Calculate normalization constant such that everything is in atomic units
    norm_const = (ng1 * ng2 * ng3) / np.linalg.det(lattice)

    # Split the density into components
    charge_density = density[:, :, :, 0] / norm_const
    mx = density[:, :, :, 1] / norm_const
    my = density[:, :, :, 2] / norm_const
    mz = density[:, :, :, 3] / norm_const

    # Close the NetCDF file
    dataset.close()

    print("ABINIT density data read successfully")

    return lattice, (ng1, ng2, ng3), charge_density, mx, my, mz