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

def VASP_get_density(input="CHGCAR"):
    if not os.path.isfile(input):
        print("CHGCAR file not found")
        exit()

    chgcar = open(input, 'r')

    # Skip lattice scaling and header lines
    for _ in range(2):
        chgcar.readline()

    # Read lattice vectors
    lattice = np.zeros((3, 3), dtype=float)
    for i in range(3):
        lattice[i] = np.array(chgcar.readline().split(), dtype=float)

    # Skip lines until we find the grid shape
    while True:
        line = chgcar.readline()
        if not line.strip():  # empty line
            break

    # Initialize storage for densities
    densities = []

    while True:
        # Read a line and check if it has exactly three integers
        line = chgcar.readline()
        if not line:  # End of file
            break
        try:
            grid = np.array(line.split(), dtype=int)
            if len(grid) != 3:
                continue
        except ValueError:
            continue  # Skip lines that can't be converted to integers

        # Extract the grid dimensions
        ng1, ng2, ng3 = grid

        # Initialize the charge/spin density matrix
        density = np.zeros(ng1 * ng2 * ng3)

        # Read density values
        num_full_lines = (ng1 * ng2 * ng3) // 5
        for i in range(num_full_lines):
            density[5 * i:5 * i + 5] = np.array(chgcar.readline().split(), dtype=float)

        # Read remaining values
        remaining_values = (ng1 * ng2 * ng3) % 5
        if remaining_values > 0:
            density[-remaining_values:] = np.array(chgcar.readline().split(), dtype=float)

        # Reshape the density into a 3D array
        density = density.reshape((ng1, ng2, ng3), order='F')

        # Normalize the density
        density /= ng1 * ng2 * ng3

        # Add the density to the list
        densities.append(density)

    chgcar.close()

    if len(densities) == 1:
        print("CHGCAR contains only charge density")
        return lattice, grid, densities[0]
    elif len(densities) == 4:
        print("CHGCAR contains charge and spin densities")
        charge, mx, my, mz = densities
        return lattice, grid, charge, mx, my, mz
    else:
        raise ValueError(f"Unexpected number of densities in CHGCAR: {len(densities)}")


def VASP_write_charge(lattice, grid, charge, input="CHGCAR", output="new_CHGCAR"):

    with open(output, "w") as f:

        chgcar = open(input, 'r')
        
        while True: #skip until newline
            line = chgcar.readline()
            f.write(line)
            if not line.strip(): #empty line
                line = chgcar.readline()
                f.write(line)
                break

        charge = charge.flatten(order='F')
        
        for i in range(int(grid[0]*grid[1]*grid[2]/5)):
            f.write( ' ' + ' '.join(map("{:.10E}".format, charge[5*i:5*i+5])) + '\n' )
               
        #last line
        if grid[0]*grid[1]*grid[2]%5 != 0:
            f.write( ' ' + ' '.join(map("{:.10E}".format, charge[-((grid[0]*grid[1]*grid[2])%5):])) )