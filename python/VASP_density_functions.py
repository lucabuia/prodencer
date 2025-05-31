import numpy as np
import os

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


