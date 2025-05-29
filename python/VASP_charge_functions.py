import numpy as np
import os

def VASP_get_charge(input="CHGCAR"):
    if not os.path.isfile(input):
        print("CHGCAR file not found")
        exit()

    chgcar = open(input, 'r')

    #read lattice parameters
    for i in range(2):
        line = chgcar.readline()

    lattice = np.zeros((3,3), dtype=float)

    line = chgcar.readline()
    lattice[0] = line.split() #Angst
    line = chgcar.readline()
    lattice[1] = line.split() #Angst
    line = chgcar.readline()
    lattice[2] = line.split() #Angst

    while True: #skip until newline
        line = chgcar.readline()
        if not line.strip(): #empty line
            line = chgcar.readline()
            break

    #read number of grid points
    grid = np.zeros(3, dtype=int)
    grid[0] = int(line.split()[0])
    grid[1] = int(line.split()[1])
    grid[2] = int(line.split()[2])

    #charge matrix
    charge = np.zeros(grid[0]*grid[1]*grid[2])

    #fill charge matrix
    for i in range(int(grid[0]*grid[1]*grid[2]/5)):
        line = chgcar.readline()
        charge[5*i:5*i+5] = np.array(line.split(), dtype=float)

    #last line
    if grid[0]*grid[1]*grid[2]%5 != 0:
        line = chgcar.readline()
        charge[-len(line.split()):] = np.array(line.split(), dtype=float)

    #reshape charge matrix
    charge = np.reshape(charge, (grid[0], grid[1], grid[2]), order='F')
    chgcar.close()

    print("charge data read successfully")

    return lattice, grid, charge

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


