import numpy as np
import netCDF4 as nc
import os
import sys
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character


def ABINIT_get_density(input="GSo_DEN.nc"):
    """
    Read an ABINIT density NetCDF file.

    Returns:
      - If the file contains only the charge density (components == 1):
          lattice, (ng1, ng2, ng3), charge
      - If the file contains charge + 3 spin components (components == 4):
          lattice, (ng1, ng2, ng3), charge, mx, my, mz

    Raises:
      - FileNotFoundError if the file is missing
      - RuntimeError for unexpected component counts
    """
    if not os.path.isfile(input):
        raise FileNotFoundError(f"ABINIT density file not found: {input}")

    # Open the NetCDF file
    try:
        dataset = nc.Dataset(input, 'r')
    except Exception as e:
        raise RuntimeError(f"Error opening NetCDF file: {e}")

    try:
        # Read lattice vectors
        if "primitive_vectors" in dataset.variables:
            lattice = dataset.variables["primitive_vectors"][:]
        else:
            raise RuntimeError("Primitive vectors not found in the file.")

        # Read density data
        if "density" in dataset.variables:
            density = dataset.variables["density"][:]
            # Reorder so dimensions become: x, y, z, components
            density = np.transpose(density, (4, 3, 2, 1, 0))
        else:
            raise RuntimeError("Density data not found in the file.")

        # Extract grid dimensions
        rc, ng1, ng2, ng3, components = density.shape

        # Calculate normalization constant such that everything is in atomic units
        norm_const = (ng1 * ng2 * ng3) / np.linalg.det(lattice)

        # Always extract charge
        charge = density[0, :, :, :, 0] / norm_const

        # Finally convert lattice to angstrom
        lattice = lattice * 0.5291772083

        if components == 1:
            # Non-magnetic / no spin data present: return only charge
            dataset.close()
            print("ABINIT density file read successfully: contains only charge density.")
            return lattice, (ng1, ng2, ng3), charge

        elif components == 4:
            # SOC / non-collinear: charge + mx,my,mz present
            mx = density[0, :, :, :, 1] / norm_const
            my = density[0, :, :, :, 2] / norm_const
            mz = density[0, :, :, :, 3] / norm_const
            dataset.close()
            print("ABINIT density file read successfully: contains both charge and spin densities.")
            return lattice, (ng1, ng2, ng3), charge, mx, my, mz

        else:
            # Unexpected number of components: inform the user
            raise RuntimeError(f"Unexpected number of density components: {components}. ")

    finally:
        # Ensure dataset is closed if not closed already
        try:
            if dataset.isopen():
                dataset.close()
        except Exception:
            # dataset may already be closed or not defined; ignore
            pass


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
        print("CHGCAR read successfully: contains only charge density.")
        return lattice, grid, densities[0]
    elif len(densities) == 4:
        print("CHGCAR read successfully: contains both charge and spin densities.")
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


def real_space_grid(lattice, Nx, Ny, Nz):
    """
    Generate a real-space grid within the unit cell based on the lattice vectors.

    Parameters:
    lattice (numpy.ndarray): A 3x3 matrix representing the lattice vectors of the unit cell.
                             Each row corresponds to one lattice vector.
    Nx, Ny, Nz (int): Number of grid points along the x, y, and z directions.

    Returns:
    tuple: Three 3D numpy arrays (rx, ry, rz) representing the x, y, and z coordinates
           of the real-space grid points, respectively.
    """
    red_rx, red_ry, red_rz = np.meshgrid(
        np.linspace(0, 1 - 1/Nx, Nx),
        np.linspace(0, 1 - 1/Ny, Ny),
        np.linspace(0, 1 - 1/Nz, Nz),
        indexing='ij'
    )
    
    rx = (
        lattice[0, 0] * red_rx +
        lattice[1, 0] * red_ry +
        lattice[2, 0] * red_rz
    )
    ry = (
        lattice[0, 1] * red_rx +
        lattice[1, 1] * red_ry +
        lattice[2, 1] * red_rz
    )
    rz = (
        lattice[0, 2] * red_rx +
        lattice[1, 2] * red_ry +
        lattice[2, 2] * red_rz
    )
    return rx, ry, rz


def project_sphere(density, lattice, center_red, radius):
    """
    Calculate the atomic multipole projections of a density onto cubic/tesseral harmonics inside a sphere.

    Parameters:
    density (numpy.ndarray): 3D array of the density.
    lattice (numpy.ndarray): 3x3 array of lattice vectors.
    center_red (numpy.ndarray): 1x3 array containing the reduced coordinates of the atom.
    radius (double): radius of the sphere centered at the atom, in atomic (Bohr) units.

    Returns:
    numpy.ndarray: Array of atomic multipole projection coefficients up to g [s, px, py, ..., g4].
    """

    # Shift the density so that the atomic center is at the center of the unit cell, 
    # ensures that the enitre sphere is contained in the unit cell (as long as the radius is reasonable)
    density, center_red = translate_density(density, center_red)

    # Convert center from reduced coordinated to cartesian
    center = np.dot(center_red, lattice)

    # Lattice parameters and real-space grid
    a1, a2, a3 = np.linalg.norm(lattice, axis=0)
    ng1, ng2, ng3 = density.shape
    rx, ry, rz = real_space_grid(lattice, ng1, ng2, ng3)
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    
    # Total density check (for debugging)
    # print(f"Total charge check: {np.sum(density)}")
    
    # Sphere - step function definition
    sphere = np.sqrt((rx - center[0])**2 + (ry - center[1])**2 + (rz - center[2])**2) < radius
    d_sphere = np.copy(density)
    d_sphere[~sphere] = 0

    # Calculate multipoles
    s = np.sum(d_sphere)
    py, pz, px = proj_p(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    dxy, dyz, dz2, dxz, dx2y2 = proj_d(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    fm3, fm2, fm1, f0, f1, f2, f3 = proj_f(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = proj_g(rx - center[0], ry - center[1], rz - center[2], d_sphere)

    return np.array([s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4])


# Definition of the multipoles (cubic/tesseral harmonics)
def proj_p(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    px = np.sum(np.sqrt(3 / (4 * np.pi)) * rx * f / r)
    py = np.sum(np.sqrt(3 / (4 * np.pi)) * ry * f / r)
    pz = np.sum(np.sqrt(3 / (4 * np.pi)) * rz * f / r)
    return py, pz, px

def proj_d(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    dz2   = np.sum( (1 / 4 * np.sqrt(5 / np.pi))  * (3 * rz**2 - r**2) * f / r**2)
    dxz   = np.sum( (1 / 2 * np.sqrt(15 / np.pi)) * rz * rx * f / r**2)
    dyz   = np.sum( (1 / 2 * np.sqrt(15 / np.pi)) * ry * rz * f / r**2)
    dxy   = np.sum( (1 / 4 * np.sqrt(15 / np.pi)) * 2 * rx * ry * f / r**2)
    dx2y2 = np.sum( (1 / 4 * np.sqrt(15 / np.pi)) * (rx**2 - ry**2) * f / r**2)
    return dxy, dyz, dz2, dxz, dx2y2

def proj_f(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    fm3 = np.sum( np.sqrt(35 / (32 * np.pi))  * (3 * rx**2 * ry - ry**3) * f / r**3)
    fm2 = np.sum( np.sqrt(105 / (16 * np.pi)) * (2 * rx * ry * rz) * f / r**3)
    fm1 = np.sum( np.sqrt(21 / (32 * np.pi))  * ry * (5 * rz**2 - r**2) * f / r**3)
    f0  = np.sum( np.sqrt(7 / (16 * np.pi))   * rz * (5 * rz**2 - r**2) * f / r**3)
    f1  = np.sum( np.sqrt(21 / (32 * np.pi))  * rx * (5 * rz**2 - r**2) * f / r**3)
    f2  = np.sum( np.sqrt(105 / (16 * np.pi)) * (rx**2 - ry**2) * rz * f / r**3)
    f3  = np.sum( np.sqrt(35 / (32 * np.pi))  * (rx**3 - 3 * rx * ry**2) * f / r**3)
    return fm3, fm2, fm1, f0, f1, f2, f3

def proj_g(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    gm4 = np.sum( (3 / 4 * np.sqrt(35 / np.pi))  * (rx**3 * ry - rx * ry**3) * f / r**4)
    gm3 = np.sum( (3 / 8 * np.sqrt(70 / np.pi))  * (3 * rx**2 * ry * rz - ry**3 * rz) * f / r**4)
    gm2 = np.sum( (3 / 8 * np.sqrt(5 / np.pi))   * (14 * rx * ry * rz**2 - 2 * rx * ry * r**2) * f / r**4)
    gm1 = np.sum( (3 / 16 * np.sqrt(5 / np.pi))  * (7 * ry * rz**3 - 3 * rz * ry * r**2) * f / r**4)
    g0  = np.sum( (3 / 16 * np.sqrt(1 / np.pi))  * (35 * rz**4 - 30 * rz**2 * r**2 + 3 * r**4) * f / r**4)
    g1  = np.sum( (3 / 16 * np.sqrt(5 / np.pi))  * (7 * rx * rz**3 - 3 * rz * rx * r**2) * f / r**4)
    g2  = np.sum( (3 / 8 * np.sqrt(5 / np.pi))   * ((rx**2 - ry**2) * (7 * rz**2 - r**2)) * f / r**4)
    g3  = np.sum( (3 / 8 * np.sqrt(70 / np.pi))  * (rx**3 * rz - 3 * rx * ry**2 * rz) * f / r**4)
    g4  = np.sum( (3 / 16 * np.sqrt(35 / np.pi)) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) * f / r**4)
    return gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4

def proj_h(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    hm5 = np.sum(np.sqrt(693/(512*np.pi)) * (ry*(5*rx**4 - 10*rx**2*ry**2 + ry**4)) * f / r**5)
    hm4 = np.sum(np.sqrt(3465/(265*np.pi)) * (4*rx*ry*rz*(rx**2 - ry**2)) * f / r**5)
    hm3 = np.sum(np.sqrt(385/(512*np.pi)) * (ry*(3*rx**2 - ry**2)*(9*rz**2 - r**2)) * f / r**5)
    hm2 = np.sum(np.sqrt(1155/(64*np.pi)) * (2*rx*ry*rz*(3*rz**2 - r**2)) * f / r**5)
    hm1 = np.sum(np.sqrt(165/(256*np.pi)) * (ry*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f / r**5)
    h0 = np.sum(np.sqrt(11/(256*np.pi)) * (rz*(63*rz**4 - 70*rz**2*r**2 + 15*r**4)) * f / r**5)
    h1 = np.sum(np.sqrt(165/(256*np.pi)) * (rx*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f / r**5)
    h2 = np.sum(np.sqrt(1155/(64*np.pi)) * ((rx**2 - ry**2)*rz*(3*rz**2 - r**2)) * f / r**5)
    h3 = np.sum(np.sqrt(385/(512*np.pi)) * (rx*(rx**2 - 3*ry**2)*(9*rz**2 - r**2)) * f / r**5)
    h4 = np.sum(np.sqrt(3465/(256*np.pi)) * (rz*(rx**4 - 6*rx**2*ry**2 + ry**4)) * f / r**5)
    h5 = np.sum(np.sqrt(693/(512*np.pi)) * (rx*(rx**4 - 10*rx**2*ry**2 + 5*ry**4)) * f / r**5)

    return hm5, hm4, hm3, hm2, hm1, h0, h1, h2, h3, h4, h5


def translate_density(f, center_red):
    """
    Adjust the density grid `f` so that the point closest to `center_red` is shifted to the center of the grid.

    Parameters:
    f (numpy.ndarray): 3D array representing the density grid.
    center_red (tuple): Fractional coordinates (x, y, z) of the desired center.

    Returns:
    tuple:
        - new_f (numpy.ndarray): The density grid after being rotated.
        - new_center_red (tuple): The updated fractional coordinates of the center.
    """
    ng1, ng2, ng3 = f.shape

    # Grid spacing in reduced coordinates
    drx, dry, drz = 1 / ng1, 1 / ng2, 1 / ng3

    # Convert the center coordinates to the closest grid indices
    idx_x = int(round(center_red[0] / drx)) % ng1
    idx_y = int(round(center_red[1] / dry)) % ng2
    idx_z = int(round(center_red[2] / drz)) % ng3

    # Closest grid point
    closest_idx = (idx_x, idx_y, idx_z)
    red_rx = idx_x * drx
    red_ry = idx_y * dry
    red_rz = idx_z * drz
    closest_point = (red_rx, red_ry, red_rz)
    difference  = center_red - closest_point

    # Calculate shifts for each axis
    target_index = (int(round(ng1/2)), int(round(ng2/2)), int(round(ng3/2)))
    shifts = [
    (target_index[axis] - closest_idx[axis]) % f.shape[axis]
    for axis in range(3)
    ]

    # Shift the density so the atomic center is at the center of the unit cell
    new_f = np.roll(f, shift=shifts, axis=(0, 1, 2))
    new_red_rx = target_index[0] * drx + difference[0]
    new_red_ry = target_index[1] * dry + difference[1]
    new_red_rz = target_index[2] * drz + difference[2]
    new_center_red = (new_red_rx, new_red_ry, new_red_rz)

    return new_f, new_center_red

def generate_xsf_file(scalar_field, lattice, output_file):
    """
    Generate an XSF file from a scalar field with proper periodic boundary conditions.

    Parameters:
    scalar_field (numpy.ndarray): 3D scalar field array of shape (Nx, Ny, Nz).
    lattice (numpy.ndarray): 3x3 matrix where each row is a lattice vector [x, y, z].
    output_file (str): Path to the output XSF file.
    """
    Nx, Ny, Nz = scalar_field.shape

    # Create periodic version by appending the first value to the end of each dimension
    # This makes the array (Nx+1) × (Ny+1) × (Nz+1)
    periodic_field = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    
    # Copy the original data
    periodic_field[:Nx, :Ny, :Nz] = scalar_field
    
    # Close periodic boundaries
    periodic_field[Nx, :Ny, :Nz] = scalar_field[0, :, :]  # x boundary
    periodic_field[:Nx, Ny, :Nz] = scalar_field[:, 0, :]  # y boundary  
    periodic_field[:Nx, :Ny, Nz] = scalar_field[:, :, 0]  # z boundary
    
    # Close edges
    periodic_field[Nx, Ny, :Nz] = scalar_field[0, 0, :]    # xy edge
    periodic_field[Nx, :Ny, Nz] = scalar_field[0, :, 0]    # xz edge
    periodic_field[:Nx, Ny, Nz] = scalar_field[:, 0, 0]    # yz edge
    
    # Close corner
    periodic_field[Nx, Ny, Nz] = scalar_field[0, 0, 0]     # xyz corner

    # Open the file for writing
    with open(output_file, 'w') as f:
        # Write the XSF header
        f.write("BEGIN_BLOCK_DATAGRID_3D\n")
        f.write("  ScalarField\n")
        f.write("  BEGIN_DATAGRID_3D_ScalarField\n")
        f.write(f"    {Nx + 1} {Ny + 1} {Nz + 1}\n")  # Note: +1 in each dimension

        # Write the origin and spanning vectors
        f.write("    0.0 0.0 0.0\n")
        for vector in lattice:
            f.write(f"    {vector[0]:.19f} {vector[1]:.19f} {vector[2]:.19f}\n")

        # Write the scalar field values in column-major order
        # Transpose to (z, y, x) order and flatten
        periodic_field = np.transpose(periodic_field, (2, 1, 0)).flatten()
        for idx, value in enumerate(periodic_field):
            f.write(f"    {value:.19f} ")
            if (idx + 1) % (Nx + 1) == 0:  # Newline every (Nx+1) values
                f.write("\n")

        # Write the end of the XSF file
        f.write("  END_DATAGRID_3D\n")
        f.write("END_BLOCK_DATAGRID_3D\n")
# def generate_xsf_file(scalar_field, lattice, output_file):
#     """
#     Generate an XSF file from a scalar field.

#     Parameters:
#     scalar_field (numpy.ndarray): 3D scalar field array of shape (Nx, Ny, Nz).
#     lattice (numpy.ndarray): 3x3 matrix where each row is a lattice vector [x, y, z].
#     output_file (str): Path to the output XSF file.
#     """
#     Nx, Ny, Nz = scalar_field.shape

#     # Open the file for writing
#     with open(output_file, 'w') as f:
#         # Write the XSF header
#         f.write("BEGIN_BLOCK_DATAGRID_3D\n")
#         f.write("  ScalarField\n")
#         f.write("  BEGIN_DATAGRID_3D_ScalarField\n")
#         f.write(f"    {Nx} {Ny} {Nz}\n")

#         # Write the origin and spanning vectors
#         f.write("    0.0 0.0 0.0\n")
#         for vector in lattice:
#             f.write(f"    {vector[0]:.19f} {vector[1]:.19f} {vector[2]:.19f}\n")

#         # Write the scalar field values in column-major order
#         scalar_field = np.transpose(scalar_field, (2, 1, 0)).flatten()
#         for idx, value in enumerate(scalar_field):
#             f.write(f"    {value:.19f} ")
#             if (idx + 1) % Nx == 0:  # Newline every Nx values
#                 f.write("\n")

#         # Write the end of the XSF file
#         f.write("  END_DATAGRID_3D\n")
#         f.write("END_BLOCK_DATAGRID_3D\n")


def inverse_project(rx, ry, rz, radius, coeffs):
    """
    Compute the real-space projection of density components using spherical harmonics
    up to the g-orbital level, modulated by an exponential suppression factor.

    Parameters:
    rx, ry, rz (numpy.ndarray): Cartesian coordinates in real space.
    radius (float): Cutoff radius beyond which the function is significantly suppressed.
    coeffs (list or array): Coefficients for each component of the density.

    Returns:
    tuple: Real-space components (s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, ...).
           Includes terms up to the g-orbital level.
    """
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    d = 100
    alpha = np.log(d)/radius # at r=radius, the function will be suppressed by a factor of d
    R = np.exp(-alpha*r)

    s     = coeffs[0]  * R
    py    = coeffs[1]  * np.sqrt(3 / (4 * np.pi)) * ry / r * R
    pz    = coeffs[2]  * np.sqrt(3 / (4 * np.pi)) * rz / r * R
    px    = coeffs[3]  * np.sqrt(3 / (4 * np.pi)) * rx / r * R
    dxy   = coeffs[4]  * (1 / 4 * np.sqrt(15 / np.pi)) * 2 * rx * ry / r**2 * R
    dyz   = coeffs[5]  * (1 / 2 * np.sqrt(15 / np.pi)) * ry * rz / r**2 * R
    dz2   = coeffs[6]  * (1 / 4 * np.sqrt(5 / np.pi)) * (3 * rz**2 - r**2) / r**2 * R
    dxz   = coeffs[7]  * (1 / 2 * np.sqrt(15 / np.pi)) * rz * rx / r**2 * R
    dx2y2 = coeffs[8]  * (1 / 4 * np.sqrt(15 / np.pi)) * (rx**2 - ry**2) / r**2 * R
    fm3   = coeffs[9]  * np.sqrt(35 / (32 * np.pi)) * (3 * rx**2 * ry - ry**3) / r**3 * R
    fm2   = coeffs[10] * np.sqrt(105 / (16 * np.pi)) * (2 * rx * ry * rz) / r**3 * R
    fm1   = coeffs[11] * np.sqrt(21 / (32 * np.pi)) * ry * (5 * rz**2 - r**2) / r**3 * R
    f0    = coeffs[12] * np.sqrt(7 / (16 * np.pi)) * rz * (5 * rz**2 - r**2) / r**3 * R
    f1    = coeffs[13] * np.sqrt(21 / (32 * np.pi)) * rx * (5 * rz**2 - r**2) / r**3 * R
    f2    = coeffs[14] * np.sqrt(105 / (16 * np.pi)) * (rx**2 - ry**2) * rz / r**3 * R
    f3    = coeffs[15] * np.sqrt(35 / (32 * np.pi)) * (rx**3 - 3 * rx * ry**2) / r**3 * R
    gm4   = coeffs[16] * (3 / 4  * np.sqrt(35 / np.pi)) * (rx**3 * ry - rx * ry**3) / r**4 * R
    gm3   = coeffs[17] * (3 / 8  * np.sqrt(70 / np.pi)) * (3 * rx**2 * ry * rz - ry**3 * rz) / r**4 * R
    gm2   = coeffs[18] * (3 / 8  * np.sqrt(5 / np.pi))  * (14 * rx * ry * rz**2 - 2 * rx * ry * r**2) / r**4 * R
    gm1   = coeffs[19] * (3 / 16 * np.sqrt(5 / np.pi))  * (7 * ry * rz**3 - 3 * rz * ry * r**2) / r**4 * R
    g0    = coeffs[20] * (3 / 16 * np.sqrt(1 / np.pi))  * (35 * rz**4 - 30 * rz**2 * r**2 + 3 * r**4) / r**4 * R
    g1    = coeffs[21] * (3 / 16 * np.sqrt(5 / np.pi))  * (7 * rx * rz**3 - 3 * rz * rx * r**2) / r**4 * R
    g2    = coeffs[22] * (3 / 8  * np.sqrt(5 / np.pi))  * ((rx**2 - ry**2) * (7 * rz**2 - r**2)) / r**4 * R
    g3    = coeffs[23] * (3 / 8  * np.sqrt(70 / np.pi)) * (rx**3 * rz - 3 * rx * ry**2 * rz) / r**4 * R
    g4    = coeffs[24] * (3 / 16 * np.sqrt(35 / np.pi)) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) / r**4 * R

    return s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4


def output_analytical_components(lattice, positions, radius, coeffs, filename_prefix, threshold=1e-6):
    """
    Compute and output the analytical components of the densities for all positions, saving each component to separate files.

    Parameters:
    lattice (numpy.ndarray): 3x3 lattice matrix.
    positions (numpy.ndarray): Nx3 array of fractional coordinates.
    radius (float): Radius for the spherical projection.
    coeffs (numpy.ndarray): Nx25 matrix of coefficients for each position.
    filename_prefix (str): Prefix for the output XSF files.
    threshold (float): Threshold for generating partial component XSF files.
    """
    # Create a real-space grid
    rx, ry, rz = real_space_grid(lattice, 100, 100, 100)

    # Define components and initialize their totals
    components = {
        "s": None, "p_y": None, "p_z": None, "p_x": None,
        "d_xy": None, "d_yz": None, "d_z^2": None, "d_xz": None, "d_x^2-y^2": None,
        "f_y(3x^2-y)": None, "f_xyz": None, "f_yz^2": None, "f_z^3": None, "f_xz^2": None, "f_z(x^2-y^2)": None, "f_x(x^2-3y^2)": None,
        "g_xy(x^2-y^2)": None, "g_yz(3x^2-y)": None, "g_xyz^2": None, "g_yz^3": None, "g_z^4": None, "g_xz^3": None, "g_(x^2-y^2)z^2": None, "g_xz(x^2-3y^2)": None, "g_x^2y^2": None
    }

    for key in components.keys():
        components[key] = np.zeros_like(rx)  # Initialize arrays

    # Iterate through all positions (included nearest neighboring cells for better visualization) and coefficients
    for pos, coeff in zip(positions, coeffs):
        for i in range(0, 2):
            for j in range(0, 2):
                for k in range(0, 2):
                    center = np.dot(pos + [i, j, k], lattice)
                    component_values = inverse_project(rx - center[0], ry - center[1], rz - center[2], radius, coeff)
                    
                    for key, value in zip(components.keys(), component_values):
                        components[key] += value  # Accumulate densities

    # Compute combined totals
    components["p_tot"] = components["p_y"] + components["p_z"] + components["p_x"]
    components["d_tot"] = components["d_xy"] + components["d_yz"] + components["d_z^2"] + components["d_xz"] + components["d_x^2-y^2"]
    components["f_tot"] = components["f_y(3x^2-y)"] + components["f_xyz"] + components["f_yz^2"] + components["f_z^3"] + components["f_xz^2"] + components["f_z(x^2-y^2)"] + components["f_x(x^2-3y^2)"]
    components["g_tot"] = components["g_xy(x^2-y^2)"] + components["g_yz(3x^2-y)"] + components["g_xyz^2"] + components["g_yz^3"] + components["g_z^4"] + components["g_xz^3"] + components["g_(x^2-y^2)z^2"] + components["g_xz(x^2-3y^2)"] + components["g_x^2y^2"]

    # Function to check if at least 2 components in a set exceed the threshold (thenit makes sense to print the total)
    def exceeds_threshold(indices):
        significant_components = np.sum(np.max(np.abs(coeffs[:, indices]), axis=0) > threshold)
        return significant_components >= 2

    # Generate combined totals only if at least 2 corresponding coefficients exceed the threshold
    if exceeds_threshold([1, 2, 3]):  # p orbitals (indices 1,2,3 for p_y, p_z, p_x)
        generate_xsf_file(components["p_tot"], lattice, f"{filename_prefix}_p_tot.xsf")
    if exceeds_threshold([4, 5, 6, 7, 8]):  # d orbitals
        generate_xsf_file(components["d_tot"], lattice, f"{filename_prefix}_d_tot.xsf")
    if exceeds_threshold([9, 10, 11, 12, 13, 14, 15]):  # f orbitals
        generate_xsf_file(components["f_tot"], lattice, f"{filename_prefix}_f_tot.xsf")
    if exceeds_threshold([16, 17, 18, 19, 20, 21, 22, 23, 24]):  # g orbitals
        generate_xsf_file(components["g_tot"], lattice, f"{filename_prefix}_g_tot.xsf")

    # Output partial density components if the corresponding coefficient exceeds the threshold
    for key, value in components.items():
        if key not in ["tot", "p_tot", "d_tot", "f_tot", "g_tot"]:  # Skip combined totals
            component_index = list(components.keys()).index(key)
            max_coeff = np.max(np.abs(coeffs[:, component_index]))
            if max_coeff > threshold:
                generate_xsf_file(value, lattice, f"{filename_prefix}_{key}.xsf")


def wyckoff(center, space_group_number):
    """
    Calculate all symmetry-equivalent positions for a given center using spglib.

    Parameters:
        center (array-like): Reduced coordinates of the center [x, y, z].
        space_group_number (int): Space group Hall number (1-530): https://yseto.net/en/sg/sg1

    Returns:
        list: List of unique symmetry-equivalent positions.
    """
    center = np.array(center)
    symmetry = spglib.get_symmetry_from_database(space_group_number)
    rotations = np.array(symmetry['rotations'])
    translations = np.array(symmetry['translations'])

    # Generate symmetry-equivalent positions
    positions = []
    for rotation, translation in zip(rotations, translations):
        new_position = np.dot(rotation, center) + translation
        # Wrap coordinates within [0, 1)
        new_position = np.mod(new_position, 1)
        positions.append(tuple(new_position))

    # Remove duplicates
    unique_positions = list(set(positions))
    return unique_positions


def project_single_irrep(f, symm, tnons, char_table, supercell_size, kpoint):
    """
    Project a charge or spin density from a distorted (primitive or super-) cell onto the 
    irreducible representations of the parent space group's primitive cell.

    Parameters
    ----------
    f : ndarray
        3D array (Nx, Ny, Nz) representing the charge or spin density on a real-space grid.
    symm : ndarray
        Array of shape (N_symm, 3, 3) containing rotation/mirror matrices (integer values).
    tnons : ndarray
        Array of shape (N_symm, 3) containing fractional translations associated with each symmetry operation.
    char_table : ndarray
        1D array of length N_symm giving the character of each symmetry operation for the target irrep.
    supercell_size : array-like, optional
        Size of the supercell relative to primitive cell, e.g., [2, 2, 2] for 2×2×2.
        If None, assumes primitive cell (equivalent to [1, 1, 1]).
    kpoint : array-like, optional
        1D array of shape (3,) representing the k-point in fractional coordinates 
        (relative to the PRIMITIVE reciprocal lattice). If None, uses [0, 0, 0].

    Returns
    -------
    proj : ndarray
        3D array (Nx, Ny, Nz) of the projected charge or spin density.
    """
    
    # Generate all supercell translation vectors in primitive coordinates
    translations_SC_primitive = []
    for i in range(supercell_size[0]):
        for j in range(supercell_size[1]):
            for k in range(supercell_size[2]):
                translations_SC_primitive.append([i, j, k])
    translations_SC_primitive = np.array(translations_SC_primitive)
    
    # Convert to supercell fractional coordinates for the grid transformation
    translations_SC_supercell = translations_SC_primitive / supercell_size
    
    grid = f.shape  # Grid dimensions (Nx, Ny, Nz)

    # Precompute phase factors: exp(i 2π R⋅k) for each supercell translation
    # Use PRIMITIVE coordinates for the phase calculation
    phase = np.exp(1j * 2 * np.pi * np.dot(translations_SC_primitive, kpoint))

    # Initialize projected density
    proj = np.zeros(f.shape)

    # Loop over supercell translations
    for t in range(translations_SC_supercell.shape[0]):
        # Loop over all symmetry operations in the parent space group
        for s in range(symm.shape[0]):
            # Generate grid of integer indices (i, j, k)
            i, j, k = np.meshgrid(
                np.arange(grid[0]),
                np.arange(grid[1]),
                np.arange(grid[2]),
                indexing='ij'
            )

            # Stack indices into vectors of shape (Nx, Ny, Nz, 3)
            v = np.stack((i, j, k), axis=-1)

            # Apply rotation to grid points
            v_new = np.tensordot(v, symm[s], axes=([3], [1])).astype(float)

            # Apply translation (tnons) and supercell translation (translations_SC_supercell[t])
            # Use SUPERCELl coordinates for the grid transformation
            v_new += (tnons[s] + translations_SC_supercell[t]) * grid

            # Wrap indices back into grid range using modulo
            i_new = v_new[..., 0] % grid[0]
            j_new = v_new[..., 1] % grid[1]
            k_new = v_new[..., 2] % grid[2]

            # Convert to integer indices
            i_new = i_new.astype(int)
            j_new = j_new.astype(int)
            k_new = k_new.astype(int)

            # Apply projection formula:
            proj[i, j, k] += np.real(
                phase[t] * char_table[s] /
                (symm.shape[0] * translations_SC_supercell.shape[0]) *
                f[i_new, j_new, k_new]
            )

    return proj


# def project_UC_irrep(f, symm, tnons, char_table):
#     """
#     Project a charge or spin density onto the irreducible representations 
#     of the parent space group's primitive cell.

#     This version assumes the distorted structure shares the same unit cell 
#     as the parent phase (no supercell translations) so the k-point must be
#     Gamma.

#     Parameters
#     ----------
#     f : ndarray
#         3D array (Nx, Ny, Nz) representing the charge or spin density 
#         on a real-space grid.
#     symm : ndarray
#         Array of shape (N_symm, 3, 3) containing rotation/mirror matrices 
#         (integer values).
#     tnons : ndarray
#         Array of shape (N_symm, 3) containing fractional translations 
#         associated with each symmetry operation.
#     char_table : ndarray
#         1D array of length N_symm giving the character of each symmetry 
#         operation for the target irrep.

#     Returns
#     -------
#     proj : ndarray
#         3D array (Nx, Ny, Nz) of the projected charge or spin density.
#     """
#     grid = f.shape  # Grid dimensions (Nx, Ny, Nz)

#     # Initialize projected density
#     proj = np.zeros(f.shape)

#     # Loop over all symmetry operations in the parent space group
#     for s in range(symm.shape[0]):
#         # Generate grid of integer indices (i, j, k)
#         i, j, k = np.meshgrid(
#             np.arange(grid[0]),
#             np.arange(grid[1]),
#             np.arange(grid[2]),
#             indexing='ij'
#         )

#         # Stack indices into vectors of shape (Nx, Ny, Nz, 3)
#         v = np.stack((i, j, k), axis=-1)

#         # Apply rotation to grid points
#         v_new = np.tensordot(v, symm[s], axes=([3], [1])).astype(float)

#         # Apply fractional translation (tnons[s])
#         v_new += tnons[s] * grid

#         # Wrap indices back into grid range using modulo
#         i_new = (v_new[..., 0] % grid[0]).astype(int)
#         j_new = (v_new[..., 1] % grid[1]).astype(int)
#         k_new = (v_new[..., 2] % grid[2]).astype(int)

#         # Apply projection formula:
#         # - char_table[s]: character for this symmetry op in the irrep
#         # - normalization: total number of operations (N_symm)
#         proj[i, j, k] += np.real(
#             char_table[s] / symm.shape[0] * f[i_new, j_new, k_new]
#         )

#     return proj


def project_irreps(
    density_file,
    dft_code,
    spacegroup,
    supercell_size=None, 
    kpoint=None,
):
    # Handle default parameters
    if kpoint is None:
        kpoint = [0.0, 0.0, 0.0]
    kpoint = np.asarray(kpoint)
    
    if supercell_size is None:
        supercell_size = [1, 1, 1]
    supercell_size = np.asarray(supercell_size, dtype=int)

    # load density explicitly according to dft_code
    ft = dft_code.lower()
    if ft == "abinit":
        out = ABINIT_get_density(density_file)
    elif ft == "vasp":
        out = VASP_get_density(density_file)
    else:
        raise ValueError("dft_code must be 'abinit' or 'vasp'")

    # Decide components depending on what the reader returned
    if len(out) == 3:
        lattice, grid, charge = out
        comp_arrays = {"charge": charge}
    elif len(out) == 6:
        lattice, grid, charge, mx, my, mz = out
        comp_arrays = {"charge": charge, "mx": mx, "my": my, "mz": mz}
    else:
        raise ValueError(
            f"Unexpected return from density reader: expected 3 or 6 items, got {len(out)}"
        )

    try:
        # get the name of the calling Python script without extension
        input_basename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    except Exception:
        # fallback if running in an interactive session
        input_basename = "output"
    output_file = input_basename + ".pdout"

    # helper function to print to screen and file
    def write(msg):
        print(msg)
        f.write(msg + "\n")

    symmetry = spglib.get_symmetry_from_database(spacegroup)
    symm = np.array(symmetry['rotations'])
    tnons = np.array(symmetry['translations']) # Non-symmorphic translations
    irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)
    little_group_symm = symm[mapping_little_group]
    little_group_tnons = tnons[mapping_little_group]
    
    with open(output_file, "w") as f:

        # --- Print lattice and positions ---
        write("\n=== Lattice vectors (Angstrom) ===")
        for i, vec in enumerate(lattice):
            write(f"Vector {i+1}: [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
        
        # --- Print symmetry elements and irrep characters ---
        write("\n=== Space Group Symmetry Operations ===")
        write(f"Space group Hall number: {spacegroup}")
        write(f"Total operations: {len(symm)}")
        write(f"Selected k-point: {kpoint}")
        write(f"Little group operations: {len(little_group_symm)}")
        write(f"Number of irreps: {len(irreps)}")
        
        write("\n--- Little group symmetry Operations (Rotation + Non-symmorphic Translation) ---")
        for i, (rot, trans) in enumerate(zip(little_group_symm, little_group_tnons)):
            write(f"Operation {i+1}:")
            write(f"  Rotation:\n{rot}")
            write(f"  Translation: {trans}")
            write("")

        write("\n--- Irrep Character Tables ---")
        for i, irrep in enumerate(irreps):
            characters = get_character(irrep)
            write(f"Irrep {i+1}: {characters}")
        
        write("\n" + "="*60)
        write("PROJECTING DENSITY ONTO IRREPS")
        write("="*60)

        # --- Project all components onto all irreps ---
        for comp_name, density in comp_arrays.items():
            write(f"\n=== Projecting {comp_name} component ===")
            
            # Calculate max of original component for magnitude normalization
            max_original = np.max(np.abs(density))
            write(f"Max absolute value of original {comp_name}: {max_original:.6f}\n")
            
            for i, irrep in enumerate(irreps):
                write(f"Projecting onto irrep {i+1}...")
                
                char_table = get_character(irrep)
                proj_density = project_single_irrep(density, little_group_symm, little_group_tnons, char_table, supercell_size, kpoint)
                
                # Calculate magnitude: max(projected) / max(original)
                max_projected = np.max(np.abs(proj_density))
                magnitude = max_projected / max_original if max_original > 0 else 0
                
                # Generate output filename
                outname = f"{input_basename}_{comp_name}_irrep{i+1}.xsf"
                generate_xsf_file(proj_density, lattice, outname)
                
                write(f"Done! Saved as {outname}")
                write(f"Magnitude (max|proj|/max|orig|): {magnitude:.6f}\n")
            
            write("-" * 40)
        
        write("\nAll projections completed successfully!")


def project_harmonics(
    density_file,
    dft_code,
    center,
    radius,
    spacegroup=1,
    output_components=False,
    decimals=4,
):
    lattice, grid, comp_arrays = load_density_file(density_file, dft_code)
    input_basename = get_output_basename()
    output_file = input_basename + ".pdout"

    center = np.asarray(center)
    positions = np.round(wyckoff(center, spacegroup), 5)

    # formatting settings
    FIELD_WIDTH = 12
    NUM_FMT = f"{{:{FIELD_WIDTH}.{decimals}f}}"
    LABEL_FMT = f"{{:<{FIELD_WIDTH}}}"

    def fmt(v):
        return NUM_FMT.format(v)

    # labels for each multipole
    MULTIPOLE_LABELS = {
        "s": ["s"],
        "p": ["y", "z", "x"],
        "d": ["xy", "yz", "z^2", "xz", "x^2-y^2"],
        "f": ["y(3x^2-y)", "xyz", "yz^2", "z^3", "xz^2", "z(x^2-y^2)", "x(x^2-3y^2)"],
        "g": ["xy(x^2-y^2)", "yz(3x^2-y)", "xyz^2", "yz^3", "z^4", "xz^3",
              "(x^2-y^2)z^2", "xz(x^2-3y^2)", "x^2y^2"]
    }

    # helper function to print to screen and file
    def write(msg):
        print(msg)
        f.write(msg + "\n")

    def print_block(label, arr, labels_list):
        write(LABEL_FMT.format(label) + " | ".join(f"{lab:>{FIELD_WIDTH}}" for lab in labels_list))
        write(LABEL_FMT.format(label) + " | ".join(fmt(v) for v in arr))

    with open(output_file, "w") as f:

        # --- Print lattice and positions ---
        write("\n=== Lattice vectors (Angstrom) ===")
        for i, vec in enumerate(lattice):
            write(f"Vector {i+1}: [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")

        write("\n=== Wyckoff-equivalent positions ===")
        for i, pos in enumerate(positions):
            write(f"Position {i+1}: [{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}]")
        write("\n" + "-"*145 + "\n")

        # --- Multipole projections ---
        for comp, arr in comp_arrays.items():
            coeffs_list = []

            write(f"\n=== Projections for component: {comp} ===\n")

            for idx, pos in enumerate(positions):
                coeffs_row = project_sphere(arr, lattice, np.asarray(pos), radius)
                coeffs_list.append(coeffs_row)

                s = coeffs_row[0:1]
                p = coeffs_row[1:4]
                d = coeffs_row[4:9]
                f_arr = coeffs_row[9:16]
                g_arr = coeffs_row[16:25]

                write(f"Position {idx+1}: {pos}")
                write("-" * 145)
                print_block("s", s, MULTIPOLE_LABELS["s"])
                print_block("p", p, MULTIPOLE_LABELS["p"])
                print_block("d", d, MULTIPOLE_LABELS["d"])
                print_block("f", f_arr, MULTIPOLE_LABELS["f"])
                print_block("g", g_arr, MULTIPOLE_LABELS["g"])
                write("-" * 145 + "\n")

            coeffs = np.array(coeffs_list)

            # Sum over positions
            sum_coeffs = np.sum(coeffs, axis=0)
            write(f"=== Sum over positions for component: {comp} ===")
            write("-" * 145)
            print_block("s", sum_coeffs[0:1], MULTIPOLE_LABELS["s"])
            print_block("p", sum_coeffs[1:4], MULTIPOLE_LABELS["p"])
            print_block("d", sum_coeffs[4:9], MULTIPOLE_LABELS["d"])
            print_block("f", sum_coeffs[9:16], MULTIPOLE_LABELS["f"])
            print_block("g", sum_coeffs[16:25], MULTIPOLE_LABELS["g"])
            write("-" * 145 + "\n")

            if output_components:
                outname = f"{input_basename}_{comp}"
                write("Outputting the analytical harmonics into .xsf files. Might take a while...\n")
                output_analytical_components(lattice, positions, radius, coeffs, outname)
                write("Done!")


def load_density_file(density_file, dft_code):
    """
    Load density from file and return standardized components.
    
    Parameters
    ----------
    density_file : str
        Path to the density file.
    dft_code : str
        'vasp' or 'abinit'
        
    Returns
    -------
    tuple: (lattice, grid, comp_arrays)
        comp_arrays is a dict with keys like 'charge', 'mx', etc.
    """
    ft = dft_code.lower()
    if ft == "abinit":
        out = ABINIT_get_density(density_file)
    elif ft == "vasp":
        out = VASP_get_density(density_file)
    else:
        raise ValueError("dft_code must be 'abinit' or 'vasp'")

    if len(out) == 3:
        lattice, grid, charge = out
        comp_arrays = {"charge": charge}
    elif len(out) == 6:
        lattice, grid, charge, mx, my, mz = out
        comp_arrays = {"charge": charge, "mx": mx, "my": my, "mz": mz}
    else:
        raise ValueError(
            f"Unexpected return from density reader: expected 3 or 6 items, got {len(out)}"
        )
    
    return lattice, grid, comp_arrays


def get_output_basename():
    """
    Get the base name for output files from the calling script name.
    
    Returns
    -------
    str: Base name for output files
    """
    try:
        # get the name of the calling Python script without extension
        return os.path.splitext(os.path.basename(sys.argv[0]))[0]
    except Exception:
        # fallback if running in an interactive session
        return "output"