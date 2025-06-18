import numpy as np
import os

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
    Calculate the atomic multipole projections of a density onto tesseral harmonics inside a sphere.

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
    density, center_red = rotate_density(density, center_red)

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
    c_sphere = np.copy(density)
    c_sphere[~sphere] = 0

    # Calculate multipoles
    s = np.sum(c_sphere)
    px, py, pz = proj_p(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    dz2, dxz, dyz, dxy, dx2y2 = proj_d(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    fm3, fm2, fm1, f0, f1, f2, f3 = proj_f(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = proj_g(rx - center[0], ry - center[1], rz - center[2], c_sphere)

    return np.array([s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4])


# Definition of the multipoles (tesseral harmonics)
def proj_p(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    px = np.sum(np.sqrt(3 / (4 * np.pi)) * rx * f / r)
    py = np.sum(np.sqrt(3 / (4 * np.pi)) * ry * f / r)
    pz = np.sum(np.sqrt(3 / (4 * np.pi)) * rz * f / r)
    return px, py, pz

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


def rotate_density(f, center_red):
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
    Generate an XSF file from a scalar field.

    Parameters:
    scalar_field (numpy.ndarray): 3D scalar field array of shape (Nx, Ny, Nz).
    lattice (numpy.ndarray): 3x3 matrix where each row is a lattice vector [x, y, z].
    output_file (str): Path to the output XSF file.
    """
    Nx, Ny, Nz = scalar_field.shape

    # Open the file for writing
    with open(output_file, 'w') as f:
        # Write the XSF header
        f.write("BEGIN_BLOCK_DATAGRID_3D\n")
        f.write("  ScalarField\n")
        f.write("  BEGIN_DATAGRID_3D_ScalarField\n")
        f.write(f"    {Nx} {Ny} {Nz}\n")

        # Write the origin and spanning vectors
        f.write("    0.0 0.0 0.0\n")
        for vector in lattice:
            f.write(f"    {vector[0]:.19f} {vector[1]:.19f} {vector[2]:.19f}\n")

        # Write the scalar field values in column-major order
        scalar_field = np.transpose(scalar_field, (2, 1, 0)).flatten()
        for idx, value in enumerate(scalar_field):
            f.write(f"    {value:.19f} ")
            if (idx + 1) % Nx == 0:  # Newline every Nx values
                f.write("\n")

        # Write the end of the XSF file
        f.write("  END_DATAGRID_3D\n")
        f.write("END_BLOCK_DATAGRID_3D\n")


def inverse_project(rx, ry, rz, radius, coeffs):
    """
    Compute the real-space projection of density components using spherical harmonics
    up to the g-orbital level, modulated by an exponential suppression factor.

    Parameters:
    rx, ry, rz (numpy.ndarray): Cartesian coordinates in real space.
    radius (float): Cutoff radius beyond which the function is significantly suppressed.
    coeffs (list or array): Coefficients for each component of the density.

    Returns:
    tuple: Real-space components (s, px, py, pz, dxy, dyz, dz2, dxz, dx2y2, ...).
           Includes terms up to the g-orbital level.
    """
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30

    d = 100
    alpha = np.log(d)/radius # at r=radius, the function will be suppressed by a factor of d
    R = np.exp(-alpha*r)

    s     = coeffs[0]  * R
    px    = coeffs[1]  * np.sqrt(3 / (4 * np.pi)) * rx / r * R
    py    = coeffs[2]  * np.sqrt(3 / (4 * np.pi)) * ry / r * R
    pz    = coeffs[3]  * np.sqrt(3 / (4 * np.pi)) * rz / r * R
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

    return s, px, py, pz, dxy, dyz, dz2, dxz, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4


def output_analytical_densities(lattice, positions, radius, coeffs, filename_prefix, threshold=1e-6):
    """
    Compute and output the analytical densities for all positions, saving each component to separate files.

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
        "s": None, "px": None, "py": None, "pz": None,
        "dxy": None, "dyz": None, "dz2": None, "dxz": None, "dx2y2": None,
        "fm3": None, "fm2": None, "fm1": None, "f0": None, "f1": None, "f2": None, "f3": None,
        "gm4": None, "gm3": None, "gm2": None, "gm1": None, "g0": None, "g1": None, "g2": None, "g3": None, "g4": None
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
    components["ptot"] = components["px"] + components["py"] + components["pz"]
    components["dtot"] = components["dxy"] + components["dyz"] + components["dz2"] + components["dxz"] + components["dx2y2"]
    components["ftot"] = components["fm3"] + components["fm2"] + components["fm1"] + components["f0"] + components["f1"] + components["f2"] + components["f3"]
    components["gtot"] = components["gm4"] + components["gm3"] + components["gm2"] + components["gm1"] + components["g0"] + components["g1"] + components["g2"] + components["g3"] + components["g4"]
    components["tot"] = sum(components.values())  # Overall total

    # Output the total density file
    generate_xsf_file(components["tot"], lattice, f"{filename_prefix}_tot.xsf")

    # Function to check if any coefficient in a set exceeds the threshold
    def exceeds_threshold(indices):
        return np.any(np.max(np.abs(coeffs[:, indices]), axis=0) > threshold)

    # Generate combined totals if at least one corresponding coefficient exceeds the threshold
    if exceeds_threshold([1, 2, 3]):  # px, py, pz
        generate_xsf_file(components["ptot"], lattice, f"{filename_prefix}_ptot.xsf")
    if exceeds_threshold([4, 5, 6, 7, 8]):  # dxy, dyz, dz2, dxz, dx2y2
        generate_xsf_file(components["dtot"], lattice, f"{filename_prefix}_dtot.xsf")
    if exceeds_threshold([9, 10, 11, 12, 13, 14, 15]):  # fm3 to f3
        generate_xsf_file(components["ftot"], lattice, f"{filename_prefix}_ftot.xsf")
    if exceeds_threshold([16, 17, 18, 19, 20, 21, 22, 23, 24]):  # gm4 to g4
        generate_xsf_file(components["gtot"], lattice, f"{filename_prefix}_gtot.xsf")

    # Output partial density components if the corresponding coefficient exceeds the threshold
    for key, value in components.items():
        if key not in ["tot", "ptot", "dtot", "ftot", "gtot"]:  # Skip combined totals
            component_index = list(components.keys()).index(key)
            max_coeff = np.max(np.abs(coeffs[:, component_index]))
            if max_coeff > threshold:
                generate_xsf_file(value, lattice, f"{filename_prefix}_{key}.xsf")