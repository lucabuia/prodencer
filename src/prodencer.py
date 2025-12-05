import numpy as np
import netCDF4 as nc
import os
import sys
import spglib
from spgrep import get_spacegroup_irreps_from_primitive_symmetry
from spgrep.representation import get_character
from numpy.fft import fftn, fftshift
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


def ABINIT_get_density(input="GSo_DEN.nc"):
    """
    Read an ABINIT density NetCDF file.

    Returns:
      - If the file contains only the charge density (non-magnetic calculation, nspden=1):
          lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge
      - If the file contains charge + 1 spin components (magnetic collinear calculation, nspden=2):
          lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mz
      - If the file contains charge + 3 spin components (magnetic non-collinear calculation, nspden=4):
          lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mx, my, mz
    """
    if not os.path.isfile(input):
        raise FileNotFoundError(f"ABINIT density file not found: {input}")

    try:
        dataset = nc.Dataset(input, 'r')
    except Exception as e:
        raise RuntimeError(f"Error opening NetCDF file: {e}")

    try:
        # ----- lattice -----
        if "primitive_vectors" in dataset.variables:
            lattice = dataset.variables["primitive_vectors"][:]
        else:
            raise RuntimeError("Primitive vectors not found.")

        # ----- atomic positions -----
        if "reduced_atom_positions" in dataset.variables:
            atomic_positions = dataset.variables["reduced_atom_positions"][:].T
        else:
            raise RuntimeError("Atomic positions not found.")

        # ----- atomic species -----
        if "reduced_atom_positions" in dataset.variables:
            atomic_species = dataset.variables["atom_species"][:].T
        else:
            raise RuntimeError("Atomic species not found.")

        # ----- density -----
        if "density" not in dataset.variables:
            raise RuntimeError("Density data not found.")

        density = dataset.variables["density"][:]
        density = np.transpose(density, (4, 3, 2, 1, 0))

        rc, ng1, ng2, ng3, components = density.shape

        # normalization factor
        norm_const = (ng1 * ng2 * ng3) / np.linalg.det(lattice)

        # charge is always component 0
        charge = density[0, :, :, :, 0] / norm_const

        # ----- branch on number of components -----

        if components == 1:
            print("ABINIT file: charge density only.")
            return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge

        elif components == 2:
            print("ABINIT file: charge density and collinear spin density.")
            mz = density[0, :, :, :, 1] / norm_const
            return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mz

        elif components == 4:
            print("ABINIT file: charge density and full spin density (mx, my, mz).")
            mx = density[0, :, :, :, 1] / norm_const
            my = density[0, :, :, :, 2] / norm_const
            mz = density[0, :, :, :, 3] / norm_const
            return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mx, my, mz

        else:
            raise RuntimeError(f"Unexpected number of density components: {components}")

    finally:
        try:
            dataset.close()
        except:
            pass


def VASP_get_density(input="CHGCAR"):
    """
    Read a VASP density CHGCAR file.

    Returns:
      - If the file contains only the charge density (non-magnetic calculation):
          lattice, atomic_positions, (ng1, ng2, ng3), charge
      - If the file contains charge + 1 spin components (magnetic collinear calculation, ISPIN=2):
          lattice, atomic_positions, (ng1, ng2, ng3), charge, mz
      - If the file contains charge + 3 spin components (magnetic non-collinear calculation, LNONCOLLINEAR=.TRUE.):
          lattice, atomic_positions, (ng1, ng2, ng3), charge, mx, my, mz
    """
    if not os.path.isfile(input):
        raise FileNotFoundError("CHGCAR file not found")

    with open(input, 'r') as chgcar:
        # Skip title and scale
        chgcar.readline()
        chgcar.readline()

        # --- lattice ---
        lattice = np.zeros((3, 3), float)
        for i in range(3):
            lattice[i] = np.array(chgcar.readline().split(), float)

        # Convert VASP Å → Bohr
        lattice = lattice / 0.5291772083

        # --- atom info ---
        atom_types = chgcar.readline().split()
        atom_counts = np.array(chgcar.readline().split(), int)
        atomic_species = np.repeat(np.arange(1, len(atom_counts) + 1), atom_counts)
        n_atoms = np.sum(atom_counts)

        coord_type = chgcar.readline().strip()
        if coord_type.lower().startswith("s"):
            coord_type = chgcar.readline().strip()

        # Atomic positions
        atomic_positions = np.zeros((3, n_atoms))
        for i in range(n_atoms):
            atomic_positions[:, i] = np.array(chgcar.readline().split()[:3], float)

        # Skip blank line(s)
        while True:
            line = chgcar.readline()
            if not line.strip():
                break

        # --- parse density blocks ---
        densities = []
        while True:
            line = chgcar.readline()
            if not line:
                break

            try:
                grid = np.array(line.split(), dtype=int)
                if len(grid) != 3:
                    continue
            except ValueError:
                continue

            ng1, ng2, ng3 = grid
            density = np.zeros(ng1 * ng2 * ng3)

            num_full_lines = (ng1 * ng2 * ng3) // 5
            for i in range(num_full_lines):
                density[5 * i:5 * i + 5] = np.array(chgcar.readline().split(), float)

            remaining = (ng1 * ng2 * ng3) % 5
            if remaining > 0:
                density[-remaining:] = np.array(chgcar.readline().split(), float)

            density = density.reshape((ng1, ng2, ng3), order='F')
            density /= (ng1 * ng2 * ng3)

            densities.append(density)

    # ---- interpret results ----
    ncomp = len(densities)

    if ncomp == 1:
        print("CHGCAR: charge only.")
        (charge,) = densities
        return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge

    elif ncomp == 2:
        print("CHGCAR: collinear spin (charge + m_z).")
        charge, mz = densities
        return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mz

    elif ncomp == 4:
        print("CHGCAR: non-collinear spin (charge + mx,my,mz).")
        charge, mx, my, mz = densities
        return lattice, atomic_positions, atomic_species, (ng1, ng2, ng3), charge, mx, my, mz

    else:
        raise RuntimeError(f"Unexpected number of densities in CHGCAR: {ncomp}")


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


def project_sphere(density, lattice, center_red, radius, units="multi"):
    """
    Calculate the atomic multipole projections of a density onto cubic/tesseral harmonics inside a sphere.

    Parameters:
    density (numpy.ndarray): 3D array of the density.
    lattice (numpy.ndarray): 3x3 array of lattice vectors.
    center_red (numpy.ndarray): 1x3 array containing the reduced coordinates of the atom.
    radius (float): Radius of the sphere centered at the atom, in atomic (Bohr radii) units.
    units (str): Specifies the unit system for the projection.
        - "multi": Uses multipolar units, expressed as |e|*a0^l for charge densities
          or μ_B*a0^l for magnetization densities, where a0 is the Bohr radius,
          |e| is the elementary charge, and μ_B is the Bohr magneton.
          These units correspond to multipole moments of order l.
        - "charge": Uses normalized units, i.e., |e| for charge
          and μ_B for magnetic moments, corresponding to using the standard definition of
          the tesseral harmonics normalized by the radial distance.

    Returns:
    numpy.ndarray: Array of atomic multipole projection coefficients up to i [s, p, d, f, g, h, i].
    """
    # Make sure it is a numpy array...
    center_red = np.array(center_red)

    # Shift density so the atomic center is at the center of the unit cell
    density, center_red = translate_density(density, center_red)

    # Convert center from reduced to cartesian coordinates
    center = np.dot(center_red, lattice)

    # Lattice and real-space grid
    ng1, ng2, ng3 = density.shape
    rx, ry, rz = real_space_grid(lattice, ng1, ng2, ng3)
    r = np.sqrt(rx**2 + ry**2 + rz**2)

    # Define sphere region
    sphere = np.sqrt((rx - center[0])**2 + (ry - center[1])**2 + (rz - center[2])**2) < radius
    d_sphere = np.copy(density)
    d_sphere[~sphere] = 0

    # Select projection functions
    if units == "multi":
        proj_p, proj_d, proj_f, proj_g, proj_h, proj_i = proj_p1, proj_d1, proj_f1, proj_g1, proj_h1, proj_i1
    elif units == "charge":
        proj_p, proj_d, proj_f, proj_g, proj_h, proj_i = proj_p2, proj_d2, proj_f2, proj_g2, proj_h2, proj_i2
    else:
        raise ValueError("Invalid units flag. Must be 'multi' or 'charge'.")

    # Calculate multipoles
    s = np.sum(d_sphere)
    py, pz, px = proj_p(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    dxy, dyz, dz2, dxz, dx2y2 = proj_d(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    fm3, fm2, fm1, f0, f1, f2, f3 = proj_f(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = proj_g(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    hm5, hm4, hm3, hm2, hm1, h0, h1, h2, h3, h4, h5 = proj_h(rx - center[0], ry - center[1], rz - center[2], d_sphere)
    im6, im5, im4, im3, im2, im1, i0, i1, i2, i3, i4, i5, i6 = proj_i(rx - center[0], ry - center[1], rz - center[2], d_sphere)

    return np.array([
        s,
        np.sqrt(4*np.pi/(3))*py, np.sqrt(4*np.pi/(3))*pz, np.sqrt(4*np.pi/(3))*px,
        np.sqrt(4*np.pi/(5))*dxy, np.sqrt(4*np.pi/(5))*dyz, np.sqrt(4*np.pi/(5))*dz2, np.sqrt(4*np.pi/(5))*dxz, np.sqrt(4*np.pi/(5))*dx2y2,
        np.sqrt(4*np.pi/(7))*fm3, np.sqrt(4*np.pi/(7))*fm2, np.sqrt(4*np.pi/(7))*fm1, np.sqrt(4*np.pi/(7))*f0, np.sqrt(4*np.pi/(7))*f1, np.sqrt(4*np.pi/(7))*f2, np.sqrt(4*np.pi/(7))*f3,
        np.sqrt(4*np.pi/(9))*gm4, np.sqrt(4*np.pi/(9))*gm3, np.sqrt(4*np.pi/(9))*gm2, np.sqrt(4*np.pi/(9))*gm1, np.sqrt(4*np.pi/(9))*g0, np.sqrt(4*np.pi/(9))*g1, np.sqrt(4*np.pi/(9))*g2, np.sqrt(4*np.pi/(9))*g3, np.sqrt(4*np.pi/(9))*g4,
        np.sqrt(4*np.pi/(11))*hm5, np.sqrt(4*np.pi/(11))*hm4, np.sqrt(4*np.pi/(11))*hm3, np.sqrt(4*np.pi/(11))*hm2, np.sqrt(4*np.pi/(11))*hm1, np.sqrt(4*np.pi/(11))*h0, np.sqrt(4*np.pi/(11))*h1, np.sqrt(4*np.pi/(11))*h2, np.sqrt(4*np.pi/(11))*h3, np.sqrt(4*np.pi/(11))*h4, np.sqrt(4*np.pi/(11))*h5,
        np.sqrt(4*np.pi/(13))*im6, np.sqrt(4*np.pi/(13))*im5, np.sqrt(4*np.pi/(13))*im4, np.sqrt(4*np.pi/(13))*im3, np.sqrt(4*np.pi/(13))*im2, np.sqrt(4*np.pi/(13))*im1, np.sqrt(4*np.pi/(13))*i0, np.sqrt(4*np.pi/(13))*i1, np.sqrt(4*np.pi/(13))*i2, np.sqrt(4*np.pi/(13))*i3, np.sqrt(4*np.pi/(13))*i4, np.sqrt(4*np.pi/(13))*i5, np.sqrt(4*np.pi/(13))*i6
    ])


# Definition of the cubic/tesseral harmonics
def proj_p2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    px = np.sum(np.sqrt(3/4/np.pi) * rx * f / r)
    py = np.sum(np.sqrt(3/4/np.pi) * ry * f / r)
    pz = np.sum(np.sqrt(3/4/np.pi) * rz * f / r)
    return py, pz, px
def proj_d2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    dz2   = np.sum( np.sqrt(15/16/np.pi) * (3 * rz**2 - r**2) * f / r**2)
    dxz   = np.sum( np.sqrt(15/16/np.pi) * 2 * rz * rx * f / r**2)
    dyz   = np.sum( np.sqrt(15/16/np.pi) * 2 * ry * rz * f / r**2)
    dxy   = np.sum( np.sqrt(15/16/np.pi) * 2 * rx * ry * f / r**2)
    dx2y2 = np.sum( np.sqrt(15/16/np.pi) * (rx**2 - ry**2) * f / r**2)
    return dxy, dyz, dz2, dxz, dx2y2
def proj_f2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    fm3 = np.sum( np.sqrt(35/32/np.pi)  * (3 * rx**2 * ry - ry**3) * f / r**3)
    fm2 = np.sum( np.sqrt(105/16/np.pi) * (2 * rx * ry * rz) * f / r**3)
    fm1 = np.sum( np.sqrt(21/32/np.pi)  * ry * (5 * rz**2 - r**2) * f / r**3)
    f0  = np.sum( np.sqrt(7/16/np.pi)   * rz * (5 * rz**2 - 3 * r**2) * f / r**3)
    f1  = np.sum( np.sqrt(21/32/np.pi)  * rx * (5 * rz**2 - r**2) * f / r**3)
    f2  = np.sum( np.sqrt(105/16/np.pi) * (rx**2 - ry**2) * rz * f / r**3)
    f3  = np.sum( np.sqrt(35/32/np.pi)  * (rx**3 - 3 * rx * ry**2) * f / r**3)
    return fm3, fm2, fm1, f0, f1, f2, f3
def proj_g2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    gm4 = np.sum( 3 / 4 * np.sqrt(35/np.pi) * (rx**3 * ry - rx * ry**3) * f / r**4)
    gm3 = np.sum( 3 / 8 * np.sqrt(70/np.pi) * (3 * rx**2 * ry * rz - ry**3 * rz) * f / r**4)
    gm2 = np.sum( 3 / 8 * np.sqrt(5/np.pi)  * (14 * rx * ry * rz**2 - 2 * rx * ry * r**2) * f / r**4)
    gm1 = np.sum( 3 / 16* np.sqrt(5/np.pi)  * (7 * ry * rz**3 - 3 * rz * ry * r**2) * f / r**4)
    g0  = np.sum( 3 / 16* np.sqrt(1/np.pi)  * (35 * rz**4 - 30 * rz**2 * r**2 + 3 * r**4) * f / r**4)
    g1  = np.sum( 3 / 16* np.sqrt(5/np.pi)  * (7 * rx * rz**3 - 3 * rz * rx * r**2) * f / r**4)
    g2  = np.sum( 3 / 8 * np.sqrt(5/np.pi)  * ((rx**2 - ry**2) * (7 * rz**2 - r**2)) * f / r**4)
    g3  = np.sum( 3 / 8 * np.sqrt(70/np.pi) * (rx**3 * rz - 3 * rx * ry**2 * rz) * f / r**4)
    g4  = np.sum( 3 / 16* np.sqrt(35/np.pi) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) * f / r**4)
    return gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4
def proj_h2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    hm5 = np.sum(np.sqrt(693/512/np.pi)  * (ry*(5*rx**4 - 10*rx**2*ry**2 + ry**4)) * f / r**5)
    hm4 = np.sum(np.sqrt(3465/256/np.pi) * (4*rx*ry*rz*(rx**2 - ry**2)) * f / r**5)
    hm3 = np.sum(np.sqrt(385/512/np.pi)  * (ry*(3*rx**2 - ry**2)*(9*rz**2 - r**2)) * f / r**5)
    hm2 = np.sum(np.sqrt(1155/64/np.pi)  * (2*rx*ry*rz*(3*rz**2 - r**2)) * f / r**5)
    hm1 = np.sum(np.sqrt(165/256/np.pi)  * (ry*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f / r**5)
    h0  = np.sum(np.sqrt(11/256/np.pi)   * (rz*(63*rz**4 - 70*rz**2*r**2 + 15*r**4)) * f / r**5)
    h1  = np.sum(np.sqrt(165/256/np.pi)  * (rx*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f / r**5)
    h2  = np.sum(np.sqrt(1155/64/np.pi)  * ((rx**2 - ry**2)*rz*(3*rz**2 - r**2)) * f / r**5)
    h3  = np.sum(np.sqrt(385/512/np.pi)  * (rx*(rx**2 - 3*ry**2)*(9*rz**2 - r**2)) * f / r**5)
    h4  = np.sum(np.sqrt(3465/256/np.pi) * (rz*(rx**4 - 6*rx**2*ry**2 + ry**4)) * f / r**5)
    h5  = np.sum(np.sqrt(693/512/np.pi)  * (rx*(rx**4 - 10*rx**2*ry**2 + 5*ry**4)) * f / r**5)
    return hm5, hm4, hm3, hm2, hm1, h0, h1, h2, h3, h4, h5
def proj_i2(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    im6 = np.sum(231/64*np.sqrt(26/231/np.pi) * rx*ry*(6*rx**4 - 20*rx**2*ry**2 + 6*ry**4) * f / r**6)
    im5 = np.sum(np.sqrt(9009/512/np.pi) * ry*rz*(5*rx**4 - 10*rx**2*ry**2 + ry**4) * f / r**6)
    im4 = np.sum(21/32*np.sqrt(13/7/np.pi) * 4*rx*ry*(rx**2 - ry**2)*(11*rz**2 - r**2) * f / r**6)
    im3 = np.sum(1/32*np.sqrt(2730/np.pi) * ry*rz*(3*rx**2-ry**2)*(11*rz**2 - 3*r**2) * f / r**6)
    im2 = np.sum(1/32*np.sqrt(2730/np.pi) * 2*rx*ry*(33*rz**4 - 18*rz**2*r**2 + r**4) * f / r**6)
    im1 = np.sum(1/8*np.sqrt(273/4/np.pi) * ry*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * f / r**6)
    i0  = np.sum(1/32*np.sqrt(13/np.pi) * (231*rz**6 - 315*rz**4*r**2 + 105*rz**2*r**5 - 5*r**6) * f / r**6)
    i1  = np.sum(1/8*np.sqrt(273/4/np.pi) * rx*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * f / r**6)
    i2  = np.sum(1/64*np.sqrt(2730/np.pi) * (rx**2-ry**2)*(33*rz**4 - 18*rz**2*r**2 + r**4) * f / r**6)
    i3  = np.sum(1/32*np.sqrt(2730/np.pi) * rx*rz*(rx**2-3*ry**2)*(11*rz**2 - 3*r**2) * f / r**6)
    i4  = np.sum(21/32*np.sqrt(13/7/np.pi) * (6*rx**2*ry**2 - rx**4 - ry**4)*(11*rz**2 - r**2) * f / r**6)
    i5  = np.sum(np.sqrt(9009/512/np.pi) * rx*rz*(rx**4 - 10*rx**2*ry**2 + 5*ry**4) * f / r**6)
    i6  = np.sum(231/64*np.sqrt(26/231/np.pi) * (rx**6-15*rx**4*ry**2 + 15*rx**2*ry**4 - ry**6) * f / r**6)
    return im6, im5, im4, im3, im2, im1, i0, i1, i2, i3, i4, i5, i6


# Definition of the cubic/tesseral harmonics without radial normalization
def proj_p1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    px = np.sum(np.sqrt(3/4/np.pi) * rx * f )
    py = np.sum(np.sqrt(3/4/np.pi) * ry * f )
    pz = np.sum(np.sqrt(3/4/np.pi) * rz * f )
    return py, pz, px
def proj_d1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    dz2   = np.sum( np.sqrt(15/16/np.pi) * (3 * rz**2 - r**2) * f )
    dxz   = np.sum( np.sqrt(15/16/np.pi) * 2 * rz * rx * f )
    dyz   = np.sum( np.sqrt(15/16/np.pi) * 2 * ry * rz * f )
    dxy   = np.sum( np.sqrt(15/16/np.pi) * 2 * rx * ry * f )
    dx2y2 = np.sum( np.sqrt(15/16/np.pi) * (rx**2 - ry**2) * f )
    return dxy, dyz, dz2, dxz, dx2y2
def proj_f1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    fm3 = np.sum( np.sqrt(35/32/np.pi)  * (3 * rx**2 * ry - ry**3) * f )
    fm2 = np.sum( np.sqrt(105/16/np.pi) * (2 * rx * ry * rz) * f )
    fm1 = np.sum( np.sqrt(21/32/np.pi)  * ry * (5 * rz**2 - r**2) * f )
    f0  = np.sum( np.sqrt(7/16/np.pi)   * rz * (5 * rz**2 - 3 * r**2) * f )
    f1  = np.sum( np.sqrt(21/32/np.pi)  * rx * (5 * rz**2 - r**2) * f )
    f2  = np.sum( np.sqrt(105/16/np.pi) * (rx**2 - ry**2) * rz * f )
    f3  = np.sum( np.sqrt(35/32/np.pi)  * (rx**3 - 3 * rx * ry**2) * f )
    return fm3, fm2, fm1, f0, f1, f2, f3
def proj_g1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    gm4 = np.sum( 3 / 4 * np.sqrt(35/np.pi) * (rx**3 * ry - rx * ry**3) * f )
    gm3 = np.sum( 3 / 8 * np.sqrt(70/np.pi) * (3 * rx**2 * ry * rz - ry**3 * rz) * f )
    gm2 = np.sum( 3 / 8 * np.sqrt(5/np.pi)  * (14 * rx * ry * rz**2 - 2 * rx * ry * r**2) * f )
    gm1 = np.sum( 3 / 16* np.sqrt(5/np.pi)  * (7 * ry * rz**3 - 3 * rz * ry * r**2) * f )
    g0  = np.sum( 3 / 16* np.sqrt(1/np.pi)  * (35 * rz**4 - 30 * rz**2 * r**2 + 3 * r**4) * f )
    g1  = np.sum( 3 / 16* np.sqrt(5/np.pi)  * (7 * rx * rz**3 - 3 * rz * rx * r**2) * f )
    g2  = np.sum( 3 / 8 * np.sqrt(5/np.pi)  * ((rx**2 - ry**2) * (7 * rz**2 - r**2)) * f )
    g3  = np.sum( 3 / 8 * np.sqrt(70/np.pi) * (rx**3 * rz - 3 * rx * ry**2 * rz) * f )
    g4  = np.sum( 3 / 16* np.sqrt(35/np.pi) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) * f )
    return gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4
def proj_h1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    hm5 = np.sum(np.sqrt(693/512/np.pi)  * (ry*(5*rx**4 - 10*rx**2*ry**2 + ry**4)) * f )
    hm4 = np.sum(np.sqrt(3465/256/np.pi) * (4*rx*ry*rz*(rx**2 - ry**2)) * f )
    hm3 = np.sum(np.sqrt(385/512/np.pi)  * (ry*(3*rx**2 - ry**2)*(9*rz**2 - r**2)) * f )
    hm2 = np.sum(np.sqrt(1155/64/np.pi)  * (2*rx*ry*rz*(3*rz**2 - r**2)) * f )
    hm1 = np.sum(np.sqrt(165/256/np.pi)  * (ry*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f )
    h0  = np.sum(np.sqrt(11/256/np.pi)   * (rz*(63*rz**4 - 70*rz**2*r**2 + 15*r**4)) * f )
    h1  = np.sum(np.sqrt(165/256/np.pi)  * (rx*(21*rz**4 - 14*rz**2*r**2 + r**4)) * f )
    h2  = np.sum(np.sqrt(1155/64/np.pi)  * ((rx**2 - ry**2)*rz*(3*rz**2 - r**2)) * f )
    h3  = np.sum(np.sqrt(385/512/np.pi)  * (rx*(rx**2 - 3*ry**2)*(9*rz**2 - r**2)) * f )
    h4  = np.sum(np.sqrt(3465/256/np.pi) * (rz*(rx**4 - 6*rx**2*ry**2 + ry**4)) * f )
    h5  = np.sum(np.sqrt(693/512/np.pi)  * (rx*(rx**4 - 10*rx**2*ry**2 + 5*ry**4)) * f )
    return hm5, hm4, hm3, hm2, hm1, h0, h1, h2, h3, h4, h5
def proj_i1(rx, ry, rz, f):
    r = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-30
    im6 = np.sum(231/64*np.sqrt(26/231/np.pi) * rx*ry*(6*rx**4 - 20*rx**2*ry**2 + 6*ry**4) * f )
    im5 = np.sum(np.sqrt(9009/512/np.pi) * ry*rz*(5*rx**4 - 10*rx**2*ry**2 + ry**4) * f )
    im4 = np.sum(21/32*np.sqrt(13/7/np.pi) * 4*rx*ry*(rx**2 - ry**2)*(11*rz**2 - r**2) * f )
    im3 = np.sum(1/32*np.sqrt(2730/np.pi) * ry*rz*(3*rx**2-ry**2)*(11*rz**2 - 3*r**2) * f )
    im2 = np.sum(1/32*np.sqrt(2730/np.pi) * 2*rx*ry*(33*rz**4 - 18*rz**2*r**2 + r**4) * f )
    im1 = np.sum(1/8*np.sqrt(273/4/np.pi) * ry*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * f )
    i0  = np.sum(1/32*np.sqrt(13/np.pi) * (231*rz**6 - 315*rz**4*r**2 + 105*rz**2*r**5 - 5*r**6) * f )
    i1  = np.sum(1/8*np.sqrt(273/4/np.pi) * rx*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * f )
    i2  = np.sum(1/64*np.sqrt(2730/np.pi) * (rx**2-ry**2)*(33*rz**4 - 18*rz**2*r**2 + r**4) * f )
    i3  = np.sum(1/32*np.sqrt(2730/np.pi) * rx*rz*(rx**2-3*ry**2)*(11*rz**2 - 3*r**2) * f )
    i4  = np.sum(21/32*np.sqrt(13/7/np.pi) * (6*rx**2*ry**2 - rx**4 - ry**4)*(11*rz**2 - r**2) * f )
    i5  = np.sum(np.sqrt(9009/512/np.pi) * rx*rz*(rx**4 - 10*rx**2*ry**2 + 5*ry**4) * f )
    i6  = np.sum(231/64*np.sqrt(26/231/np.pi) * (rx**6-15*rx**4*ry**2 + 15*rx**2*ry**4 - ry**6) * f )
    return im6, im5, im4, im3, im2, im1, i0, i1, i2, i3, i4, i5, i6


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


def read_xsf_file(xsf_path):
    """
    Read a 3D scalar field from an XSF file generated by generate_xsf_file().
    
    Returns:
        scalar_field (np.ndarray): Array of shape (Nx, Ny, Nz) with periodic padding removed.
        lattice (np.ndarray): 3x3 lattice matrix.
        origin (np.ndarray): 3-element origin vector.
    """
    with open(xsf_path, 'r') as f:
        lines = f.readlines()

    # ---- Find the DATAGRID block ----
    start = None
    for i, line in enumerate(lines):
        if "BEGIN_DATAGRID_3D" in line:
            start = i
            break

    if start is None:
        raise ValueError("No BEGIN_DATAGRID_3D section found in XSF file.")

    # ---- Read grid dimensions ----
    dims = list(map(int, lines[start + 1].split()))
    Nx1, Ny1, Nz1 = dims  # these are Nx+1 etc.

    # ---- Read origin ----
    origin = np.array(list(map(float, lines[start + 2].split())))

    # ---- Read lattice vectors (3 lines) ----
    lattice = np.zeros((3, 3))
    for j in range(3):
        lattice[j] = list(map(float, lines[start + 3 + j].split()))

    # ---- Read scalar field ----
    data_start = start + 6  # first line of the grid values

    raw_vals = []
    for line in lines[data_start:]:
        if "END_DATAGRID_3D" in line:
            break
        parts = line.split()
        for p in parts:
            raw_vals.append(float(p))

    raw_vals = np.array(raw_vals)

    # ---- Reshape back into (Nz+1, Ny+1, Nx+1) ----
    if raw_vals.size != Nx1 * Ny1 * Nz1:
        raise ValueError("Incorrect number of grid points in XSF file.")

    periodic_field = raw_vals.reshape((Nz1, Ny1, Nx1))
    periodic_field = np.transpose(periodic_field, (2, 1, 0))  # back to (Nx+1, Ny+1, Nz+1)

    # ---- Remove periodic padding (inverse of your +1 extension) ----
    scalar_field = periodic_field[:Nx1-1, :Ny1-1, :Nz1-1]

    return scalar_field, lattice, origin



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
    py    = coeffs[1]  * np.sqrt(3) * ry / r * R
    pz    = coeffs[2]  * np.sqrt(3) * rz / r * R
    px    = coeffs[3]  * np.sqrt(3) * rx / r * R
    dxy   = coeffs[4]  * (1 / 2 * np.sqrt(15)) * 2 * rx * ry / r**2 * R
    dyz   = coeffs[5]  * (1 / 2 * np.sqrt(15)) * 2 * ry * rz / r**2 * R
    dz2   = coeffs[6]  * (1 / 2 * np.sqrt(5))  * (3 * rz**2 - r**2) / r**2 * R
    dxz   = coeffs[7]  * (1 / 2 * np.sqrt(15)) * 2 * rz * rx / r**2 * R
    dx2y2 = coeffs[8]  * (1 / 2 * np.sqrt(15)) * (rx**2 - ry**2) / r**2 * R
    fm3   = coeffs[9]  * np.sqrt(35 / 16) * (3 * rx**2 * ry - ry**3) / r**3 * R
    fm2   = coeffs[10] * np.sqrt(105 / 4) * (2 * rx * ry * rz) / r**3 * R
    fm1   = coeffs[11] * np.sqrt(21 / 16) * ry * (5 * rz**2 - r**2) / r**3 * R
    f0    = coeffs[12] * np.sqrt(7 / 5) * rz * (5 * rz**2 - 3 * r**2) / r**3 * R
    f1    = coeffs[13] * np.sqrt(21 / 16) * rx * (5 * rz**2 - r**2) / r**3 * R
    f2    = coeffs[14] * np.sqrt(105 / 4) * (rx**2 - ry**2) * rz / r**3 * R
    f3    = coeffs[15] * np.sqrt(35 / 16) * (rx**3 - 3 * rx * ry**2) / r**3 * R
    gm4   = coeffs[16] * (3 / 2  * np.sqrt(35)) * (rx**3 * ry - rx * ry**3) / r**4 * R
    gm3   = coeffs[17] * (3 / 4  * np.sqrt(70)) * (3 * rx**2 * ry * rz - ry**3 * rz) / r**4 * R
    gm2   = coeffs[18] * (3 / 4  * np.sqrt(5))  * (14 * rx * ry * rz**2 - 2 * rx * ry * r**2) / r**4 * R
    gm1   = coeffs[19] * (3 / 8 * np.sqrt(5))  * (7 * ry * rz**3 - 3 * rz * ry * r**2) / r**4 * R
    g0    = coeffs[20] * 3 / 8 * (35 * rz**4 - 30 * rz**2 * r**2 + 3 * r**4) / r**4 * R
    g1    = coeffs[21] * 3 / 8 * np.sqrt(5)   * (7 * rx * rz**3 - 3 * rz * rx * r**2) / r**4 * R
    g2    = coeffs[22] * (3 / 4  * np.sqrt(5))  * ((rx**2 - ry**2) * (7 * rz**2 - r**2)) / r**4 * R
    g3    = coeffs[23] * (3 / 4  * np.sqrt(70)) * (rx**3 * rz - 3 * rx * ry**2 * rz) / r**4 * R
    g4    = coeffs[24] * (3 / 8 * np.sqrt(35)) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) / r**4 * R
    hm5   = coeffs[25] * np.sqrt(693/128) * (ry*(5*rx**4 - 10*rx**2*ry**2 + ry**4)) * R / r**5
    hm4   = coeffs[26] * np.sqrt(3465/64) * (4*rx*ry*rz*(rx**2 - ry**2)) * R / r**5
    hm3   = coeffs[27] * np.sqrt(385/128) * (ry*(3*rx**2 - ry**2)*(9*rz**2 - r**2)) * R / r**5
    hm2   = coeffs[28] * np.sqrt(1155/16) * (2*rx*ry*rz*(3*rz**2 - r**2)) * R / r**5
    hm1   = coeffs[29] * np.sqrt(165/64) * (ry*(21*rz**4 - 14*rz**2*r**2 + r**4)) * R / r**5
    h0    = coeffs[30] * np.sqrt(11/64) * (rz*(63*rz**4 - 70*rz**2*r**2 + 15*r**4)) * R / r**5
    h1    = coeffs[31] * np.sqrt(165/64) * (rx*(21*rz**4 - 14*rz**2*r**2 + r**4)) * R / r**5
    h2    = coeffs[32] * np.sqrt(1155/16) * ((rx**2 - ry**2)*rz*(3*rz**2 - r**2)) * R / r**5
    h3    = coeffs[33] * np.sqrt(385/128) * (rx*(rx**2 - 3*ry**2)*(9*rz**2 - r**2)) * R / r**5
    h4    = coeffs[34] * np.sqrt(3465/64) * (rz*(rx**4 - 6*rx**2*ry**2 + ry**4)) * R / r**5
    h5    = coeffs[35] * np.sqrt(693/128) * (rx*(rx**4 - 10*rx**2*ry**2 + 5*ry**4)) * R / r**5
    im6   = coeffs[36] * 231/32*np.sqrt(26/231) * rx*ry*(6*rx**4 - 20*rx**2*ry**2 + 6*ry**4) * R / r**6
    im5   = coeffs[37] * np.sqrt(9009/128) * ry*rz*(5*rx**4 - 10*rx**2*ry**2 + ry**4) * R / r**6
    im4   = coeffs[38] * 21/16*np.sqrt(13/7) * 4*rx*ry*(rx**2 - ry**2)*(11*rz**2 - r**2) * R / r**6
    im3   = coeffs[39] * 1/16*np.sqrt(2730) * ry*rz*(3*rx**2-ry**2)*(11*rz**2 - 3*r**2) * R / r**6
    im2   = coeffs[40] * 1/32*np.sqrt(2730) * 2*rx*ry*(33*rz**4 - 18*rz**2*r**2 + r**4) * R / r**6
    im1   = coeffs[41] * 1/4*np.sqrt(273/4) * ry*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * R / r**6
    i0    = coeffs[42] * 1/16*np.sqrt(13) * (231*rz**6 - 315*rz**4*r**2 + 105*rz**2*r**5 - 5*r**6) * R / r**6
    i1    = coeffs[43] * 1/4*np.sqrt(273/4) * rx*rz*(33*rz**4 - 30*rz**2*r**2 + 5*r**4) * R / r**6
    i2    = coeffs[44] * 1/32*np.sqrt(2730) * (rx**2-ry**2)*(33*rz**4 - 18*rz**2*r**2 + r**4) * R / r**6
    i3    = coeffs[45] * 1/16*np.sqrt(2730) * rx*rz*(rx**2-3*ry**2)*(11*rz**2 - 3*r**2) * R / r**6
    i4    = coeffs[46] * 21/16*np.sqrt(13/7) * (6*rx**2*ry**2 - rx**4 - ry**4)*(11*rz**2 - r**2) * R / r**6
    i5    = coeffs[47] * np.sqrt(9009/128) * rx*rz*(rx**4 - 10*rx**2*ry**2 + 5*ry**4) * R / r**6
    i6    = coeffs[48] * 231/32*np.sqrt(26/231) * (rx**6-15*rx**4*ry**2 + 15*rx**2*ry**4 - ry**6) * R / r**6

    return s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4, hm5, hm4, hm3, hm2, hm1, h0, h1, h2, h3, h4, h5, im6, im5, im4, im3, im2, im1, i0, i1, i2, i3, i4, i5, i6


def output_analytical_components(lattice, positions, radius, coeffs, filename_prefix, threshold=1e-6):
    """
    Compute and output the analytical components of the densities for all positions, saving each component to separate files.

    Parameters:
    lattice (numpy.ndarray): 3x3 lattice matrix.
    positions (numpy.ndarray): Nx3 array of fractional coordinates.
    radius (float): Radius for the spherical projection.
    coeffs (numpy.ndarray): Nx49 matrix of coefficients for each position.
    filename_prefix (str): Prefix for the output XSF files.
    threshold (float): Threshold for generating partial component XSF files.
    """

    # Make sure they are numpy arrrays and have the correct size (in case of only one position being given)
    positions = np.array(positions, dtype=float)
    if positions.ndim == 1:
        positions = positions[np.newaxis, :]   # make it 1×3

    coeffs = np.array(coeffs, dtype=float)
    if coeffs.ndim == 1:
        coeffs = coeffs[np.newaxis, :]         # make it 1×49

    # Create a real-space grid
    rx, ry, rz = real_space_grid(lattice, 100, 100, 100)

    # Define components and initialize their totals
    components = {
    # s and p (1 + 3 components)
    "s": None, "p_y": None, "p_z": None, "p_x": None,

    # d (5 components)
    "d_xy": None, "d_yz": None, "d_z^2": None, "d_xz": None, "d_x^2-y^2": None,

    # f (7 components)
    "f_y(3x^2-y^2)": None, "f_xyz": None, "f_yz^2": None, "f_z^3": None, "f_xz^2": None, "f_z(x^2-y^2)": None, "f_x(x^2-3y^2)": None,

    # g (9 components)
    "g_xy(x^2-y^2)": None, "g_yz(3x^2-y^2)": None, "g_xyz^2": None, "g_yz^3": None, "g_z^4": None, "g_xz^3": None, "g_(x^2-y^2)z^2": None, "g_xz(x^2-3y^2)": None, "g_x^2y^2": None,

    # h (11 components)
    "h_m5": None, "h_m4": None, "h_m3": None, "h_m2": None, "h_m1": None, "h_0": None, "h_1": None, "h_2": None, "h_3": None, "h_4": None, "h_5": None,

    # i (13 components)
    "i_m6": None, "i_m5": None, "i_m4": None, "i_m3": None, "i_m2": None, "i_m1": None, "i_0": None, "i_1": None, "i_2": None, "i_3": None, "i_4": None, "i_5": None, "i_6": None
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
    components["f_tot"] = components["f_y(3x^2-y^2)"] + components["f_xyz"] + components["f_yz^2"] + components["f_z^3"] + components["f_xz^2"] + components["f_z(x^2-y^2)"] + components["f_x(x^2-3y^2)"]
    components["g_tot"] = components["g_xy(x^2-y^2)"] + components["g_yz(3x^2-y^2)"] + components["g_xyz^2"] + components["g_yz^3"] + components["g_z^4"] + components["g_xz^3"] + components["g_(x^2-y^2)z^2"] + components["g_xz(x^2-3y^2)"] + components["g_x^2y^2"]
    components["h_tot"] = components["h_m5"] + components["h_m4"] + components["h_m3"] + components["h_m2"] + components["h_m1"] + components["h_0"] + components["h_1"] + components["h_2"] + components["h_3"] + components["h_4"] + components["h_5"]
    components["i_tot"] = components["i_m6"] + components["i_m5"] + components["i_m4"] + components["i_m3"] + components["i_m2"] + components["i_m1"] + components["i_0"] + components["i_1"] + components["i_2"] + components["i_3"] + components["i_4"] + components["i_5"] + components["i_6"]

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
    if exceeds_threshold(list(range(25, 36))): # h orbitals
        generate_xsf_file(components["h_tot"], lattice, f"{filename_prefix}_h_tot.xsf")
    if exceeds_threshold(list(range(36, 49))): # i orbitals
        generate_xsf_file(components["i_tot"], lattice, f"{filename_prefix}_i_tot.xsf")

    # Output partial density components if the corresponding coefficient exceeds the threshold
    for key, value in components.items():
        if key not in ["tot", "p_tot", "d_tot", "f_tot", "g_tot", "h_tot", "i_tot"]:  # Skip combined totals
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


def find_Hall(lattice, atomic_positions, atomic_species):

    structure = (lattice, np.transpose(atomic_positions), atomic_species)
    str_object = spglib.get_symmetry_dataset(structure, symprec=1e-6)
    Hall = str_object.hall_number

    return Hall


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


def project_irreps(
    density_file,
    dft_code,
    spacegroup=1,
    auto_symmetry=False,
    supercell_size=None,
    kpoint=None,
):
    """
    Project real-space density components onto the irreducible representations (irreps)
    of the little group of a specified k-point.

    This function:
    1. Loads the density data from a DFT code output.
    2. Identifies the space group symmetry operations (rotations and translations).
    3. Determines the little group of the given k-point and its irreps.
    4. Projects each component of the density onto each irrep.
    5. Saves the projected densities into `.xsf` files for visualization.
    6. Writes symmetry information, character tables, and projection weights
       into an output log file (`.pdout`).

    Parameters
    ----------
    density_file : str
        Path to the density file produced by the DFT calculation.
    dft_code : str
        Identifier for the DFT code used (used by `load_density_file` to parse the file).
    spacegroup : int
        Hall number specifying the space group symmetry. Default is 1.
        For full list see: https://yseto.net/en/sg/sg1
    auto_symmetry: bool, optional
        Automatic detection of space group Hall number. May not be useful in many cases, since this function
        is thought to be used to project onto the irreps of the PARENT high-symmetry space group. If you use
        automatic detection, you will likely only get the trivial irrep. Thus, defult is False.
    supercell_size : list[int], optional
        Size of the supercell used for projection in each lattice direction,
        defaults to [1, 1, 1]. Needed for commensurate points away from Gamma,
        for example one needs [2, 1, 1] for the k-point [0.5, 0, 0].
    kpoint : list[float], optional
        Target k-point in reciprocal coordinates. Defaults to Gamma point [0, 0, 0].

    Returns
    -------
    - A `.pdout` text file containing:
        * Lattice vectors
        * Space group operations
        * Little group operations
        * Irrep character tables
        * Projection weights for each density component
    - `.xsf` files for each projected component, named
      `{basename}_{component}_irrep{i}.xsf`.
    """
    # Handle default parameters
    if kpoint is None:
        kpoint = [0.0, 0.0, 0.0]
    kpoint = np.asarray(kpoint)
    
    if supercell_size is None:
        supercell_size = [1, 1, 1]
    supercell_size = np.asarray(supercell_size, dtype=int)

    # load density explicitly according to dft_code
    lattice, atomic_positions, atomic_species, grid, comp_arrays = load_density_file(density_file, dft_code)

    # If user has not set space group number and set auto_symmetry on, then find symmetry with spglib
    # Not suggested since this function is thought to be used to project onto irreps of parent phase
    if auto_symmetry and spacegroup == 1:
        spacegroup = find_Hall(lattice, atomic_positions, atomic_species)

    # give output file name
    input_basename = get_output_basename()
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
        write("\n=== Lattice vectors (Bohr radii) ===")
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
            
            # Calculate max of original component for weight normalization
            max_original = np.max(np.abs(density))
            write(f"Max absolute value of original {comp_name}: {max_original:.6f}\n")
            
            for i, irrep in enumerate(irreps):
                write(f"Projecting onto irrep {i+1}...")
                
                char_table = get_character(irrep)
                proj_density = project_single_irrep(density, little_group_symm, little_group_tnons, char_table, supercell_size, kpoint)
                
                # Calculate weight: max(projected) / max(original)
                max_projected = np.max(np.abs(proj_density))
                weight = max_projected / max_original if max_original > 0 else 0
                
                # Generate output filename
                outname = f"{input_basename}_{comp_name}_irrep{i+1}.xsf"
                generate_xsf_file(proj_density, lattice, outname)
                
                write(f"Done! Saved as {outname}")
                write(f"Weight (max|proj|/max|orig|): {weight:.6f}\n")
            
            write("-" * 40)
        
        write("\nAll projections completed successfully!")


def project_harmonics(
    density_file,
    dft_code,
    center,
    radius,
    spacegroup=1,
    auto_symmetry=True,
    output_components=False,
    decimals=4,
    units="multi"
):
    """
    Project real-space density components onto tesseral harmonics 
    (multipole expansion) around a given center within a sphere of radius R.

    This function:
    1. Loads the density data from a DFT code output.
    2. Expands the density inside a sphere of radius `radius` centered at 
       the specified atomic/site position(s).
    3. Groups the expansion coefficients into s, p, d, f, g, h, and i 
       harmonics.
    4. Prints results for each symmetry-equivalent Wyckoff position 
       (from the specified or automatically detected spacegroup).
    5. Optionally outputs analytical harmonics into `.xsf` files for 
       visualization.

    Parameters
    ----------
    density_file : str
        Path to the density file produced by the DFT calculation.
    dft_code : str
        Identifier for the DFT code used (parsed by `load_density_file`).
    center : list[float]
        Reference position (in fractional coordinates) around which 
        the spherical expansion is performed.
    radius : float
        Radius of the sphere (in Bohr radii) within which the density is projected.
    spacegroup : int, optional
        Space group Hall number (default = 1, i.e. P1 symmetry). Used to generate
        Wyckoff-equivalent positions. For full list see: https://yseto.net/en/sg/sg1
    auto_symmetry : bool, optional
        Automatic detection of Hall number through spglib. Default is True.
        NOTICE: Only works if spacegroup=1.
    output_components : bool, optional
        If True, writes analytical harmonics for each component into `.xsf` 
        files for visualization. Default is False.
    decimals : int, optional
        Number of decimal places to use when formatting printed results. 
        Default is 4.

    Returns
    -------
    - A `.pdout` text file containing:
        * Lattice vectors
        * Wyckoff-equivalent positions
        * Multipole expansion coefficients for s–i harmonics at each position
        * Sum over all positions (if multiple sites are present)
    - `.xsf` files (if `output_components=True`), containing real-space 
      harmonics for each component of the density.
    """

    lattice, atomic_positions, atomic_species, grid, comp_arrays = load_density_file(density_file, dft_code)

    # If user has not set space group number and left auto_symmetry on, then find symmetry with spglib
    if auto_symmetry and spacegroup == 1:
        spacegroup = find_Hall(lattice, atomic_positions, atomic_species)

    input_basename = get_output_basename()
    output_file = input_basename + ".pdout"

    center = np.asarray(center)
    positions = np.round(wyckoff(center, spacegroup), 5)

    # formatting settings
    LABEL_FIELD_WIDTH = 3    # narrower for "s", "p", "d", ...
    VALUE_FIELD_WIDTH = 14   # enough space for numbers and multipole names

    NUM_FMT = f"{{:{VALUE_FIELD_WIDTH}.{decimals}f}}"
    LABEL_FMT = f"{{:<{LABEL_FIELD_WIDTH}}}"

    def fmt(v):
        return NUM_FMT.format(v)

    def print_block(label, arr, labels_list):
        # header row
        write(
            LABEL_FMT.format(label)
            + " | ".join(f"{lab:>{VALUE_FIELD_WIDTH}}" for lab in labels_list)
        )
        # value row
        write(
            LABEL_FMT.format(label)
            + " | ".join(fmt(v) for v in arr)
        )


    # labels for each multipole
    MULTIPOLE_LABELS = {
    "s": ["s"],
    "p": ["y", "z", "x"],
    "d": ["xy", "yz", "z^2", "xz", "x^2-y^2"],
    "f": ["y(3x^2-y^2)", "xyz", "yz^2", "z^3", "xz^2", "z(x^2-y^2)", "x(x^2-3y^2)"],
    "g": ["xy(x^2-y^2)", "yz(3x^2-y^2)", "xyz^2", "yz^3", "z^4", "xz^3",
          "(x^2-y^2)z^2", "xz(x^2-3y^2)", "x^2y^2"],
    "h": ["x^2y^3", "xyz(x^2-y^2)", "yz^2(3x^2-y^2)", "xyz^3", "yz^4", "z^5",
          "xz^4", "(x^2-y^2)z^3", "xz^2(x^2-3y^2)", "x^2y^2z", "x^3y^2"],
    "i": ["x^3y^3", "x^2y^3z", "xy(x^2-y^2)z^2", "yz^3(3x^2-y^2)", "xyz^4", "yz^5",
          "z^6", "xz^5", "(x^2-y^2)z^4", "xz^3(x^2-3y^2)", "x^2y^2z^2",
          "x^3y^2z", "x^2y^2(x^2-y^2)"]
    }

    # helper function to print to screen and file
    def write(msg):
        print(msg)
        f.write(msg + "\n")        

    with open(output_file, "w") as f:

        # --- Print lattice and positions ---
        write("\n=== Lattice vectors (Bohr radii) ===")
        for i, vec in enumerate(lattice):
            write(f"Vector {i+1}: [{vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}]")
        
        write(f"\n=== Space Group Hall number: {spacegroup:.0f} ===")
        write("(Visit https://yseto.net/en/sg/sg1 for full list)")

        write("\n=== Wyckoff-equivalent positions ===")
        for i, pos in enumerate(positions):
            write(f"Position {i+1}: [{pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f}]")
        write("\n" + "-"*145 + "\n")

        # --- Multipole projections ---
        for comp, arr in comp_arrays.items():
            coeffs_list = []

            write(f"\n=== Projections for component: {comp} ===\n")

            for idx, pos in enumerate(positions):
                coeffs_row = project_sphere(arr, lattice, np.asarray(pos), radius, units)
                coeffs_list.append(coeffs_row)

                s_coeff = coeffs_row[0:1]
                p_coeff = coeffs_row[1:4]
                d_coeff = coeffs_row[4:9]
                f_coeff = coeffs_row[9:16]
                g_coeff = coeffs_row[16:25]
                h_coeff = coeffs_row[25:36]
                i_coeff = coeffs_row[36:49]

                write(f"Position {idx+1}: {pos}")
                write("-" * 145)
                print_block("s", s_coeff, MULTIPOLE_LABELS["s"])
                print_block("p", p_coeff, MULTIPOLE_LABELS["p"])
                print_block("d", d_coeff, MULTIPOLE_LABELS["d"])
                print_block("f", f_coeff, MULTIPOLE_LABELS["f"])
                print_block("g", g_coeff, MULTIPOLE_LABELS["g"])
                print_block("h", h_coeff, MULTIPOLE_LABELS["h"])
                print_block("i", i_coeff, MULTIPOLE_LABELS["i"])
                write("-" * 145 + "\n")

            coeffs = np.array(coeffs_list)

            # Only print sum if there is more than one site
            if len(positions) > 1:
                sum_coeffs = np.sum(coeffs, axis=0)
                write(f"=== Sum over positions for component: {comp} ===")
                write("-" * 145)
                print_block("s", sum_coeffs[0:1], MULTIPOLE_LABELS["s"])
                print_block("p", sum_coeffs[1:4], MULTIPOLE_LABELS["p"])
                print_block("d", sum_coeffs[4:9], MULTIPOLE_LABELS["d"])
                print_block("f", sum_coeffs[9:16], MULTIPOLE_LABELS["f"])
                print_block("g", sum_coeffs[16:25], MULTIPOLE_LABELS["g"])
                print_block("h", sum_coeffs[25:36], MULTIPOLE_LABELS["h"])
                print_block("i", sum_coeffs[36:49], MULTIPOLE_LABELS["i"])
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

    if len(out) == 5:
        lattice, atomic_positions, atomic_species, grid, charge = out
        comp_arrays = {"charge": charge}
    elif len(out) == 6:
        lattice, atomic_positions, atomic_species, grid, charge, mz = out
        comp_arrays = {"charge": charge, "mz": mz}
    elif len(out) == 8:
        lattice, atomic_positions, atomic_species, grid, charge, mx, my, mz = out
        comp_arrays = {"charge": charge, "mx": mx, "my": my, "mz": mz}
    else:
        raise ValueError(
            f"Unexpected return from density reader: expected 5, 6 or 8 items, got {len(out)}"
        )
    
    return lattice, atomic_positions, atomic_species, grid, comp_arrays


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



def xrd_powder(charge, lattice, lambda_x=1.5406, do_plot=True):
    """
    Compute a simulated XRD powder pattern from a 3D charge density.

    Parameters
    ----------
    charge : 3D array
        Charge density on a real-space grid.
    lattice : array-like (3x3)
        Direct lattice vectors in Bohr.
    lambda_x : float
        X-ray wavelength in Å (default: Cu Kα = 1.5406 Å).
    do_plot : bool
        If True, plots the intensity vs 2θ.

    Returns
    -------
    centers_deg : 1D array
        2θ bin centers in degrees.
    I_binned : 1D array
        Binned diffraction intensity.
    """

    Nx, Ny, Nz = charge.shape

    # Convert lattice to Angstrom
    a1, a2, a3 = lattice[0]*0.529177, lattice[1]*0.529177, lattice[2]*0.529177

    V = np.dot(a1, np.cross(a2, a3))
    astar = 2*np.pi * np.cross(a2, a3) / V
    bstar = 2*np.pi * np.cross(a3, a1) / V
    cstar = 2*np.pi * np.cross(a1, a2) / V

    # Convert charge to n_electrons / Ang^3
    charge *= 1/(0.529177)**3

    hr = np.arange(-Nx//2, Nx//2)
    kr = np.arange(-Ny//2, Ny//2)
    lr = np.arange(-Nz//2, Nz//2)
    H, K, L = np.meshgrid(hr, kr, lr, indexing='ij')

    Kx = H*astar[0] + K*bstar[0] + L*cstar[0]
    Ky = H*astar[1] + K*bstar[1] + L*cstar[1]
    Kz = H*astar[2] + K*bstar[2] + L*cstar[2]
    Kmag = np.sqrt(Kx**2 + Ky**2 + Kz**2)

    F = fftshift(fftn(charge))
    I = np.abs(F)**2

    arg = (Kmag * lambda_x) / (4*np.pi)
    valid_arg = np.isfinite(arg) & (arg >= -1) & (arg <= 1)

    two_theta_all = 2 * np.arcsin(arg[valid_arg])
    Ivals_all = I[valid_arg]

    mask_good = np.isfinite(two_theta_all)
    two_theta = two_theta_all[mask_good]
    Ivals = Ivals_all[mask_good]

    # Binning
    dtheta = 0.001 * np.pi/180
    edges = np.arange(two_theta.min() - 0.5*dtheta, two_theta.max() + 0.5*dtheta + dtheta, dtheta)
    centers = 0.5*(edges[:-1] + edges[1:])

    # use numpy.histogram (it handles weights and edges cleanly)
    I_binned, _ = np.histogram(two_theta, bins=edges, weights=Ivals)

    # # Gaussian smoothing?
    # FWHM = 0.001
    # sigma = FWHM / 2.355
    # halfw = int(np.ceil(4*sigma/dtheta))
    # xk = np.arange(-halfw, halfw+1) * dtheta
    # kernel = np.exp(-0.5*(xk/sigma)**2)
    # kernel /= kernel.sum()

    # I_smooth = np.convolve(I_binned, kernel, mode='same')

    centers_deg = centers * 180/np.pi

    if do_plot:
        fig = plt.figure(figsize=(8,4))
        gs = fig.add_gridspec(10, 1, hspace=0.0)
        ax1 = fig.add_subplot(gs[:9, 0])
        ax1.plot(centers_deg, I_binned, 'k-', linewidth=1)
        ax1.set_ylabel("Intensity [arb. units]")
        ax1.set_xlim(0, 120)
        ax1.grid()
        ax1.set_xticklabels([])

        ax2 = fig.add_subplot(gs[9:, 0])
        ax2.plot(centers_deg, I_binned, 'r-', linewidth=0.5)
        ax2.set_xlabel("2θ [deg]")
        ax2.set_ylabel("Peaks")
        ax2.set_xlim(0, 120)
        ax2.set_ylim(1e-7, 2e-7)
        ax2.grid()
        ax2.set_yticklabels([])
        ax2.set_yticks([])

        plt.show()

    return centers_deg, I_binned


# import mplcursors
def xrd_crystal(charge, lattice, plane='001', shift=0.0, do_plot=True):
    Nx, Ny, Nz = charge.shape

    # Convert lattice to Angstrom
    a1, a2, a3 = lattice[0]*0.529177, lattice[1]*0.529177, lattice[2]*0.529177

    V = np.dot(a1, np.cross(a2, a3))
    astar = 2*np.pi * np.cross(a2, a3) / V
    bstar = 2*np.pi * np.cross(a3, a1) / V
    cstar = 2*np.pi * np.cross(a1, a2) / V

    # Convert charge to e / Å^3
    charge = charge * (1/(0.529177)**3)

    hr = np.arange(-Nx//2, Nx//2)
    kr = np.arange(-Ny//2, Ny//2)
    lr = np.arange(-Nz//2, Nz//2)
    H, K, L = np.meshgrid(hr, kr, lr, indexing='ij')

    Kx = H*astar[0] + K*bstar[0] + L*cstar[0]
    Ky = H*astar[1] + K*bstar[1] + L*cstar[1]
    Kz = H*astar[2] + K*bstar[2] + L*cstar[2]

    F = fftshift(fftn(charge))
    I = np.abs(F)**2
    I[Nx//2, Ny//2, Nz//2] = 0.0


    if plane=='001':
        # # Display intensity at specific Miller indices
        # h_target, k_target, l_target = 0, 0, 1
        
        # # Find the array indices corresponding to these Miller indices
        # # Since hr, kr, lr go from -N/2 to N/2-1, we need to find the index
        # h_idx = np.where(hr == h_target)[0][0]
        # k_idx = np.where(kr == k_target)[0][0] 
        # l_idx = np.where(lr == l_target)[0][0]
        
        # intensity_111 = I[h_idx, k_idx, l_idx]
        # print(f"Intensity at H={h_target}, K={k_target}, L={l_target}: {intensity_111:.30f}")

        # Choose plane
        mask = np.abs(L - shift) < 0.1

        K1_slice = Kx[mask]
        K2_slice = Ky[mask]
        I_slice  = I[mask]

        if do_plot:
            plt.figure(figsize=(6,5))
            plt.rcParams.update({'font.size': 16})

            # Plot the diffraction pattern
            scatter = plt.scatter(K1_slice, K2_slice, c=np.log10(I_slice), cmap="hot_r",
                                s=50, edgecolor="none", alpha=0.7)

            L = L[mask]
            H = H[mask]
            K = K[mask]
            absences = ~(L % 2 == 0) | ~((H - K) % 3 == 1) | ~((H - K) % 3 == 2)
            import matplotlib.patches as patches
            ax = plt.gca()
            for x, y, Lval in zip(K1_slice[absences], K2_slice[absences], L[absences]):
                circ = patches.Circle((x, y), 0.15, fill=False, linestyle='-', linewidth=1.2, edgecolor='green')
                ax.add_patch(circ)

            # Plot Brillouin Zone boundaries
            plot_brillouin_zone_2d(astar[0:2], bstar[0:2], shift*cstar[0:2])
            
            # mplcursors.cursor(hover=True)

            plt.colorbar(scatter, label="Intensity")
            plt.xlabel(f"H in [H K L={shift:.0f}] (Å⁻¹)")
            plt.ylabel(f"K in [H K L={shift:.0f}] (Å⁻¹)")
            plt.clim(-10,2)
            plt.axis("equal")
            plt.tight_layout()
            plt.xlim(-2*np.linalg.norm(astar), 2*np.linalg.norm(astar))
            plt.ylim(-2*np.linalg.norm(bstar), 2*np.linalg.norm(bstar))
            plt.savefig("myplot.png", dpi=600)
            plt.show()

    if plane=='100':
        # Choose plane
        mask = np.abs(H - shift) < 0.1

        K1_slice = Ky[mask]
        K2_slice = Kz[mask]
        I_slice  = I[mask]

        if do_plot:
            plt.figure(figsize=(6,5))
            
            # Plot the diffraction pattern
            scatter = plt.scatter(K1_slice, K2_slice, c=np.log10(I_slice), cmap="hot_r",
                                s=50, edgecolor="none", alpha=0.7)

            # Plot Brillouin Zone boundaries
            plot_brillouin_zone_2d(bstar[1:3], cstar[1:3], shift*astar[1:3])
            
            mplcursors.cursor(hover=True)

            plt.colorbar(scatter, label="Intensity")
            plt.xlabel(f"K in [H={shift:.0f} K L] (Å⁻¹)")
            plt.ylabel(f"L in [H={shift:.0f} K L] (Å⁻¹)")
            plt.clim(-10,2)
            plt.axis("equal")
            plt.tight_layout()
            plt.xlim(-2*np.linalg.norm(bstar), 2*np.linalg.norm(bstar))
            plt.ylim(-2*np.linalg.norm(cstar), 2*np.linalg.norm(cstar))
            plt.show()
        
    if plane=='010':
        # Choose plane
        mask = np.abs(K - shift) < 0.1

        K1_slice = Kz[mask]
        K2_slice = Kx[mask]
        I_slice  = I[mask]

        if do_plot:
            plt.figure(figsize=(6,5))
            
            # Plot the diffraction pattern
            scatter = plt.scatter(K1_slice, K2_slice, c=np.log10(I_slice), cmap="hot_r",
                                s=50, edgecolor="none", alpha=0.7)

            # Plot Brillouin Zone boundaries
            plot_brillouin_zone_2d(np.array([cstar[2], cstar[0]]), np.array([astar[2], astar[0]]), shift*np.array([bstar[2], bstar[0]]))
            
            mplcursors.cursor(hover=True)

            plt.colorbar(scatter, label="Intensity")
            plt.xlabel(f"L in [H K={shift:.0f} L] (Å⁻¹)")
            plt.ylabel(f"H in [H K={shift:.0f} L] (Å⁻¹)")
            plt.clim(-10,2)
            plt.axis("equal")
            plt.tight_layout()
            plt.xlim(-2*np.linalg.norm(cstar), 2*np.linalg.norm(cstar))
            plt.ylim(-2*np.linalg.norm(astar), 2*np.linalg.norm(astar))
            plt.show()

    return K1_slice, K2_slice, I_slice

def plot_brillouin_zone_2d(a, b, c, n_cells=4):
    """
    Plot Brillouin Zones by translating the first BZ around origin
    """
    
    # Generate first Brillouin Zone using Voronoi of nearest neighbors
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            vec = i * a + j * b
            neighbors.append(vec)
    
    points = np.array([[0, 0]] + neighbors)  # origin + neighbors
    
    vor = Voronoi(points)
    
    # Get the first Brillouin Zone (Voronoi cell around origin)
    origin_idx = 0  # first point is origin
    region = vor.regions[vor.point_region[origin_idx]]
    
    if not -1 in region:  # -1 indicates infinite region
        bz_polygon = [vor.vertices[i] for i in region]
        
        # Plot translated Brillouin Zones
        for i in range(-n_cells, n_cells+1):
            for j in range(-n_cells, n_cells+1):
                translation = i * a + j * b
                translated_bz = [point + c + translation for point in bz_polygon]
                
                poly = plt.Polygon(translated_bz, fill=False, 
                                 edgecolor='black', linewidth=0.5, alpha=0.3)
                plt.gca().add_patch(poly)