import numpy as np
import os

def project_sphere(density, lattice_vectors, center, radius):
    """
    Calculate the multipole projections of a density onto tesseral harmonics inside a sphere.

    Parameters:
    density (numpy.ndarray): 3D array of the density.
    lattice_vectors (numpy.ndarray): 3x3 array of lattice vectors.

    Returns:
    numpy.ndarray: Array of multipole projections up to g [s, px, py, ..., g4].
    """
    # Lattice parameters and real-space grid
    a1, a2, a3 = np.linalg.norm(lattice_vectors, axis=0)
    ng1, ng2, ng3 = density.shape
    drx, dry, drz = 1 / ng1, 1 / ng2, 1 / ng3
    red_rx, red_ry, red_rz = np.meshgrid(
        np.linspace(0, 1 - drx, ng1),
        np.linspace(0, 1 - dry, ng2),
        np.linspace(0, 1 - drz, ng3),
        indexing='ij'
    )
    
    rx = (
        lattice_vectors[0, 0] * red_rx +
        lattice_vectors[1, 0] * red_ry +
        lattice_vectors[2, 0] * red_rz
    )
    ry = (
        lattice_vectors[0, 1] * red_rx +
        lattice_vectors[1, 1] * red_ry +
        lattice_vectors[2, 1] * red_rz
    )
    rz = (
        lattice_vectors[0, 2] * red_rx +
        lattice_vectors[1, 2] * red_ry +
        lattice_vectors[2, 2] * red_rz
    )
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    
    # Total density check (for debugging)
    # print(f"Total charge check: {np.sum(density)}")
    
    # Sphere definition
    sphere = np.sqrt((rx - center[0])**2 + (ry - center[1])**2 + (rz - center[2])**2) < radius
    
    c_sphere = np.copy(density)
    c_sphere[~sphere] = 0

    # Calculate multipoles
    s = np.sum(c_sphere)
    px, py, pz = calc_p_orbitals(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    dz2, dxz, dyz, dxy, dx2y2 = calc_d_orbitals(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    fm3, fm2, fm1, f0, f1, f2, f3 = calc_f_orbitals(rx - center[0], ry - center[1], rz - center[2], c_sphere)
    gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4 = calc_g_orbitals(rx - center[0], ry - center[1], rz - center[2], c_sphere)

    return np.array([s, px, py, pz, dz2, dxz, dyz, dxy, dx2y2, fm3, fm2, fm1, f0, f1, f2, f3, gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4])


# Definition of the multipoles (tesseral harmonics)
def calc_p_orbitals(rx, ry, rz, f):
    rx[rx == 0] = 1e-30
    ry[ry == 0] = 1e-30
    rz[rz == 0] = 1e-30
    px = np.sum(np.sqrt(3 / (4 * np.pi)) * rx * f)
    py = np.sum(np.sqrt(3 / (4 * np.pi)) * ry * f)
    pz = np.sum(np.sqrt(3 / (4 * np.pi)) * rz * f)
    return px, py, pz

def calc_d_orbitals(rx, ry, rz, f):
    r2 = rx**2 + ry**2 + rz**2
    dz2 = np.sum((1 / 4 * np.sqrt(5 / np.pi)) * (3 * rz**2 - r2) * f)
    dxz = np.sum((1 / 2 * np.sqrt(15 / np.pi)) * rz * rx * f)
    dyz = np.sum((1 / 2 * np.sqrt(15 / np.pi)) * ry * rz * f)
    dxy = np.sum((1 / 4 * np.sqrt(15 / np.pi)) * 2 * rx * ry * f)
    dx2y2 = np.sum((1 / 4 * np.sqrt(15 / np.pi)) * (rx**2 - ry**2) * f)
    return dz2, dxz, dyz, dxy, dx2y2

def calc_f_orbitals(rx, ry, rz, f):
    r2 = rx**2 + ry**2 + rz**2
    fm3 = np.sum(np.sqrt(35 / (32 * np.pi)) * (3 * rx**2 * ry - ry**3) * f)
    fm2 = np.sum(np.sqrt(105 / (16 * np.pi)) * (2 * rx * ry * rz) * f)
    fm1 = np.sum(np.sqrt(21 / (32 * np.pi)) * ry * (5 * rz**2 - r2) * f)
    f0 = np.sum(np.sqrt(7 / (16 * np.pi)) * rz * (5 * rz**2 - r2) * f)
    f1 = np.sum(np.sqrt(21 / (32 * np.pi)) * rx * (5 * rz**2 - r2) * f)
    f2 = np.sum(np.sqrt(105 / (16 * np.pi)) * (rx**2 - ry**2) * rz * f)
    f3 = np.sum(np.sqrt(35 / (32 * np.pi)) * (rx**3 - 3 * rx * ry**2) * f)
    return fm3, fm2, fm1, f0, f1, f2, f3

def calc_g_orbitals(rx, ry, rz, f):
    r2 = rx**2 + ry**2 + rz**2
    r4 = r2**2
    gm4 = np.sum((3 / 4 * np.sqrt(35 / np.pi)) * (rx**3 * ry - rx * ry**3) * f)
    gm3 = np.sum((3 / 8 * np.sqrt(70 / np.pi)) * (3 * rx**2 * ry * rz - ry**3 * rz) * f)
    gm2 = np.sum((3 / 8 * np.sqrt(5 / np.pi)) * (14 * rx * ry * rz**2 - 2 * rx * ry * r2) * f)
    gm1 = np.sum((3 / 16 * np.sqrt(5 / np.pi)) * (7 * ry * rz**3 - 3 * rz * ry * r2) * f)
    g0 = np.sum((3 / 16 * np.sqrt(1 / np.pi)) * (35 * rz**4 - 30 * rz**2 * r2 + 3 * r4) * f)
    g1 = np.sum((3 / 16 * np.sqrt(5 / np.pi)) * (7 * rx * rz**3 - 3 * rz * rx * r2) * f)
    g2 = np.sum((3 / 8 * np.sqrt(5 / np.pi)) * ((rx**2 - ry**2) * (7 * rz**2 - r2)) * f)
    g3 = np.sum((3 / 8 * np.sqrt(70 / np.pi)) * (rx**3 * rz - 3 * rx * ry**2 * rz) * f)
    g4 = np.sum((3 / 16 * np.sqrt(35 / np.pi)) * (rx**4 + ry**4 - 6 * rx**2 * ry**2) * f)
    return gm4, gm3, gm2, gm1, g0, g1, g2, g3, g4
