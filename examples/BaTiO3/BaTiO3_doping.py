import numpy as np
import matplotlib.pyplot as plt
import prodencer as pd

O_pz_tetra = []
O_pz_cubic = []
x_vals = [32.00, 32.02, 32.04, 32.06, 32.08, 32.10, 32.12, 32.14, 32.16]

for n in range(9):
    input_file = f"doping_tetragonal/{2*n}_CHGCAR"
    dft_code = "vasp"

    lattice, atomic_positions, grid, charge = pd.VASP_get_density(input_file)
#     if n==0 or n==8:
#         pd.generate_xsf_file(charge, lattice, f"32p{2*n}_charge_tetra.xsf")

    # sg_tetragonal = 376
    O_coords = atomic_positions[2,:]
    radius = 1.41

    coeffs = pd.project_sphere(charge, lattice, O_coords, radius)
    O_pz_tetra.append(coeffs[2])


for n in range(9):
    input_file = f"doping_cubic/{2*n}_CHGCAR"
    dft_code = "vasp"

    lattice, atomic_positions, grid, charge = pd.VASP_get_density(input_file)
#     if n==0 or n==8:
#         pd.generate_xsf_file(charge, lattice, f"32p{2*n}_charge_cubic.xsf")

    # sg_tetragonal = 376
    O_coords = atomic_positions[2,:]
    radius = 1.41

    coeffs = pd.project_sphere(charge, lattice, O_coords, radius)
    O_pz_cubic.append(coeffs[2])



# import spglib
# from spgrep import get_spacegroup_irreps_from_primitive_symmetry
# from spgrep.representation import get_character
# sg_cubic = 517
# symmetry = spglib.get_symmetry_from_database(sg_cubic)
# symm = np.array(symmetry['rotations'])
# tnons = np.array(symmetry['translations']) # Non-symmorphic translations
# kpoint = [0.0, 0.0, 0.0]
# irreps, mapping_little_group = get_spacegroup_irreps_from_primitive_symmetry(symm, tnons, kpoint)
# char_table = get_character(irreps[7]) #Get characters
# symm = symm[mapping_little_group]

# GM4m = []
# O_pz_GM4m = []
# for n in range(9):
#     input_file = f"doping_cubic/{2*n}_CHGCAR"
#     dft_code = "vasp"
#     lattice, atomic_positions, grid, charge = pd.VASP_get_density(input_file)
#     proj_charge = pd.project_single_irrep(charge, symm, tnons, char_table, [1,1,1], kpoint)
#     pd.generate_xsf_file(proj_charge, lattice, f"32p{2*n}_charge_GM4m.xsf")
#     weight = np.max(np.abs(proj_charge)) / np.max(np.abs(charge))
#     GM4m.append(weight)

#     coeffs = pd.project_sphere(proj_charge, lattice, np.array([0.5,0.5,0]), 1.41)
#     O_pz_GM4m.append(coeffs[2])




# --- Plot ---
plt.figure(figsize=(6,4))

plt.plot(x_vals, O_pz_tetra, marker="o", linestyle="-", color="b", label="Tetragonal")
plt.plot(x_vals, O_pz_cubic, marker="o", linestyle="-", color="r", label="Cubic")

# plt.plot(x_vals, GM4m, marker="o", linestyle="-", color="g", label=r"$\Gamma_4^-$")
# plt.plot(x_vals, O_pz_GM4m, marker="o", linestyle="-", color="g", label=r"$\Gamma_4^- \rightarrow p_z$")

plt.title("$p_z$ rojection vs electron count")
# plt.title(r"$\Gamma_4^-$ rojection vs electron count")

plt.xlabel("Number of electrons (|e|)")
plt.ylabel("Projection (arb. un.)")
plt.grid(True); plt.xlim(32, 32.16)
plt.tight_layout()
plt.legend()
# plt.ylim(0,0.03)
plt.show()