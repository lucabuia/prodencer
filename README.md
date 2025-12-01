# ProDenCeR

**Pro**ject **Den**sities onto **C**ubic/tesseral harmonics & **R**epresentations of space/point groups

<p align="center">
  <img src="Logo.png" alt="ProDenCeR logo" width="350"/>
</p>

ProDenCeR is a **Python package** to project **charge** and **spin densities** from *VASP* and *Abinit*.  

---

## Key Features
The code contains several useful functions, but the **two main, high level functions** are:

1. **`project_harmonics`**  
   Projects the charge/spin density inside spheres around atoms onto **atomic multipoles** (cubic/tesseral harmonics).  
   *Example:* In d-wave altermagnetic MnF$_2$, the magnetic dipoles $M_z$ are out of phase, but the magnetic octupoles $xyM_z$ are in phase.

2. **`project_irreps`**  
   Projects the charge/spin density of a **distorted cell** (either primitive or supercell) onto the **irreducible representations** of the parent space group at any commensurate k-point (via `spglib` and `spgrep`).  
   *Example:* In ferroelectric BaTiO$_3$, the primary order parameter transforms as the $\Gamma_4^-$ irrep of $m\bar{3}m$, which looks like a charge-dipole on the Ti ion.

---

## Installation

1. Clone this repository.
2. Move into the project’s root directory (the one containing `pyproject.toml`).
3. Install the package in editable mode:

```bash
pip install -e .
```

---

##
2025 – Luca Buiarelli, Hyeonseo Park, Seongjoo Jung and Turan Birol  
University of Minnesota, CEMS
