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
   *Example:* In *d-wave altermagnetic MnF₂*, the magnetic dipoles *Mz* are out of phase, but the magnetic octupoles *xyMz* are in phase.

2. **`project_irreps`**  
   Projects the charge/spin density of a **distorted cell** (either primitive or supercell) onto the **irreducible representations** of the parent space group at any commensurate k-point (via `spglib` and `spgrep`).  
   *Example:* ...

---

##
2025 – Luca Buiarelli, Seongjoo Jung, Turan Birol  
University of Minnesota, CEMS
