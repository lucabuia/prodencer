# ProDenCeR

**Pro**ject **Den**sities onto **C**ubic/tesseral harmonics & **R**epresentations of space/point groups

<p align="center">
  <img src="Logo.png" alt="ProDenCeR logo" width="350"/>
</p>

ProDenCeR is a **Python package** to project **charge** and **spin densities** from *VASP* and *Abinit*.  

---

## Key Features
The code contains several useful functions, but the **three main functions** are:

1. **`project_harmonics`**  
   Projects the charge/spin density inside spheres around atoms onto **atomic multipoles** (cubic/tesseral harmonics).  
   *Example:* In *d-wave altermagnetic MnF₂*, the magnetic dipoles *Mz* are out of phase, but the magnetic octupoles *xyMz* are in phase.

2. **`project_UC_irrep`**  
   Projects the charge/spin density of a **distorted unit cell** onto the **irreducible representations** of the parent space group (via `spglib`).  
   Use this when the distortion *does not* enlarge the unit cell (*Γ distortion*).  
   *Example:* ...

3. **`project_SC_irrep`**  
   Same as above, but for cases where the distortion **enlarges the unit cell** (supercell needed).  
   *Example:* ...

---

##
2025 – Luca Buiarelli, Seongjoo Jung, Turan Birol  
University of Minnesota, CEMS
