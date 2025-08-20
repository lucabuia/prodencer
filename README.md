# ProDenCeR

Project Densities onto Cubic/tesseral harmonics & Representations of space/point groups.

<img src="Logo.png" alt="ProDenCeR logo" width="250"/>

This Python code projects charge/spin densities from VASP and Abinit. 

The code contains several useful functions, however the three main functions are:

1) project_harmonics: Pojects the charge/spin density inside spheres around atoms onto the atomic multipoles (cubic/tesseral harmonics). See the example of d-wave altermagnetic MnF2 where one can check that the magnetic dipoles Mz are out of phase but the magnetic ocutpoles xyMz are in-phase.

2) project_UC_irrep or project_SC_irrep: Pojects the charge/spin density of a distored unit cell onto the irreducible representation of the parent space groups (through spglib). Can be used in the case of a GM distortion that does not enlarge the unit cell (project_UC_irrep) or in the case of a commensurate distortion that enlarges the unit cell and thus requires the density in a supercell (project_SC_irrep). See the examples of ferroelectric HfO2 or the chiral phase of TiSe2.



2025 - Luca Buiarelli, Seongjoo Jung, Turan Birol. University of Minnesota, CEMS.
