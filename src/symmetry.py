import numpy as np
import spglib

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