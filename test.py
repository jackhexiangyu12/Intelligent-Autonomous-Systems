"""Calculate the molecular mass of a given formula"""
import sys

periodic = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "S": 32.07
}

def get_num_str(in_str):
    """Return the full number that starts the string"""
    out_str = ""
    for char in in_str:
        if char.isnumeric():
            out_str += char
        else:
            break
    return int(out_str)

def mol_mass(in_str):
    """
    Return the molecular mass from a formula

    Parameters
    ----------
    in_str : str
        The molecular formula respecting case conventions e.g. H3OCOH
    Returns
    -------
    total_mass : float
        Molecular mass of the molecule

    """

    # running total
    total_mass = 0
    for i, char in enumerate(in_str):
        # if it's the start of an element
        if char.isupper():
            # if it's the last character in the string
            if i == len(in_str) - 1:
                total_mass += periodic[char]
                break
            else:
                # if it's a 1 char element with no number
                if in_str[i + 1].isupper():
                    total_mass += periodic[char]
                # if it's a 2 char element
                if in_str[i + 1].islower():
                    # if it's the last bit of the string
                    if i + 1 == len(in_str) - 1:  # +1 for 2 char, -1 for not inclusive
                        total_mass += periodic[in_str[i : i + 2]]
                    else:
                        # if the next bit is a number
                        if in_str[i + 2].isnumeric():
                            total_mass += (
                                get_num_str(in_str[i + 2 :])
                                * periodic[in_str[i : i + 2]]
                            )
                        # if the next bit is a letter
                        else:
                            total_mass += periodic[in_str[i : i + 2]]

                # if it's a 1 char element with a number
                if in_str[i + 1].isnumeric():
                    total_mass += get_num_str(in_str[i + 1 :]) * periodic[char]
        else:
            if i == len(in_str) - 1:
                break
    return total_mass

# test these

print(mol_mass(sys.argv[1]))
