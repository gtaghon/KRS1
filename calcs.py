import math
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral

def get_phi_psi(residue, pdb_file, chain="A"):
    """
    Calculates phi-psi angles for a given residue in a PDB file.

    Args:
        residue: Residue index to fetch the phi and psi angles for.
        pdb_file (str): PDB file path to calculate phi and psi angles.
        chain (str, optional): The chain to process. Defaults to "A" for models.

    Returns:
        tuple: A tuple containing the phi and psi angles in degrees.
    """
    if pdb_file is None:
        print(f"Warning: PDB file '{pdb_file}' not found.")
        return 0.0, 0.0
    
    if residue == '-':
        print(f"No residue at position, assigning 0,0")
        return 0.0, 0.0

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    for model in structure:
        for chain_obj in model:
            if chain_obj.id != chain:
                continue

            residues = list(chain_obj.get_residues())
            for i in range(1, len(residues) - 1):
                if residues[i].id[1] != residue:
                    continue

                prev_res = residues[i - 1]
                curr_res = residues[i]
                next_res = residues[i + 1]

                try:
                    prev_c = prev_res["C"].get_vector()
                    curr_n = curr_res["N"].get_vector()
                    curr_ca = curr_res["CA"].get_vector()
                    curr_c = curr_res["C"].get_vector()
                    next_n = next_res["N"].get_vector()

                    phi = calc_dihedral(prev_c, curr_n, curr_ca, curr_c)
                    psi = calc_dihedral(curr_n, curr_ca, curr_c, next_n)

                    #print(f"Residue {chain}:{residue} - Phi: {math.degrees(phi)}, Psi: {math.degrees(psi)}")
                    return math.degrees(phi), math.degrees(psi)

                except KeyError:
                    # Skip residues with missing atoms
                    continue

    #print(f"Residue {chain}:{residue} not found or angles not calculated.")
    return 0.0, 0.0