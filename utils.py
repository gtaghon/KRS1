import os
import requests

def extract_uniprot_id(search_text):

    # Construct the API URL
    api_url = f"https://rest.uniprot.org/uniprotkb/search?query={search_text}&format=json&size=1"

    # Send a GET request to the API URL
    response = requests.get(api_url)

    # Parse the JSON response
    data = response.json()

    # Check if the response contains search results
    if "results" in data and len(data["results"]) > 0:
        # Extract the first search result
        result = data["results"][0]

        # Extract the Uniprot ID and entry name
        uniprot_id = result["primaryAccession"]
        entry_name = result["uniProtkbId"] 
        #protein_name = result["proteinDescription"]["recommendedName"]["fullName"]["value"]

        return uniprot_id, entry_name #, protein_name
    else:
        return None, None #, None


def download_alphafold_pdb(uniprot_id, output_dir="pdbs"):
    """Downloads the AlphaFold PDB file for a given UniProt ID.

    Args:
        uniprot_id (str): The UniProt ID of the protein.
        output_dir (str, optional): Directory to save the PDB file. Defaults to "pdbs".
    """
    existing = f"{output_dir}/{uniprot_id}.pdb"
    if os.path.isfile(existing):
        #print("File already exists")
        return existing
    
    base_url = "https://alphafold.ebi.ac.uk/files/AF-"

    pdb_url = f"{base_url}{uniprot_id}-F1-model_v4.pdb"  # Assuming full-length model

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(pdb_url)
    if response.status_code == 200:
        pdb_file = os.path.join(output_dir, f"{uniprot_id}.pdb")
        with open(pdb_file, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PDB file for {uniprot_id} to {pdb_file}")
        return pdb_file
    else:
        print(f"PDB file not found for {uniprot_id}")

def get_AF_model(query):
    """Downloads the top AlphaFold PDB file for a given text search query.

    Args:
        uniprot_id (str): A text search query.
    """
    uniprot_id, _ = extract_uniprot_id(query)
    if uniprot_id:
        pdb_file = download_alphafold_pdb(uniprot_id)
        #print("Model acquired")
        return pdb_file
    else:
        print("Uniprot ID error")
