import re
import numpy as np
import matplotlib.pyplot as plt
from kolzur_filter import *
from calcs import *
from utils import *
from tqdm import tqdm

def fft_on_alignment(alignment):
    fft_results = {}

    for gene in tqdm(alignment, desc="Creating FFT table"):
        mass_series = []
        charge_series = []
        hydrophobicity_series = []
        phi_series = []
        psi_series = []
        alignment_positions = []

        pdb_file = get_AF_model(gene)

        for gpcrdb_numbering in sorted(alignment[gene].keys()):
            data = alignment[gene][gpcrdb_numbering]
            if data['residue'] != '-':
                residue = int(re.sub('[A-Z]', '', data['residue']))  # Extract the residue number
                phi, psi = get_phi_psi(residue, pdb_file)
            else:
                phi, psi = 0.0, 0.0
            mass_series.append(data['mass'])
            charge_series.append(data['charge'])
            hydrophobicity_series.append(data['hydrophobicity'])
            phi_series.append(phi)
            psi_series.append(psi)
            alignment_positions.append(gpcrdb_numbering)

        mass_series = np.array(mass_series)
        charge_series = np.array(charge_series)
        hydrophobicity_series = np.array(hydrophobicity_series)
        phi_series = np.array(phi_series)
        psi_series = np.array(psi_series)

        mass_fft = np.fft.fft(mass_series)
        charge_fft = np.fft.fft(charge_series)
        hydrophobicity_fft = np.fft.fft(hydrophobicity_series)
        phi_fft = np.fft.fft(phi_series)
        psi_fft = np.fft.fft(psi_series)

        fft_results[gene] = {
            'mass': mass_fft,
            'charge': charge_fft,
            'hydrophobicity': hydrophobicity_fft,
            'phi': phi_fft,
            'psi': psi_fft,
            'alignment_positions': alignment_positions
        }

    return fft_results

def parse_alignment_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    headers = lines[0].strip().split('\t')[2:]  # Extract header row and remove first two columns
    alignment = {}

    for line in lines[1:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        fields = line.split('\t')
        gpcrdb_numbering = fields[0]

        for i, residue in enumerate(fields[2:]):  # Start from the third column
            gene = headers[i].replace("..", "") #split(' ')[0]  # Remove the "Human" text from the gpcr name
            
            rescode = residue.strip('0123456789') # single letter code

            if gene not in alignment:
                alignment[gene] = {}

            alignment[gene][gpcrdb_numbering] = {
                'residue': residue,
                'mass': get_mass(rescode),
                'charge': get_charge(rescode),
                'hydrophobicity': get_hydrophobicity(rescode),
                'secondary_structure': get_secondary_structure(rescode),
            }

    return alignment

def get_mass(residue):
    mass_dict = {
        'A': 71.08, 'C': 103.14, 'D': 115.09, 'E': 129.12, 'F': 147.18,
        'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16,
        'M': 131.20, 'N': 114.10, 'P': 97.12, 'Q': 128.13, 'R': 156.19,
        'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.21, 'Y': 163.18
    }
    return mass_dict.get(residue.upper(), 0.0)

def get_charge(residue, pH=7.4):
    """
    Calculate the charge of an amino acid at a given pH.
    
    Args:
        residue (str): Single letter amino acid code
        pH (float): pH value (default: 7.4 for physiological pH)
        
    Returns:
        float: Estimated charge of the amino acid at the given pH
    """
    # pKa values for ionizable groups
    pKa_values = {
        # Side chains
        'D': 3.9,    # Aspartic acid
        'E': 4.3,    # Glutamic acid
        'H': 6.0,    # Histidine
        'C': 8.3,    # Cysteine
        'Y': 10.1,   # Tyrosine
        'K': 10.5,   # Lysine
        'R': 12.5,   # Arginine
        
        # N-terminus and C-terminus pKa values not used in this simplified model
        # as we're looking at internal residues in a protein
    }
    
    # Default charge for non-ionizable residues
    if residue.upper() not in pKa_values:
        return 0.0
    
    # Calculate charge using Henderson-Hasselbalch equation
    pKa = pKa_values[residue.upper()]
    
    # For acidic residues (D, E, C, Y)
    if residue.upper() in ['D', 'E', 'C', 'Y']:
        charge = -1.0 / (1.0 + 10 ** (pKa - pH))
    # For basic residues (H, K, R)
    else:
        charge = 1.0 / (1.0 + 10 ** (pH - pKa))
        
    return charge

def get_hydrophobicity(residue):
    hydrophobicity_dict = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    return hydrophobicity_dict.get(residue.upper(), 0.0)

def get_secondary_structure(residue):
    secondary_structure_dict = {
        'A': 'H', 'C': 'C', 'D': 'C', 'E': 'C', 'F': 'H',
        'G': 'C', 'H': 'C', 'I': 'H', 'K': 'H', 'L': 'H',
        'M': 'H', 'N': 'C', 'P': 'T', 'Q': 'C', 'R': 'C',
        'S': 'C', 'T': 'C', 'V': 'H', 'W': 'H', 'Y': 'H'
    }
    return secondary_structure_dict.get(residue.upper(), 'C')

def plot_gene_heatmaps(alignment, gene):
    if gene not in alignment:
        print(f"Gene '{gene}' not found in the alignment.")
        return

    gene_data = alignment[gene]
    num_positions = len(gene_data)

    properties = ['mass', 'charge', 'hydrophobicity', 'secondary_structure']
    num_properties = len(properties)

    data_matrix = np.zeros((num_positions, num_properties))

    for i, position in enumerate(sorted(gene_data.keys())):
        for j, prop in enumerate(properties):
            if prop == 'secondary_structure':
                data_matrix[i, j] = ord(gene_data[position][prop])
            else:
                data_matrix[i, j] = gene_data[position][prop]

    fig, axs = plt.subplots(1, num_properties, figsize=(4 * num_properties, 6))

    for j, prop in enumerate(properties):
        if prop == 'charge':
            # Use a diverging colormap for charge (blue-white-red)
            im = axs[j].imshow(data_matrix[:, j].reshape(-1, 1), cmap='coolwarm', aspect='auto')
        else:
            im = axs[j].imshow(data_matrix[:, j].reshape(-1, 1), cmap='viridis', aspect='auto')
        axs[j].set_title(prop.capitalize())
        axs[j].set_xticks([])
        axs[j].set_yticks(range(num_positions))
        axs[j].set_yticklabels(sorted(gene_data.keys()))

    plt.tight_layout()
    plt.show()

def plot_gene_fft(fft_results, gene, window_size=5, num_iterations=3):
    if gene not in fft_results:
        print(f"Gene '{gene}' not found in the FFT results.")
        return

    gene_fft = fft_results[gene]
    properties = ['mass', 'charge', 'hydrophobicity', 'phi', 'psi']
    num_properties = len(properties)

    alignment_positions = gene_fft['alignment_positions']

    fig, axs = plt.subplots(2, num_properties, figsize=(4 * num_properties, 8))

    for j, prop in enumerate(properties):
        fft_data = gene_fft[prop]
        freq = np.fft.fftfreq(len(fft_data))

        filtered_fft_data = kza(np.abs(fft_data), window_size, num_iterations)

        freq_filtered = np.linspace(freq[0], freq[-1], len(filtered_fft_data))

        axs[0, j].plot(freq, np.abs(fft_data), label='Original FFT')
        axs[0, j].set_title(f"{prop.capitalize()} FFT - Original")
        axs[0, j].set_xlabel("Frequency")
        axs[0, j].set_ylabel("Magnitude")
        axs[0, j].legend()

        axs[1, j].plot(freq_filtered, filtered_fft_data, label='KZ Filtered FFT')
        axs[1, j].set_title(f"{prop.capitalize()} FFT - KZ Filtered")
        axs[1, j].set_xlabel("Frequency")
        axs[1, j].set_ylabel("Magnitude")
        axs[1, j].legend()

        for pos in alignment_positions:
            if pos.endswith('x'):
                segment = pos[:-1]
                if segment.isdigit():
                    label = f"Segment {segment}"
                else:
                    label = f"Segment {segment.upper()}"
                idx = alignment_positions.index(pos)
                axs[0, j].annotate(label, (freq[idx], np.abs(fft_data[idx])), textcoords="offset points", xytext=(0, 10), ha='center')
                axs[1, j].annotate(label, (freq_filtered[idx], filtered_fft_data[idx]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout()
    plt.show()

def plot_multiple_gene_fft(fft_results, genes, window_size=5, num_iterations=3, property_list=None):
    """
    Plot FFT signatures for multiple genes on the same plot for easier comparison.
    
    Args:
        fft_results (dict): Dictionary containing FFT results for each gene
        genes (list): List of gene names to compare
        window_size (int): Window size for KZ filter
        num_iterations (int): Number of iterations for KZ filter
        property_list (list, optional): List of properties to plot. If None, all properties are plotted.
    """
    if not property_list:
        property_list = ['mass', 'charge', 'hydrophobicity', 'phi', 'psi']
    
    num_properties = len(property_list)
    
    # Create a figure with 2 rows (original and filtered) and columns for each property
    fig, axs = plt.subplots(2, num_properties, figsize=(5 * num_properties, 10))
    
    # Set up a colormap for different genes
    colors = plt.cm.tab10.colors
    
    # Track which genes are actually plotted (for legend)
    plotted_genes = []
    
    for i, gene in enumerate(genes):
        if gene not in fft_results:
            print(f"Gene '{gene}' not found in the FFT results. Skipping.")
            continue
        
        plotted_genes.append(gene)
        gene_fft = fft_results[gene]
        color = colors[i % len(colors)]
        
        for j, prop in enumerate(property_list):
            fft_data = gene_fft[prop]
            freq = np.fft.fftfreq(len(fft_data))
            
            # Apply KZ filter to smooth the FFT data
            filtered_fft_data = kza(np.abs(fft_data), window_size, num_iterations)
            freq_filtered = np.linspace(freq[0], freq[-1], len(filtered_fft_data))
            
            # Plot original FFT
            axs[0, j].plot(freq, np.abs(fft_data), color=color, alpha=0.7, label=gene)
            
            # Plot filtered FFT
            axs[1, j].plot(freq_filtered, filtered_fft_data, color=color, alpha=0.7, label=gene)
    
    # Add titles and labels
    for j, prop in enumerate(property_list):
        axs[0, j].set_title(f"{prop.capitalize()} FFT - Original")
        axs[0, j].set_xlabel("Frequency")
        axs[0, j].set_ylabel("Magnitude")
        
        axs[1, j].set_title(f"{prop.capitalize()} FFT - KZ Filtered")
        axs[1, j].set_xlabel("Frequency")
        axs[1, j].set_ylabel("Magnitude")
        
        # Only add legend to the first column to avoid redundancy
        if j == 0:
            axs[0, j].legend(loc='best')
            axs[1, j].legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def interactive_comparison_generator(alignment):
    """
    Interactive interface for generating comparison plots between multiple genes.
    
    Args:
        alignment (dict): Dictionary containing alignment data
    """
    fft_results = fft_on_alignment(alignment)
    
    print("\nAvailable genes in the alignment:")
    for i, gene in enumerate(sorted(alignment.keys())):
        if i > 0 and i % 5 == 0:
            print("")  # Add newline every 5 genes for readability
        print(f"{gene}", end=", " if (i+1) % 5 != 0 and i != len(alignment.keys())-1 else "\n")
    
    while True:
        genes_input = input("\nEnter gene names to compare, separated by commas (or 'q' to quit): ")
        if genes_input.lower() == 'q':
            break
        
        genes = [gene.strip() for gene in genes_input.split(',')]
        valid_genes = [gene for gene in genes if gene in alignment]
        
        if not valid_genes:
            print("None of the entered genes were found in the alignment.")
            continue
        
        if len(valid_genes) != len(genes):
            missing = set(genes) - set(valid_genes)
            print(f"Warning: The following genes were not found and will be skipped: {', '.join(missing)}")
        
        # Get window size and iterations from user
        try:
            window_size = int(input("Enter the window size for KZ filter (default 5): ") or "5")
            num_iterations = int(input("Enter the number of iterations for KZ filter (default 3): ") or "3")
        except ValueError:
            print("Invalid input. Using default values: window_size=5, num_iterations=3")
            window_size = 5
            num_iterations = 3
        
        # Get properties to plot
        print("\nAvailable properties: mass, charge, hydrophobicity, phi, psi")
        prop_input = input("Enter properties to plot, separated by commas (leave blank for all): ")
        
        if prop_input.strip():
            properties = [p.strip() for p in prop_input.split(',')]
            # Validate properties
            valid_props = ['mass', 'charge', 'hydrophobicity', 'phi', 'psi']
            properties = [p for p in properties if p in valid_props]
            if not properties:
                print("No valid properties entered. Using all properties.")
                properties = None
        else:
            properties = None
        
        # Plot the comparison
        plot_multiple_gene_fft(fft_results, valid_genes, window_size, num_iterations, properties)

def interactive_plot_generator(alignment):
    fft_results = fft_on_alignment(alignment)

    while True:
        print("\nMenu Options:")
        print("1. Plot single gene")
        print("2. Compare multiple genes")
        print("3. Quit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '3':
            break
        
        if choice == '1':
            gene = input("Enter the gene name to plot: ")
            if gene not in alignment:
                print(f"Gene '{gene}' not found in the alignment.")
                continue

            print("Select plot type:")
            print("1. Heatmap")
            print("2. FFT")

            plot_type = input("Enter the plot type (1 or 2): ")

            if plot_type == '1':
                plot_gene_heatmaps(alignment, gene)
            elif plot_type == '2':
                window_size = int(input("Enter the window size for KZ filter: "))
                num_iterations = int(input("Enter the number of iterations for KZ filter: "))
                plot_gene_fft(fft_results, gene, window_size, num_iterations)
            else:
                print("Invalid plot type. Please try again.")
        
        elif choice == '2':
            interactive_comparison_generator(alignment)
        
        else:
            print("Invalid choice. Please try again.")
            
# Example usage
if __name__ == "__main__":
    alignment_file = 'input_alignments/aminergic_alignment.txt'
    alignment = parse_alignment_data(alignment_file)

    interactive_plot_generator(alignment)