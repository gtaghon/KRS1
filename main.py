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
        volume_series = []
        hydrophobicity_series = []
        phi_series = []
        psi_series = []
        alignment_positions = []

        pdb_file = get_AF_model(gene)

        for gpcrdb_numbering in sorted(alignment[gene].keys()):
            data = alignment[gene][gpcrdb_numbering]
            if data['residue'] is not '-':
                residue = int(re.sub('[A-Z]', '', data['residue']))  # Extract the residue number
                phi, psi = get_phi_psi(residue, pdb_file)
            else:
                phi, psi = 0.0, 0.0
            mass_series.append(data['mass'])
            volume_series.append(data['volume'])
            hydrophobicity_series.append(data['hydrophobicity'])
            phi_series.append(phi)
            psi_series.append(psi)
            alignment_positions.append(gpcrdb_numbering)

        mass_series = np.array(mass_series)
        volume_series = np.array(volume_series)
        hydrophobicity_series = np.array(hydrophobicity_series)
        phi_series = np.array(phi_series)
        psi_series = np.array(psi_series)

        mass_fft = np.fft.fft(mass_series)
        volume_fft = np.fft.fft(volume_series)
        hydrophobicity_fft = np.fft.fft(hydrophobicity_series)
        phi_fft = np.fft.fft(phi_series)
        psi_fft = np.fft.fft(psi_series)

        fft_results[gene] = {
            'mass': mass_fft,
            'volume': volume_fft,
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
                'volume': get_volume(rescode),
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

def get_volume(residue):
    volume_dict = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
    }
    return volume_dict.get(residue.upper(), 0.0)

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

    properties = ['mass', 'volume', 'hydrophobicity', 'secondary_structure']
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
    properties = ['mass', 'volume', 'hydrophobicity', 'phi', 'psi']
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

def interactive_plot_generator(alignment):
    fft_results = fft_on_alignment(alignment)

    while True:
        gene = input("Enter the gene name to plot (or 'q' to quit): ")
        if gene == 'q':
            break

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
            
# Example usage
alignment_file = 'input_alignments/aminergic_alignment.txt'
alignment = parse_alignment_data(alignment_file)

interactive_plot_generator(alignment)