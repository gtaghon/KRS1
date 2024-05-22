# Kolmogorov Receptor Signatures 1.0 (KRS1)

KRS1 uses Kolmogorov-Zurbenko adaptive filtered [1] Fourier transform on structural alignments aiming to uncover distinct signatures. The software is currently in a highly experimental alpha phase. Example properties plotted in the frequency domain across an alignment include:
- amino acid mass
- amino acid volume
- average hydrophobicity
- phi and psi

Empty positions in an alignment (-) are currently represented as 0.0.

## To test the software, create a virtual environment. Then run
```
pip install -r requirements.txt

python main.py
```
*Requires an internet connection for initial Alphafold model download and phi psi angle calculation.

## Interaction
An interactive text UI will start to guide plot generation. To generate a plot, enter a column name/receptor name from the alignment file, ie. 'alignments/aminergic_alignment.txt' in the demo. Closing a plot will allow for generating the next plot.

### Reference
1. https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.71
