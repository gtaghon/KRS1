# Kolmogorov Receptor Signatures 1.1 (KRS1)

KRS1 uses Kolmogorov-Zurbenko adaptive filtered [1] Fourier transform on structural alignments aiming to uncover distinct signatures. Properties are calculated from either textbook values (mass, charge) or from structural models downloaded from the AlphaFold server (phi/psi angles). The software is currently in a highly experimental alpha phase. Example properties plotted in the frequency domain across an alignment include:
- amino acid mass
- amino acid charge (approximation at pH 7.4)
- average hydrophobicity
- phi and psi

Empty positions in an alignment (-) are currently represented as 0.0.

## To test the software, create a virtual environment. Then run
```
pip install -r requirements.txt

python main.py
```

*Requires an internet connection for initial AlphaFold model download and phi psi angle calculation.

## Interaction
An interactive text UI will start to guide plot generation. Closing a plot will allow for generating the next plot. To change alignment files, edit the last section of `main.py`.

### Reference
1. https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.71
