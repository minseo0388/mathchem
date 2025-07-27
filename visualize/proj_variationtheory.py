# proj_variationtheory.py
# Visualization for Variation Theory results.

import matplotlib.pyplot as plt
from src.variationtheory import VariationTheory
import numpy as np

def plot_variation_theory(H: np.ndarray, S: np.ndarray):
    """
    Plot molecular orbitals and energies using Variation Theory.

    Parameters:
        H: Hamiltonian matrix (n x n).
        S: Overlap matrix (n x n).
    """
    mol = VariationTheory(H, S)
    energy, mo_coeff = mol.solve()

    plt.figure(figsize=(10, 6))
    plt.title("Molecular Orbitals and Energies (Variation Theory)")
    for i in range(len(energy)):
        plt.plot(mo_coeff[:, i], label=f"MO {i+1} (E={energy[i]:.2f})")
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.xlabel("Atomic Orbital Index")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid()
    plt.show()
    