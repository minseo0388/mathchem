# proj_huckelapprox.py
# Visualization for Huckel approximation results.

import matplotlib.pyplot as plt
from src.huckelapprox import Huckel
from typing import List

def plot_molecular_orbitals(adj_matrix: List[List[float]], alpha: float, beta: float, figwidth: float = 10, figlength: float = 6):
    """
    Plot molecular orbitals and energies using Huckel approximation.

    Parameters:
        adj_matrix: Adjacency matrix (list of lists).
        alpha: Coulomb integral (float).
        beta: Resonance integral (float).
        figwidth: Figure width (float).
        figlength: Figure length (float).
    """
    mol = Huckel(adj_matrix, alpha, beta)
    energy, mo_coeff = mol.solve()
    if energy is None or mo_coeff is None:
        raise ValueError("Huckel approximation setup is invalid.")

    plt.figure(figsize=(figwidth, figlength))
    plt.title("Molecular Orbitals and Energies (Huckel Approximation)")
    for i in range(len(energy)):
        plt.plot(mo_coeff[:, i], label=f"MO {i+1} (E={energy[i]:.2f})")
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.xlabel("Atomic Orbital Index")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid()
    plt.show()
    