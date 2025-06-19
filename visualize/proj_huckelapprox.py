# projection_huckel.py
# This file contains the implementation of the projection method 
# for visualizing the results of the Huckel approximation.
# Show the graphical representation of the molecular orbitals 
# and their energies using the Huckel approximation method.

import matplotlib.pyplot as plt
from src.huckelapprox import Huckel

def plot_molecular_orbitals(adj_matrix, alpha, beta, figwidth, figlength):
    """
    Plot the molecular orbitals and their energies using the Huckel approximation.
    
    Parameters:
        adj_matrix: Adjacency matrix of the molecule (list of lists).
        alpha: Coulomb integral (π orbital on-site energy, commonly 0) (number).
        beta: Resonance integral (adjacent π coupling, commonly -1) (number).
        figwidth: Width of the figure (float).
        figlength: Length of the figure (float).
    """
    mol = Huckel(adj_matrix, alpha, beta)
    energy, mo_coeff = mol.solve()
    if energy is None or mo_coeff is None:
        raise ValueError("Please RECHECK the Huckel approximation setup.")

    plt.figure(figsize=(figwidth, figlength))
    plt.title("Molecular Orbitals and Energies")

    for i in range(len(energy)):
        plt.plot(mo_coeff[:, i], label=f"MO {i+1} (E={energy[i]:.2f})")

    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.xlabel("Atomic Orbital Index")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid()
    plt.show()
