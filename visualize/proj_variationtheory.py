# proj_variationtheory.py
# This file contains the implementation of the Variation Theory method 
# for visualizing the results of molecular orbital calculations.
# Show the graphical representation of the molecular orbitals and their energies 
# using the Variation Theory method

import matplotlib.pyplot as plt
from src.variationtheory import VariationTheory

def plot_variation_theory(H, S):
    """
    Plot the molecular orbitals and their energies using the Variation Theory method.
    
    Parameters:
        H: Hamiltonian matrix (n by n).
        S: Overlap matrix (n by n). If not provided, identity matrix is used.
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
