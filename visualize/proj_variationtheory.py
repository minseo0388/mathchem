"""
proj_variationtheory.py

This module provides visualization functions for variation theory results,
enabling graphical representation of molecular orbitals and energy levels
calculated using the variation method.

Author: Choi Minseo
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from cnumathchem.variationtheory import VariationTheory
from typing import Union, Tuple, List, Optional, Dict, Any

def plot_energy_levels(H: np.ndarray, S: Optional[np.ndarray] = None,
                      electrons: Optional[int] = None,
                      figwidth: float = 8, figlength: float = 10) -> plt.Figure:
    """
    Create an energy level diagram for molecular orbitals using Variation Theory.
    
    This function visualizes the energy levels of molecular orbitals calculated
    using the variation method, optionally indicating occupied and unoccupied orbitals.
    
    Parameters:
        H: Hamiltonian matrix (n x n)
        S: Overlap matrix (n x n), defaults to identity if not provided
        electrons: Total number of electrons (if provided, occupied orbitals will be highlighted)
        figwidth: Width of the figure in inches (default = 8)
        figlength: Height of the figure in inches (default = 10)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - Horizontal lines represent energy levels
        - Blue lines indicate occupied orbitals (if electrons parameter is provided)
        - Red lines indicate unoccupied orbitals (if electrons parameter is provided)
        - The energy levels are solutions to the generalized eigenvalue problem H c = E S c
    """
    # Create VariationTheory object and solve
    mol = VariationTheory(H, S)
    energy, _ = mol.solve()
    
    # Determine HOMO index if electrons are specified
    homo_index = -1
    if electrons is not None:
        if not isinstance(electrons, int) or electrons <= 0:
            import warnings
            warnings.warn("Invalid electron count. Treating all orbitals as unoccupied.")
        else:
            homo_index = electrons // 2 - 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    ax.set_title("Molecular Orbital Energy Levels (Variation Theory)")
    
    # Plot each energy level
    for i, e in enumerate(energy):
        if i <= homo_index:
            # Occupied orbital
            color = 'blue'
            label = "Occupied" if i == 0 else None
        else:
            # Unoccupied orbital
            color = 'red'
            label = "Unoccupied" if i == homo_index + 1 else None
            
        ax.plot([-0.5, 0.5], [e, e], color=color, linewidth=2, label=label)
        ax.text(0.6, e, f"MO {i+1}: {e:.4f}", verticalalignment='center')
    
    # Set axis properties
    ax.set_xticks([])  # No ticks on x-axis
    ax.set_xlim(-1, 3)
    ax.set_ylabel("Energy")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    if electrons is not None:
        ax.legend(loc='upper right')
    
    # Add information about matrix dimensions
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    n = H.shape[0]
    textstr = f"Hamiltonian: {n}×{n} matrix\nBasis functions: {n}"
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig


def plot_convergence_study(H_series: List[np.ndarray], S_series: Optional[List[np.ndarray]] = None,
                          basis_labels: Optional[List[str]] = None,
                          figwidth: float = 10, figlength: float = 6) -> plt.Figure:
    """
    Plot energy convergence as a function of basis set size using Variation Theory.
    
    This function visualizes how energy eigenvalues converge as the basis set size
    increases, which is a fundamental aspect of the variation method.
    
    Parameters:
        H_series: List of Hamiltonian matrices of increasing size
        S_series: List of corresponding overlap matrices (if None, identity matrices are used)
        basis_labels: Labels for each basis set (if None, numeric indices are used)
        figwidth: Width of the figure in inches (default = 10)
        figlength: Height of the figure in inches (default = 6)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - The x-axis represents different basis sets
        - The y-axis represents energy values
        - Multiple lines show different energy eigenvalues
        - The variation theorem guarantees that energy eigenvalues decrease
          (or remain constant) as the basis set size increases
    """
    # Input validation
    if not H_series:
        raise ValueError("H_series cannot be empty")
        
    n_basis_sets = len(H_series)
    
    # Initialize S_series if not provided
    if S_series is None:
        S_series = [None] * n_basis_sets
    elif len(S_series) != n_basis_sets:
        raise ValueError("H_series and S_series must have the same length")
    
    # Initialize basis labels if not provided
    if basis_labels is None:
        basis_labels = [f"Basis {i+1}" for i in range(n_basis_sets)]
    elif len(basis_labels) != n_basis_sets:
        raise ValueError("basis_labels must have the same length as H_series")
    
    # Calculate energies for each basis set
    all_energies = []
    min_orbitals = float('inf')
    
    for i in range(n_basis_sets):
        mol = VariationTheory(H_series[i], S_series[i])
        energy, _ = mol.solve()
        all_energies.append(energy)
        min_orbitals = min(min_orbitals, len(energy))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    ax.set_title("Energy Convergence with Increasing Basis Set Size")
    
    # Plot energy convergence for each orbital
    x = np.arange(n_basis_sets)
    for orbital in range(min_orbitals):
        orbital_energies = [all_energies[i][orbital] for i in range(n_basis_sets)]
        ax.plot(x, orbital_energies, 'o-', label=f"MO {orbital+1}")
    
    # Set axis properties
    ax.set_xlabel("Basis Set")
    ax.set_ylabel("Energy")
    ax.set_xticks(x)
    ax.set_xticklabels(basis_labels)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
    