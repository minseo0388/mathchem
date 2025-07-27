"""
proj_huckelapprox.py

This module provides visualization functions for Hückel molecular orbital theory results,
enabling graphical representation of molecular orbitals and energy levels.

Author: Choi Minseo
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from src.huckelapprox import Huckel
from typing import List, Tuple, Optional, Union, Dict, Any
import warnings

def plot_energy_levels(adj_matrix: List[List[float]], alpha: float = 0.0, beta: float = -1.0,
                      electrons: Optional[int] = None, figwidth: float = 8, figlength: float = 10) -> plt.Figure:
    """
    Create an energy level diagram for molecular orbitals using Hückel approximation.
    
    This function visualizes the energy levels of molecular orbitals calculated
    using the Hückel method, optionally indicating occupied and unoccupied orbitals.
    
    Parameters:
        adj_matrix: Adjacency matrix representing molecular structure
        alpha: Coulomb integral (π orbital on-site energy), default = 0.0
        beta: Resonance integral (adjacent π coupling), default = -1.0
        electrons: Total number of π electrons (if provided, occupied orbitals will be highlighted)
        figwidth: Width of the figure in inches, default = 8
        figlength: Height of the figure in inches, default = 10
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - Horizontal lines represent energy levels
        - Blue lines indicate occupied orbitals (if electrons parameter is provided)
        - Red lines indicate unoccupied orbitals (if electrons parameter is provided)
        - The energy is given in terms of α and β parameters
    """
    # Create Huckel object and solve
    mol = Huckel(adj_matrix, alpha, beta)
    energy, _ = mol.solve()
    
    # Create figure for energy levels
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    ax.set_title("Molecular Orbital Energy Levels (Hückel Approximation)")
    
    # Determine orbital occupancy if electrons are specified
    if electrons is not None:
        if not isinstance(electrons, int) or electrons < 0:
            warnings.warn("Invalid electron count. Showing all orbitals as unoccupied.")
            electrons = 0
            
        homo_index = electrons // 2 - 1  # Highest Occupied Molecular Orbital
    else:
        homo_index = -1  # No occupied orbitals
    
    # Plot each energy level
    y_positions = np.arange(len(energy))
    
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
        ax.text(0.6, e, f"MO {i+1}: {e:.3f}α{'+' if e > 0 else ''}{beta/alpha if alpha != 0 else beta:.3f}β", 
                verticalalignment='center')
    
    # Remove x-axis ticks and set y-axis limits
    ax.set_xticks([])
    ax.set_xlim(-1, 3)
    ax.set_ylabel("Energy (in terms of α and β)")
    
    if electrons is not None:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_electron_density(adj_matrix: List[List[float]], electrons: int, 
                         alpha: float = 0.0, beta: float = -1.0,
                         figwidth: float = 10, figlength: float = 6) -> plt.Figure:
    """
    Visualize the π-electron density distribution across atoms in the molecule.
    
    This function calculates and plots the electron density at each atom
    based on Hückel molecular orbital theory.
    
    Parameters:
        adj_matrix: Adjacency matrix representing molecular structure
        electrons: Total number of π electrons in the system
        alpha: Coulomb integral (π orbital on-site energy), default = 0.0
        beta: Resonance integral (adjacent π coupling), default = -1.0
        figwidth: Width of the figure in inches, default = 10
        figlength: Height of the figure in inches, default = 6
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - The x-axis represents atom indices
        - The y-axis represents electron density
        - For neutral alternant hydrocarbons, all atoms typically have density close to 1.0
        - Deviations indicate charge accumulation or depletion
    """
    # Create Huckel object
    mol = Huckel(adj_matrix, alpha, beta)
    
    # Calculate electron densities
    mol.solve()
    densities = mol.electron_density(electrons)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    ax.set_title(f"π-Electron Density Distribution ({electrons} electrons)")
    
    # Plot densities
    atoms = np.arange(1, len(densities) + 1)
    ax.bar(atoms, densities, width=0.6, color='skyblue', edgecolor='navy')
    
    # Add reference line at density=1.0
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Neutral reference')
    
    ax.set_xlabel("Atom Index")
    ax.set_ylabel("π-Electron Density")
    ax.set_xticks(atoms)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig
    