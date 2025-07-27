"""
proj_particle.py

This module provides visualization functions for quantum mechanical particle models,
including 1D and 2D particle in a box systems, probability densities, and energy spectra.

Author: Choi Minseo
Date: 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, Tuple, List, Optional, Any
import matplotlib.cm as cm
from src.particle import ParticleInABox1D, ParticleInABox2D

def plot_wavefunction_1d(particle: ParticleInABox1D, n: int, x: np.ndarray, 
                       show_probability: bool = False, figwidth: float = 10, figlength: float = 6) -> plt.Figure:
    """
    Plot the wavefunction of a particle in a 1D infinite potential well.
    
    This function visualizes the quantum mechanical wavefunction and optionally
    the probability density for a given quantum state.
    
    Parameters:
        particle: ParticleInABox1D instance representing the quantum system
        n: Quantum number of the state to plot (positive integer)
        x: Array of position values to evaluate the wavefunction at
        show_probability: Whether to show the probability density |ψ|² (default = False)
        figwidth: Width of the figure in inches (default = 10)
        figlength: Height of the figure in inches (default = 6)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - The wavefunction ψ(x) is plotted in blue
        - If show_probability is True, the probability density |ψ(x)|² is plotted in red
        - The x-axis represents position within the box
        - For positions outside the box [0,L], the wavefunction is zero
    """
    # Calculate wavefunction
    psi = particle.wavefunction(n, x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    
    # Plot wavefunction
    ax.plot(x, psi, 'b-', label=f"ψ(x), n={n}")
    
    # Optionally plot probability density
    if show_probability:
        prob = psi**2
        ax.plot(x, prob, 'r-', label=f"|ψ(x)|², n={n}")
        ax.set_title(f"Wavefunction and Probability Density for n={n}")
    else:
        ax.set_title(f"Wavefunction for n={n}")
    
    # Add box boundaries as vertical lines
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(particle.L, color='black', linestyle='-', alpha=0.3)
    
    # Add zero reference line
    ax.axhline(0, color='black', lw=0.5, ls='--')
    
    # Labels and grid
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    # Add energy information
    energy = particle.energy(n)
    ax.text(0.02, 0.95, f"Energy: {energy:.4f}", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_energy_spectrum_1d(particle: ParticleInABox1D, n_max: int = 10, 
                          figwidth: float = 8, figlength: float = 10) -> plt.Figure:
    """
    Plot the energy spectrum for a particle in a 1D infinite potential well.
    
    This function visualizes the energy levels and their spacings for the
    first n_max quantum states.
    
    Parameters:
        particle: ParticleInABox1D instance representing the quantum system
        n_max: Maximum quantum number to include (default = 10)
        figwidth: Width of the figure in inches (default = 8)
        figlength: Height of the figure in inches (default = 10)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - Horizontal lines represent energy levels
        - The y-axis shows energy values
        - Energy increases quadratically with quantum number: E_n ∝ n²
        - Values are in the same units as specified in the particle parameters
    """
    # Calculate energy spectrum
    ns, energies = particle.spectrum(n_max)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    ax.set_title("Energy Spectrum for Particle in a 1D Box")
    
    # Plot energy levels as horizontal lines
    for n, E in zip(ns, energies):
        ax.plot([-0.5, 0.5], [E, E], 'b-', linewidth=2)
        ax.text(0.6, E, f"n={n}: E={E:.4f}", verticalalignment='center')
    
    # Set axis properties
    ax.set_ylabel("Energy")
    ax.set_xticks([])  # No ticks on x-axis
    ax.set_xlim(-1, 3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add information about box parameters
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    textstr = f"Box length: {particle.L}\nEffective mass: {particle.m}\nℏ: {particle.hbar}"
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig


def plot_probability_density_1d(particle: ParticleInABox1D, n: int, x: np.ndarray,
                               figwidth: float = 10, figlength: float = 6) -> plt.Figure:
    """
    Plot the probability density for a particle in a 1D infinite potential well.
    
    This function visualizes the probability density |ψ(x)|² for a given quantum state.
    
    Parameters:
        particle: ParticleInABox1D instance representing the quantum system
        n: Quantum number of the state to plot (positive integer)
        x: Array of position values to evaluate the probability density at
        figwidth: Width of the figure in inches (default = 10)
        figlength: Height of the figure in inches (default = 6)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - The probability density |ψ(x)|² represents the probability of finding
          the particle at position x
        - The integral of the probability density over the box equals 1
        - The number of nodes (zeros) in the probability density is n-1
    """
    # Calculate wavefunction and probability density
    psi = particle.wavefunction(n, x)
    prob_density = psi**2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    
    # Plot probability density
    ax.plot(x, prob_density, 'r-', linewidth=2)
    ax.fill_between(x, prob_density, alpha=0.3, color='red')
    
    # Add box boundaries as vertical lines
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(particle.L, color='black', linestyle='-', alpha=0.3)
    
    # Labels and grid
    ax.set_title(f"Probability Density for n={n}")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Probability Density |ψ(x)|²")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add information about the quantum state
    energy = particle.energy(n)
    nodes = n - 1
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    textstr = f"Quantum number: {n}\nEnergy: {energy:.4f}\nNodes: {nodes}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig


def plot_multiple_wavefunctions_1d(particle: ParticleInABox1D, n_values: List[int], x: np.ndarray,
                                  figwidth: float = 12, figlength: float = 8) -> plt.Figure:
    """
    Plot multiple wavefunctions for a particle in a 1D infinite potential well.
    
    This function visualizes several quantum states on the same plot for comparison.
    
    Parameters:
        particle: ParticleInABox1D instance representing the quantum system
        n_values: List of quantum numbers for states to plot
        x: Array of position values to evaluate the wavefunctions at
        figwidth: Width of the figure in inches (default = 12)
        figlength: Height of the figure in inches (default = 8)
        
    Returns:
        Matplotlib figure object
        
    Notes:
        - Each wavefunction is plotted with a different color
        - The legend shows the quantum number and energy for each state
        - Comparing multiple states helps visualize how the wavefunction 
          changes with increasing quantum number
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth, figlength))
    
    # Colors for different states
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))
    
    # Plot each wavefunction
    for i, n in enumerate(n_values):
        psi = particle.wavefunction(n, x)
        energy = particle.energy(n)
        ax.plot(x, psi, color=colors[i], linewidth=2, 
                label=f"n={n}, E={energy:.4f}")
    
    # Add box boundaries as vertical lines
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(particle.L, color='black', linestyle='-', alpha=0.3)
    
    # Add zero reference line
    ax.axhline(0, color='black', lw=0.5, ls='--')
    
    # Labels and grid
    ax.set_title("Multiple Wavefunctions for Particle in a 1D Box")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Wavefunction ψ(x)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
    