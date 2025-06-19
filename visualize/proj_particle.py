# proj_particle.py
# This file contains the implementation of the ParticleInABox1D and ParticleInABox2D classes
# for 1D and 2D quantum mechanical particle models.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_wavefunction_1d(particle, n, x):
    """
    Plot the wavefunction of a 1D particle in a box.
    
    Parameters:
        particle: ParticleInABox1D instance.
        n: Quantum number (integer).
        x: Array of positions (numpy array).
    """    
    psi = particle.wavefunction(n, x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, psi, label=f"n={n}")
    plt.title("Wavefunction of Particle in a 1D Box")
    plt.xlabel("Position (x)")
    plt.ylabel("Wavefunction ψ(x)")
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

def plot_wavefunction_2d(particle, nx, ny, x, y):
    """
    Plot the wavefunction of a 2D particle in a box.
    
    Parameters:
        particle: ParticleInABox2D instance.
        nx: Quantum number in x direction (integer).
        ny: Quantum number in y direction (integer).
        x: Array of x positions (numpy array).
        y: Array of y positions (numpy array).
    """
    
    X, Y = np.meshgrid(x, y)
    psi = particle.wavefunction(nx, ny, X, Y)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, psi, cmap='viridis')
    
    ax.set_title(f"Wavefunction of Particle in a 2D Box (nx={nx}, ny={ny})")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Wavefunction ψ(x,y)")
    
    plt.show()