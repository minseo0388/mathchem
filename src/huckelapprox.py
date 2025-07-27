"""
huckelapprox.py

This module provides an implementation of the Hückel molecular orbital theory,
which is a semi-empirical quantum mechanical method for predicting properties
of π-electron systems in conjugated hydrocarbons.

Author: Choi Minseo
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Any

class Huckel:
    """
    Hückel Approximation for Molecular Orbital Calculations.
    
    This class implements the Hückel molecular orbital (HMO) theory, which is a simple
    method for calculating energies and wavefunctions of π electrons in conjugated
    hydrocarbon systems. The method uses a simplified Hamiltonian where:
    - Diagonal elements (H_ii) are set to α (Coulomb integral)
    - Off-diagonal elements (H_ij) are set to β (resonance integral) for bonded atoms
      and 0 for non-bonded atoms
    
    Parameters:
        adj_matrix: Adjacency matrix (n x n), where 1 indicates bonded atoms and 0 indicates
                   non-bonded atoms. This defines the molecular structure.
        alpha: Coulomb integral (π orbital on-site energy), typically set to 0 as a reference.
        beta: Resonance integral (adjacent π coupling), typically negative (~ -2.8 eV).
    
    Notes:
        - The Hückel method assumes that σ and π electrons can be treated separately
        - Only π orbitals are considered explicitly
        - The method works best for planar, conjugated hydrocarbon systems
        - α and β are empirical parameters often determined by fitting to experimental data
    """
    def __init__(self, adj_matrix: List[List[float]], alpha: float = 0.0, beta: float = -1.0):
        """
        Initialize the Hückel calculation with molecular structure and parameters.
        
        Parameters:
            adj_matrix: Adjacency matrix where 1 indicates bonded atoms and 0 indicates non-bonded atoms
            alpha: Coulomb integral (π orbital on-site energy), default = 0.0
            beta: Resonance integral (adjacent π coupling), default = -1.0
            
        Raises:
            ValueError: If inputs are None or if adjacency matrix is not square
        """
        if adj_matrix is None or alpha is None or beta is None:
            raise ValueError("Adjacency matrix, alpha, and beta must be provided.")
        self.adj = np.array(adj_matrix, dtype=float)
        if self.adj.ndim != 2 or self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        self.n = self.adj.shape[0]
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._init_hamiltonian()
        self.energy: Optional[np.ndarray] = None
        self.mo_coeff: Optional[np.ndarray] = None

    def _init_hamiltonian(self) -> None:
        """
        Initialize the Hückel Hamiltonian matrix.
        
        This private method constructs the Hamiltonian matrix based on the adjacency matrix:
        - Diagonal elements (H_ii) are set to α (Coulomb integral)
        - Off-diagonal elements (H_ij) are set to β (resonance integral) if atoms i and j are bonded
          (i.e., if the adjacency matrix has a 1 at position [i,j])
        """
        new_matrix = np.full((self.n, self.n), 0.0, dtype=float)
        new_matrix[self.adj == 1] = self.beta
        np.fill_diagonal(new_matrix, self.alpha)
        self.H = new_matrix

    def set_params(self, alpha: Optional[float] = None, beta: Optional[float] = None) -> None:
        """
        Change α and β parameters and regenerate the Hamiltonian matrix.
        
        Parameters:
            alpha: New value for Coulomb integral α (if None, keeps current value)
            beta: New value for resonance integral β (if None, keeps current value)
        """
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        self._init_hamiltonian()
        self.energy = None
        self.mo_coeff = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hückel Hamiltonian to get molecular orbital energies and coefficients.
        
        This method solves the eigenvalue problem H c = E c, where:
        - H is the Hückel Hamiltonian matrix
        - E are the eigenvalues (molecular orbital energies)
        - c are the eigenvectors (molecular orbital coefficients)
        
        The eigenvalues and eigenvectors are sorted in ascending order of energy.
        
        Returns:
            energy: Eigenvalues (molecular orbital energies) in ascending order
            mo_coeff: Eigenvectors (molecular orbital coefficients) corresponding to the energies
            
        Notes:
            - The results are also stored as instance attributes (self.energy, self.mo_coeff)
            - The Hückel method uses a simple one-electron Hamiltonian with no explicit
              electron-electron interactions
        """
        e, c = np.linalg.eigh(self.H)
        idx = np.argsort(e)
        self.energy = e[idx]
        self.mo_coeff = c[:, idx]
        return self.energy, self.mo_coeff

    def total_pi_energy(self, electrons: int) -> float:
        """
        Calculate total π electron energy by filling the lowest energy molecular orbitals.
        
        This method calculates the sum of the energies of all occupied molecular orbitals.
        Each orbital can hold up to 2 electrons (due to spin).
        
        Parameters:
            electrons: Total number of π electrons in the system
                      (should be an even number for closed-shell systems)
            
        Returns:
            Total π electron energy (in units consistent with α and β)
            
        Raises:
            ValueError: If electrons is negative or not an integer
            ValueError: If self.energy is None (solve() must be called first)
            Warning: If electrons is odd (open-shell system)
        """
        if not isinstance(electrons, int) or electrons < 0:
            raise ValueError("Number of electrons must be a non-negative integer.")
            
        if electrons % 2 != 0:
            import warnings
            warnings.warn("Odd number of electrons specified. Using fractional occupation.")
            
        if self.energy is None:
            self.solve()
            
        if self.energy is None:  # This should not happen after solve() is called
            raise ValueError("Energy levels not available. Solve failed.")
            
        occ = electrons // 2
        return 2 * np.sum(self.energy[:occ])