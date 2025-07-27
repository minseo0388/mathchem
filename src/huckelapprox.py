# huckelapprox.py
# Huckel approximation for molecular orbital calculations.

import numpy as np
from typing import List, Tuple, Optional

class Huckel:
    """
    Huckel Approximation for Molecular Orbital Calculations.

    Parameters:
        adj_matrix: Adjacency matrix (n x n), 1 if bonded, 0 otherwise.
        alpha: Coulomb integral (π orbital on-site energy).
        beta: Resonance integral (adjacent π coupling).
    """
    def __init__(self, adj_matrix: List[List[float]], alpha: float = 0.0, beta: float = -1.0):
        if adj_matrix is None or alpha is None or beta is None:
            raise ValueError("Adjacency matrix, alpha, and beta must be provided.")
        self.adj = np.array(adj_matrix, dtype=float)
        if self.adj.ndim != 2 or self.adj.shape[0] != self.adj.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        self.n = self.adj.shape[0]
        self.alpha = alpha
        self.beta = beta
        self._init_hamiltonian()
        self.energy: Optional[np.ndarray] = None
        self.mo_coeff: Optional[np.ndarray] = None

    def _init_hamiltonian(self):
        new_matrix = np.full((self.n, self.n), self.alpha, dtype=float)
        new_matrix[self.adj == 1] = self.beta
        np.fill_diagonal(new_matrix, self.alpha)
        self.H = new_matrix

    def set_params(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        """Change alpha, beta values and regenerate the Hamiltonian."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self._init_hamiltonian()
        self.energy = None
        self.mo_coeff = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize Hamiltonian to get energies and MO coefficients.

        Returns:
            energy: Eigenvalues (MO energies)
            mo_coeff: Eigenvectors (MO coefficients)
        """
        e, c = np.linalg.eigh(self.H)
        idx = np.argsort(e)
        self.energy = e[idx]
        self.mo_coeff = c[:, idx]
        return self.energy, self.mo_coeff

    def total_pi_energy(self, electrons: int) -> float:
        """
        Calculate total π energy by filling lowest orbitals.

        Parameters:
            electrons: Total number of π electrons (even number)
        Returns:
            Total π energy (float)
        """
        if self.energy is None:
            self.solve()
        occ = electrons // 2
        return 2 * np.sum(self.energy[:occ])
    