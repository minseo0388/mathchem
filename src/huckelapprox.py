# huckelapprox.py
# This file contains the implementation of the Huckel approximation method for molecular orbital calculations.

import numpy as np

class Huckel:
    def __init__(self, adj_matrix, alpha, beta):
        """
        Huckel Approximation for Molecular Orbital Calculations
        This class implements the Huckel approximation method for calculating molecular orbital energies and coefficients.

        Parameters:
            adj_matrix: adjoint matrix (n by n), if bonded = 1, otherwise 0 (list of lists)
            alpha: Coulomb integral (π orbital on-site energy) (number, commonly 0.0)
            beta: Resonance integral (adjacent π coupling) (number, commonly -1.0)
        """
        if adj_matrix is None or alpha is None or beta is None:
            raise ValueError("Please RECHECK the Huckel approximation setup.")
        self.adj = np.array(adj_matrix, dtype=float)
        self.n = self.adj.shape[0]
        self.alpha = alpha
        self.beta = beta
        self._init_hamiltonian()

        self.energy = None
        self.mo_coeff = None

    def _init_hamiltonian(self):
        new_matrix = np.zeros((self.n, self.n), dtype=float)
        np.fill_diagonal(new_matrix, self.alpha)
        new_matrix[self.adj == 1] = self.beta
        self.H = new_matrix

    def set_params(self, alpha=None, beta=None):
        """
        Change alpha, beta values and regenerate the Hamiltonian
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self._init_hamiltonian()

        self.energy = None
        self.mo_coeff = None

    def solve(self):
        """
        Hamiltonian Diagonalization
        Energies and Molecular Orbital Coefficients

        Returns:
            energy: Eigenvalues (molecular orbital energies)
            mo_coeff: Eigenvectors (molecular orbital coefficients)
        """
        e, c = np.linalg.eigh(self.H)
        idx = np.argsort(e)
        self.energy = e[idx]
        self.mo_coeff = c[:, idx]
        return self.energy, self.mo_coeff

    def total_pi_energy(self, electrons):
        """
        Fill the lowest orbitals with electrons and calculate total π energy

        Parameters:
            electrons: Total number of π electrons (even number)
        """
        if self.energy is None:
            self.solve()
        occ = electrons // 2
        return 2 * np.sum(self.energy[:occ])
