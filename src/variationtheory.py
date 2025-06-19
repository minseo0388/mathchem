# variationtheory.py
# This file contains the implementation of the Variation Theory method for molecular orbital calculations.

import numpy as np
from scipy.linalg import eigh

class VariationTheory:
    def __init__(self, H, S=None):
        """
        Variation Theory for Molecular Orbital Calculations
        This class implements the Variation Theory method for solving the generalized eigenvalue problem.

        Parameters:
            H: Hamiltonian Array (n by n)
            S: Overlapped Array (n by n). If not provided, identity matrix is used.
        """
        self.H = np.array(H, dtype=float)
        if S is None:
            self.S = np.eye(self.H.shape[0])
        else:
            self.S = np.array(S, dtype=float)

    def solve(self):
        """
        Solve the generalized eigenvalue problem H c = E S c

        Parameters:
            E: Eigenvalue (molecular orbital energies)
            C: Eigenvector (molecular orbital coefficients)

        Returns:
            Energies: Eigenvalues (molecular orbital energies)
            Coefficients: Eigenvectors (molecular orbital coefficients)
        """
        E, C = eigh(self.H, self.S)
        idx = np.argsort(E)
        self.energy = E[idx]
        self.coeff = C[:, idx]
        return self.energy, self.coeff
