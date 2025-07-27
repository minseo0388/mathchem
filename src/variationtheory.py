# variationtheory.py
# Variation Theory for molecular orbital calculations.

import numpy as np
from scipy.linalg import eigh
from typing import Optional, Tuple

class VariationTheory:
    """
    Variation Theory for Molecular Orbital Calculations.

    Parameters:
        H: Hamiltonian matrix (n x n)
        S: Overlap matrix (n x n), defaults to identity if not provided.
    """
    def __init__(self, H, S: Optional[np.ndarray] = None):
        self.H = np.array(H, dtype=float)
        if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
            raise ValueError("Hamiltonian matrix must be square.")
        if S is None:
            self.S = np.eye(self.H.shape[0])
        else:
            self.S = np.array(S, dtype=float)
            if self.S.shape != self.H.shape:
                raise ValueError("Overlap matrix must match Hamiltonian shape.")

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem H c = E S c.

        Returns:
            Energies: Eigenvalues (MO energies)
            Coefficients: Eigenvectors (MO coefficients)
        """
        E, C = eigh(self.H, self.S)
        idx = np.argsort(E)
        self.energy = E[idx]
        self.coeff = C[:, idx]
        return self.energy, self.coeff
    