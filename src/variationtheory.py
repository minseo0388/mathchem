"""
variationtheory.py

This module implements the variation method for quantum mechanical calculations,
specifically focused on molecular orbital theory with non-orthogonal basis sets.

Author: Choi Minseo
Date: 2025
"""

import numpy as np
from scipy.linalg import eigh
from typing import Optional, Tuple, Union, List, Dict, Any
import warnings

class VariationTheory:
    """
    Variation Theory for Molecular Orbital Calculations.
    
    This class implements the variation method for solving the electronic structure
    of molecules using a linear combination of atomic orbitals (LCAO) approach with
    possibly non-orthogonal basis functions.
    
    The variation method seeks to minimize the energy expectation value by optimizing
    the coefficients of the basis functions. For non-orthogonal basis sets, this leads
    to the generalized eigenvalue problem: H c = E S c
    
    Parameters:
        H: Hamiltonian matrix (n x n), representing the energy integrals between basis functions
        S: Overlap matrix (n x n), representing the overlap between basis functions
           If None, an identity matrix is used (orthogonal basis)
           
    Notes:
        - The variation method provides an upper bound to the true ground state energy
        - The accuracy depends on the choice of basis functions
        - For a complete basis set, the method gives exact results
        - The lowest eigenvalue corresponds to the ground state energy
    """
    def __init__(self, H: np.ndarray, S: Optional[np.ndarray] = None):
        """
        Initialize the variation theory calculation with Hamiltonian and overlap matrices.
        
        Parameters:
            H: Hamiltonian matrix (n x n), representing energy integrals
            S: Overlap matrix (n x n), representing basis function overlaps
               If None, an identity matrix is assumed (orthogonal basis)
               
        Raises:
            ValueError: If H is not a square matrix or if S dimensions don't match H
        """
        self.H = np.array(H, dtype=float)
        if self.H.ndim != 2 or self.H.shape[0] != self.H.shape[1]:
            raise ValueError("Hamiltonian matrix must be square.")
        if S is None:
            self.S = np.eye(self.H.shape[0])
        else:
            self.S = np.array(S, dtype=float)
            if self.S.shape != self.H.shape:
                raise ValueError("Overlap matrix must match Hamiltonian shape.")
                
        # Check for symmetry of matrices
        if not np.allclose(self.H, self.H.T):
            warnings.warn("Hamiltonian matrix is not symmetric. Results may be physically incorrect.")
        if not np.allclose(self.S, self.S.T):
            warnings.warn("Overlap matrix is not symmetric. Results may be physically incorrect.")
            
        self.energy: Optional[np.ndarray] = None
        self.coeff: Optional[np.ndarray] = None

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem H c = E S c.
        
        This method diagonalizes the Hamiltonian in the given basis, accounting for
        the non-orthogonality of the basis functions through the overlap matrix S.
        
        Returns:
            energies: Eigenvalues (molecular orbital energies) in ascending order
            coefficients: Eigenvectors (molecular orbital coefficients) corresponding to the energies
            
        Notes:
            - The results are also stored as instance attributes (self.energy, self.coeff)
            - The generalized eigenvalue problem is solved using scipy.linalg.eigh
            - For orthogonal basis (S = I), this reduces to the standard eigenvalue problem
        """
        E, C = eigh(self.H, self.S)
        idx = np.argsort(E)
        self.energy = E[idx]
        self.coeff = C[:, idx]
        return self.energy, self.coeff
        
    def expectation_value(self, operator: np.ndarray, state_idx: int = 0) -> float:
        """
        Calculate the expectation value of an operator for a given eigenstate.
        
        Parameters:
            operator: Matrix representation of the operator in the same basis as H and S
            state_idx: Index of the eigenstate to use (default=0 for ground state)
            
        Returns:
            Expectation value <ψ|O|ψ>
            
        Raises:
            ValueError: If operator dimensions don't match H and S
            ValueError: If state_idx is out of range
            ValueError: If solve() has not been called
        """
        if self.coeff is None or self.energy is None:
            self.solve()
            
        if operator.shape != self.H.shape:
            raise ValueError("Operator matrix must have same dimensions as Hamiltonian.")
            
        n = self.H.shape[0]
        if state_idx < 0 or state_idx >= n:
            raise ValueError(f"State index must be between 0 and {n-1}.")
            
        c = self.coeff[:, state_idx]
        return c.T @ operator @ c
        
    def transition_moment(self, operator: np.ndarray, i: int, j: int) -> float:
        """
        Calculate the transition moment between two eigenstates.
        
        Parameters:
            operator: Matrix representation of the operator in the same basis as H and S
            i: Index of the initial state
            j: Index of the final state
            
        Returns:
            Transition moment <ψ_i|O|ψ_j>
            
        Notes:
            - For electric dipole transitions, the operator would be the dipole operator
            - The transition probability is proportional to the square of the transition moment
        """
        if self.coeff is None:
            self.solve()
            
        n = self.H.shape[0]
        if i < 0 or i >= n or j < 0 or j >= n:
            raise ValueError(f"State indices must be between 0 and {n-1}.")
            
        c_i = self.coeff[:, i]
        c_j = self.coeff[:, j]
        
        return c_i.T @ operator @ c_j