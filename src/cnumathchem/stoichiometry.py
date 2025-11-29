"""
stoichiometry.py

This module provides functionality for stoichiometric calculations,
including molar mass calculation and equation balancing.

Author: Choi Minseo
Date: 2025
"""

import re
from typing import Dict, List, Tuple, Optional

class Stoichiometry:
    """
    Class for performing stoichiometric calculations.
    """
    
    # Atomic masses of common elements (g/mol)
    ATOMIC_MASSES = {
        'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
        'S': 32.06, 'Cl': 35.45, 'K': 39.098, 'Ar': 39.948, 'Ca': 40.078,
        'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Ag': 107.87, 'Au': 196.97,
        'Hg': 200.59, 'Pb': 207.2
    }

    @staticmethod
    def calculate_molar_mass(formula: str) -> float:
        """
        Calculate the molar mass of a chemical formula.
        
        Parameters:
            formula: Chemical formula string (e.g., "H2O", "C6H12O6")
            
        Returns:
            Molar mass in g/mol
            
        Raises:
            ValueError: If formula contains unknown elements or invalid format
        """
        # Parse formula
        elements = {}
        pattern = r"([A-Z][a-z]*)(\d*)"
        for element, count in re.findall(pattern, formula):
            count = int(count) if count else 1
            if element not in Stoichiometry.ATOMIC_MASSES:
                raise ValueError(f"Unknown element: {element}")
            elements[element] = elements.get(element, 0) + count
            
        # Calculate mass
        mass = 0.0
        for element, count in elements.items():
            mass += Stoichiometry.ATOMIC_MASSES[element] * count
            
        return mass

    @staticmethod
    def parse_reaction(reaction: str) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        """
        Parse a chemical reaction string into reactants and products.
        
        Parameters:
            reaction: String like "H2 + O2 -> H2O"
            
        Returns:
            Tuple of (reactants, products), where each is a list of element counts
        """
        # Implementation placeholder for complex balancing logic
        # For now, just basic parsing
        parts = reaction.split('->')
        if len(parts) != 2:
            raise ValueError("Invalid reaction format. Use '->' to separate reactants and products.")
            
        return ([], [])  # Placeholder
