import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('src'))
from cnumathchem.particle import ParticleInABox1D
from cnumathchem.stoichiometry import Stoichiometry

class TestParticleInABox1D(unittest.TestCase):
    def setUp(self):
        self.particle = ParticleInABox1D(L=1.0, m_eff=1.0)

    def test_energy(self):
        # E_1 = pi^2 * hbar^2 / (2 * m * L^2)
        # With L=1, m=1, hbar=1, E_1 = pi^2 / 2
        expected = np.pi**2 / 2
        self.assertAlmostEqual(self.particle.energy(1), expected)

    def test_wavefunction_normalization(self):
        # Check normalization for n=1
        x = np.linspace(0, 1, 1000)
        dx = x[1] - x[0]
        psi = self.particle.wavefunction(1, x)
        prob = np.sum(psi**2) * dx
        self.assertAlmostEqual(prob, 1.0, places=3)

class TestStoichiometry(unittest.TestCase):
    def test_molar_mass(self):
        self.assertAlmostEqual(Stoichiometry.calculate_molar_mass("H2O"), 18.015, places=3)
        self.assertAlmostEqual(Stoichiometry.calculate_molar_mass("C6H12O6"), 180.156, places=2)

    def test_invalid_element(self):
        with self.assertRaises(ValueError):
            Stoichiometry.calculate_molar_mass("Xy")

if __name__ == '__main__':
    unittest.main()
