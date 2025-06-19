# particle.py
# This file contains the implementation of various quantum mechanical particle models.
# ParticleInABox1D, ParticleInABox2D, Rotational2D, and TunnelingBarrier classes are defined here.

import numpy as np

class ParticleInABox1D:
    """
    Particle in a 1D infinite potential box

    Parameters:
        L: Length of the box
        m_eff: Effective mass of the particle
        hbar: Reduced Planck's constant (h/2π)
    """
    def __init__(self, L, m_eff, hbar=1.0):
        self.L = L
        self.m = m_eff
        self.hbar = hbar

    # E_n = n² π² ħ² / (2 m L²)
    def energy(self, n):
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)

    # ψ_n(x) = √(2/L) sin(n π x / L), 0 ≤ x ≤ L
    def wavefunction(self, n, x):
        return np.sqrt(2/self.L) * np.sin(n * np.pi * x / self.L)

    def spectrum(self, n_max):
        ns = np.arange(1, n_max+1)
        Es = (ns**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)
        return ns, Es


class ParticleInABox2D:
    """
    Particle in a 2D infinite potential box (rectangular box)

    Parameters:
        Lx, Ly: Lengths of the box in x and y directions (number)
        m_eff: Effective mass of the particle (number)
        hbar: Reduced Planck's constant (h/2π) (number)
    """
    def __init__(self, Lx, Ly, m_eff, hbar=1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.m = m_eff
        self.hbar = hbar

    # E_nx,ny = (hbar² π² / 2m) (nx²/Lx² + ny²/Ly²)
    def energy(self, nx, ny):
        return (self.hbar**2 * np.pi**2 / (2*self.m)) * (nx**2/self.Lx**2 + ny**2/self.Ly**2)

    # ψ = √(2/Lx) sin(nx π x / Lx) · √(2/Ly) sin(ny π y / Ly)
    def wavefunction(self, nx, ny, x, y):
        ψx = np.sqrt(2/self.Lx) * np.sin(nx * np.pi * x / self.Lx)
        ψy = np.sqrt(2/self.Ly) * np.sin(ny * np.pi * y / self.Ly)
        return ψx * ψy

    def spectrum(self, n_max_x, n_max_y):
        levels = []
        for nx in range(1, n_max_x+1):
            for ny in range(1, n_max_y+1):
                E = self.energy(nx, ny)
                levels.append((nx, ny, E))
        levels.sort(key=lambda t: t[2])
        return levels


class Rotational2D:
    """
    Rotational motion in 2D (r = constant, V = 0)

    Parameters:
        r: Radius of the circular path
        m_eff: Effective mass of the particle
        hbar: Reduced Planck's constant (h/2π)
    """
    def __init__(self, r, m_eff, hbar=1.0):
        self.r = r
        self.m = m_eff
        self.hbar = hbar
        self.I = self.m * self.r**2

    # E = ħ² m_l² / (2 I)
    def energy(self, m_l):
        return (self.hbar**2 * m_l**2) / (2 * self.I)

    # ψ(φ) = 1/√(2π) e^{i m_l φ}
    def wavefunction(self, m_l, phi):
        return (1/np.sqrt(2*np.pi)) * np.exp(1j * m_l * phi)

    def spectrum(self, m_max):
        ms = np.arange(-m_max, m_max+1)
        Es = [self.energy(m) for m in ms]
        return ms, np.array(Es)


class TunnelingBarrier:
    """
    Tunneling through a 1D potential barrier

    Parameters:
        V0: Height of the barrier
        a: Width of the barrier
        m_eff: Effective mass of the particle
        hbar: Reduced Planck's constant (h/2π)
    """
    def __init__(self, V0, a, m_eff, hbar=1.0):
        self.V0 = V0
        self.a = a
        self.m = m_eff
        self.hbar = hbar

    # κ = sqrt(2 m (V0 - E)) / ħ
    def kappa(self, E):
        return np.sqrt(2 * self.m * (self.V0 - E)) / self.hbar

    # E < V0: T ≈ exp(-2 κ a), E ≥ V0: T = 1
    def transmission(self, E):
        if E >= self.V0:
            return 1.0
        κ = self.kappa(E)
        return np.exp(-2 * κ * self.a)
