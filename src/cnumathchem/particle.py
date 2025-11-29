"""
particle.py

This module provides implementations of fundamental quantum mechanical particle models
used in quantum chemistry and physics education. It includes classes for particles
in various potential environments, with analytical solutions to the Schrödinger equation.

Author: Choi Minseo
Date: 2025
"""

import numpy as np
from typing import Union, Tuple, List, Optional, Any, Dict
import warnings

# Physical constants (in atomic units unless specified)
HBAR_DEFAULT = 1.0  # ℏ = h/2π (reduced Planck's constant)
ELECTRON_MASS = 1.0  # Electron mass
BOHR_RADIUS = 1.0  # Bohr radius (a₀)

class ParticleInABox1D:
    """
    Particle in a 1D infinite potential box (quantum well).
    
    The quantum system represents a particle confined to a one-dimensional
    region with infinitely high potential barriers at x=0 and x=L.
    
    Parameters:
        L: Length of the box (positive number)
        m_eff: Effective mass of the particle (positive number)
        hbar: Reduced Planck's constant (h/2π) (positive number, default=1.0)
        
    Notes:
        - Energy eigenvalues: E_n = (n²π²ℏ²)/(2mL²)
        - Normalized wavefunctions: ψ_n(x) = √(2/L)·sin(nπx/L) for 0 ≤ x ≤ L
        - Valid quantum numbers: n = 1, 2, 3, ...
    """
    def __init__(self, L: float, m_eff: float, hbar: float = 1.0):
        if L <= 0 or m_eff <= 0 or hbar <= 0:
            raise ValueError("Length, mass, and ℏ must be positive.")
        self.L = float(L)
        self.m = float(m_eff)
        self.hbar = float(hbar)

    def energy(self, n: int) -> float:
        """
        Calculate the energy eigenvalue for a given quantum number.
        
        Parameters:
            n: Principal quantum number (positive integer)
            
        Returns:
            Energy eigenvalue E_n = (n²π²ℏ²)/(2mL²)
            
        Raises:
            ValueError: If n is not a positive integer
        """
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Quantum number n must be a positive integer.")
        return (n**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)

    def wavefunction(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the normalized wavefunction at position x.
        
        Parameters:
            n: Principal quantum number (positive integer)
            x: Position or array of positions
            
        Returns:
            Wavefunction value(s) ψ_n(x) = √(2/L)·sin(nπx/L) for 0 ≤ x ≤ L,
            and zero outside this range
            
        Notes:
            - Returns zero for x outside the box [0,L]
            - Can handle scalar x or numpy array of positions
            - The wavefunction satisfies the boundary conditions ψ(0) = ψ(L) = 0
            - The wavefunction is normalized: ∫|ψ_n(x)|²dx = 1 over [0,L]
            - Wavefunctions with different n are orthogonal to each other
        
        Raises:
            ValueError: If n is not a positive integer
        """
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Quantum number n must be a positive integer.")
        
        # Convert to numpy array for consistent handling
        x_array = np.asarray(x)
        
        # Create mask for valid positions
        mask = (x_array >= 0) & (x_array <= self.L)
        
        # Initialize result array with zeros
        result = np.zeros_like(x_array, dtype=float)
        
        # Calculate wavefunction only for valid positions
        result[mask] = np.sqrt(2/self.L) * np.sin(n * np.pi * x_array[mask] / self.L)
        
        # Return scalar if input was scalar
        if np.isscalar(x):
            return float(result.item())
        return result

    def probability_density(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the probability density |ψ(x)|² at position x.
        
        The probability density gives the probability per unit length of finding
        the particle at position x.
        
        Parameters:
            n: Principal quantum number (positive integer)
            x: Position or array of positions
            
        Returns:
            Probability density |ψ_n(x)|² = (2/L)·sin²(nπx/L) for 0 ≤ x ≤ L,
            and zero outside this range
            
        Notes:
            - The probability of finding the particle in a small region dx around x
              is approximately |ψ(x)|²dx
            - The probability density integrates to 1 over the entire box: 
              ∫|ψ(x)|²dx = 1 over [0,L]
            - The probability density has n-1 nodes (zeros) inside the box
        
        Raises:
            ValueError: If n is not a positive integer
        """
        psi = self.wavefunction(n, x)
        return psi ** 2
        
    def expectation_x(self, n: int) -> float:
        """
        Calculate the expectation value of position for a given quantum state.
        
        Parameters:
            n: Principal quantum number (positive integer)
            
        Returns:
            Expectation value ⟨x⟩ = ∫ψ*(x)·x·ψ(x)dx = L/2 for all n
            
        Notes:
            - For the infinite square well, the expectation value of position
              is always at the center of the well (L/2) due to symmetry
        """
        return self.L / 2.0
        
    def expectation_x_squared(self, n: int) -> float:
        """
        Calculate the expectation value of position squared for a given quantum state.
        
        Parameters:
            n: Principal quantum number (positive integer)
            
        Returns:
            Expectation value ⟨x²⟩ = ∫ψ*(x)·x²·ψ(x)dx = (L²/2)·(1 - 1/(n²π²))
            
        Notes:
            - This is useful for calculating the position uncertainty Δx = √(⟨x²⟩ - ⟨x⟩²)
        
        Raises:
            ValueError: If n is not a positive integer
        """
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Quantum number n must be a positive integer.")
            
        # ⟨x²⟩ = (L²/2)·(1 - 1/(n²π²))
        return (self.L**2 / 2) * (1 - 1/(n**2 * np.pi**2))
        
    def position_uncertainty(self, n: int) -> float:
        """
        Calculate the position uncertainty (standard deviation) for a given quantum state.
        
        Parameters:
            n: Principal quantum number (positive integer)
            
        Returns:
            Position uncertainty Δx = √(⟨x²⟩ - ⟨x⟩²)
            
        Notes:
            - The position uncertainty decreases as n increases, indicating that
              higher energy states have more localized wavefunctions
            - This demonstrates the Heisenberg uncertainty principle: as the energy (and momentum)
              becomes more certain, the position becomes less certain
        
        Raises:
            ValueError: If n is not a positive integer
        """
        x_avg = self.expectation_x(n)
        x2_avg = self.expectation_x_squared(n)
        return np.sqrt(x2_avg - x_avg**2)
        
    def momentum_uncertainty(self, n: int) -> float:
        """
        Calculate the momentum uncertainty (standard deviation) for a given quantum state.
        
        Parameters:
            n: Principal quantum number (positive integer)
            
        Returns:
            Momentum uncertainty Δp = n·π·ℏ/L·√(1/3 - 1/(2n²π²))
            
        Notes:
            - The momentum uncertainty increases with n
            - When multiplied by the position uncertainty, this demonstrates the
              Heisenberg uncertainty principle: Δx·Δp ≥ ℏ/2
        
        Raises:
            ValueError: If n is not a positive integer
        """
        if not isinstance(n, (int, np.integer)) or n <= 0:
            raise ValueError("Quantum number n must be a positive integer.")
            
        # Δp = n·π·ℏ/L·√(1/3 - 1/(2n²π²))
        return (n * np.pi * self.hbar / self.L) * np.sqrt(1/3 - 1/(2 * n**2 * np.pi**2))
        
    def spectrum(self, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energy eigenvalues for quantum numbers from 1 to n_max.
        
        Parameters:
            n_max: Maximum quantum number to include (positive integer)
            
        Returns:
            Tuple of (quantum_numbers, energies) as numpy arrays
            
        Raises:
            ValueError: If n_max is not a positive integer
        """
        if not isinstance(n_max, (int, np.integer)) or n_max <= 0:
            raise ValueError("n_max must be a positive integer.")
            
        ns = np.arange(1, n_max+1)
        Es = (ns**2 * np.pi**2 * self.hbar**2) / (2 * self.m * self.L**2)
        return ns, Es


class ParticleInABox2D:
    """
    Particle in a 2D infinite potential box (rectangular quantum well).
    
    The quantum system represents a particle confined to a two-dimensional
    rectangular region with infinitely high potential barriers at the boundaries.
    
    Parameters:
        Lx: Length of the box in x direction (positive number)
        Ly: Length of the box in y direction (positive number)
        m_eff: Effective mass of the particle (positive number)
        hbar: Reduced Planck's constant (h/2π) (positive number, default=1.0)
        
    Notes:
        - Energy eigenvalues: E_{nx,ny} = (π²ℏ²/2m)·(nx²/Lx² + ny²/Ly²)
        - Normalized wavefunctions: ψ_{nx,ny}(x,y) = √(4/LxLy)·sin(nxπx/Lx)·sin(nyπy/Ly)
          for 0 ≤ x ≤ Lx and 0 ≤ y ≤ Ly
        - Valid quantum numbers: nx, ny = 1, 2, 3, ...
    """
    def __init__(self, Lx: float, Ly: float, m_eff: float, hbar: float = 1.0):
        if Lx <= 0 or Ly <= 0 or m_eff <= 0 or hbar <= 0:
            raise ValueError("Box dimensions, mass, and ℏ must be positive.")
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.m = float(m_eff)
        self.hbar = float(hbar)

    def energy(self, nx: int, ny: int) -> float:
        """
        Calculate the energy eigenvalue for given quantum numbers.
        
        Parameters:
            nx: Quantum number in x direction (positive integer)
            ny: Quantum number in y direction (positive integer)
            
        Returns:
            Energy eigenvalue E_{nx,ny} = (π²ℏ²/2m)·(nx²/Lx² + ny²/Ly²)
            
        Raises:
            ValueError: If nx or ny is not a positive integer
        """
        if not isinstance(nx, (int, np.integer)) or nx <= 0 or not isinstance(ny, (int, np.integer)) or ny <= 0:
            raise ValueError("Quantum numbers nx and ny must be positive integers.")
        return (self.hbar**2 * np.pi**2 / (2*self.m)) * (nx**2/self.Lx**2 + ny**2/self.Ly**2)

    def wavefunction(self, nx: int, ny: int, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the normalized wavefunction at position (x,y).
        
        Parameters:
            nx: Quantum number in x direction (positive integer)
            ny: Quantum number in y direction (positive integer)
            x: Position or array of positions in x direction
            y: Position or array of positions in y direction
            
        Returns:
            Wavefunction value(s) ψ_{nx,ny}(x,y) = √(4/LxLy)·sin(nxπx/Lx)·sin(nyπy/Ly)
            for 0 ≤ x ≤ Lx and 0 ≤ y ≤ Ly, and zero outside this range
            
        Notes:
            - Returns zero for positions outside the box [0,Lx]×[0,Ly]
            - Can handle scalar x,y or numpy arrays of positions
            - The wavefunction satisfies the boundary conditions ψ = 0 at all edges
            - The wavefunction is normalized: ∫∫|ψ_{nx,ny}(x,y)|²dxdy = 1 over [0,Lx]×[0,Ly]
            - Wavefunctions with different quantum numbers are orthogonal to each other
        
        Raises:
            ValueError: If nx or ny is not a positive integer
        """
        if not isinstance(nx, (int, np.integer)) or nx <= 0 or not isinstance(ny, (int, np.integer)) or ny <= 0:
            raise ValueError("Quantum numbers nx and ny must be positive integers.")
            
        # Convert to numpy arrays for consistent handling
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        
        # Calculate the wavefunctions in x and y directions
        # Apply boundary conditions by using masks
        x_mask = (x_array >= 0) & (x_array <= self.Lx)
        y_mask = (y_array >= 0) & (y_array <= self.Ly)
        
        # Initialize with zeros
        psi_x = np.zeros_like(x_array, dtype=float)
        psi_y = np.zeros_like(y_array, dtype=float)
        
        # Calculate wavefunction components only for valid positions
        psi_x[x_mask] = np.sqrt(2/self.Lx) * np.sin(nx * np.pi * x_array[x_mask] / self.Lx)
        psi_y[y_mask] = np.sqrt(2/self.Ly) * np.sin(ny * np.pi * y_array[y_mask] / self.Ly)
        
        # Total wavefunction is the product
        psi = psi_x * psi_y
        
        # Return scalar if both inputs were scalars
        if np.isscalar(x) and np.isscalar(y):
            return float(psi.item())
        return psi

    def probability_density_2d(self, nx: int, ny: int, x: Union[float, np.ndarray], 
                           y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the probability density |ψ(x,y)|² at position (x,y).
        
        The probability density gives the probability per unit area of finding
        the particle at position (x,y).
        
        Parameters:
            nx: Quantum number in x direction (positive integer)
            ny: Quantum number in y direction (positive integer)
            x: Position or array of positions in x direction
            y: Position or array of positions in y direction
            
        Returns:
            Probability density |ψ_{nx,ny}(x,y)|² = (4/LxLy)·sin²(nxπx/Lx)·sin²(nyπy/Ly)
            for 0 ≤ x ≤ Lx and 0 ≤ y ≤ Ly, and zero outside this range
            
        Notes:
            - The probability of finding the particle in a small region dxdy around (x,y)
              is approximately |ψ(x,y)|²dxdy
            - The probability density integrates to 1 over the entire box: 
              ∫∫|ψ(x,y)|²dxdy = 1 over [0,Lx]×[0,Ly]
            - The probability density has (nx-1) nodes in the x-direction and
              (ny-1) nodes in the y-direction
        
        Raises:
            ValueError: If nx or ny is not a positive integer
        """
        psi = self.wavefunction(nx, ny, x, y)
        return psi ** 2
        
    def expectation_position(self, nx: int, ny: int) -> Tuple[float, float]:
        """
        Calculate the expectation values of position for a given quantum state.
        
        Parameters:
            nx: Quantum number in x direction (positive integer)
            ny: Quantum number in y direction (positive integer)
            
        Returns:
            Tuple (⟨x⟩, ⟨y⟩) = (Lx/2, Ly/2) for all nx, ny
            
        Notes:
            - For the 2D infinite square well, the expectation values of position
              are always at the center of the well (Lx/2, Ly/2) due to symmetry
        """
        return (self.Lx / 2.0, self.Ly / 2.0)
        
    def spectrum(self, n_max_x: int, n_max_y: int) -> List[Tuple[int, int, float]]:
        """
        Calculate energy eigenvalues for quantum numbers up to n_max_x and n_max_y.
        
        Parameters:
            n_max_x: Maximum quantum number in x direction (positive integer)
            n_max_y: Maximum quantum number in y direction (positive integer)
            
        Returns:
            List of tuples (nx, ny, energy) sorted by increasing energy
            
        Raises:
            ValueError: If n_max_x or n_max_y is not a positive integer
        """
        if not isinstance(n_max_x, (int, np.integer)) or n_max_x <= 0 or not isinstance(n_max_y, (int, np.integer)) or n_max_y <= 0:
            raise ValueError("Maximum quantum numbers must be positive integers.")
            
        levels = []
        for nx in range(1, n_max_x+1):
            for ny in range(1, n_max_y+1):
                E = self.energy(nx, ny)
                levels.append((nx, ny, E))
        levels.sort(key=lambda t: t[2])
        return levels


class Rotational2D:
    """
    Rotational motion in 2D (rigid rotor in two dimensions).
    
    The quantum system represents a particle constrained to move in a circular path
    with constant radius and no potential energy.
    
    Parameters:
        r: Radius of the circular path (positive number)
        m_eff: Effective mass of the particle (positive number)
        hbar: Reduced Planck's constant (h/2π) (positive number, default=1.0)
        
    Notes:
        - Energy eigenvalues: E_{m_l} = (ℏ²m_l²)/(2I) where I = mr²
        - Normalized wavefunctions: ψ_{m_l}(φ) = (1/√(2π))·e^{im_lφ}
        - Valid quantum numbers: m_l = 0, ±1, ±2, ... (integer)
        - The wavefunctions are eigenfunctions of the angular momentum operator
          with eigenvalues m_l·ℏ
    """
    def __init__(self, r: float, m_eff: float, hbar: float = 1.0):
        if r <= 0 or m_eff <= 0 or hbar <= 0:
            raise ValueError("Radius, mass, and ℏ must be positive.")
        self.r = float(r)
        self.m = float(m_eff)
        self.hbar = float(hbar)
        self.I = self.m * self.r**2  # Moment of inertia

    def energy(self, m_l: int) -> float:
        """
        Calculate the energy eigenvalue for a given angular momentum quantum number.
        
        Parameters:
            m_l: Angular momentum quantum number (integer)
            
        Returns:
            Energy eigenvalue E_{m_l} = (ℏ²m_l²)/(2I)
            
        Notes:
            - The energy depends only on the magnitude of m_l, not its sign
            - This leads to degenerate energy levels for ±m_l pairs (except m_l=0)
            - The energy increases quadratically with the angular momentum
            
        Raises:
            ValueError: If m_l is not an integer
        """
        if not isinstance(m_l, (int, np.integer)):
            raise ValueError("Angular momentum quantum number m_l must be an integer.")
        return (self.hbar**2 * m_l**2) / (2 * self.I)

    def wavefunction(self, m_l: int, phi: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        """
        Calculate the normalized wavefunction at angle phi.
        
        Parameters:
            m_l: Angular momentum quantum number (integer)
            phi: Angle or array of angles (in radians)
            
        Returns:
            Complex wavefunction value(s) ψ_{m_l}(φ) = (1/√(2π))·e^{im_lφ}
            
        Notes:
            - Returns complex values (wavefunctions are complex exponentials)
            - Can handle scalar phi or numpy array of angles
            - The wavefunction is periodic with period 2π: ψ(φ+2π) = ψ(φ)
            - The wavefunction is normalized: ∫|ψ(φ)|²dφ = 1 over [0,2π]
            - Different m_l wavefunctions are orthogonal to each other
            - These wavefunctions are eigenstates of the angular momentum operator L_z
              with eigenvalues m_l·ℏ
            
        Raises:
            ValueError: If m_l is not an integer
        """
        if not isinstance(m_l, (int, np.integer)):
            raise ValueError("Angular momentum quantum number m_l must be an integer.")
        return (1/np.sqrt(2*np.pi)) * np.exp(1j * m_l * phi)

    def probability_density(self, m_l: int, phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the probability density |ψ(φ)|² at angle phi.
        
        The probability density gives the probability per unit angle of finding
        the particle at angle phi.
        
        Parameters:
            m_l: Angular momentum quantum number (integer)
            phi: Angle or array of angles (in radians)
            
        Returns:
            Probability density |ψ_{m_l}(φ)|² = 1/(2π) for all phi
            
        Notes:
            - For rotational motion in 2D, the probability density is uniform
              across all angles, regardless of the quantum number m_l
            - This reflects the rotational symmetry of the system
            - The probability density integrates to 1 over the full circle:
              ∫|ψ(φ)|²dφ = 1 over [0,2π]
        
        Raises:
            ValueError: If m_l is not an integer
        """
        if not isinstance(m_l, (int, np.integer)):
            raise ValueError("Angular momentum quantum number m_l must be an integer.")
            
        # The probability density is constant for all phi
        if np.isscalar(phi):
            return 1.0 / (2 * np.pi)
        else:
            return np.full_like(phi, 1.0 / (2 * np.pi), dtype=float)
    
    def angular_momentum(self, m_l: int) -> float:
        """
        Calculate the z-component of angular momentum for a given quantum state.
        
        Parameters:
            m_l: Angular momentum quantum number (integer)
            
        Returns:
            Angular momentum L_z = m_l·ℏ
            
        Notes:
            - This is an eigenvalue of the angular momentum operator L_z = -iℏ·∂/∂φ
            - The angular momentum is quantized in units of ℏ
        
        Raises:
            ValueError: If m_l is not an integer
        """
        if not isinstance(m_l, (int, np.integer)):
            raise ValueError("Angular momentum quantum number m_l must be an integer.")
        return m_l * self.hbar
        
    def spectrum(self, m_max: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energy eigenvalues for angular momentum quantum numbers from -m_max to m_max.
        
        Parameters:
            m_max: Maximum absolute value of angular momentum quantum number (non-negative integer)
            
        Returns:
            Tuple of (quantum_numbers, energies) as numpy arrays
            
        Notes:
            - The spectrum includes both positive and negative m_l values
            - Energy levels are degenerate for ±m_l pairs (except m_l=0)
            - The energy increases quadratically with |m_l|
            
        Raises:
            ValueError: If m_max is not a non-negative integer
        """
        if not isinstance(m_max, (int, np.integer)) or m_max < 0:
            raise ValueError("m_max must be a non-negative integer.")
            
        ms = np.arange(-m_max, m_max+1)
        Es = np.array([self.energy(m) for m in ms])
        return ms, Es


class TunnelingBarrier:
    """
    Quantum tunneling through a 1D rectangular potential barrier.
    
    The quantum system represents a particle encountering a rectangular potential
    barrier of height V0 and width a. This class calculates tunneling probabilities
    using the WKB (Wentzel-Kramers-Brillouin) approximation.
    
    Parameters:
        V0: Height of the barrier (positive number)
        a: Width of the barrier (positive number)
        m_eff: Effective mass of the particle (positive number)
        hbar: Reduced Planck's constant (h/2π) (positive number, default=1.0)
        
    Notes:
        - For E < V0: Transmission coefficient T ≈ exp(-2κa)
        - For E ≥ V0: Transmission coefficient T = 1
        - Where κ = √(2m(V0-E))/ℏ is the wave vector in the barrier region
        - This model demonstrates the quantum tunneling effect where particles
          can penetrate barriers that would be impossible in classical mechanics
        - The transmission probability decreases exponentially with barrier width
          and with the square root of the barrier height
    """
    def __init__(self, V0: float, a: float, m_eff: float, hbar: float = 1.0):
        if V0 <= 0 or a <= 0 or m_eff <= 0 or hbar <= 0:
            raise ValueError("Barrier height, width, mass, and ℏ must be positive.")
        self.V0 = float(V0)
        self.a = float(a)
        self.m = float(m_eff)
        self.hbar = float(hbar)

    def kappa(self, E: float) -> float:
        """
        Calculate the wave vector κ in the barrier region.
        
        Parameters:
            E: Energy of the incident particle (non-negative number)
            
        Returns:
            Wave vector κ = √(2m(V0-E))/ℏ for E < V0, or 0 for E ≥ V0
            
        Notes:
            - For E < V0, κ is real and represents the exponential decay rate inside the barrier
            - For E ≥ V0, κ would be imaginary in the full quantum solution, but we return 0
              as we're only using it for tunneling calculations
            - The wave function inside the barrier has the form ψ(x) ∝ e^{-κx} for E < V0
            
        Raises:
            ValueError: If E is negative
        """
        if E < 0:
            raise ValueError("Energy must be non-negative.")
        if E >= self.V0:
            return 0.0
        return np.sqrt(2 * self.m * (self.V0 - E)) / self.hbar

    def transmission(self, E: float) -> float:
        """
        Calculate the transmission coefficient through the barrier.
        
        Parameters:
            E: Energy of the incident particle (non-negative number)
            
        Returns:
            Transmission coefficient:
              - T = exp(-2κa) for E < V0 (tunneling regime)
              - T = 1 for E ≥ V0 (classical regime)
            
        Notes:
            - This uses the WKB approximation for tunneling
            - For E < V0, the particle quantum tunnels through the barrier with
              probability less than 1
            - For E ≥ V0, the particle classically passes over the barrier with
              probability 1 (reflection is neglected in this simple model)
            - The transmission coefficient decreases exponentially with barrier width
              and with the square root of the effective barrier height (V0-E)
            
        Raises:
            ValueError: If E is negative
        """
        if E < 0:
            raise ValueError("Energy must be non-negative.")
        if E >= self.V0:
            return 1.0
        κ = self.kappa(E)
        return np.exp(-2 * κ * self.a)
        
    def reflection(self, E: float) -> float:
        """
        Calculate the reflection coefficient from the barrier.
        
        Parameters:
            E: Energy of the incident particle (non-negative number)
            
        Returns:
            Reflection coefficient R = 1 - T
            
        Notes:
            - By conservation of probability, R + T = 1
            - For E < V0, most particles are reflected (R close to 1)
            - For E ≥ V0, no particles are reflected in this simple model (R = 0)
            
        Raises:
            ValueError: If E is negative
        """
        if E < 0:
            raise ValueError("Energy must be non-negative.")
        return 1.0 - self.transmission(E)
        
    def tunneling_current(self, E: float, incident_flux: float) -> float:
        """
        Calculate the tunneling current (transmitted flux) through the barrier.
        
        Parameters:
            E: Energy of the incident particle (non-negative number)
            incident_flux: Flux of incident particles (particles per unit time)
            
        Returns:
            Transmitted flux = incident_flux × transmission_coefficient
            
        Notes:
            - This is useful for modeling quantum devices like tunnel diodes,
              scanning tunneling microscopes, and resonant tunneling structures
            - The tunneling current is directly proportional to the transmission probability
            
        Raises:
            ValueError: If E is negative or incident_flux is not positive
        """
        if incident_flux <= 0:
            raise ValueError("Incident flux must be positive.")
        return incident_flux * self.transmission(E)
        
    def barrier_penetration(self, E: float) -> float:
        """
        Calculate the characteristic penetration depth into the barrier.
        
        Parameters:
            E: Energy of the incident particle (non-negative number, E < V0)
            
        Returns:
            Penetration depth d = 1/κ = ℏ/√(2m(V0-E))
            
        Notes:
            - This represents how far the wavefunction extends into the barrier
              before decaying significantly
            - The penetration depth increases as E approaches V0
            - For E ≥ V0, the concept of penetration depth doesn't apply
              (returns float('inf'))
            
        Raises:
            ValueError: If E is negative
        """
        if E < 0:
            raise ValueError("Energy must be non-negative.")
        if E >= self.V0:
            return float('inf')
        
        κ = self.kappa(E)
        return 1.0 / κ
        
    def transmission_spectrum(self, E_range: np.ndarray) -> np.ndarray:
        """
        Calculate the transmission spectrum over a range of energies.
        
        Parameters:
            E_range: Array of energy values (non-negative numbers)
            
        Returns:
            Array of transmission coefficients corresponding to each energy
            
        Notes:
            - Useful for visualizing how tunneling probability varies with energy
            - Shows the characteristic exponential increase in transmission as energy approaches V0
            - For educational purposes, can demonstrate quantum vs. classical behavior
            
        Example:
            ```python
            import numpy as np
            import matplotlib.pyplot as plt
            
            barrier = TunnelingBarrier(V0=1.0, a=2.0, m_eff=1.0)
            energies = np.linspace(0, 1.5, 100)  # Energies from 0 to 1.5*V0
            transmission = barrier.transmission_spectrum(energies)
            
            plt.figure(figsize=(8, 5))
            plt.plot(energies, transmission)
            plt.axvline(x=1.0, color='r', linestyle='--', label='V0')
            plt.xlabel('Energy (E/V0)')
            plt.ylabel('Transmission Coefficient')
            plt.title('Quantum Tunneling: Transmission vs. Energy')
            plt.legend()
            plt.grid(True)
            plt.show()
            ```
            
        Raises:
            ValueError: If any energy value is negative
        """
        if np.any(E_range < 0):
            raise ValueError("All energy values must be non-negative.")
        
        transmission_values = np.zeros_like(E_range, dtype=float)
        
        # Vectorized calculation
        tunneling_mask = E_range < self.V0
        classical_mask = ~tunneling_mask
        
        # Calculate for tunneling regime (E < V0)
        if np.any(tunneling_mask):
            κ_values = np.sqrt(2 * self.m * (self.V0 - E_range[tunneling_mask])) / self.hbar
            transmission_values[tunneling_mask] = np.exp(-2 * κ_values * self.a)
        
        # Set to 1 for classical regime (E ≥ V0)
        if np.any(classical_mask):
            transmission_values[classical_mask] = 1.0
            
        return transmission_values
        
    def exact_transmission(self, E: float) -> float:
        """
        Calculate the exact transmission coefficient through a rectangular barrier.
        
        This method uses the full quantum solution rather than the WKB approximation,
        accounting for wave-like behavior including reflection and interference.
        
        Parameters:
            E: Energy of the incident particle (non-negative number)
            
        Returns:
            Exact transmission coefficient calculated from the full quantum solution
            
        Notes:
            - This is more accurate than the WKB approximation, especially for thin barriers
            - For E < V0, includes interference effects not captured by simple exponential decay
            - For E ≥ V0, includes quantum reflection which the simple model neglects
            - The formula is derived by matching boundary conditions for the wavefunction
            - Mathematical form:
              T = [1 + (V0²sinh²(κa))/(4E(V0-E))]⁻¹ for E < V0
              T = [1 + (V0²sin²(ka))/(4E(V0-E))]⁻¹ for E > V0
              Where k = √(2mE)/ℏ and κ = √(2m(V0-E))/ℏ
            
        Raises:
            ValueError: If E is negative
        """
        if E < 0:
            raise ValueError("Energy must be non-negative.")
            
        # Special case: resonance at E = V0
        if np.isclose(E, self.V0):
            return 1.0 / (1.0 + (self.V0 * self.a)**2 / 4)
            
        # Wave vectors
        k = np.sqrt(2 * self.m * E) / self.hbar  # Outside barrier
        
        if E < self.V0:
            # Tunneling regime
            κ = self.kappa(E)
            Z = (self.V0**2 * np.sinh(κ * self.a)**2) / (4 * E * (self.V0 - E))
            return 1.0 / (1.0 + Z)
        else:
            # Over-barrier regime
            k_inside = np.sqrt(2 * self.m * (E - self.V0)) / self.hbar  # Inside barrier
            Z = (self.V0**2 * np.sin(k_inside * self.a)**2) / (4 * E * (E - self.V0))
            return 1.0 / (1.0 + Z)
        
    def comparison_plot(self, E_max: float, num_points: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate data for comparing WKB and exact transmission coefficients.
        
        Parameters:
            E_max: Maximum energy value (typically 1.5-2 times V0)
            num_points: Number of energy points to calculate
            
        Returns:
            Dictionary with keys 'energies', 'wkb', and 'exact' containing numpy arrays
            
        Notes:
            - Useful for educational demonstrations of the accuracy of the WKB approximation
            - Shows where WKB breaks down (typically for thin barriers or energies near V0)
            - The returned dictionary can be used directly for plotting
            
        Example:
            ```python
            import matplotlib.pyplot as plt
            
            barrier = TunnelingBarrier(V0=1.0, a=2.0, m_eff=1.0)
            data = barrier.comparison_plot(E_max=1.5)
            
            plt.figure(figsize=(10, 6))
            plt.plot(data['energies'], data['wkb'], 'b-', label='WKB Approximation')
            plt.plot(data['energies'], data['exact'], 'r--', label='Exact Solution')
            plt.axvline(x=barrier.V0, color='k', linestyle=':', label='Barrier Height (V0)')
            plt.xlabel('Energy')
            plt.ylabel('Transmission Coefficient')
            plt.title('Quantum Tunneling: WKB vs. Exact Solution')
            plt.legend()
            plt.grid(True)
            plt.show()
            ```
        """
        energies = np.linspace(0.01, E_max, num_points)  # Avoid E=0 for numerical stability
        
        # Calculate both methods
        wkb_values = self.transmission_spectrum(energies)
        exact_values = np.array([self.exact_transmission(E) for E in energies])
        
        return {
            'energies': energies,
            'wkb': wkb_values,
            'exact': exact_values
        }


class PotentialEnergy:
    """
    Utility class for creating common potential energy profiles for quantum systems.
    
    This class provides static methods to generate various potential energy profiles
    commonly used in quantum mechanics problems.
    
    Available potentials:
    - Infinite square well
    - Finite square well
    - Harmonic oscillator
    - Step potential
    - Rectangular barrier
    - Double well
    - Morse potential
    - Kronig-Penney model
    """
    
    @staticmethod
    def infinite_square_well(x: np.ndarray, L: float) -> np.ndarray:
        """
        Generate an infinite square well potential (particle in a box).
        
        Parameters:
            x: Array of position values
            L: Width of the well (positive number)
            
        Returns:
            Array of potential values (0 inside the well, np.inf outside)
            
        Notes:
            - This potential represents a particle confined to a region 0 ≤ x ≤ L
            - The eigenvalues (energy levels) are E_n = (n²π²ℏ²)/(2mL²)
            - The eigenfunctions (wavefunctions) are ψ_n(x) = √(2/L)sin(nπx/L)
            - This is the simplest quantum confinement model with analytical solutions
            - Used extensively in quantum chemistry as a first approximation for conjugated systems
            
        Raises:
            ValueError: If L is not positive
        """
        if L <= 0:
            raise ValueError("Well width must be positive.")
        V = np.full_like(x, np.inf, dtype=float)
        mask = (x >= 0) & (x <= L)
        V[mask] = 0.0
        return V
    
    @staticmethod
    def finite_square_well(x: np.ndarray, L: float, V0: float) -> np.ndarray:
        """
        Generate a finite square well potential (quantum well).
        
        Parameters:
            x: Array of position values
            L: Width of the well (positive number)
            V0: Height of the potential outside the well (positive number)
            
        Returns:
            Array of potential values (0 inside, V0 outside)
            
        Notes:
            - This potential represents a quantum well with finite barriers
            - Unlike the infinite well, the wavefunction penetrates into the barriers
            - The energy eigenvalues must be found numerically by solving a transcendental equation
            - The number of bound states depends on the well parameters (L and V0)
            - For a very deep well (V0 → ∞), the solutions approach those of the infinite well
            - Finite wells are fundamental in semiconductor physics (quantum wells, heterostructures)
            - Applications include LEDs, lasers, and quantum computing devices
            
        Raises:
            ValueError: If L or V0 are not positive
        """
        if L <= 0 or V0 <= 0:
            raise ValueError("Well width and height must be positive.")
        V = np.full_like(x, V0, dtype=float)
        mask = (x >= 0) & (x <= L)
        V[mask] = 0.0
        return V
    
    @staticmethod
    def harmonic_oscillator(x: np.ndarray, k: float, x0: float = 0.0) -> np.ndarray:
        """
        Generate a harmonic oscillator potential.
        
        Parameters:
            x: Array of position values
            k: Force constant (spring constant) (positive number)
            x0: Equilibrium position (default = 0.0)
            
        Returns:
            Array of potential values V(x) = (1/2)k(x-x0)²
            
        Notes:
            - This is the quantum version of a spring oscillator (Hooke's law)
            - The energy eigenvalues are E_n = ℏω(n + 1/2), where ω = √(k/m)
            - The eigenfunctions are Hermite polynomials multiplied by a Gaussian
            - One of the few exactly solvable problems in quantum mechanics
            - Foundation for understanding molecular vibrations and phonons in solids
            - In chemistry, used to model molecular vibrations near equilibrium
            - Also serves as approximation for any potential at its minimum (Taylor expansion)
            
        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            raise ValueError("Force constant must be positive.")
        return 0.5 * k * (x - x0)**2
    
    @staticmethod
    def double_well(x: np.ndarray, a: float, b: float, V0: float) -> np.ndarray:
        """
        Generate a double well potential.
        
        Parameters:
            x: Array of position values
            a: Distance between wells (positive number)
            b: Width of each well (positive number)
            V0: Height of the barrier between wells (positive number)
            
        Returns:
            Array of potential values (double well profile)
            
        Notes:
            - This potential features two minima separated by a barrier
            - The system exhibits quantum tunneling between the wells
            - For low barriers, the ground state wavefunction spreads across both wells
            - For high barriers, nearly degenerate pairs of states exist 
              (symmetric and antisymmetric combinations)
            - Describes phenomena like ammonia inversion and proton transfer
            - Models molecular conformational changes and chemical reactions
            - Important in studying quantum coherence and decoherence
            
        Raises:
            ValueError: If a, b, or V0 are not positive
        """
        if a <= 0 or b <= 0 or V0 <= 0:
            raise ValueError("Distance, width, and height must be positive.")
        V = np.full_like(x, V0, dtype=float)
        well1 = (x >= -a/2 - b/2) & (x <= -a/2 + b/2)
        well2 = (x >= a/2 - b/2) & (x <= a/2 + b/2)
        V[well1 | well2] = 0.0
        return V
    
    @staticmethod
    def morse_potential(x: np.ndarray, D: float, a: float, r0: float = 0.0) -> np.ndarray:
        """
        Generate a Morse potential for molecular vibration.
        
        Parameters:
            x: Array of position values (representing nuclear separation)
            D: Dissociation energy (positive number)
            a: Controls width of potential well (positive number)
            r0: Equilibrium bond length (default = 0.0)
            
        Returns:
            Array of potential values V(r) = D(1-exp(-a(r-r0)))²
            
        Notes:
            - Realistic model for diatomic molecular vibrations that includes anharmonicity
            - Unlike harmonic oscillator, correctly predicts molecular dissociation
            - Energy eigenvalues: E_n = ℏω(n+1/2) - ℏω(n+1/2)²/(4D)
            - Has a finite number of bound states (unlike harmonic oscillator)
            - Parameter a relates to force constant k by a = √(k/2D)
            - Widely used in spectroscopy to model vibrational spectra
            - Can be extended to polyatomic molecules
            - Shows vibrational energy level spacing decreases with n (anharmonicity)
            
        Raises:
            ValueError: If D or a are not positive
            
        Example:
            ```python
            # Modeling H2 molecule
            x = np.linspace(-0.5, 3.0, 1000)  # Distance in Ångstroms
            D = 4.52  # eV
            a = 1.95  # Å⁻¹
            r0 = 0.74  # Å
            V = PotentialEnergy.morse_potential(x, D, a, r0)
            ```
        """
        if D <= 0 or a <= 0:
            raise ValueError("Dissociation energy and width parameter must be positive.")
        return D * (1 - np.exp(-a * (x - r0)))**2
        
    @staticmethod
    def step_potential(x: np.ndarray, V0: float, x0: float = 0.0) -> np.ndarray:
        """
        Generate a step potential.
        
        Parameters:
            x: Array of position values
            V0: Height of the step (positive number)
            x0: Position of the step (default = 0.0)
            
        Returns:
            Array of potential values (0 for x < x0, V0 for x ≥ x0)
            
        Notes:
            - One of the simplest problems exhibiting quantum reflection and transmission
            - For E < V0, classical particles are completely reflected
            - For E < V0, quantum particles have some probability of transmission (tunneling)
            - For E > V0, quantum particles show both reflection and transmission
            - The transmission coefficient depends on energy: T = 4k₁k₂/((k₁+k₂)²)
              where k₁ = √(2mE)/ℏ and k₂ = √(2m(E-V0))/ℏ
            - Useful for modeling heterojunctions in semiconductors
            
        Raises:
            ValueError: If V0 is not positive
        """
        if V0 <= 0:
            raise ValueError("Step height must be positive.")
        V = np.zeros_like(x, dtype=float)
        V[x >= x0] = V0
        return V
        
    @staticmethod
    def rectangular_barrier(x: np.ndarray, V0: float, a: float, x0: float = 0.0) -> np.ndarray:
        """
        Generate a rectangular barrier potential.
        
        Parameters:
            x: Array of position values
            V0: Height of the barrier (positive number)
            a: Width of the barrier (positive number)
            x0: Position of the barrier center (default = 0.0)
            
        Returns:
            Array of potential values (V0 inside the barrier, 0 elsewhere)
            
        Notes:
            - The canonical system for demonstrating quantum tunneling
            - The transmission coefficient for E < V0 is given by:
              T = [1 + (V0²sinh²(κa))/(4E(V0-E))]⁻¹
            - For thin barriers (κa < 1), T ≈ exp(-2κa)
            - Models electron tunneling in STM, alpha decay, and tunnel diodes
            
        Raises:
            ValueError: If V0 or a are not positive
        """
        if V0 <= 0 or a <= 0:
            raise ValueError("Barrier height and width must be positive.")
        V = np.zeros_like(x, dtype=float)
        barrier = (x >= x0 - a/2) & (x <= x0 + a/2)
        V[barrier] = V0
        return V
        
    @staticmethod
    def kronig_penney(x: np.ndarray, V0: float, a: float, b: float) -> np.ndarray:
        """
        Generate a Kronig-Penney periodic potential.
        
        Parameters:
            x: Array of position values
            V0: Height of the barriers (positive number)
            a: Width of each barrier (positive number)
            b: Distance between barriers (positive number)
            
        Returns:
            Array of potential values (periodic barriers of height V0)
            
        Notes:
            - A model for electrons in a periodic crystal lattice
            - Demonstrates formation of energy bands and band gaps
            - The period of the potential is (a+b)
            - The band structure depends on the parameters:
              cos(k(a+b)) = cos(αa)cosh(βb) - (α²+β²)/(2αβ)sin(αa)sinh(βb)
              where α = √(2mE)/ℏ and β = √(2m(V0-E))/ℏ
            - Important for understanding electronic properties of solids
            - Shows why solids have allowed and forbidden energy bands
            
        Raises:
            ValueError: If V0, a, or b are not positive
        """
        if V0 <= 0 or a <= 0 or b <= 0:
            raise ValueError("Barrier height, width, and separation must be positive.")
        
        period = a + b
        V = np.zeros_like(x, dtype=float)
        
        # Create one period and then repeat
        for i in range(len(x)):
            # Map to first period
            x_mod = x[i] % period
            if x_mod < a:
                V[i] = V0
        
        return V

    @staticmethod
    def calculate_eigenstates(x: np.ndarray, V: np.ndarray, m: float = 1.0, 
                             hbar: float = 1.0, num_states: int = 5) -> Dict[str, np.ndarray]:
        """
        Calculate energy eigenstates for a given potential using the finite difference method.
        
        Parameters:
            x: Array of position values (uniform grid)
            V: Potential energy at each position
            m: Mass of the particle (positive number, default = 1.0)
            hbar: Reduced Planck's constant (positive number, default = 1.0)
            num_states: Number of lowest eigenstates to calculate (default = 5)
            
        Returns:
            Dictionary containing:
            - 'energies': Array of eigenvalues (energies)
            - 'wavefunctions': 2D array where each row is an eigenfunction
            
        Notes:
            - Uses the finite difference method to discretize the Hamiltonian
            - The Schrödinger equation becomes a matrix eigenvalue problem
            - Useful for potentials without analytical solutions
            - Accuracy depends on the grid spacing (finer grid = better accuracy)
            - Returns the lowest `num_states` eigenstates
            - Wavefunctions are normalized such that ∫|ψ|² dx = 1
            
        Raises:
            ValueError: If m or hbar are not positive or if x is not uniformly spaced
            
        Example:
            ```python
            # Solve for harmonic oscillator eigenstates
            x = np.linspace(-5, 5, 1000)
            V = PotentialEnergy.harmonic_oscillator(x, k=1.0)
            result = PotentialEnergy.calculate_eigenstates(x, V, num_states=4)
            
            # Plot results
            plt.figure(figsize=(10, 8))
            for i in range(4):
                plt.plot(x, result['wavefunctions'][i] + result['energies'][i], 
                         label=f'E{i} = {result["energies"][i]:.3f}')
            plt.plot(x, V, 'k--', label='Potential')
            plt.legend()
            plt.grid(True)
            plt.xlabel('Position')
            plt.ylabel('Energy / Wavefunction')
            plt.title('Harmonic Oscillator Eigenstates')
            plt.show()
            ```
        """
        if m <= 0 or hbar <= 0:
            raise ValueError("Mass and ℏ must be positive.")
            
        # Check for uniform grid
        dx = x[1] - x[0]
        if not np.allclose(np.diff(x), dx):
            raise ValueError("Position grid must be uniformly spaced.")
            
        n_points = len(x)
        
        # Construct the Hamiltonian matrix using finite difference for kinetic energy
        H = np.zeros((n_points, n_points))
        
        # Diagonal terms: potential energy + kinetic energy term
        coeff = -hbar**2 / (2 * m * dx**2)
        H[np.arange(n_points), np.arange(n_points)] = V + 2 * coeff
        
        # Off-diagonal terms: kinetic energy
        for i in range(n_points-1):
            H[i, i+1] = -coeff
            H[i+1, i] = -coeff
            
        # Solve the eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Sort by eigenvalue and take the lowest num_states
        idx = np.argsort(eigenvalues)[:num_states]
        energies = eigenvalues[idx]
        wavefunctions = eigenvectors[:, idx].T  # Each row is a wavefunction
        
        # Normalize the wavefunctions
        for i in range(len(wavefunctions)):
            # Normalization: ∫|ψ|² dx = 1
            norm = np.sqrt(np.trapz(np.abs(wavefunctions[i])**2, x))
            wavefunctions[i] /= norm
            
        return {
            'energies': energies,
            'wavefunctions': wavefunctions
        }
        
    @staticmethod
    def expectation_value(x: np.ndarray, psi: np.ndarray, 
                         observable: Optional[Union[np.ndarray, callable]] = None) -> float:
        """
        Calculate the expectation value of an observable for a given wavefunction.
        
        Parameters:
            x: Array of position values (uniform grid)
            psi: Wavefunction values at each position
            observable: Either a function f(x) or array of values representing the observable
                       If None, calculates <x> (position expectation value)
            
        Returns:
            Expectation value <ψ|O|ψ>
            
        Notes:
            - Uses the definition <O> = ∫ψ*(x)O(x)ψ(x)dx
            - For position expectation: <x> = ∫ψ*(x)xψ(x)dx
            - For momentum expectation, use the momentum operator -iℏ d/dx
            - Wavefunction should be normalized (∫|ψ|²dx = 1)
            
        Example:
            ```python
            # Calculate position expectation value
            x = np.linspace(-5, 5, 1000)
            psi = np.exp(-(x-1.0)**2)  # Gaussian centered at x=1.0
            psi /= np.sqrt(np.trapz(np.abs(psi)**2, x))  # Normalize
            
            # Position expectation
            x_expect = PotentialEnergy.expectation_value(x, psi)
            
            # Potential energy expectation for harmonic oscillator
            V = 0.5 * x**2  # V(x) = 1/2 kx²
            V_expect = PotentialEnergy.expectation_value(x, psi, V)
            
            # Kinetic energy expectation using finite difference for d²ψ/dx²
            def kinetic_energy(psi, x, m=1.0, hbar=1.0):
                dx = x[1] - x[0]
                d2psi = np.zeros_like(psi)
                # Second derivative using central difference
                d2psi[1:-1] = (psi[:-2] - 2*psi[1:-1] + psi[2:]) / dx**2
                return -0.5 * hbar**2 / m * d2psi
                
            T_values = kinetic_energy(psi, x)
            T_expect = PotentialEnergy.expectation_value(x, psi, T_values)
            ```
        """
        # Ensure the array is normalized
        norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
        if not np.isclose(norm, 1.0, rtol=1e-3):
            warnings.warn("Wavefunction not normalized, normalizing automatically.")
            psi = psi / norm
            
        if observable is None:
            # Position expectation value <x>
            observable = x
        elif callable(observable):
            # If a function is provided, evaluate it
            observable = observable(x)
            
        # Calculate <ψ|O|ψ>
        integrand = np.conj(psi) * observable * psi
        return np.real(np.trapz(integrand, x))
