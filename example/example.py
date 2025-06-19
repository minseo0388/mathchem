# 대충 예시 긁어와서 맞게 수정만 한 것입니다.
# Try free

import numpy as np
from src.huckelapprox import Huckel

# Adjacency matrix for the benzene molecule (C6H6)
adj_benzene = [
    [0,1,0,0,0,1],
    [1,0,1,0,0,0],
    [0,1,0,1,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1],
    [1,0,0,0,1,0],
]

if __name__ == "__main__":
    mol = Huckel(adj_benzene)  # common alpha=0.0, common beta=-1.0
    print("Basic parameter calculation")
    energies, _ = mol.solve()
    print("MO Energies:", np.round(energies, 4))
    print("Total π energy:", np.round(mol.total_pi_energy(6), 4))

    # Change parameters to realistic values
    mol.set_params(alpha=-11.0, beta=-2.5)
    print("\nCalculation after substituting actual values (unit: eV)")
    energies2, _ = mol.solve()
    print("MO Energies:", np.round(energies2, 4))
    print("Total π energy:", np.round(mol.total_pi_energy(6), 4))


from src.particle import ParticleInABox1D, ParticleInABox2D, Rotational2D, TunnelingBarrier
import numpy as np

if __name__ == "__main__":
    # 1D box
    box1 = ParticleInABox1D(L=1.0, m_eff=1.0)
    ns, Es1 = box1.spectrum(5)
    print("1D n:", ns)
    print("1D Energies:", np.round(Es1,4))

    # 2D box
    box2 = ParticleInABox2D(Lx=1.0, Ly=2.0, m_eff=1.0)
    levels2 = box2.spectrum(3,3)
    print("2D levels (nx,ny,E):", [(nx,ny,round(E,4)) for nx,ny,E in levels2])

    # 2D rotation
    rot = Rotational2D(r=1.0, m_eff=1.0)
    ms, Er = rot.spectrum(3)
    print("Rot m:", ms)
    print("Rot Energies:", np.round(Er,4))

    # Tunneling barrier
    barrier = TunnelingBarrier(V0=5.0, a=1.0, m_eff=1.0)
    for E in [1.0,3.0,5.0,7.0]:
        print(f"E={E}, T=", round(barrier.transmission(E),4))
