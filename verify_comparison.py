import numpy as np
import psi_solve2.functions as f
import sys

# Try importing qmsolve
try:
    from qmsolve import Hamiltonian, SingleParticle, init_visualization
except ImportError:
    print("Error: qmsolve not found. Please install it via 'pip install qmsolve'")
    sys.exit(1)

def run_comparison():
    with open("comparison_log.txt", "w") as log_file:
        sys.stdout = log_file
        print("========================================")
        print("CROSS-VERIFICATION: PSI_SOLVE2 vs QMSOLVE")
        print("========================================")

        # ---------------------------------------------------------
        # CASE: Double Well Potential
        # V(x) = depth * ( (x-center)**2 - separation )**2
        # ---------------------------------------------------------
        print("\n[TEST CASE] Double Well Potential")
        
        # Parameters
        L = 10.0
        N = 512 # QMSolve default is often 512 or similar, let's match
        depth = 2.0
        separation = 1.0
        center = 0.0
        m_particle = 1.0
        
        print(f"Parameters: L={L}, N={N}, depth={depth}, separation={separation}, m={m_particle}")

        # ---------------------------------------------------------
        # 1. Run psi_solve2
        # ---------------------------------------------------------
        print("\n--- Running psi_solve2 ---")
        x_full, dx, x_internal = f.make_grid(L=L, N=N)
        
        # Construct Potential
        # Note: functions.V_double_well signature: (x, depth=20, separation=1, center=0.0)
        V_internal = f.V_double_well(x_internal, depth=depth, separation=separation, center=center)
        
        # Pad for solver
        V_full = np.zeros_like(x_full)
        V_full[1:-1] = V_internal
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = f.kinetic_operator(N, dx, m=m_particle)
        E_psi, psi_psi = f.solve(T, V_full, dx)
        
        print(f"psi_solve2 Energies (first 5): {E_psi[:5]}")

        # ---------------------------------------------------------
        # 2. Run QMSolve
        # ---------------------------------------------------------
        print("\n--- Running QMSolve ---")
        
        # Define potential function for QMSolve
        def double_well(particle):
            x = particle.x
            return depth * ( (x - center)**2 - separation )**2

        # Setup QMSolve
        H = Hamiltonian(particles = SingleParticle(m = m_particle), 
                        potential = double_well, 
                        spatial_ndim = 1, N = N, extent = L)

        # Diagonalize
        eigenstates = H.solve(max_states = 10)
        E_qm_eV = eigenstates.energies
        
        # Convert QMSolve (eV) to Hartree
        # 1 Hartree = 27.211386 eV
        Hartree_to_eV = 27.211386
        E_qm = E_qm_eV / Hartree_to_eV

        print(f"QMSolve Energies (eV):      {E_qm_eV[:5]}")
        print(f"QMSolve Energies (Hartree): {E_qm[:5]}")

        # ---------------------------------------------------------
        # 3. Compare
        # ---------------------------------------------------------
        print("\n--- Comparison Results ---")
        print("-" * 65)
        print(f"| n | psi_solve2 E | QMSolve E    | Diff         | % Diff   |")
        print("-" * 65)
        
        for i in range(5):
            e1 = E_psi[i]
            e2 = E_qm[i]
            diff = abs(e1 - e2)
            p_diff = (diff / e2) * 100 if e2 != 0 else 0.0
            
            print(f"| {i:<1} | {e1:<12.6f} | {e2:<12.6f} | {diff:<12.2e} | {p_diff:<7.4f}% |")
        print("-" * 65)
        
        # ---------------------------------------------------------
        # DEBUG CASE: Harmonic Oscillator
        # ---------------------------------------------------------
        print("\n[DEBUG CASE] Harmonic Oscillator (k=1)")
        k_debug = 1.0
        
        # psi_solve2
        V_internal_HO = 0.5 * k_debug * x_internal**2
        V_full_HO = np.zeros_like(x_full)
        V_full_HO[1:-1] = V_internal_HO
        V_full_HO[0] = 1e10
        V_full_HO[-1] = 1e10
        
        E_psi_HO, _ = f.solve(T, V_full_HO, dx)
        print(f"psi_solve2 HO Energies: {E_psi_HO[:5]}")
        
        # QMSolve
        def harmonic_potential(particle):
            return 0.5 * k_debug * particle.x**2
            
        H_HO = Hamiltonian(particles = SingleParticle(m = m_particle), 
                        potential = harmonic_potential, 
                        spatial_ndim = 1, N = N, extent = L)
        eigenstates_HO = H_HO.solve(max_states = 10)
        E_qm_HO = eigenstates_HO.energies
        print(f"QMSolve HO Energies:    {E_qm_HO[:5]}")
        
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    run_comparison()
