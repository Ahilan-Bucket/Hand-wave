import numpy as np
from functions import make_grid, kinetic_operator, solve, harmonic, verify_solver_analytic, benchmark_user_potential

def test_new_benchmarks():
    print("--- Testing New Benchmarks ---")
    
    # 1. Setup Grid & Harmonic Potential
    L = 10.0
    N = 200
    x, dx, x_int = make_grid(L, N)
    k = 1.0
    V = harmonic(x, k=k)
    
    # 2. Solve with Hand-wave
    T = kinetic_operator(len(x), dx) # Note: harmonic uses full grid usually, let's check make_grid usage
    # make_grid returns x (full), dx, x_int (internal). 
    # kinetic_operator needs size of the vector it acts on.
    # If we solve on full grid (standard for harmonic in this code?), we need T size N.
    # Let's verify solve usage in functions.py or notebook.
    # Usually solve takes T and V. If V is size N, T must be size N.
    
    T = kinetic_operator(N, dx)
    E_sim, psi_sim = solve(T, V, dx)
    
    # 3. Test verify_solver_analytic
    print("\n[Test] verify_solver_analytic (Harmonic)...")
    verify_solver_analytic(E_sim, 'Harmonic', {'k': k})
    
    # 4. Test benchmark_user_potential (requires qmsolve)
    print("\n[Test] benchmark_user_potential...")
    benchmark_user_potential(V, x)

if __name__ == "__main__":
    test_new_benchmarks()
