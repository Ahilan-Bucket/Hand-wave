import numpy as np
from functions import make_grid, kinetic_operator, solve, inf_square_well, check_ISW_analytic

def reproduce_isw_error():
    print("--- Reproducing ISW Verification Error ---")
    
    # 1. Setup Grid
    L = 10.0
    N = 500
    x, dx, x_int = make_grid(L, N)
    
    # 2. Create Potential (Infinite Square Well)
    # Case A: Well width = Grid width (L=10)
    V_full = inf_square_well(x_int, -L/2, L/2)
    
    # Pad for solver (solver expects full grid V)
    # Note: inf_square_well returns V for x_int if passed x_int.
    # We need to construct V_full carefully as in the notebook.
    V_solver = np.zeros_like(x)
    V_solver[1:-1] = V_full
    V_solver[0] = 1e10
    V_solver[-1] = 1e10
    
    # 3. Solve
    T = kinetic_operator(len(x_int), dx)
    E_sim, psi_sim = solve(T, V_solver, dx)
    
    print(f"Simulated Energies (first 5): {E_sim[:5]}")
    
    # 4. Run Verification
    # This is where we expect it to fail or print wrong values
    print("\nRunning check_ISW_analytic...")
    # Now we pass L explicitly or bounds correctly
    check_ISW_analytic(E_sim, L=L)

if __name__ == "__main__":
    reproduce_isw_error()
