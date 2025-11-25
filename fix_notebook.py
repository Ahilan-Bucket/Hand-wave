import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

def fix_and_expand_notebook():
    nb_path = r"d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave\ModularedPresentation_Educational.ipynb"
    
    print(f"Reading notebook: {nb_path}")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    # 1. Find the "Educational Analysis Playground" section
    insert_idx = -1
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and "Educational Analysis Playground" in cell.source:
            insert_idx = i
            break
            
    if insert_idx != -1:
        print("Found Educational Section. Inserting imports...")
        # Insert import cell BEFORE the first example
        import_cell = new_code_cell("""# Setup: Import necessary functions
import numpy as np
import matplotlib.pyplot as plt
from functions import (
    make_grid, 
    inf_square_well, 
    finite_square_well, 
    harmonic,
    kinetic_operator, 
    solve, 
    plot_educational,
    verify_solver_analytic,
    benchmark_user_potential
)
%load_ext autoreload
%autoreload 2
""")
        nb.cells.insert(insert_idx + 1, import_cell)
    else:
        print("Warning: Educational Section not found. Appending imports at end.")
        nb.cells.append(import_cell)

    # 2. Add Verification Demo Section
    print("Adding Verification Playground...")
    
    nb.cells.append(new_markdown_cell("""## 5. Verification Playground
    
Trust but verify! Here we check our solver's accuracy against:
1.  **Analytic Formulas**: For known potentials (Infinite Well, Harmonic Oscillator).
2.  **QMSolve**: A professional quantum solver package (for custom potentials).
"""))

    # Demo 1: Analytic Verification (ISW)
    nb.cells.append(new_markdown_cell("### 5.1 Analytic Verification: Infinite Square Well"))
    nb.cells.append(new_code_cell("""# 1. Setup Grid & Potential
L_verify = 10
x_v, dx_v, x_int_v = make_grid(L=L_verify, N=500)
V_isw = inf_square_well(x_int_v, -L_verify/2, L_verify/2)

# Pad for solver
V_full_isw = np.zeros_like(x_v)
V_full_isw[1:-1] = V_isw
V_full_isw[0] = 1e10; V_full_isw[-1] = 1e10

# 2. Solve
T_v = kinetic_operator(len(x_int_v), dx_v)
E_isw, psi_isw = solve(T_v, V_full_isw, dx_v)

# 3. Verify
# This compares our E_isw against the exact formula
verify_solver_analytic(E_isw, 'ISW', {'L': L_verify})
"""))

    # Demo 2: Analytic Verification (Harmonic)
    nb.cells.append(new_markdown_cell("### 5.2 Analytic Verification: Harmonic Oscillator"))
    nb.cells.append(new_code_cell("""# 1. Setup Harmonic Potential
k_spring = 1.0
V_harm = harmonic(x_v, k=k_spring)

# 2. Solve
# Note: Harmonic oscillator uses the full grid usually
T_harm = kinetic_operator(len(x_v), dx_v)
E_harm, psi_harm = solve(T_harm, V_harm, dx_v)

# 3. Verify
verify_solver_analytic(E_harm, 'Harmonic', {'k': k_spring})
"""))

    # Demo 3: QMSolve Benchmark
    nb.cells.append(new_markdown_cell("### 5.3 Benchmark vs QMSolve (Custom Potential)"))
    nb.cells.append(new_markdown_cell("We can define *any* crazy potential and see if QMSolve agrees with us."))
    nb.cells.append(new_code_cell("""# 1. Define a custom potential (e.g., a double well with a tilt)
def custom_V(x):
    return 0.5 * (x**2 - 2.5)**2 + 0.1 * x

V_custom = custom_V(x_v)

# 2. Solve with Hand-wave
E_custom, psi_custom = solve(T_harm, V_custom, dx_v)

# 3. Benchmark against QMSolve
# This feeds our V_custom array into QMSolve and compares eigenvalues
try:
    benchmark_user_potential(V_custom, x_v)
except ImportError:
    print("QMSolve is not installed. Run 'pip install qmsolve' to use this feature.")
"""))

    print(f"Saving updated notebook to: {nb_path}")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Done!")

if __name__ == "__main__":
    fix_and_expand_notebook()
