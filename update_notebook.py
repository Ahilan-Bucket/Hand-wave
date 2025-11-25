import json
import nbformat

def create_code_cell(source):
    return nbformat.v4.new_code_cell(source)

def create_markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source)

def update_notebook():
    input_path = r"d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave\ModularedPresentation.ipynb"
    output_path = r"d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\Hand-wave\ModularedPresentation_Educational.ipynb"

    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Find and replace plot_alive with plot_educational in the main solve cell
    # We look for the cell containing "from functions import plot_alive" and "solve(T, V, dx)"
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if "plot_alive" in cell.source and "solve" in cell.source:
                print("Found target cell. Updating to use plot_educational...")
                cell.source = """from functions import plot_educational, solve

# Solve the Schrödinger Equation
E, psi = solve(T, V, dx)

# Use the new Educational Plotter
# This will analyze the potential and the state to tell you what's happening!
plot_educational(E, psi, V, x, no=0);"""
                break

    # 2. Add Educational Playground Section
    print("Adding Educational Analysis Playground...")
    
    new_cells = []
    
    new_cells.append(create_markdown_cell("""## 4. Educational Analysis Playground

Here we demonstrate the new **Educational Features**. The solver now explains *why* the results look the way they do.

### Case 1: "Why is my potential flat?"
If you set the well width equal to the grid width, the walls are pushed to the very edge, and the potential looks like a flat line (Free Particle). The analyzer will detect this."""))

    new_cells.append(create_code_cell("""# Example: Infinite Well filling the entire grid
L_demo = 20
x_demo, dx_demo, x_int_demo = make_grid(L=L_demo, N=1000)

# Well boundaries match grid boundaries -> Walls are hidden!
V_flat = inf_square_well(x_int_demo, -L_demo/2, L_demo/2)
V_full_flat = np.zeros_like(x_demo)
V_full_flat[1:-1] = V_flat
V_full_flat[0] = 1e10; V_full_flat[-1] = 1e10

T_demo = kinetic_operator(len(x_int_demo), dx_demo)
E_flat, psi_flat = solve(T_demo, V_full_flat, dx_demo)

plot_educational(E_flat, psi_flat, V_full_flat, x_demo, no=0);"""))

    new_cells.append(create_markdown_cell("""### Case 2: Quantum Tunneling
If we have a finite well, the particle can penetrate the walls. The analyzer detects when probability exists in "classically forbidden" regions ($E < V$)."""))

    new_cells.append(create_code_cell("""# Example: Finite Well with Tunneling
V_tunnel = finite_square_well(x_int_demo, -2, 2, depth_V=5.0)
V_full_tunnel = np.zeros_like(x_demo)
V_full_tunnel[1:-1] = V_tunnel
V_full_tunnel[0] = 1e10; V_full_tunnel[-1] = 1e10

E_tunnel, psi_tunnel = solve(T_demo, V_full_tunnel, dx_demo)

# Plot the ground state (should be bound)
plot_educational(E_tunnel, psi_tunnel, V_full_tunnel, x_demo, no=0);"""))

    new_cells.append(create_markdown_cell("""### Case 3: Scattering State
If the energy is higher than the potential barrier, the particle is not bound. It acts like a free wave."""))

    new_cells.append(create_code_cell("""# Example: Shallow Well (High energy state)
# We look at a higher excited state that might be above the well depth
plot_educational(E_tunnel, psi_tunnel, V_full_tunnel, x_demo, no=15);"""))

    # Append new cells to the notebook
    nb.cells.extend(new_cells)

    # Save the new notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Successfully created {output_path}")

if __name__ == "__main__":
    update_notebook()
