# test_controls.py
# This script programmatically tests the potential controls for the Hand-wave app.
# It generates each static potential, solves the Schrödinger equation, and creates a Plotly figure.
# The script will save the figures as PNG files in the current directory for inspection.

import numpy as np
import os

# Import the physics engine from the Hand-wave repo
import functions as f
from app import plot_interactive

# Helper to run a test for a given potential
def run_test(potential_name, V_physics, x_full, dx, x_internal, **kwargs):
    # Pad the potential with high walls (infinite walls)
    V_full = np.pad(V_physics, (1, 1), constant_values=1e10)
    # Solve
    T = f.kinetic_operator(len(x_internal), dx)
    E, psi = f.solve(T, V_full, dx)
    # Plot
    fig = plot_interactive(E, psi, V_full, x_full, nos=5)
    # Save figure as static image (requires kaleido)
    out_path = f"{potential_name.replace(' ', '_').lower()}.png"
    try:
        fig.write_image(out_path)
        print(f"[OK] Saved {out_path}")
    except Exception as e:
        print(f"[ERROR] Could not save {out_path}: {e}")

# Common grid
L = 50
N_GRID = 1000
x_full, dx, x_internal = f.make_grid(L, N_GRID)

# 1. Static Square Well
width = 10.0
V_sq = np.zeros_like(x_internal)
V_sq[np.abs(x_internal) > width/2] = 200
run_test("Static Square Well", V_sq, x_full, dx, x_internal)

# 2. Static Harmonic Oscillator
k = 0.5
V_ho = 0.5 * k * x_internal**2
V_ho = V_ho / np.max(V_ho) * 50
run_test("Static Harmonic Oscillator", V_ho, x_full, dx, x_internal)

# 3. Double Well
sep = 2.0
depth = 1.0
V_dw = depth * ((x_internal**2 - sep**2)**2)
V_dw = V_dw / np.max(V_dw) * 50
run_test("Double Well", V_dw, x_full, dx, x_internal)

print("All tests completed.")
