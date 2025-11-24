"""
Demonstration of Enhanced Analytic Check Functions
===================================================
This script shows how to use the improved check_*_analytic functions
with custom parameters.
"""

import numpy as np
import functions as f

# Setup grid
L = 50
N_GRID = 2000
x_full, dx, x_internal = f.make_grid(L, N_GRID)
T = f.kinetic_operator(len(x_internal), dx)

print("="*70)
print("DEMONSTRATION: Enhanced Analytic Check Functions")
print("="*70)

# ==========================================
# 1. INFINITE SQUARE WELL - Custom boundaries
# ==========================================
print("\n\n" + "="*70)
print("1. INFINITE SQUARE WELL - Custom Boundaries")
print("="*70)

# Create well from -10 to 10
V_ISW = f.inf_sqaure_well(x_internal, lower_bound=-10, upper_bound=10)
V_ISW_full = np.pad(V_ISW, (1, 1), constant_values=1e10)

E_ISW, psi_ISW = f.solve(T, V_ISW_full, dx)

# NEW WAY: Specify boundaries directly
print("\nUsing NEW function signature:")
f.check_ISW_analytic(E_ISW, lower_bound=-10, upper_bound=10, max_levels=5)

# ==========================================
# 2. HARMONIC OSCILLATOR - Custom k and center
# ==========================================
print("\n\n" + "="*70)
print("2. HARMONIC OSCILLATOR - Custom Parameters")
print("="*70)

# Example 1: Standard QHO at origin
k1 = 1.0
V_QHO1 = f.harmonic(x_internal, k=k1, center=0.0)
V_QHO1_full = np.pad(V_QHO1, (1, 1), constant_values=1e10)

E_QHO1, psi_QHO1 = f.solve(T, V_QHO1_full, dx)

print("\n--- Example 1: k=1.0, center=0 ---")
f.check_harmonic_analytic(E_QHO1, k=1.0, center=0.0, max_levels=5)

# Example 2: Stiffer spring
k2 = 10.0
V_QHO2 = f.harmonic(x_internal, k=k2, center=0.0)
V_QHO2_full = np.pad(V_QHO2, (1, 1), constant_values=1e10)

E_QHO2, psi_QHO2 = f.solve(T, V_QHO2_full, dx)

print("\n--- Example 2: k=10.0, center=0 ---")
f.check_harmonic_analytic(E_QHO2, k=10.0, center=0.0, max_levels=5)

# Example 3: Off-center oscillator (energies same, but center specified)
k3 = 5.0
center3 = 10.0
V_QHO3 = f.harmonic(x_internal, k=k3, center=center3)
V_QHO3_full = np.pad(V_QHO3, (1, 1), constant_values=1e10)

E_QHO3, psi_QHO3 = f.solve(T, V_QHO3_full, dx)

print(f"\n--- Example 3: k={k3}, center={center3} ---")
f.check_harmonic_analytic(E_QHO3, k=k3, center=center3, max_levels=5)

# ==========================================
# 3. FINITE SQUARE WELL - NEW FUNCTION!
# ==========================================
print("\n\n" + "="*70)
print("3. FINITE SQUARE WELL - Analytical Comparison")
print("="*70)

# Example 1: Deep well (should have many bound states)
V0_deep = 2.0
V_FSW_deep = f.finite_square_well(x_internal, lower_bound=-10, upper_bound=10, depth_V=V0_deep)
V_FSW_deep_full = np.pad(V_FSW_deep, (1, 1), constant_values=1e10)

E_FSW_deep, psi_FSW_deep = f.solve(T, V_FSW_deep_full, dx)

print("\n--- Example 1: Deep Well (V0=2.0) ---")
f.check_finite_well_analytic(E_FSW_deep, V0=V0_deep, lower_bound=-10, upper_bound=10, max_levels=10)

# Example 2: Shallow well (might have few or no bound states)
V0_shallow = 0.01
V_FSW_shallow = f.finite_square_well(x_internal, lower_bound=-10, upper_bound=10, depth_V=V0_shallow)
V_FSW_shallow_full = np.pad(V_FSW_shallow, (1, 1), constant_values=1e10)

E_FSW_shallow, psi_FSW_shallow = f.solve(T, V_FSW_shallow_full, dx)

print("\n--- Example 2: Shallow Well (V0=0.01) - YOUR ORIGINAL CASE ---")
f.check_finite_well_analytic(E_FSW_shallow, V0=V0_shallow, lower_bound=-10, upper_bound=10, max_levels=10)

# Example 3: Medium well
V0_medium = 0.5
V_FSW_medium = f.finite_square_well(x_internal, lower_bound=-10, upper_bound=10, depth_V=V0_medium)
V_FSW_medium_full = np.pad(V_FSW_medium, (1, 1), constant_values=1e10)

E_FSW_medium, psi_FSW_medium = f.solve(T, V_FSW_medium_full, dx)

print("\n--- Example 3: Medium Well (V0=0.5) ---")
f.check_finite_well_analytic(E_FSW_medium, V0=V0_medium, lower_bound=-10, upper_bound=10, max_levels=10)

# ==========================================
# SUMMARY
# ==========================================
print("\n\n" + "="*70)
print("SUMMARY OF NEW FUNCTION SIGNATURES")
print("="*70)

print("""
1. check_ISW_analytic(E, lower_bound=-10, upper_bound=10, max_levels=6)
   - Now accepts well boundaries instead of just length
   - Example: check_ISW_analytic(E, lower_bound=-5, upper_bound=5)

2. check_harmonic_analytic(E, k=None, center=0.0, max_levels=6)
   - Now accepts custom k and center parameters
   - Falls back to global Last_k_value if k=None
   - Example: check_harmonic_analytic(E, k=10, center=0)

3. check_finite_well_analytic(E, V0, lower_bound=-10, upper_bound=10, max_levels=10)
   - NEW FUNCTION for finite square wells!
   - Solves transcendental equations to find analytical energies
   - Example: check_finite_well_analytic(E, V0=2.0, lower_bound=-10, upper_bound=10)

All functions now return (E_analytic, E_numerical) for further analysis.
""")

print("\n✓ Demonstration complete!")
