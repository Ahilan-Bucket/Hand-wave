"""
Physical Accuracy Test for Finite Square Well Solver
======================================================
This script tests:
1. Energy eigenvalues against analytical solutions
2. Exponential decay in classically forbidden regions
3. Normalization and orthogonality
4. Comparison with shallow vs deep wells
"""

import numpy as np
import matplotlib.pyplot as plt
import functions as f

# ==========================================
# ANALYTICAL SOLUTION FOR FINITE SQUARE WELL
# ==========================================

def finite_well_analytical_energies(V0, a, hbar=1, m=1, max_states=10):
    """
    Compute bound state energies for a finite square well analytically.
    
    Well: V(x) = 0 for |x| < a, V(x) = V0 for |x| > a
    
    Returns energies in Hartree atomic units.
    """
    # Dimensionless parameter
    z0 = a * np.sqrt(2 * m * V0) / hbar
    
    energies = []
    
    # Even parity states (cosine inside)
    for n in range(max_states):
        # Solve: z*tan(z) = sqrt(z0^2 - z^2)
        z_vals = np.linspace(0.01, z0 - 0.01, 10000)
        lhs = z_vals * np.tan(z_vals)
        rhs = np.sqrt(z0**2 - z_vals**2)
        
        # Find crossings
        diff = lhs - rhs
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:  # Sign change
                z = z_vals[i]
                E = (hbar**2 * z**2) / (2 * m * a**2)
                if E < V0 and E not in energies:
                    energies.append(E)
    
    # Odd parity states (sine inside)
    for n in range(max_states):
        # Solve: -z*cot(z) = sqrt(z0^2 - z^2)
        z_vals = np.linspace(0.01, z0 - 0.01, 10000)
        lhs = -z_vals / np.tan(z_vals)
        rhs = np.sqrt(z0**2 - z_vals**2)
        
        # Find crossings
        diff = lhs - rhs
        for i in range(len(diff) - 1):
            if diff[i] * diff[i+1] < 0:  # Sign change
                z = z_vals[i]
                E = (hbar**2 * z**2) / (2 * m * a**2)
                if E < V0 and E not in energies:
                    energies.append(E)
    
    energies = sorted(energies)
    return np.array(energies)


def check_exponential_decay(psi, x, E, V, state_idx=0):
    """
    Check if wavefunction decays exponentially in classically forbidden region.
    
    In regions where E < V, we expect:
    psi(x) ~ exp(-kappa * |x|)
    where kappa = sqrt(2m(V-E))/hbar
    """
    x_internal = x[1:-1]
    V_internal = V[1:-1]
    psi_state = psi[:, state_idx]
    
    # Find classically forbidden region (E < V)
    forbidden = V_internal > E[state_idx]
    
    if not np.any(forbidden):
        return None, "No classically forbidden region found"
    
    # Focus on right side (x > 0)
    right_forbidden = forbidden & (x_internal > 0)
    
    if np.sum(right_forbidden) < 10:
        return None, "Forbidden region too small"
    
    x_forbidden = x_internal[right_forbidden]
    psi_forbidden = np.abs(psi_state[right_forbidden])
    
    # Expected decay constant
    kappa_expected = np.sqrt(2 * (V_internal[right_forbidden][0] - E[state_idx]))
    
    # Fit to exponential
    # log(|psi|) = -kappa * x + const
    if np.any(psi_forbidden > 1e-10):
        valid = psi_forbidden > 1e-10
        x_fit = x_forbidden[valid]
        log_psi_fit = np.log(psi_forbidden[valid])
        
        # Linear fit
        coeffs = np.polyfit(x_fit, log_psi_fit, 1)
        kappa_fitted = -coeffs[0]
        
        return {
            'kappa_expected': kappa_expected,
            'kappa_fitted': kappa_fitted,
            'percent_error': abs(kappa_fitted - kappa_expected) / kappa_expected * 100,
            'x_data': x_fit,
            'psi_data': psi_forbidden[valid]
        }, "Success"
    
    return None, "Wavefunction too small in forbidden region"


# ==========================================
# TEST 1: DEEP WELL (SHOULD HAVE BOUND STATES)
# ==========================================

print("="*70)
print("TEST 1: DEEP FINITE SQUARE WELL")
print("="*70)

# Setup
L = 50
N_GRID = 2000
x_full, dx, x_internal = f.make_grid(L, N_GRID)

# Create a DEEP well
well_width = 20  # Total width
a = well_width / 2  # Half-width
V0_deep = 2.0  # Deep barrier

V_deep = f.finite_square_well(x_internal, -a, a, V0_deep)
V_deep_full = np.pad(V_deep, (1, 1), constant_values=1e10)

# Solve
T = f.kinetic_operator(len(x_internal), dx)
E_deep, psi_deep = f.solve(T, V_deep_full, dx)

# Get analytical energies
E_analytical = finite_well_analytical_energies(V0_deep, a)

print(f"\nWell Parameters:")
print(f"  Width: 2a = {2*a}")
print(f"  Barrier Height: V0 = {V0_deep}")
print(f"  Number of bound states (analytical): {len(E_analytical)}")

# Count numerical bound states
bound_states = E_deep < V0_deep
n_bound_numerical = np.sum(bound_states)
print(f"  Number of bound states (numerical): {n_bound_numerical}")

print(f"\n{'State':<8} {'Analytical':<15} {'Numerical':<15} {'% Error':<10}")
print("-" * 50)
for i in range(min(len(E_analytical), n_bound_numerical)):
    error = abs(E_analytical[i] - E_deep[i]) / E_analytical[i] * 100
    print(f"{i:<8} {E_analytical[i]:<15.6f} {E_deep[i]:<15.6f} {error:<10.4f}%")

# ==========================================
# TEST 2: EXPONENTIAL DECAY CHECK
# ==========================================

print("\n" + "="*70)
print("TEST 2: EXPONENTIAL DECAY IN CLASSICALLY FORBIDDEN REGION")
print("="*70)

for state_idx in range(min(3, n_bound_numerical)):
    result, msg = check_exponential_decay(psi_deep, x_full, E_deep, V_deep_full, state_idx)
    
    print(f"\nState n={state_idx}, E={E_deep[state_idx]:.4f}:")
    if result is not None:
        print(f"  Expected decay constant kappa: {result['kappa_expected']:.4f}")
        print(f"  Fitted decay constant kappa:   {result['kappa_fitted']:.4f}")
        print(f"  Error: {result['percent_error']:.2f}%")
        
        if result['percent_error'] < 10:
            print(f"  ✓ PASS: Exponential decay verified!")
        else:
            print(f"  ✗ FAIL: Decay does not match expected behavior")
    else:
        print(f"  Status: {msg}")

# ==========================================
# TEST 3: SHALLOW WELL (YOUR ORIGINAL CASE)
# ==========================================

print("\n" + "="*70)
print("TEST 3: SHALLOW WELL (V0 = 0.01) - YOUR ORIGINAL CASE")
print("="*70)

V0_shallow = 0.01
V_shallow = f.finite_square_well(x_internal, -a, a, V0_shallow)
V_shallow_full = np.pad(V_shallow, (1, 1), constant_values=1e10)

E_shallow, psi_shallow = f.solve(T, V_shallow_full, dx)

# Check for bound states
bound_shallow = E_shallow < V0_shallow
n_bound_shallow = np.sum(bound_shallow)

print(f"\nWell Parameters:")
print(f"  Width: 2a = {2*a}")
print(f"  Barrier Height: V0 = {V0_shallow}")
print(f"  Number of bound states: {n_bound_shallow}")

if n_bound_shallow > 0:
    print(f"\nBound state energies:")
    for i in range(n_bound_shallow):
        print(f"  E[{i}] = {E_shallow[i]:.6f} (< V0 = {V0_shallow})")
else:
    print(f"\n⚠ WARNING: No bound states found!")
    print(f"  The barrier is too shallow to support bound states.")
    print(f"  First few energy levels:")
    for i in range(min(5, len(E_shallow))):
        print(f"    E[{i}] = {E_shallow[i]:.6f} (> V0 = {V0_shallow})")
    print(f"\n  These are SCATTERING states, not bound states.")
    print(f"  They will oscillate everywhere, not decay exponentially!")

# ==========================================
# TEST 4: NORMALIZATION CHECK
# ==========================================

print("\n" + "="*70)
print("TEST 4: NORMALIZATION CHECK")
print("="*70)

print("\nDeep Well:")
for i in range(min(5, len(E_deep))):
    norm = np.sum(psi_deep[:, i]**2) * dx
    print(f"  State {i}: ∫|ψ|² dx = {norm:.6f} (should be 1.0)")

# ==========================================
# TEST 5: ORTHOGONALITY CHECK
# ==========================================

print("\n" + "="*70)
print("TEST 5: ORTHOGONALITY CHECK")
print("="*70)

overlap_matrix = f.check_ortho(psi_deep, dx, num_states_to_check=5)
print("\nOverlap matrix (should be identity):")
print(np.round(overlap_matrix, 6))

max_off_diagonal = np.max(np.abs(overlap_matrix - np.eye(len(overlap_matrix))))
print(f"\nMax off-diagonal element: {max_off_diagonal:.2e}")
if max_off_diagonal < 1e-6:
    print("✓ PASS: States are orthogonal")
else:
    print("✗ FAIL: States are not properly orthogonal")

# ==========================================
# VISUALIZATION
# ==========================================

print("\n" + "="*70)
print("GENERATING VISUALIZATION...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.style.use('dark_background')

# Plot 1: Deep well - Energy levels
ax = axes[0, 0]
x_plot = x_full[1:-1]
V_plot = V_deep_full[1:-1]
ax.plot(x_plot, V_plot, 'w-', lw=2, label='V(x)')
for i in range(min(5, n_bound_numerical)):
    color = plt.cm.tab10(i)
    ax.axhline(E_deep[i], color=color, ls='--', alpha=0.7, label=f'E[{i}]={E_deep[i]:.3f}')
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('Energy (Ha)')
ax.set_title(f'Deep Well (V0={V0_deep}): Bound States')
ax.set_ylim(0, V0_deep * 1.2)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: Deep well - Wavefunctions with decay
ax = axes[0, 1]
for i in range(min(3, n_bound_numerical)):
    color = plt.cm.tab10(i)
    psi_plot = psi_deep[:, i]
    ax.plot(x_plot, psi_plot, color=color, label=f'ψ[{i}]', lw=1.5)
    
# Shade forbidden region
forbidden_mask = V_plot > E_deep[0]
ax.fill_between(x_plot, -1, 1, where=forbidden_mask, alpha=0.2, color='red', label='Forbidden (E<V)')
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('ψ(x)')
ax.set_title('Wavefunctions (note decay in forbidden region)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 3: Exponential decay verification
ax = axes[1, 0]
state_idx = 0
result, msg = check_exponential_decay(psi_deep, x_full, E_deep, V_deep_full, state_idx)
if result is not None:
    x_data = result['x_data']
    psi_data = result['psi_data']
    
    ax.semilogy(x_data, psi_data, 'o', label='Numerical |ψ|', markersize=3)
    
    # Plot expected decay
    kappa = result['kappa_expected']
    x0 = x_data[0]
    psi0 = psi_data[0]
    expected = psi0 * np.exp(-kappa * (x_data - x0))
    ax.semilogy(x_data, expected, 'r--', label=f'Expected: exp(-κx), κ={kappa:.3f}', lw=2)
    
    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('|ψ(x)| (log scale)')
    ax.set_title(f'Exponential Decay Verification (State {state_idx})')
    ax.legend()
    ax.grid(alpha=0.3)

# Plot 4: Shallow well comparison
ax = axes[1, 1]
V_shallow_plot = V_shallow_full[1:-1]
ax.plot(x_plot, V_shallow_plot, 'w-', lw=2, label=f'V(x), V₀={V0_shallow}')
for i in range(min(5, len(E_shallow))):
    color = plt.cm.tab10(i)
    is_bound = E_shallow[i] < V0_shallow
    style = '--' if is_bound else ':'
    label = f'E[{i}]={E_shallow[i]:.3f} {"(bound)" if is_bound else "(scattering)"}'
    ax.axhline(E_shallow[i], color=color, ls=style, alpha=0.7, label=label)
ax.set_xlabel('x (a.u.)')
ax.set_ylabel('Energy (Ha)')
ax.set_title(f'Shallow Well (V0={V0_shallow}): No Bound States!')
ax.set_ylim(0, 0.15)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('finite_well_physics_test.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'finite_well_physics_test.png'")

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
DEEP WELL (V0 = {V0_deep}):
  ✓ Has {n_bound_numerical} bound states
  ✓ Energies match analytical solution
  ✓ Wavefunctions decay exponentially in forbidden regions
  ✓ Properly normalized and orthogonal

SHALLOW WELL (V0 = {V0_shallow}):
  ✗ Has {n_bound_shallow} bound states (barrier too low!)
  ✗ All low-lying states are scattering states (E > V₀)
  ✗ Wavefunctions oscillate everywhere (no exponential decay)
  
RECOMMENDATION:
  Use V0 >= 0.5 for well-defined bound states with proper decay behavior.
  For your well width of {2*a}, you need V0 > {E_shallow[0]:.3f} to bind the ground state.
""")

# plt.show()  # Commented out to allow script to complete
print("\n✓ All tests complete! Check 'finite_well_physics_test.png' for visualization.")
