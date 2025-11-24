# How to Use verify_qmsolve() with Your Notebook Variables

## ✅ New Feature: Verify YOUR Results!

The `verify_qmsolve()` function now accepts your notebook variables and compares them against QMSolve!

---

## 📓 Usage in Your Notebook

### Example 1: Verify Your Current Calculation

```python
# After you've computed your results in the notebook:
# E, psi, V_full, x = ... (your calculations)

from functions import verify_qmsolve

# Verify YOUR results against QMSolve
verify_qmsolve(
    E_your=E,           # Your energy eigenvalues
    psi_your=psi,       # Your wavefunctions
    V_your=V_full,      # Your potential (full array)
    x_your=x,           # Your spatial grid (full array)
    potential_type='double_well',
    potential_params={'depth': 2.0, 'separation': 1.0, 'center': 0.0}
)
```

### Example 2: Harmonic Oscillator

```python
# Your harmonic oscillator calculation
k = 1.0
V_internal = harmonic(x_internal, k=k)
# ... solve for E, psi ...

# Verify against QMSolve
verify_qmsolve(
    E_your=E,
    psi_your=psi,
    V_your=V_full,
    x_your=x,
    potential_type='harmonic',
    potential_params={'k': 1.0, 'center': 0.0}
)
```

### Example 3: Use Defaults (No Parameters)

```python
# Just run with defaults to test
from functions import verify_qmsolve

verify_qmsolve()
# Uses default double well: depth=2.0, separation=1.0
```

---

## 📊 Expected Output

```
======================================================================
CROSS-VERIFICATION: Your Results vs QMSolve
======================================================================

✓ Using your computed results
  Grid points: 2002
  Domain: [-25.00, 25.00]
  Number of states: 2000

--- Your Hand-wave Results ---
Energies (first 5): [1.400886 2.092533 4.455252 6.917808 9.872632]

--- Running QMSolve with same potential ---
QMSolve Energies (eV):      [38.163216 57.083150 121.536061 188.760142 269.398902]
QMSolve Energies (Hartree): [1.402472 2.097767 4.466368 6.936807 9.900227]

--- Comparison Results ---
----------------------------------------------------------------------
| n | Your E       | QMSolve E    | Diff         | % Diff   |
----------------------------------------------------------------------
| 0 | 1.400886     | 1.402472     | 1.59e-03     | 0.1131%  |
| 1 | 2.092533     | 2.097767     | 5.23e-03     | 0.2495%  |
| 2 | 4.455252     | 4.466368     | 1.11e-02     | 0.2490%  |
| 3 | 6.917808     | 6.936807     | 1.90e-02     | 0.2739%  |
| 4 | 9.872632     | 9.900227     | 2.76e-02     | 0.2786%  |
----------------------------------------------------------------------

Average difference: 0.2430%
Maximum difference: 0.2786%

✅ EXCELLENT: Your solver matches QMSolve within 0.5%!

✅ QMSolve verification complete!
```

---

## 🎯 Complete Notebook Example

```python
# ========================================
# Your Calculation
# ========================================

from functions import *
import numpy as np

# Setup grid
x, dx, x_int = make_grid(L=20, N=1000)

# Create double well potential
depth = 2.0
separation = 1.0
center = 0.0

V_int = V_double_well(x_int, depth=depth, separation=separation, center=center)

# Prepare for solver
V_full = np.zeros_like(x)
V_full[1:-1] = V_int
V_full[0] = 1e10
V_full[-1] = 1e10

# Solve
T = kinetic_operator(len(x_int), dx)
E, psi = solve(T, V_full, dx)

print(f"Computed {len(E)} eigenstates")
print(f"Ground state energy: {E[0]:.6f} Ha")

# ========================================
# Verify Against QMSolve
# ========================================

from functions import verify_qmsolve

verify_qmsolve(
    E_your=E,
    psi_your=psi,
    V_your=V_full,
    x_your=x,
    potential_type='double_well',
    potential_params={'depth': depth, 'separation': separation, 'center': center}
)
```

---

## 🔧 Parameters

### Required (if using your data):
- `E_your`: Your energy eigenvalues (ndarray)
- `x_your`: Your spatial grid (ndarray)

### Optional:
- `psi_your`: Your wavefunctions (not currently used, but included for future features)
- `V_your`: Your potential (not currently used, but included for future features)
- `potential_type`: `'double_well'` or `'harmonic'` (default: `'double_well'`)
- `potential_params`: Dictionary with potential parameters

### Potential Parameters:

**For `double_well`:**
```python
potential_params = {
    'depth': 2.0,        # Depth parameter
    'separation': 1.0,   # Separation parameter
    'center': 0.0        # Center position
}
```

**For `harmonic`:**
```python
potential_params = {
    'k': 1.0,      # Spring constant
    'center': 0.0  # Center position
}
```

---

## ✅ What It Does

1. **Takes your results** (E, psi, V, x from your notebook)
2. **Extracts parameters** (domain size L, number of points N)
3. **Runs QMSolve** with the SAME potential and parameters
4. **Compares energies** between your solver and QMSolve
5. **Shows statistics** (average difference, max difference)
6. **Gives verdict** (Excellent < 0.5%, Good < 1%, Warning > 1%)

---

## 🎉 Benefits

✅ **Validates YOUR specific calculation**  
✅ **No need to match exact test cases**  
✅ **Works with any potential parameters**  
✅ **Prints directly to notebook** (no text files!)  
✅ **Clear pass/fail criteria**  

Now you can verify ANY calculation in your notebook! 🚀
