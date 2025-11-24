# Enhanced Functions Documentation Summary

## Overview

All functions in `functions.py` now have comprehensive NumPy-style docstrings that provide detailed information when you hover over them in your IDE (VS Code, PyCharm, Jupyter, etc.) or call `help(function_name)`.

## What's Included in Each Docstring

Each function now includes:

### 1. **Brief Description**
   - One-line summary of what the function does
   - Detailed explanation of the physics/mathematics

### 2. **Parameters Section**
   ```
   Parameters
   ----------
   param_name : type
       Description of the parameter
       Additional notes about valid ranges, units, etc.
   ```

### 3. **Returns Section**
   ```
   Returns
   -------
   return_name : type
       Description of what is returned
       Shape information for arrays
       Units where applicable
   ```

### 4. **Notes Section**
   - Important implementation details
   - Physical interpretation
   - Caveats and warnings
   - Related functions

### 5. **Examples Section**
   - Practical usage examples
   - Expected output
   - Common use cases

---

## Enhanced Functions List

### Grid Functions

#### `make_grid(L=50, N=2000)`
- **Purpose**: Create spatial grid for solving Schrödinger equation
- **Returns**: Full grid (x_full), spacing (dx), internal points (x_internal)
- **Example**:
  ```python
  x, dx, x_int = make_grid(L=20, N=1000)
  ```

---

### Potential Generators

#### `constant(x, c)`
- **Purpose**: Create constant potential V(x) = c
- **Use case**: Adding baseline energy offset

#### `harmonic(x, k, center=0.0)`
- **Purpose**: Quantum harmonic oscillator potential
- **Formula**: V(x) = (1/2)k(x - center)²
- **Notes**: Sets global `Last_k_value` for analytic checks
- **Example**:
  ```python
  V = harmonic(x, k=10, center=0)  # Stiff spring at origin
  V_offset = harmonic(x, k=1, center=5)  # Centered at x=5
  ```

#### `gaussian_well(x, center=0.0, width=1.0, depth=50)`
- **Purpose**: Smooth Gaussian-shaped potential well
- **Formula**: V(x) = -depth × exp(-(x-center)²/(2×width²))
- **Use case**: Modeling smooth trapping potentials

#### `inf_sqaure_well(x, lower_bound, upper_bound)`
- **Purpose**: Infinite square well (particle in a box)
- **Formula**: 
  - V(x) = 0 inside [lower_bound, upper_bound]
  - V(x) = 10¹⁰ outside (penalty method)
- **Example**:
  ```python
  V = inf_sqaure_well(x, -10, 10)  # Well from -10 to 10
  check_ISW_analytic(E, lower_bound=-10, upper_bound=10)
  ```

#### `inf_wall(x, side, bound)`
- **Purpose**: Place infinite wall on one side
- **Parameters**:
  - `side`: 'left' or 'right'
  - `bound`: Position of the wall
- **Example**:
  ```python
  V_left = inf_wall(x, 'left', bound=-5)  # Blocks x < -5
  ```

#### `finite_barrier(x, center, width, height)`
- **Purpose**: Rectangular potential barrier for tunneling studies
- **Use case**: Quantum tunneling experiments

#### `V_double_well(x, depth=20, separation=1, center=0.0)`
- **Purpose**: Quartic double-well potential
- **Formula**: V(x) = depth × ((x-center)² - separation)²
- **Use case**: Tunneling splitting, symmetric/antisymmetric states

#### `finite_square_well(x, lower_bound, upper_bound, depth_V)`
- **Purpose**: Finite square well with barrier height depth_V
- **Key points**:
  - Bound states only when E < depth_V
  - Exponential decay in barriers for bound states
  - Oscillatory everywhere for scattering states (E > depth_V)
- **Example**:
  ```python
  V_deep = finite_square_well(x, -10, 10, depth_V=2.0)  # Many bound states
  V_shallow = finite_square_well(x, -10, 10, depth_V=0.01)  # Few/no bound states
  check_finite_well_analytic(E, V0=2.0, lower_bound=-10, upper_bound=10)
  ```

---

### Core Solver Functions

#### `kinetic_operator(N, dx, hbar=1, m=1)`
- **Purpose**: Build kinetic energy operator matrix T = -(ℏ²/2m) d²/dx²
- **Method**: 3-point central difference stencil
- **Returns**: Symmetric tridiagonal matrix (N × N)
- **Formula**: d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
- **Example**:
  ```python
  T = kinetic_operator(N=1000, dx=0.025)
  print(f"Symmetric: {np.allclose(T, T.T)}")  # True
  ```

#### `solve(T, V_full, dx)`
- **Purpose**: Solve time-independent Schrödinger equation Hψ = Eψ
- **Method**: Eigenvalue decomposition using `np.linalg.eigh()`
- **Returns**: 
  - `E`: Eigenvalues (energies) sorted ascending
  - `psi`: Normalized eigenvectors (wavefunctions) as columns
- **Normalization**: ∫|ψ|² dx = 1 for each state
- **Example**:
  ```python
  # Setup
  x, dx, x_int = make_grid(L=20, N=1000)
  T = kinetic_operator(len(x_int), dx)
  
  # Create potential
  V = inf_sqaure_well(x_int, -10, 10)
  V_full = np.pad(V, (1,1), constant_values=1e10)
  
  # Solve
  E, psi = solve(T, V_full, dx)
  
  # Verify
  norm = np.sum(psi[:, 0]**2) * dx  # Should be 1.0
  ```

---

### Plotting Functions

#### `plot_V(V_raw_input)`
- **Purpose**: Quick visualization of potential profile
- **Style**: Dark background, cyan curve
- **Returns**: Matplotlib figure object

#### `plot_alive(E, psi, V, x, no=1, nos=5, mode='')`
- **Purpose**: Plot probability densities |ψ|² with dual y-axes
- **Features**:
  - Left axis: Energy levels and potential
  - Right axis: Probability density (separate scale)
  - Color-synchronized between curves and energy levels
- **Modes**:
  - `mode='all'`: Plot first `nos` states
  - `mode=''`: Plot single state `no`
- **Example**:
  ```python
  # Plot first 5 states
  fig = plot_alive(E, psi, V_full, x_full, nos=5, mode='all')
  
  # Plot only ground state
  fig = plot_alive(E, psi, V_full, x_full, no=0)
  ```

#### `plot_dead(E, psi, V, x, nos=5)`
- **Purpose**: Textbook-style plot with wavefunctions shifted by energy
- **Features**: Energy spectrum sidebar

---

### Benchmarking Functions

#### `check_ortho(psi, dx, num_states_to_check=20)`
- **Purpose**: Verify orthonormality of wavefunctions
- **Returns**: Overlap matrix (should be identity)
- **Example**:
  ```python
  overlap = check_ortho(psi, dx, num_states_to_check=5)
  print(np.round(overlap, 6))  # Should be I
  ```

#### `check_ISW_analytic(E, lower_bound=-10, upper_bound=10, max_levels=6)`
- **Purpose**: Compare numerical energies to infinite square well analytical solution
- **Formula**: E_n = (ℏ²π²n²)/(2mL²) where L = upper_bound - lower_bound
- **Returns**: (E_analytic, E_numerical)
- **Example**:
  ```python
  check_ISW_analytic(E, lower_bound=-10, upper_bound=10, max_levels=5)
  ```

#### `check_harmonic_analytic(E, k=None, center=0.0, max_levels=6)`
- **Purpose**: Compare numerical energies to harmonic oscillator analytical solution
- **Formula**: E_n = ℏω(n + 1/2) where ω = √(k/m)
- **Parameters**:
  - `k`: Spring constant (if None, uses global `Last_k_value`)
  - `center`: Center position (for documentation, doesn't affect energies)
- **Returns**: (E_analytic, E_numerical)
- **Example**:
  ```python
  # Explicit k
  check_harmonic_analytic(E, k=10, center=0, max_levels=5)
  
  # Use global Last_k_value
  V = harmonic(x, k=1.0)  # Sets Last_k_value
  check_harmonic_analytic(E)  # Uses k=1.0
  ```

#### `check_finite_well_analytic(E, V0, lower_bound=-10, upper_bound=10, max_levels=10)` **[NEW!]**
- **Purpose**: Compare numerical energies to finite square well analytical solution
- **Method**: Solves transcendental equations numerically
  - Even states: z×tan(z) = √(z₀² - z²)
  - Odd states: -z×cot(z) = √(z₀² - z²)
  - where z₀ = a√(2mV₀)/ℏ
- **Returns**: (E_analytic, E_numerical) for bound states only
- **Warning**: Returns (None, None) if no bound states exist
- **Example**:
  ```python
  # Deep well (many bound states)
  check_finite_well_analytic(E, V0=2.0, lower_bound=-10, upper_bound=10)
  
  # Shallow well (may have no bound states)
  check_finite_well_analytic(E, V0=0.01, lower_bound=-10, upper_bound=10)
  # WARNING: No bound states found! Barrier too shallow.
  ```

---

## How to Use Enhanced Docstrings

### In Jupyter Notebook:
```python
import functions as f

# View docstring
?f.harmonic

# Or
help(f.harmonic)

# Shift+Tab while cursor is on function name shows inline help
```

### In VS Code / PyCharm:
- **Hover** over any function name to see full documentation
- **Ctrl+Click** (or Cmd+Click on Mac) to jump to definition
- IntelliSense will show parameter hints as you type

### In Python REPL:
```python
>>> import functions as f
>>> help(f.solve)
```

---

## Complete Workflow Example

```python
from functions import *
import numpy as np

# 1. Setup grid
x, dx, x_int = make_grid(L=20, N=1000)
T = kinetic_operator(len(x_int), dx)

# 2. Create potential (hover over each function for details!)
V = finite_square_well(x_int, -10, 10, depth_V=2.0)
V_full = np.pad(V, (1,1), constant_values=1e10)

# 3. Solve
E, psi = solve(T, V_full, dx)

# 4. Verify accuracy
check_finite_well_analytic(E, V0=2.0, lower_bound=-10, upper_bound=10)

# 5. Check orthonormality
overlap = check_ortho(psi, dx, num_states_to_check=5)

# 6. Visualize
fig = plot_alive(E, psi, V_full, x, nos=5, mode='all')
```

---

## Summary

✅ **All functions now have comprehensive docstrings**
✅ **IDE hover shows full documentation**
✅ **Parameters, returns, and examples clearly documented**
✅ **Physics formulas and units specified**
✅ **Usage examples provided**

When you hover over any function in your IDE or Jupyter notebook, you'll see:
- What it does
- All parameters with types and descriptions
- What it returns
- Practical examples
- Important notes and warnings

This makes the codebase much more professional and easier to use!
