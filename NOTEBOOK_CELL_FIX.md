# Fix for ModulatedPresentation.ipynb Camera Cells

## Problem

The variables `E_vals`, `psi_vecs`, `V_full`, etc. are defined **inside** the `verify()` function, making them inaccessible outside. This causes:

```python
NameError: name 'E_vals' is not defined
```

## Solution

You have two options:

---

## Option 1: Return Values from verify() (Recommended)

Modify the `verify()` function to **return** the computed values:

```python
# Cell 1: Capture Potential
from functions import capture_potential_notebook

print("Starting camera stream...")
A_MIN = 0
A_MAX = 100 
V_raw_input = capture_potential_notebook(mode='', tune=1, A_MIN=A_MIN, A_MAX=A_MAX)
```

```python
# Cell 2: Process and Solve (MODIFIED)
def verify():
    if V_raw_input is None:
        print("Error: Run the Camera Input cell first to get V_raw_input!")
        return None, None, None  # Return None values
    
    # --- 1. Map Input to Physics Grid ---
    x_solver = x[1:-1]
    
    V_interpolated = np.interp(
        x_solver, 
        np.linspace(-L/2, L/2, len(V_raw_input)),
        V_raw_input
    )
    
    V_max_height = 100.0
    V_internal = V_interpolated * V_max_height
    V_full = np.pad(V_internal, (1, 1), 'constant', constant_values=0.0)
    
    # --- 2. Solve ---
    try:
        E_vals, psi_vecs = solve(T, V_full, dx)
        print(f"Solver complete. Found {E_vals.size} eigenstates.")
        return E_vals, psi_vecs, V_full  # RETURN the values!
    
    except Exception as e:
        print(f"Error during solve: {e}")
        if np.max(V_internal) > 1e9:
            print("Potential may be too steep or high.")
        return None, None, None

# Call verify and capture returned values
E_vals, psi_vecs, V_full = verify()
```

```python
# Cell 3: Plot (now E_vals is defined!)
if E_vals is not None:
    plot_dead(E=E_vals, psi=psi_vecs, V=V_full, x=x, nos=5)
else:
    print("No results to plot. Check previous cells.")
```

---

## Option 2: Remove the Function Wrapper (Simpler)

Just run the code **directly** without wrapping it in `verify()`:

```python
# Cell 1: Capture Potential
from functions import capture_potential_notebook

print("Starting camera stream...")
A_MIN = 0
A_MAX = 100 
V_raw_input = capture_potential_notebook(mode='', tune=1, A_MIN=A_MIN, A_MAX=A_MAX)
```

```python
# Cell 2: Process and Solve (NO FUNCTION WRAPPER)
if V_raw_input is None:
    print("Error: Run the Camera Input cell first!")
else:
    # --- 1. Map Input to Physics Grid ---
    x_solver = x[1:-1]
    
    V_interpolated = np.interp(
        x_solver, 
        np.linspace(-L/2, L/2, len(V_raw_input)),
        V_raw_input
    )
    
    V_max_height = 100.0
    V_internal = V_interpolated * V_max_height
    V_full = np.pad(V_internal, (1, 1), 'constant', constant_values=0.0)
    
    # --- 2. Solve ---
    try:
        E_vals, psi_vecs = solve(T, V_full, dx)
        print(f"Solver complete. Found {E_vals.size} eigenstates.")
    
    except Exception as e:
        print(f"Error during solve: {e}")
        if np.max(V_internal) > 1e9:
            print("Potential may be too steep or high.")
```

```python
# Cell 3: Plot (E_vals is now in global scope!)
plot_dead(E=E_vals, psi=psi_vecs, V=V_full, x=x, nos=5)
```

---

## Why This Happens

In Python, variables defined inside a function are **local** to that function:

```python
def my_function():
    x = 10  # Local variable
    return x

my_function()
print(x)  # ❌ NameError: name 'x' is not defined
```

To access them outside:

```python
def my_function():
    x = 10
    return x

x = my_function()  # ✅ Now x is in global scope
print(x)  # Works!
```

---

## Recommended Fix

Use **Option 2** (remove function wrapper) for notebook cells - it's simpler and more natural for interactive work.

Only use functions when you need to:
- Reuse code multiple times
- Organize complex logic
- Create a library/module

For exploratory notebook work, direct execution is clearer!
