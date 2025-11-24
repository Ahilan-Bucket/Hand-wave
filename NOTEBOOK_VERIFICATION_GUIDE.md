# How to Use run_comparison() and run_verification() in Notebooks

## ✅ Functions Now Available in functions.py

Both `run_comparison()` and `run_verification()` are now available as importable functions!

---

## 📓 Usage in Jupyter Notebooks

### Cell 1: Import and Run QMSolve Comparison

```python
from functions import run_comparison

# Run comparison against QMSolve
run_comparison()

# View results
with open('comparison_log.txt', 'r') as f:
    print(f.read())
```

**What it does**:
- Compares your solver vs QMSolve for Double Well potential
- Compares Harmonic Oscillator (debug test)
- Saves results to `comparison_log.txt`

**Expected Output**:
```
========================================
CROSS-VERIFICATION: PSI_SOLVE2 vs QMSOLVE
========================================

[TEST CASE] Double Well Potential
Parameters: L=10.0, N=512, depth=2.0, separation=1.0, m=1.0

--- Running psi_solve2 ---
psi_solve2 Energies (first 5): [1.400886 2.092533 4.455252 6.917808 9.872632]

--- Running QMSolve ---
QMSolve Energies (eV):      [38.163216 57.083150 121.536061 188.760142 269.398902]
QMSolve Energies (Hartree): [1.402472 2.097767 4.466368 6.936807 9.900227]

--- Comparison Results ---
-----------------------------------------------------------------
| n | psi_solve2 E | QMSolve E    | Diff         | % Diff   |
-----------------------------------------------------------------
| 0 | 1.400886     | 1.402472     | 1.59e-03     | 0.1131%  |
| 1 | 2.092533     | 2.097767     | 5.23e-03     | 0.2495%  |
...
```

---

### Cell 2: Import and Run Physics Verification

```python
from functions import run_verification

# Run comprehensive physics tests
run_verification()

# View results
with open('verification_log.txt', 'r') as f:
    print(f.read())
```

**What it tests**:
1. ✅ Infinite Square Well
2. ✅ Harmonic Oscillator
3. ✅ Half-Harmonic Oscillator
4. ✅ Triangular Potential
5. ✅ Hamiltonian Construction

**Expected Output**:
```
========================================
PHYSICS ENGINE VERIFICATION
========================================

[TEST 1] Infinite Square Well (Particle in a Box)

### ENERGY BENCHMARK: Infinite Square Well ###
Well boundaries: x = [-10.0, 10.0], Width L = 20.0
-------------------------------------------------------
| n | Analytic E | Numerical E | % Error |
-------------------------------------------------------
| 1 | 0.012337   | 0.012337    | 0.0001% |
| 2 | 0.049348   | 0.049348    | 0.0003% |
...

[TEST 2] Harmonic Oscillator

### ENERGY BENCHMARK: Harmonic Oscillator ###
Spring constant k = 1.0, Center = 0.0, omega = 1.0000
-------------------------------------------------------
| n | Analytic E | Numerical E | % Error |
-------------------------------------------------------
| 0 | 0.500000   | 0.499980    | 0.0039% |
| 1 | 1.500000   | 1.499902    | 0.0065% |
...
```

---

## 🔧 Alternative: Run Without Viewing Output

If you just want to run the tests without printing to notebook:

```python
from functions import run_comparison, run_verification

# Run tests (output goes to log files)
run_comparison()
run_verification()

print("✓ Tests complete!")
print("  - comparison_log.txt")
print("  - verification_log.txt")
```

---

## 📊 View Specific Results

### View only QMSolve comparison:

```python
from functions import run_comparison

run_comparison()

# Read and display
import pandas as pd

print("QMSolve Comparison Results:")
with open('comparison_log.txt', 'r') as f:
    content = f.read()
    # Find comparison table
    start = content.find("--- Comparison Results ---")
    end = content.find("DEBUG CASE")
    print(content[start:end])
```

### View only specific verification test:

```python
from functions import run_verification

run_verification()

# Read and display only Harmonic Oscillator test
with open('verification_log.txt', 'r') as f:
    content = f.read()
    # Find HO test
    start = content.find("[TEST 2] Harmonic Oscillator")
    end = content.find("[TEST 3]")
    print(content[start:end])
```

---

## ⚠️ Important Notes

### For run_comparison():

**Requires QMSolve**:
```bash
pip install qmsolve
```

If QMSolve is not installed, you'll get:
```
Error: qmsolve not found. Please install it via 'pip install qmsolve'
```

### For run_verification():

**No external dependencies** - uses only your solver and analytical solutions!

---

## 🎯 Complete Notebook Example

```python
# Cell 1: Setup
import numpy as np
import matplotlib.pyplot as plt
from functions import run_comparison, run_verification

# Cell 2: Run all tests
print("Running QMSolve comparison...")
run_comparison()

print("\nRunning physics verification...")
run_verification()

print("\n✓ All tests complete!")

# Cell 3: Display results
print("="*70)
print("QMSOLVE COMPARISON")
print("="*70)
with open('comparison_log.txt', 'r') as f:
    print(f.read())

print("\n" + "="*70)
print("PHYSICS VERIFICATION")
print("="*70)
with open('verification_log.txt', 'r') as f:
    print(f.read())
```

---

## 📁 Output Files

Both functions create log files in your current directory:

- **`comparison_log.txt`** - QMSolve comparison results
- **`verification_log.txt`** - Physics verification results

These files are **gitignored** by default (they're in `.gitignore`).

---

## Summary

✅ **`run_comparison()`** - Compare against QMSolve (requires qmsolve package)  
✅ **`run_verification()`** - Comprehensive physics tests (no external deps)  
✅ **Both functions** can be imported directly in notebooks  
✅ **Results saved** to log files for later review  

Use these in your `ModulatedPresentation.ipynb` or any other notebook!
