# QMSolve Comparison Code - Location Guide

## 📍 Where is the QMSolve Comparison Code?

The QMSolve cross-verification code is located in **multiple places** across your projects:

---

## 1. **Main Comparison Script** ⭐

**Location**: `psi_solve2/verify_comparison.py`

**Purpose**: Automated comparison between your solver and QMSolve package

**What it does**:
- Compares Double Well potential results
- Compares Harmonic Oscillator results (debug case)
- Outputs results to `comparison_log.txt`

**How to run**:
```bash
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2
python verify_comparison.py
```

**Results saved to**: `psi_solve2/comparison_log.txt`

---

## 2. **Jupyter Notebook Comparison** 📓

**Location**: `psi_solve2/Comparison_Notebook.ipynb`

**Purpose**: Interactive comparison with visualizations

**What it includes**:
- Step-by-step comparison
- Visual plots of wavefunctions
- Side-by-side energy comparisons
- Unit conversion explanations (eV ↔ Hartree)

**How to use**:
```bash
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2
jupyter notebook Comparison_Notebook.ipynb
```

---

## 3. **Streamlit App Display**

**Location**: `psi_solve2/app.py` (lines 391-429)

**Purpose**: Shows QMSolve comparison in the web app

**Tab**: "Benchmarks & Verification" → "QMSolve Comparison"

**Displays**:
```python
State | psi_solve2 (Ha) | QMSolve (Ha) | % Difference
------|-----------------|--------------|-------------
0     | 1.400886        | 1.402472     | 0.11%
1     | 2.092533        | 2.097767     | 0.25%
2     | 4.455252        | 4.466368     | 0.25%
3     | 6.917808        | 6.936807     | 0.27%
4     | 9.872632        | 9.900227     | 0.28%
```

---

## 4. **Hand-wave Project** (Copy)

The same files also exist in `Hand-wave/`:
- `Hand-wave/verify_comparison.py`
- `Hand-wave/Comparison_Notebook.ipynb`
- `Hand-wave/comparison_log.txt`

---

## 📊 What the Comparison Tests

### Test Case 1: Double Well Potential

**Potential**: V(x) = 2.0 × ((x - 0)² - 1)²

**Parameters**:
- Domain: L = 10.0 a.u.
- Grid points: N = 512
- Depth: 2.0
- Separation: 1.0

**Results**:
- Agreement within **0.11% - 0.28%**
- Excellent validation!

### Test Case 2: Harmonic Oscillator (Debug)

**Potential**: V(x) = 0.5 × k × x²

**Parameters**:
- Spring constant: k = 1.0
- Same grid as above

**Purpose**: Sanity check with known analytical solution

---

## 🔧 How to Run QMSolve Comparison

### Option 1: Run the Script

```bash
# Navigate to psi_solve2
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2

# Run comparison
python verify_comparison.py

# View results
type comparison_log.txt  # Windows
# or
cat comparison_log.txt   # Linux/Mac
```

### Option 2: Use Jupyter Notebook

```bash
# Navigate to psi_solve2
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2

# Launch Jupyter
jupyter notebook

# Open: Comparison_Notebook.ipynb
# Run all cells
```

### Option 3: View in Streamlit App

```bash
# Navigate to psi_solve2
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2

# Run app
streamlit run app.py

# Navigate to: "Benchmarks & Verification" → "QMSolve Comparison" tab
```

---

## 📦 Requirements

To run QMSolve comparison, you need:

```bash
pip install qmsolve
```

**Note**: QMSolve is already in `psi_solve2/requirements.txt`

---

## 🎯 Key Findings from QMSolve Comparison

### ✅ Validation Results

**Double Well Potential**:
- State 0: **0.11% difference**
- State 1: **0.25% difference**
- State 2: **0.25% difference**
- State 3: **0.27% difference**
- State 4: **0.28% difference**

**Conclusion**: Your solver is **highly accurate** and agrees with the established QMSolve package!

### 📝 Important Notes

1. **Unit Conversion**: QMSolve uses **eV** (electron-volts), your solver uses **Hartree**
   - Conversion: 1 Hartree = 27.211386 eV
   - The script handles this automatically

2. **Grid Matching**: Both solvers use N=512 grid points for fair comparison

3. **Potential Definition**: Same mathematical potential used in both solvers

---

## 📁 File Structure

```
psi_solve2/
├── verify_comparison.py          ← Main comparison script
├── Comparison_Notebook.ipynb     ← Interactive notebook
├── comparison_log.txt            ← Output results
├── app.py                        ← Streamlit app (includes QMSolve tab)
└── requirements.txt              ← Includes qmsolve

Hand-wave/
├── verify_comparison.py          ← Copy of comparison script
├── Comparison_Notebook.ipynb     ← Copy of notebook
└── comparison_log.txt            ← Copy of results
```

---

## 🚀 Quick Start

**To see QMSolve comparison results right now**:

```bash
cd d:\Documents\SFU\PHYS385_Quantum2\CodeProjects\psi_solve2
python verify_comparison.py
```

This will:
1. Run your solver on Double Well potential
2. Run QMSolve on same potential
3. Compare results
4. Save to `comparison_log.txt`

**Expected output**:
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
| 2 | 4.455252     | 4.466368     | 1.11e-02     | 0.2490%  |
| 3 | 6.917808     | 6.936807     | 1.90e-02     | 0.2739%  |
| 4 | 9.872632     | 9.900227     | 2.76e-02     | 0.2786%  |
-----------------------------------------------------------------
```

---

## Summary

✅ **QMSolve comparison code exists** in `psi_solve2/verify_comparison.py`  
✅ **Interactive notebook** available at `psi_solve2/Comparison_Notebook.ipynb`  
✅ **Results show < 0.3% difference** - excellent validation!  
✅ **Also displayed** in Streamlit app under "Benchmarks & Verification"  

Your solver has been **cross-verified** against an established quantum mechanics package and shows excellent agreement! 🎉
