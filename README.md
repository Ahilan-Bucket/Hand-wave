
# ⚛️ Quantum Potential Solver

**Author:** Ahilan Kumaresan  
**Institution:** Simon Fraser University  
**Field:** Mathematical & Computational Physics

---

## 🎯 Overview

An advanced numerical solver for the **Time-Independent Schrödinger Equation** (TISE) using the Finite Difference Method. This project demonstrates rigorous computational physics methodology with comprehensive verification against analytical solutions and external libraries.

## 🔬 Key Features

### 1. **Accurate Numerical Solver**
- **Method:** 3-point Central Difference stencil for the Laplacian operator
- **Grid:** Adaptive resolution (1000-2000 points)
- **Units:** Hartree Atomic Units (ℏ=1, m=1)
- **Boundary Conditions:** Dirichlet (infinite walls)

### 2. **Multiple Potential Types**
- Infinite Square Well
- Harmonic Oscillator
- Double Well Potential
- Custom potentials via hand gestures (camera input)

### 3. **Interactive Visualizations**
- Plotly-based interactive charts
- Wavefunction probability densities
- Energy level diagrams
- Real-time solver performance metrics

### 4. **Rigorous Verification**

#### Analytical Benchmarks
| System | Max Error |
|--------|-----------|
| Infinite Square Well | < 0.003% |
| Harmonic Oscillator | < 0.02% |
| Half-Harmonic Oscillator | < 0.8% |
| Triangular Potential | < 0.003% |

#### External Library Comparison
- **Cross-verified with QMSolve** (Python quantum mechanics package)
- **Double Well Potential:** Agreement within 0.25%
- Demonstrates accuracy for systems without analytical solutions

## 📐 Mathematical Foundation

The solver discretizes the TISE:

$$\hat{H}\psi(x) = E\psi(x)$$

$$\left[ -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x) \right]\psi(x) = E\psi(x)$$

Using finite differences:

$$\frac{d^2\psi}{dx^2} \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{\Delta x^2}$$

This transforms the problem into a matrix eigenvalue equation solved via `numpy.linalg.eigh`.

## 🛠️ Technical Implementation

- **Language:** Python 3.12
- **Core Libraries:** NumPy, SciPy
- **Visualization:** Plotly, Matplotlib
- **UI Framework:** Streamlit
- **Computer Vision:** MediaPipe (for hand gesture input)

## 📊 Verification Scripts

The project includes comprehensive verification:
- `verify_physics.py`: Analytical benchmarks
- `verify_comparison.py`: QMSolve cross-verification
- `Comparison_Notebook.ipynb`: Jupyter notebook with detailed analysis

## 🎓 Academic Applications

This project demonstrates:
1. **Numerical Methods:** Finite difference discretization of differential operators
2. **Linear Algebra:** Eigenvalue problems for Hermitian matrices
3. **Quantum Mechanics:** Stationary states and energy quantization
4. **Software Engineering:** Modular design, comprehensive testing, professional documentation

## 📖 Usage

### Interactive Simulator
1. Select a potential type from the sidebar
2. Adjust parameters using sliders
3. View real-time solutions with interactive plots

### Verification Dashboard
- Navigate to "Benchmarks & Verification" to see accuracy metrics
- Compare against analytical solutions and QMSolve

### Theory Section
- Detailed mathematical background
- Implementation methodology
- Code verification

## 🔗 Repository Structure

```
psi_solve2/
├── app.py                      # Main Streamlit application
├── functions.py                # Physics engine (solver + potentials)
├── verify_physics.py           # Analytical verification script
├── verify_comparison.py        # QMSolve comparison script
├── Comparison_Notebook.ipynb   # Jupyter analysis notebook
└── requirements.txt            # Python dependencies
```

## 📝 Citation

If you use this solver in your research, please cite:

```
Kumaresan, A. (2024). Quantum Potential Solver: A Verified Finite Difference 
Implementation for the Time-Independent Schrödinger Equation. 
Simon Fraser University.
```

## 📧 Contact

**Ahilan Kumaresan**  
Mathematical & Computational Physics  
Simon Fraser University

---

*This project showcases advanced computational physics methodology suitable for graduate-level research in quantum mechanics and numerical analysis.*
