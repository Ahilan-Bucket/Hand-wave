---
title: Hand-Wave Quantum Solver
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# ⚛️ Hand-Wave Quantum Solver

**Author:** Ahilan Kumaresan  
**Institution:** Simon Fraser University  
**Field:** Mathematical & Computational Physics

---

## 🎯 Overview

An **interactive quantum mechanics solver** that combines rigorous numerical methods with intuitive hand gesture input. This project demonstrates solving the **Time-Independent Schrödinger Equation** (TISE) using the Finite Difference Method, with comprehensive verification against analytical solutions.

**Unique Feature:** Use your **hands** to define quantum potentials via webcam!
- **ONE HAND (Pinch)** → Quantum Harmonic Oscillator
- **TWO HANDS (Spread)** → Square Well Potential

---

## 🔬 Physics Background

### What is Quantum Mechanics?

At the atomic scale, particles behave very differently from our everyday experience:
- **Energy is quantized** (only certain values allowed)
- **Wave-particle duality**: Particles exhibit wave-like properties
- **Heisenberg Uncertainty**: Cannot know position and momentum simultaneously
- **Probabilistic nature**: We describe systems with probability distributions

### The Schrödinger Equation

The **Time-Independent Schrödinger Equation (TISE)** governs stationary quantum states:

$$\hat{H}\psi(x) = E\psi(x)$$

where:
- $\hat{H}$ is the **Hamiltonian operator** (total energy)
- $\psi(x)$ is the **wavefunction** (quantum state)
- $E$ is the **energy eigenvalue**

Expanding the Hamiltonian:

$$\left[ -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x) \right]\psi(x) = E\psi(x)$$

**Components:**
- **Kinetic Energy**: $-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}$ (curvature of wavefunction)
- **Potential Energy**: $V(x)$ (external forces/confinement)

### Physical Interpretation

**Wavefunction** $\psi(x)$:
- Complex-valued function describing the quantum state
- Must be continuous and normalizable
- **Normalization**: $\int_{-\infty}^{\infty} |\psi(x)|^2 dx = 1$

**Probability Density** $|\psi(x)|^2$:
- $|\psi(x)|^2 dx$ = probability of finding particle between $x$ and $x+dx$

**Energy Quantization**:
- Only discrete energy values $E_n$ are allowed (for bound states)
- Ground state: $n=0$ (lowest energy)
- Excited states: $n=1, 2, 3, ...$ (higher energies)

**Nodes and Curvature**:
- Higher energy states have more **nodes** (zeros of $\psi$)
- More nodes → higher curvature → higher kinetic energy

---

## 📐 Numerical Method: Finite Differences

### Discretization

We approximate the continuous space with a **discrete grid**:

$$x_i = x_{\text{min}} + i\Delta x, \quad i = 0, 1, 2, ..., N$$

The wavefunction becomes a vector: $\psi(x) \rightarrow \vec{\psi} = [\psi_0, \psi_1, ..., \psi_N]$

### Second Derivative Approximation

Using the **3-point central difference stencil**:

$$\frac{d^2\psi}{dx^2}\bigg|_{x_i} \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{\Delta x^2}$$

This transforms the differential operator into a **tridiagonal matrix**:

$$\mathbf{D2} = \frac{1}{\Delta x^2}\begin{bmatrix}
-2 & 1 & 0 & \cdots \\
1 & -2 & 1 & \cdots \\
0 & 1 & -2 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

### Matrix Eigenvalue Problem

The TISE becomes:

$$\mathbf{H}\vec{\psi} = E\vec{\psi}$$

where the **Hamiltonian matrix** is:

$$\mathbf{H} = -\frac{\hbar^2}{2m}\mathbf{D2} + \text{diag}(V_1, V_2, ..., V_N)$$

**Solution Method:**
- Use `numpy.linalg.eigh()` (optimized for Hermitian matrices)
- Returns eigenvalues $E_n$ and eigenvectors $\vec{\psi}_n$
- Automatically sorted by energy

### Boundary Conditions

**Dirichlet Boundaries** (infinite walls):
- $\psi(x_{\text{min}}) = \psi(x_{\text{max}}) = 0$
- Particle cannot exist outside the simulation box
- Implemented by excluding boundary points from the eigenvalue problem

---

## 🛠️ Implementation Details

### Atomic Units

We use **Hartree Atomic Units** for simplicity:
- $\hbar = 1$ (reduced Planck constant)
- $m = 1$ (electron mass)
- $e = 1$ (elementary charge)

**Advantages:**
- Simplifies equations (no physical constants)
- Natural units for atomic-scale systems
- Easy conversion to SI units when needed

### Grid Parameters

**Default Configuration:**
- **Domain**: $x \in [-25, 25]$ a.u. (≈ 2.6 nm)
- **Grid Points**: $N = 2000$ (interior points)
- **Spacing**: $\Delta x \approx 0.025$ a.u. (≈ 1.3 pm)

**Convergence:**
- Tested with $N = 500, 1000, 2000$
- Energy errors < 0.003% for well-behaved potentials

### Potential Types

#### 1. Infinite Square Well
$$V(x) = \begin{cases} 0 & |x| < a/2 \\ \infty & |x| \geq a/2 \end{cases}$$

**Analytical Solution:**
$$E_n = \frac{\hbar^2\pi^2 n^2}{2ma^2}, \quad \psi_n(x) = \sqrt{\frac{2}{a}}\sin\left(\frac{n\pi x}{a}\right)$$

#### 2. Quantum Harmonic Oscillator
$$V(x) = \frac{1}{2}kx^2$$

**Analytical Solution:**
$$E_n = \hbar\omega\left(n + \frac{1}{2}\right), \quad \omega = \sqrt{\frac{k}{m}}$$

#### 3. Finite Square Well
$$V(x) = \begin{cases} -V_0 & |x| < a/2 \\ 0 & |x| \geq a/2 \end{cases}$$

**Features:**
- Bound states: $E < 0$
- **Tunneling**: Wavefunction extends into classically forbidden region
- Number of bound states depends on $V_0$ and $a$

#### 4. Hand-Gesture Potentials
- **ONE HAND (Pinch)**: Creates QHO with adjustable curvature
- **TWO HANDS (Spread)**: Creates square well with adjustable width
- Real-time capture using MediaPipe hand tracking

---

## 📊 Verification & Accuracy

### Analytical Benchmarks

| System | Grid Points | Max Error | Test Cases |
|--------|-------------|-----------|------------|
| Infinite Square Well | 2000 | < 0.003% | $n = 1-10$ |
| Harmonic Oscillator | 2000 | < 0.02% | $n = 0-9$ |
| Half-Harmonic | 2000 | < 0.8% | $n = 0-5$ |
| Triangular Potential | 2000 | < 0.003% | $n = 1-5$ |

### External Library Comparison

**Cross-verification with QMSolve:**
- Double Well Potential: Agreement within 0.25%
- Demonstrates accuracy for systems without analytical solutions

### Orthonormality Check

Eigenstates satisfy:
$$\langle \psi_m | \psi_n \rangle = \delta_{mn}$$

**Measured Overlap Matrix:**
- Diagonal elements: $\approx 1.0000$
- Off-diagonal elements: $< 10^{-10}$

---

## 🎓 Educational Applications

### Concepts Demonstrated

1. **Energy Quantization**
   - Only discrete energies allowed for bound states
   - Emerges from boundary conditions

2. **Wavefunction Nodes**
   - Ground state: No interior nodes
   - $n$-th excited state: $n$ interior nodes
   - Higher energy → more oscillations

3. **Quantum Tunneling**
   - Finite wells: Wavefunction penetrates barriers
   - Exponential decay in forbidden region
   - Probability of finding particle where classically impossible

4. **Uncertainty Principle**
   - Tighter confinement → higher kinetic energy
   - Cannot localize particle without increasing momentum spread

### Classroom Use

**Suitable for:**
- Undergraduate Quantum Mechanics (PHYS 385)
- Computational Physics courses
- Graduate-level numerical methods

**Learning Outcomes:**
- Understand finite difference discretization
- Visualize quantum wavefunctions
- Explore parameter effects interactively
- Verify numerical methods against theory

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Hand-wave.git
cd Hand-wave

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
streamlit run app.py
```

### Using the Jupyter Notebook

```bash
jupyter notebook Handwave_Presentation.ipynb
```

**Notebook Workflow:**
1. Initialize grid and operators
2. Capture hand gesture (or use predefined potential)
3. Solve Schrödinger equation
4. Visualize results

---

## 📁 Repository Structure

```
Hand-wave/
├── app.py                          # Streamlit web application
├── functions.py                    # Core physics engine
├── camera_functions.py             # Hand gesture capture
├── Handwave_Presentation.ipynb     # Interactive notebook
├── ModularedPresentation.ipynb     # Detailed theory notebook
├── verify_physics.py               # Analytical verification
├── verify_comparison.py            # QMSolve comparison
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🎯 Usage Examples

### Example 1: Infinite Square Well

```python
from functions import *

# Setup
x, dx, x_int = make_grid(x_min=-10, x_max=10, N_GRID=1000)
T = kinetic_operator(len(x_int), dx)

# Define potential
V = inf_square_well(x_int, lower_bound=-5, upper_bound=5)
V_full = np.pad(V, (1, 1), constant_values=1e10)

# Solve
E, psi = solve(T, V_full, dx)

# Plot
plot_alive(E, psi, V_full, x, nos=5, mode='all')
```

### Example 2: Hand Gesture Input

```python
from camera_functions import *

# Capture potential
V_raw = capture_hand_potential(A_MIN=0, A_MAX=100)

# Solve
E, psi, V_full = verify_and_plot_capture(V_raw, x, dx, T)

# Visualize
plot_alive(E, psi, V_full, x, nos=5, mode='all')
```

---

## 📝 Citation

If you use this solver in your research, please cite:

```bibtex
@software{kumaresan2024handwave,
  author = {Kumaresan, Ahilan},
  title = {Hand-Wave Quantum Solver: Interactive Finite Difference 
           Implementation for the Schrödinger Equation},
  year = {2024},
  institution = {Simon Fraser University},
  url = {https://github.com/yourusername/Hand-wave}
}
```

---

## 📧 Contact

**Ahilan Kumaresan**  
Mathematical & Computational Physics  
Simon Fraser University


<<<<<<< HEAD
## 🙏 Acknowledgments

- **MediaPipe** (Google) for hand tracking
- **QMSolve** library for verification
- **NumPy/SciPy** for numerical linear algebra
- **Streamlit** for web framework

---

*This project showcases advanced computational physics methodology suitable for graduate-level research in quantum mechanics and numerical analysis.*
=======
>>>>>>> a615dc9a62d4ab49ed3711b119d99694cd987197
