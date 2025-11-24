# %% [markdown]

# # Numerical Schrödinger Solver – Just an Infinite Potential Square Well

#

# This is my working notes for building a **numerical Schrödinger solver** from first principles, starting with the simplest case:

#

# - A **1D infinite square well**, and

# - A clean path from the **differential equation** to a **matrix eigenvalue problem** that the computer can solve.

#

# Ultimately I want:

#

# - A clean, modular solver that works for **custom 1D potentials**.

# - Later: **GPU acceleration**, a nicer **UI**, and interactive knobs.

#

# Code-wise, I imagine something like:

#

# - `potentials.py` – all the functions that define different ( V(x) )

# - `solver.py` – builds the Hamiltonian and solves ( H\psi = E\psi )

# - `main_app.py` – glues things together (UI, MediaPipe input, plotting)

#

# For now, I’m focusing on:

#

# - Getting the **math and numerics** right for the infinite square well.

# - Explaining the full chain so another student can start from here and extend it.

#

# Technical / numerical TODO that this write-up connects to:

#

# - Rigorous boundary-condition (BC) enforcement (Dirichlet for infinite well)

# - Orthonormality diagnostic plots

# - Convergence analysis

# - e.g. compare ( N = 500 ) vs ( N = 2000 ) and see how the discrete operator ( \mathbf{D2} ) behaves

# - Analytic benchmarking

# - For the infinite square well: compare numerical ( E_n ) to

#

# $$

# E_n = \frac{\hbar^2 \pi^2 n^2}{2 m L^2}

# $$

#

# - Documentation + modular design (this markdown is part of that)

# %% [markdown]

# ## 1. First-principles picture of ( V(x) )

#

# I want to keep an intuitive, almost classical picture of what the potential ( V(x) ) is doing.

#

# - ( V(x) ) is the **potential energy landscape** created by the environment.

# - I like to think of it as a **height map** or **hill profile** in 1D.

#

# Concretely:

#

# - If ( V(x) = 2 ) (some constant), then:

# - Placing a particle there means its potential energy is 2 (in my chosen units).

# - If the total energy is ( E ), then the kinetic energy is

#

# $$

# K(x) = E - V(x).

# $$

#

# - So where ( V(x) ) is larger, there is less kinetic energy “available”.

#

# Rough intuition:

#

# - ( V(x) = 0 )

# → flat ground, “easy to move”.

# - ( V(x) = 2 )

# → you are walking on a slope with height 2; you need more total energy to have the same kinetic energy.

# - It is always “easier” (in energy terms) for the particle to be where ( V(x) ) is low.

#

# Now, the extreme case:

#

# - If ( V(x) = \infty ) in some region, then:

# - A particle would need **infinite total energy** to be there.

# - In quantum mechanics this is implemented by forcing the wavefunction to be **exactly zero** in that region.

# - So the particle simply **cannot exist** there.

#

# This is exactly how the **infinite square well** is defined: a finite region where ( V = 0 ), and “infinite walls” outside.

# %% [markdown]

# ## 2. The infinite square well setup

#

# Think of an electron (mass ( m )) trapped in an infinite square well.

#

# I define the potential:

#

# - Inside the well, for ( 0 < x < a ):

#

# $$

# V(x) = 0

# $$

#

# - Outside the well:

#

# $$

# V(x) = \infty

# $$

#

# So the particle is strictly confined to the region ( 0 < x < a ).

# The wavefunction will live only on that interval, and will be zero outside.

# %% [markdown]

# ### 2.1 Time-Independent Schrödinger Equation (TISE)

#

# The time-independent Schrödinger equation is

#

# $$

# H \Psi(x) = E \Psi(x)

# $$

#

# where the Hamiltonian is

#

# $$

# H = -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} + V(x).

# $$

#

# Inside the infinite well, ( V(x) = 0 ), so the equation becomes:

#

# $$

# -\frac{\hbar^2}{2m} \frac{d^2 \Psi}{dx^2} = E \Psi.

# $$

#

# This is the differential equation I want the **computer** to solve numerically on a grid, instead of solving it analytically.

# %% [markdown]

# ### 2.2 Boundary conditions and physical constraints

#

# The shape of the wavefunction is not arbitrary; it is constrained by physics:

#

# **1. Dirichlet boundary conditions**

#

# For the infinite well, we impose

#

# $$

# \Psi(0) = \Psi(a) = 0.

# $$

#

# Why?

#

# - Outside the well, ( V(x) = \infty ), so the wavefunction must vanish:

#

# $$

# \Psi(0^-) = 0, \quad \Psi(a^+) = 0.

# $$

#

# - Schrödinger’s equation requires ( \Psi(x) ) to be **continuous**.

# - If ( \Psi(x) ) had a jump, its second derivative would involve something like a delta function,

# which would blow up the kinetic energy term.

# - Continuity across the boundaries then implies

#

# $$

# \Psi(0^+) = 0, \quad \Psi(a^-) = 0,

# $$

#

# so we simply write

#

# $$

# \Psi(0) = \Psi(a) = 0.

# $$

#

# These are Dirichlet boundary conditions: the wavefunction is pinned to zero at the walls.

#

# ---

#

# **2. Normalization**

#

# The wavefunction must also be normalized:

#

# $$

# \int_{-\infty}^{\infty} |\Psi(x)|^2 , dx = 1.

# $$

#

# For the infinite well, this becomes

#

# $$

# \int_0^a |\Psi(x)|^2 , dx = 1.

# $$

#

# This is the statement that the total probability of finding the particle somewhere in the well is 1.

#

# ---

#

# **3. Probabilistic interpretation**

#

# - ( |\Psi(x)|^2 , dx ) is the probability of finding the particle between ( x ) and ( x + dx ).

# - The shape of ( |\Psi(x)|^2 ) tells us “where the particle likes to be” inside the well.

# %% [markdown]

# ### 2.3 Analytic solutions (for benchmarking the numerics)

#

# For the infinite square well, the analytic solutions are standard and very clean.

# We will use them later to **check** that the numerical solver is doing the right thing.

#

# **Wavefunctions:**

#

# $$

# \Psi_n(x) = \sqrt{\frac{2}{a}} \sin\left(\frac{n \pi x}{a}\right), \quad n = 1, 2, 3, \dots

# $$

#

# These obey:

#

# - ( \Psi_n(0) = \Psi_n(a) = 0 ) (boundary conditions),

# - and are orthonormal on ( [0,a] ).

#

# **Energy levels:**

#

# $$

# E_n = \frac{\hbar^2 \pi^2 n^2}{2 m a^2}.

# $$

#

# In the code, after I diagonalize the Hamiltonian matrix, I can compare the numerical eigenvalues ( E_n^\text{num} ) to this analytic formula.

# If they match (up to small numerical error), I know my discretization, boundary conditions, and solver are behaving properly.

# %% [markdown]

# ## 3. Turning the TISE into something the computer understands

#

# The Schrödinger equation is a **differential equation**.

# The computer is happier with **matrices and vectors**.

#

# The plan:

#

# 1. Break continuous space into **grid points**.

# 2. Approximate derivatives using **finite differences**.

# 3. Turn the second derivative operator into a **matrix**.

# 4. Add the potential ( V(x) ) as a diagonal matrix.

# 5. Solve the matrix eigenvalue problem

#

# $$

# H \Psi = E \Psi

# $$

#

# with standard linear algebra tools (`numpy.linalg.eigh`).

#

# This section is about steps 1–3.

# %% [markdown]

# ### 3.1 Discretizing the derivative

#

# Start from the usual derivative definition:

#

# $$

# f'(x_i) = \lim_{\Delta x \to 0} \frac{f(x_i + \Delta x) - f(x_i)}{\Delta x}.

# $$

#

# On a computer, I **do not** send ( \Delta x \to 0 ). Instead:

#

# - I choose a **fixed grid spacing** ( \Delta x ).

# - I approximate derivatives using finite differences.

#

# A simple **forward-difference** formula for the first derivative is:

#

# $$

# f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{\Delta x}.

# $$

#

# There are also backward and central differences.

# For the **second derivative**, we usually like the **central difference** because it is more symmetric and more accurate for the same grid spacing.

# %% [markdown]

# ### 3.2 Building the second derivative ( f''(x_i) )

#

# Because ( \Delta x ) is constant and differentiation is a linear operation, we can write:

#

# $$

# f''(x_i) \approx \frac{f'(x_{i+1}) - f'(x_i)}{\Delta x}.

# $$

#

# Now plug in finite-difference expressions for the first derivative:

#

# $$

# f'(x_{i+1}) \approx \frac{f(x_{i+2}) - f(x_{i+1})}{\Delta x},

# $$

#

# $$

# f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{\Delta x}.

# $$

#

# Then

#

# $$

# f''(x_i)

# \approx

# \frac{

# \dfrac{f(x_{i+2}) - f(x_{i+1})}{\Delta x}

# -

# \dfrac{f(x_{i+1}) - f(x_i)}{\Delta x}

# }{\Delta x}

# =

# \frac{f(x_{i+1}) - 2 f(x_i) + f(x_{i-1})}{\Delta x^2}.

# $$

#

# This is the standard **central difference** formula for the second derivative.

# It is exactly what I use to approximate the operator ( \dfrac{d^2}{dx^2} ) on my grid.

# %% [markdown]

# ### 3.3 Writing the second derivative as a matrix

#

# Now I reorganize the finite-difference formula into **matrix form**.

#

# Let the grid points be

#

# $$

# x_1, x_2, x_3, x_4, \dots

# $$

#

# and the sampled function values be collected into a vector

#

# $$

# f(x_i) \rightarrow

# f_i =

# \begin{bmatrix}

# f_1 \

# f_2 \

# f_3 \

# f_4 \

# \vdots

# \end{bmatrix}.

# $$

#

# The central difference formula says:

#

# $$

# \begin{cases}

# f''_1 = \dfrac{f_2 - 2 f_1 + f_0}{(\Delta x)^2}, \

# f''_2 = \dfrac{f_3 - 2 f_2 + f_1}{(\Delta x)^2}, \

# f''_3 = \dfrac{f_4 - 2 f_3 + f_2}{(\Delta x)^2}, \

# \vdots

# \end{cases}

# $$

#

# so the pattern is a stencil ([1, -2, 1]) sliding across the vector.

#

# In matrix form:

#

# $$

# \frac{d^2}{dx^2} f(x_i)

# \approx

# f''(x_i)

# =

# \frac{1}{(\Delta x)^2}

# \begin{bmatrix}

# -2 & 1  & 0  & 0  & \cdots \

# 1  & -2 & 1  & 0  & \cdots \

# 0  & 1  & -2 & 1  & \cdots \

# 0  & 0  & 1  & -2 & \cdots \

# \vdots & \vdots & \vdots & \vdots & \ddots

# \end{bmatrix}

# \begin{bmatrix}

# f_1 \

# f_2 \

# f_3 \

# f_4 \

# \vdots

# \end{bmatrix}.

# $$

#

# This tridiagonal matrix is my discrete version of the second derivative operator.

# %% [markdown]

# ### 3.4 Schrödinger equation in matrix form

#

# Start from the continuous TISE:

#

# $$

# -\frac{\hbar^2}{2m} \frac{d^2}{dx^2} \Psi(x) + V(x) \Psi(x) = E \Psi(x).

# $$

#

# After discretization:

#

# - The kinetic term

#

# $$

# -\frac{\hbar^2}{2m} \frac{d^2}{dx^2}

# $$

#

# becomes (using the matrix from the previous section)

#

# $$

# -\frac{\hbar^2}{2m}

# \frac{1}{(\Delta x)^2}

# \begin{bmatrix}

# -2 & 1  & 0  & 0  & \cdots \

# 1  & -2 & 1  & 0  & \cdots \

# 0  & 1  & -2 & 1  & \cdots \

# 0  & 0  & 1  & -2 & \cdots \

# \vdots & \vdots & \vdots & \vdots & \ddots

# \end{bmatrix}.

# $$

#

# - The potential term ( V(x) \Psi(x) ) becomes a **diagonal matrix** acting on the same vector:

#

# $$

# V(x) \Psi(x)

# ;\longrightarrow;

# \text{diag}\big(V(x_1), V(x_2), \dots\big)

# \begin{bmatrix}

# \Psi_1 \ \Psi_2 \ \vdots

# \end{bmatrix}.

# $$

#

# Putting this together, the matrix version of the TISE is

#

# $$

# H \Psi = E \Psi

# $$

#

# with

#

# $$

# H =

# -\frac{\hbar^2}{2m}

# \frac{1}{(\Delta x)^2}

# \begin{bmatrix}

# -2 & 1  & 0  & 0  & \cdots \

# 1  & -2 & 1  & 0  & \cdots \

# 0  & 1  & -2 & 1  & \cdots \

# 0  & 0  & 1  & -2 & \cdots \

# \vdots & \vdots & \vdots & \vdots & \ddots

# \end{bmatrix}

# +

# \text{diag}\big(V(x_1), V(x_2), \dots \big).

# $$

#

# Notes:

#

# - This matrix acts on the **internal grid points** ( \Psi_1, \dots, \Psi_{N-1} ).

# - The boundary values ( \Psi_0 ) and ( \Psi_N ) are fixed by the **Dirichlet BC**:

#

# $$

# \Psi_0 = \Psi_N = 0.

# $$

#

# - In other words:

# - The computer solves for the interior values.

# - The boundary conditions are imposed by construction (the walls are built into the grid and the potential).

#

# From here, the numerical method is:

#

# - Build this ( H ) matrix for our chosen ( V(x) ).

# - Feed it to `numpy.linalg.eigh`.

# - Interpret the eigenvalues as energies ( E_n ) and eigenvectors as sampled wavefunctions ( \Psi_n(x_i) ).

# - Then benchmark against the analytic infinite square well solution.
