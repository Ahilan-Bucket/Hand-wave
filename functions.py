# psi_solve2/functions.py

import numpy as np
import matplotlib.pyplot as plt
import cv2 # Kept for potential vision functions you might use later
import mediapipe as mp # Kept for potential vision functions you might use later
import math 
from matplotlib.ticker import MultipleLocator 

# ==========================================
# 1. PHYSICS CONSTANTS
# ==========================================
hbar = 1
m = 1
L = 50
N_GRID = 2000
global Last_k_value # Used by harmonic() and check_harmonic_analytic()

# ==========================================
# 2. GRID FUNCTIONS
# ==========================================
def make_grid(L=L, N=N_GRID):
    """
    Create a spatial grid for solving the Schrödinger equation.
    
    Parameters
    ----------
    L : float, optional
        Total length of the spatial domain (default: 50 a.u.)
    N : int, optional
        Number of internal grid points (default: 2000)
    
    Returns
    -------
    x_full : ndarray
        Full grid with N+2 points from -L/2 to L/2, including boundary points
    dx : float
        Grid spacing (distance between adjacent points)
    x_internal : ndarray
        Internal grid points (N points) where the wavefunction is solved
        Excludes the boundary points at x[0] and x[-1]
    
    Notes
    -----
    The boundary points are used to enforce boundary conditions (typically ψ=0)
    while x_internal contains the points where we actually solve for ψ.
    
    Examples
    --------
    >>> x, dx, x_int = make_grid(L=20, N=1000)
    >>> print(f"Domain: [{x[0]:.1f}, {x[-1]:.1f}], spacing: {dx:.4f}")
    Domain: [-10.0, 10.0], spacing: 0.0200
    """
    x = np.linspace(-L/2, L/2, N+2)
    dx = x[1] - x[0]
    x_internal = x[1:-1]
    return x, dx, x_internal

# ==========================================
# 3. POTENTIAL GENERATORS (V(x))
# ==========================================
def constant(x, c):
    """
    Create a constant potential across the entire domain.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    c : float
        Constant potential value (in Hartree atomic units)
    
    Returns
    -------
    V : ndarray
        Constant potential array of same shape as x, with value c everywhere
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 100)
    >>> V = constant(x, 5.0)  # V(x) = 5.0 everywhere
    """
    return np.ones_like(x) * c

def harmonic(x, k, center=0.0):
    """
    Create a harmonic oscillator (parabolic) potential.
    
    Generates V(x) = (1/2)k(x - center)² representing a quantum harmonic
    oscillator potential centered at the specified position.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    k : float
        Spring constant (curvature parameter) in atomic units
        Larger k → stiffer spring → more tightly bound states
    center : float, optional
        Center position of the parabola (default: 0.0)
    
    Returns
    -------
    V : ndarray
        Harmonic potential array: V(x) = 0.5 * k * (x - center)²
    
    Notes
    -----
    - Sets global variable Last_k_value for use by check_harmonic_analytic()
    - Energy levels: E_n = ℏω(n + 1/2) where ω = √(k/m)
    - In atomic units (ℏ=1, m=1): ω = √k
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V = harmonic(x, k=1.0, center=0.0)  # Standard QHO
    >>> V_stiff = harmonic(x, k=10.0, center=0.0)  # Stiffer spring
    >>> V_offset = harmonic(x, k=1.0, center=5.0)  # Centered at x=5
    """
    global Last_k_value
    Last_k_value = k
    
    constant_factor = 1 
    potential = 0.5 * k * (x - center)**2
    return constant_factor * potential

def gaussian_well(x, center=0.0, width=1.0, depth=50): 
    """
    Create a Gaussian-shaped potential well.
    
    Generates a smooth, bell-shaped potential dip that can trap particles.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    center : float, optional
        Center position of the well (default: 0.0)
    width : float, optional
        Width parameter (standard deviation) of the Gaussian (default: 1.0)
        Larger width → broader well
    depth : float, optional
        Depth of the well at the center (default: 50)
        Positive depth creates a well (attractive potential)
    
    Returns
    -------
    V : ndarray
        Gaussian well potential: V(x) = -depth * exp(-(x-center)²/(2*width²))
    
    Notes
    -----
    - Minimum potential is -depth at x = center
    - Potential approaches 0 as |x - center| → ∞
    - Smooth potential (infinitely differentiable)
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V = gaussian_well(x, center=0, width=2.0, depth=10)
    """
    return -depth * np.exp(-(x - center)**2 / (2 * width**2))

def inf_sqaure_well(x, lower_bound, upper_bound):
    """
    Create an infinite square well (particle in a box) potential.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    lower_bound : float
        Left boundary of the well
    upper_bound : float
        Right boundary of the well
    
    Returns
    -------
    V : ndarray
        Infinite square well potential:
        - V(x) = 0 for lower_bound ≤ x ≤ upper_bound (inside well)
        - V(x) = 10¹⁰ for x < lower_bound or x > upper_bound (outside well)
    
    Notes
    -----
    - Uses penalty method: "infinite" walls are approximated by very large
      potential (10¹⁰) to enforce ψ ≈ 0 outside the well
    - Well width: L = upper_bound - lower_bound
    - Analytical energies: E_n = (ℏ²π²n²)/(2mL²) for n = 1, 2, 3, ...
    
    Examples
    --------
    >>> x = np.linspace(-15, 15, 1000)
    >>> V = inf_sqaure_well(x, lower_bound=-10, upper_bound=10)  # L = 20
    >>> # Use with check_ISW_analytic(E, lower_bound=-10, upper_bound=10)
    """
    HUGE_NUMBER = 1e10
    V = np.zeros_like(x) 
    V[x < lower_bound] = HUGE_NUMBER
    V[x > upper_bound] = HUGE_NUMBER
    return V

def inf_wall(x, side, bound):
    """
    Place an infinite potential wall on one side of the domain.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    side : str
        Which side to place the wall: 'left' or 'right'
        (case-insensitive, strips whitespace and punctuation)
    bound : float
        Position of the wall boundary
    
    Returns
    -------
    V : ndarray
        Potential with infinite wall:
        - If side='left': V(x) = 10¹⁰ for x < bound, V(x) = 0 for x ≥ bound
        - If side='right': V(x) = 10¹⁰ for x > bound, V(x) = 0 for x ≤ bound
    
    Notes
    -----
    Uses penalty method with V = 9×10¹⁰ to approximate infinite potential.
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V_left = inf_wall(x, 'left', bound=-5)  # Wall at x=-5, blocks left side
    >>> V_right = inf_wall(x, 'right', bound=5)  # Wall at x=5, blocks right side
    """
    V = np.zeros_like(x)
    HUGE_NUMBER = 9e10 
    side = side.strip(', . ').lower() 

    if side == 'left':
        V[x < bound] = HUGE_NUMBER
    elif side == 'right':
        V[x > bound] = HUGE_NUMBER
    return V

def finite_barrier(x, center, width, height):
    """
    Create a finite rectangular potential barrier.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    center : float
        Center position of the barrier
    width : float
        Total width of the barrier
    height : float
        Height of the potential barrier
    
    Returns
    -------
    V : ndarray
        Rectangular barrier potential:
        - V(x) = height for |x - center| < width/2
        - V(x) = 0 elsewhere
    
    Notes
    -----
    Useful for studying quantum tunneling phenomena. Particles with E < height
    can tunnel through the barrier with exponentially decaying probability.
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V = finite_barrier(x, center=0, width=2, height=5)  # Barrier from x=-1 to x=1
    """
    V = np.zeros_like(x)
    mask = (x > (center - width/2)) & (x < (center + width/2))
    V[mask] = height
    return V

def V_double_well(x, depth=20, separation=1, center=0.0):
    """
    Create a quartic double-well potential.
    
    Generates V(x) = depth × ((x-center)² - separation)² which has two minima
    separated by a central barrier.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    depth : float, optional
        Depth parameter controlling overall potential strength (default: 20)
    separation : float, optional
        Controls the distance between the two wells (default: 1)
        Well minima are approximately at x = center ± separation
    center : float, optional
        Center position of the double well system (default: 0.0)
    
    Returns
    -------
    V : ndarray
        Double well potential: V(x) = depth × ((x-center)² - separation)²
    
    Notes
    -----
    - Creates symmetric double well with barrier at x = center
    - Useful for studying tunneling splitting and symmetric/antisymmetric states
    - Ground state and first excited state form tunneling doublet
    
    Examples
    --------
    >>> x = np.linspace(-5, 5, 1000)
    >>> V = V_double_well(x, depth=2, separation=1, center=0)
    """
    V = depth * ((x - center)**2 - separation)**2
    return V

def custom2(value,x):
    """Helper function from the notebook."""
    return value * np.ones_like(x)

# In psi_solve2/functions.py

def finite_square_well(x, lower_bound, upper_bound, depth_V):
    """
    Create a finite square well potential.
    
    The potential is zero inside the well and has finite height depth_V outside.
    Unlike the infinite square well, particles can exist in the barrier region
    with exponentially decaying wavefunctions.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    lower_bound : float
        Left boundary of the well
    upper_bound : float
        Right boundary of the well
    depth_V : float
        Height of the potential barriers outside the well (V₀)
    
    Returns
    -------
    V : ndarray
        Finite square well potential:
        - V(x) = 0 for lower_bound ≤ x ≤ upper_bound (inside well)
        - V(x) = depth_V for x < lower_bound or x > upper_bound (barrier regions)
    
    """
    """    
    Notes
    -----
    - Bound states exist only when E < depth_V
    - Number of bound states depends on well width and depth_V
    - For bound states, wavefunction decays exponentially in barrier (E < V)
    - For scattering states (E > depth_V), wavefunction oscillates everywhere
    - Use check_finite_well_analytic() to verify numerical results
    
    Examples
    --------
    >>> x = np.linspace(-15, 15, 1000)
    >>> V_deep = finite_square_well(x, -10, 10, depth_V=2.0)  # Deep well, many bound states
    >>> V_shallow = finite_square_well(x, -10, 10, depth_V=0.01)  # Shallow, few/no bound states
    """
    # Start with a baseline of zero potential
    V = np.zeros_like(x) 
    
    # The walls outside the well are set to the height/depth V_0
    V[x < lower_bound] = depth_V
    V[x > upper_bound] = depth_V
    
    # The potential *inside* the well remains V=0 (or whatever you set the baseline to)
    return V

# ==========================================
# 4. SCHRÖDINGER EQUATION SOLVER
# ==========================================
def kinetic_operator(N, dx, hbar=hbar, m=m):
    """
    Build the kinetic energy operator matrix using finite difference method.
    
    Constructs the discrete representation of the kinetic energy operator
    T = -(ℏ²/2m) d²/dx² using a 3-point central difference stencil.
    
    Parameters
    ----------
    N : int
        Number of internal grid points (size of the matrix)
    dx : float
        Grid spacing (distance between adjacent points)
    hbar : float, optional
        Reduced Planck constant (default: 1.0 in atomic units)
    m : float, optional
        Particle mass (default: 1.0 in atomic units)
    
    Returns
    -------
    T : ndarray, shape (N, N)
        Kinetic energy operator matrix (symmetric, tridiagonal)
        - Diagonal elements: -(ℏ²/2m) × (-2/dx²)
        - Off-diagonal elements: -(ℏ²/2m) × (1/dx²)
    
    """
    """    
    Notes
    -----
    The second derivative is approximated using central differences:
        d²ψ/dx² ≈ (ψ_{i+1} - 2ψ_i + ψ_{i-1}) / dx²
    
    This creates a tridiagonal matrix:
        - Main diagonal: -2/dx²
        - Upper/lower diagonals: +1/dx²
    
    The kinetic energy operator is then: T = -(ℏ²/2m) × D2
    
    Examples
    --------
    >>> N = 1000
    >>> dx = 0.025
    >>> T = kinetic_operator(N, dx)
    >>> print(f"Matrix shape: {T.shape}, Symmetric: {np.allclose(T, T.T)}")
    Matrix shape: (1000, 1000), Symmetric: True
    """
    main_diagonal = (1/dx**2) * np.diag(-2 * np.ones(N))
    off_diagonal1 = (1/dx**2) * np.diag(np.ones(N-1), -1)
    off_diagonal2 = (1/dx**2) * np.diag(np.ones(N-1), 1)
    D2 = (main_diagonal + off_diagonal1 + off_diagonal2)

    T = (-(hbar**2 / (2*m)) * D2)
    return T

def solve(T, V_full, dx):
    """
    Solve the time-independent Schrödinger equation for eigenvalues and eigenvectors.
    
    Solves the eigenvalue problem Hψ = Eψ where H = T + V is the Hamiltonian.
    Returns normalized eigenstates sorted by energy.
    
    Parameters
    ----------
    T : ndarray, shape (N, N)
        Kinetic energy operator matrix from kinetic_operator()
    V_full : ndarray, shape (N+2,)
        Full potential array including boundary points
        V_full[0] and V_full[-1] are boundary values (typically very large)
        V_full[1:-1] are the internal potential values
    dx : float
        Grid spacing used for normalization
    
    Returns
    -------
    E : ndarray, shape (N,)
        Eigenvalues (energy levels) sorted in ascending order
        Units: Hartree (atomic units)
    psi : ndarray, shape (N, N)
        Eigenvectors (wavefunctions) as columns
        psi[:, i] is the wavefunction for energy E[i]
        Each wavefunction is normalized: ∫|ψ|² dx = 1
    
    """
    """    
    Notes
    -----
    - Uses np.linalg.eigh() which assumes Hermitian matrix (guaranteed for H)
    - Automatically sorts eigenvalues and eigenvectors by energy
    - Normalizes each eigenstate using trapezoidal rule: ∫|ψ|² dx = 1
    - Boundary conditions are enforced by V_full having large values at edges
    
    The Hamiltonian is constructed as:
        H = T + diag(V_internal)
    where V_internal = V_full[1:-1]
    
    Examples
    --------
    >>> # Setup
    >>> x, dx, x_int = make_grid(L=20, N=1000)
    >>> T = kinetic_operator(len(x_int), dx)
    >>> 
    >>> # Create infinite square well
    >>> V = inf_sqaure_well(x_int, -10, 10)
    >>> V_full = np.pad(V, (1,1), constant_values=1e10)
    >>> 
    >>> # Solve
    >>> E, psi = solve(T, V_full, dx)
    >>> print(f"Ground state energy: {E[0]:.6f} Ha")
    >>> 
    >>> # Verify normalization
    >>> norm = np.sum(psi[:, 0]**2) * dx
    >>> print(f"Normalization: {norm:.6f}")  # Should be 1.0
    """
    V_internal = V_full[1:-1]
    H = T + np.diag(V_internal)

    E, psi = np.linalg.eigh(H) 

    # Normalize each state individually
    for i in range(psi.shape[1]):
        # sum( |psi|^2 * dx )
        norm_factor = np.sum(psi[:, i]**2) * dx
        # Divide by the square root of the integral
        psi[:, i] = psi[:, i] / np.sqrt(norm_factor)

    return E, psi

# ==========================================
# 5. PLOTTING FUNCTIONS (STREAMLIT/JUPYTER SAFE)
# ==========================================
def plot_V(V_raw_input):
    """
    Plot a 1D potential profile.
    
    Creates a simple matplotlib figure showing the potential energy landscape.
    
    Parameters
    ----------
    V_raw_input : ndarray or None
        1D array representing the potential V(x)
        If None or scalar, returns None
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object containing the potential plot
        Returns None if input is invalid
    """
    """    
    Notes
    -----
    - Uses dark background style
    - Cyan color for potential curve
    - Useful for quick visualization of potential shapes
    
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V = harmonic(x, k=1.0)
    >>> fig = plot_V(V)
    >>> plt.show()
    """
    if V_raw_input is None or np.ndim(V_raw_input) == 0:
        return None

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(V_raw_input, lw=1.5, color="cyan")
    ax.set_title("Potential Input")
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Potential")
    fig.tight_layout()
    return fig


def plot_alive(E, psi, V, x, no=1, nos=5, mode=''):
    """
    Plot wavefunctions as probability densities with separate energy and probability axes.
    
    Creates a physically accurate plot showing:
    - Potential V(x) and energy levels on left y-axis
    - Probability densities |ψ|² on right y-axis (separate scale)
    - Color-synchronized between probability curves and energy levels
    
    Parameters
    ----------
    E : ndarray
        Energy eigenvalues (in Hartree)
    psi : ndarray, shape (N, M)
        Wavefunction array where psi[:, i] is the i-th eigenstate
    V : ndarray, shape (N+2,)
        Full potential array including boundaries
    x : ndarray, shape (N+2,)
        Full spatial grid including boundaries
    no : int, optional
        State index to plot if mode != 'all' (default: 1)
    nos : int, optional
        Number of states to plot if mode == 'all' (default: 5)
    mode : str, optional
        Plot mode:
        - 'all': Plot multiple states (first nos states)
        - '': Plot single state (state no)
        Default: '' (single state)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with dual y-axes
        - ax1 (left): Energy/Potential scale
        - ax2 (right): Probability density scale
    
    Notes
    -----
    - Uses dark background theme
    - Probability densities are plotted as |ψ|², not ψ
    - Each state has matching colors for its probability curve and energy level
    - Regions where V > 10⁵ are hidden (infinite walls)
    
    """
    """    
    Examples
    --------
    >>> # Plot first 5 states
    >>> fig = plot_alive(E, psi, V_full, x_full, nos=5, mode='all')
    >>> plt.show()
    >>> 
    >>> # Plot only ground state
    >>> fig = plot_alive(E, psi, V_full, x_full, no=0)
    >>> plt.show()
    """
    import matplotlib.pyplot as plt
    
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax2 = ax1.twinx()  # Right axis for probability
    
    states = min(nos, len(E))
    x_solver = x[1:-1]
    V_internal = V[1:-1]

    # --- Plot Potential ---
    ax1.plot(x, V, color="white", lw=2, label="V(x)", alpha=0.7)

    # --- Plot wavefunctions ---
    if mode == 'all':
        for n in range(states):
            # 1. Get the synchronized color for this state
            color = plt.colormaps["tab20"].colors[n % 20]
            
            psi_n_sq = psi[:, n]**2
            
            # 2. Plot probability density on ax2 with the chosen color
            ax2.plot(
                x_solver, psi_n_sq,
                label=rf"$|\psi_{n}|^2$ (E={E[n]:.2f})",
                lw=1.2,
                color=color # <-- EXPLICIT COLOR SET
            )
            
            # 3. Plot energy line on ax1 with the same color
            ax1.axhline(E[n], linestyle="--", lw=0.8, alpha=0.5, color=color) # <-- EXPLICIT COLOR SET
    else:
        # For single state mode, we still need a color. Use 'no' as the index.
        n = no
        color = plt.colormaps["tab20"].colors[n % 20]

        psi_n_sq = psi[:, n]**2
        
        # Plot probability density on ax2 with the chosen color
        ax2.plot(
            x_solver, psi_n_sq,
            label=rf"$|\psi_{no}|^2$ (E={E[no]:.2f})",
            lw=1.2,
            color=color # <-- EXPLICIT COLOR SET
        )
        
        # Plot energy line on ax1 with the same color
        ax1.axhline(E[no], linestyle="--", lw=0.8, alpha=0.5, color=color) # <-- EXPLICIT COLOR SET
        
    # Formatting
    ax1.set_xlabel("x [a.u.]")
    ax1.set_ylabel("Energy / V(x)")
    ax2.set_ylabel(r"Probability Density $|\psi|^2$")

    ax1.set_title("Physically Accurate Eigenstates and Potential")

    # The fig.legend() might show the color/style of the *last* plot line. 
    # For robust legend handling with twinx, you often need to combine the handles.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right", fontsize=8) # <-- Improved Legend
    
    plt.tight_layout()
    return fig

def plot_dead(E, psi, V, x, nos=5):
    """Textbook: wavefunctions vertically shifted by energy."""
    plt.style.use("dark_background")
    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [5, 1]}
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.4)

    states = min(nos, len(E))
    x_solver = x[1:-1]
    V_internal = V[1:-1]

    if states <= 0:
        return fig

    scale = (E[1] - E[0]) * 0.4 if states > 1 else max(E[0] * 0.1, 0.5)
    max_E = E[states - 1]
    window_height = max_E * 1.5

    # Plot shifted wavefunctions
    for n in range(states):
        psi_n = psi[:, n]
        maxabs = np.max(np.abs(psi_n))
        psi_norm = psi_n / (maxabs if maxabs != 0 else 1)
        y = psi_norm * scale + E[n]
        y[V_internal > 1e5] = np.nan  # hide where potential is infinite

        color = plt.colormaps["tab20"].colors[n % 20]
        ax_main.plot(x_solver, y, lw=1.3, color=color, label=f"n={n+1}, E={E[n]:.2f}")

    # Plot potential
    V_clip = np.clip(V, 0, window_height)
    ax_main.plot(x, V_clip, color="white", lw=2, label="V(x)")

    ax_main.set_title("Eigenstates + Potential")
    ax_main.set_xlabel("x [a.u.]")
    ax_main.set_ylabel("Energy / ψ")
    ax_main.set_ylim(0, max_E * 1.2)
    ax_main.legend(fontsize=8)

    # Energy levels
    ax_bar.set_title("Energy Spectrum")
    ax_bar.set_xticks([])
    ax_bar.set_ylim(0, np.max(E[:states]) * 1.1)
    for n in range(states):
        ax_bar.axhline(E[n], lw=1, color=plt.colormaps["tab20"].colors[n % 20])

    return fig


# ==========================================
# 6. BENCHMARKING FUNCTIONS
# ==========================================
def check_ortho(psi, dx, num_states_to_check=20):
    """
    Checks the orthonormality of the first 'num_states_to_check' wave functions.
    """
    N_CHECK = min(psi.shape[1], num_states_to_check) 
    overlap_matrix = np.zeros((N_CHECK, N_CHECK))

    for i in range(N_CHECK):
        for j in range(N_CHECK):
            # Riemann Sum: integral(psi_i * psi_j) dx
            Rsum = np.sum(psi[:, i] * psi[:, j]) * dx
            overlap_matrix[i, j] = Rsum

    print(f"\n--- Orthonormality Check (First {N_CHECK} states) ---")
    print("Overlap Matrix should approximate the Identity Matrix:")
    return overlap_matrix

def show_matrix(overlap_matrix,how='normal',round_value=10):
    '''
    how = normal, round , plot
    '''
    if how == 'normal':      
        print(overlap_matrix[:3])
    elif how == 'round':
        print(np.round(overlap_matrix, round_value))
    elif how == 'plot':
        plt.figure(figsize=(6,5))
        plt.imshow(overlap_matrix, cmap='coolwarm', origin='lower')
        plt.colorbar(label="Overlap Value")
        plt.title("Orthonormality Check Matrix")
        plt.xlabel("State Index m")
        plt.ylabel("State Index n")
        plt.gca().invert_yaxis()
        plt.locator_params(axis='y', integer=True)
        plt.locator_params(axis='x', integer=True)
        plt.show()

def check_ISW_analytic(E, lower_bound=-10, upper_bound=10, hbar=1.0, m=1.0, max_levels=6):
    """
    Compares numerical energies to the Infinite Square Well analytic formula.
    
    Parameters:
    -----------
    E : array
        Numerical eigenvalues
    lower_bound : float
        Lower boundary of the well (default: -10)
    upper_bound : float
        Upper boundary of the well (default: 10)
    hbar : float
        Reduced Planck constant (default: 1.0)
    m : float
        Particle mass (default: 1.0)
    max_levels : int
        Number of levels to check (default: 6)

    """
    """            
    Example:
    --------
    check_ISW_analytic(E, lower_bound=-10, upper_bound=10)
    """
    L = upper_bound - lower_bound  # Well width
    CHECK_N = min(max_levels, len(E))
    E_numerical = E[:CHECK_N]
    E_analytic = np.zeros(CHECK_N)

    for i in range(CHECK_N):
        n = i + 1 
        E_analytic[i] = (hbar**2 * np.pi**2 * n**2) / (2*m*L**2)

    print("\n### ENERGY BENCHMARK: Infinite Square Well ###")
    print(f"Well boundaries: x = [{lower_bound}, {upper_bound}], Width L = {L}")
    print("-" * 55)
    print(f"| n | Analytic E | Numerical E | % Error |")
    print("-" * 55)

    for i in range(CHECK_N):
        percent_error = np.abs((E_numerical[i] - E_analytic[i]) / E_analytic[i]) * 100
        print(
            f"| {i+1:<1} | {E_analytic[i]:<10.6f} | {E_numerical[i]:<11.6f} | {percent_error:<7.4f}% |"
        )
    print("-" * 55)
    
    return E_analytic, E_numerical

def check_harmonic_analytic(E, k=None, center=0.0, hbar=1.0, m=1.0, max_levels=6):
    """
    Compares numerical energies to the Harmonic Oscillator analytic formula.
    
    Parameters:
    -----------
    E : array
        Numerical eigenvalues
    k : float, optional
        Spring constant. If None, uses Last_k_value global variable
    center : float
        Center position of the harmonic oscillator (default: 0.0)
    hbar : float
        Reduced Planck constant (default: 1.0)
    m : float
        Particle mass (default: 1.0)
    max_levels : int
        Number of levels to check (default: 6)
    
    Example:
    --------
    check_harmonic_analytic(E, k=10, center=0)
    """
    CHECK_N = min(max_levels, len(E))
    
    try: 
        # Use provided k or fall back to global Last_k_value
        if k is None:
            k = Last_k_value 
            if k is None:
                print("ERROR: k is not set. Please provide k parameter or run harmonic() first.")
                return
        
        w = np.sqrt(k/m)
        E_numerical = E[:CHECK_N]
        E_analytic = np.zeros(CHECK_N)

        for i in range(CHECK_N):
            n_quantum = i 
            E_analytic[i] = (n_quantum + 0.5) * hbar * w

        print("\n### ENERGY BENCHMARK: Harmonic Oscillator ###")
        print(f"Spring constant k = {k}, Center = {center}, omega = {w:.4f}")
        print("-" * 55)
        print(f"| n | Analytic E | Numerical E | % Error |") 
        print("-" * 55)

        for i in range(CHECK_N):
            n_label = i 
            percent_error = np.abs((E_numerical[i] - E_analytic[i]) / E_analytic[i]) * 100
            
            print(
                f"| {n_label:<1} | {E_analytic[i]:<10.6f} | {E_numerical[i]:<11.6f} | {percent_error:<7.4f}% |"
            )
        print("-" * 55)
        
        return E_analytic, E_numerical

    except Exception as e:
        print(f"Error in harmonic oscillator check: {e}")


def check_finite_well_analytic(E, V0, lower_bound=-10, upper_bound=10, hbar=1.0, m=1.0, max_levels=10):
    """
    Compares numerical energies to the Finite Square Well analytical solution.
    
    The finite square well has no simple closed-form solution, but bound state
    energies can be found by solving transcendental equations numerically.
    
    Parameters:
    -----------
    E : array
        Numerical eigenvalues from your solver
    V0 : float
        Barrier height (potential outside the well)
    lower_bound : float
        Lower boundary of the well (default: -10)
    upper_bound : float
        Upper boundary of the well (default: 10)
    hbar : float
        Reduced Planck constant (default: 1.0)
    m : float
        Particle mass (default: 1.0)
    max_levels : int
        Maximum number of levels to check (default: 10)
    
    Example:
    --------
    check_finite_well_analytic(E, V0=2.0, lower_bound=-10, upper_bound=10)
    """
    a = (upper_bound - lower_bound) / 2  # Half-width
    z0 = a * np.sqrt(2 * m * V0) / hbar  # Dimensionless parameter
    
    # Find analytical energies by solving transcendental equations
    E_analytic = []
    
    # Even parity states: z*tan(z) = sqrt(z0^2 - z^2)
    z_vals = np.linspace(0.01, z0 - 0.01, 10000)
    for n in range(max_levels):
        try:
            lhs = z_vals * np.tan(z_vals)
            rhs = np.sqrt(z0**2 - z_vals**2)
            diff = lhs - rhs
            
            # Find sign changes (crossings)
            for i in range(len(diff) - 1):
                if diff[i] * diff[i+1] < 0:
                    z = z_vals[i]
                    E_candidate = (hbar**2 * z**2) / (2 * m * a**2)
                    if E_candidate < V0 and not any(np.isclose(E_candidate, E_a, rtol=1e-3) for E_a in E_analytic):
                        E_analytic.append(E_candidate)
                        break
        except:
            pass
    
    # Odd parity states: -z*cot(z) = sqrt(z0^2 - z^2)
    for n in range(max_levels):
        try:
            lhs = -z_vals / np.tan(z_vals)
            rhs = np.sqrt(z0**2 - z_vals**2)
            diff = lhs - rhs
            
            for i in range(len(diff) - 1):
                if diff[i] * diff[i+1] < 0:
                    z = z_vals[i]
                    E_candidate = (hbar**2 * z**2) / (2 * m * a**2)
                    if E_candidate < V0 and not any(np.isclose(E_candidate, E_a, rtol=1e-3) for E_a in E_analytic):
                        E_analytic.append(E_candidate)
                        break
        except:
            pass
    
    E_analytic = sorted(E_analytic)
    
    # Filter numerical energies to only bound states
    E_numerical_bound = E[E < V0]
    
    CHECK_N = min(len(E_analytic), len(E_numerical_bound), max_levels)
    
    if CHECK_N == 0:
        print("\n### ENERGY BENCHMARK: Finite Square Well ###")
        print(f"Well: x in [{lower_bound}, {upper_bound}], V0 = {V0}, z0 = {z0:.4f}")
        print("WARNING: No bound states found!")
        print(f"  Barrier too shallow. Need V0 > {E[0]:.4f} to bind the ground state.")
        return None, None
    
    print("\n### ENERGY BENCHMARK: Finite Square Well ###")
    print(f"Well: x in [{lower_bound}, {upper_bound}], V0 = {V0}, z0 = {z0:.4f}")
    print(f"Number of bound states: {CHECK_N}")
    print("-" * 55)
    print(f"| n | Analytic E | Numerical E | % Error |")
    print("-" * 55)
    
    for i in range(CHECK_N):
        percent_error = np.abs((E_numerical_bound[i] - E_analytic[i]) / E_analytic[i]) * 100
        print(
            f"| {i:<1} | {E_analytic[i]:<10.6f} | {E_numerical_bound[i]:<11.6f} | {percent_error:<7.4f}% |"
        )
    print("-" * 55)
    
    return np.array(E_analytic[:CHECK_N]), E_numerical_bound[:CHECK_N]









def display_params(frame, params_list, start_y=80, line_height=25, color=(255, 255, 255)):
    for i, text in enumerate(params_list):
        y = start_y + i * line_height
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)


# ---------------------------------------------------------------------
# MAIN FUNCTION: HAND-CONTROLLED POTENTIAL CAPTURE
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
drawer = mp.solutions.drawing_utils


def capture_potential(tune, A_MIN, A_MAX, mode='wait'):

    cap = cv2.VideoCapture(0)
    captured_V = None

    # Stability tracking -----------------------------------------------
    stability_counter = 0
    REQUIRED_STABLE_FRAMES = 45
    MOVEMENT_THRESHOLD = 0.015
    prev_landmarks = []

    # Landmark indices --------------------------------------------------
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8

    # QHO Mapping constants --------------------------------------------
    D_MIN = 0.001
    D_MAX = 0.2

    D_RANGE = D_MAX - D_MIN
    A_RANGE = A_MAX - A_MIN

    SLOPE = -A_RANGE / D_RANGE
    INTERCEPT = A_MAX - SLOPE * D_MIN

    # Fixed visual scale (independent of physics range)
    PLOT_CEILING_A = 10.0
    EPS = 1e-9

    print("Controls: HOLD STILL to capture, or press 'q' to quit.")

    # =================================================================
    # MAIN LOOP
    # =================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pot_profile = None
        mode_msg = "No Hands"
        params_to_display = []
        current_landmarks_flat = []

        # --------------------------------------------------------------
        # LANDMARK PROCESSING
        # --------------------------------------------------------------
        if res.multi_hand_landmarks:

            # Flatten positions for stability detection
            for hand_lms in res.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    current_landmarks_flat.extend([lm.x, lm.y])

            # Draw detected hands
            for lm in res.multi_hand_landmarks:
                drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # ----------------------------------------------------------
            # TWO HANDS = SQUARE WELL (AUTO-CENTERED)
            # ----------------------------------------------------------
            if len(res.multi_hand_landmarks) >= 2:
                mode_msg = "Mode: Square Well (Auto-Centered)"

                # 1. Get Hand Positions
                x_coords = [
                    lm.landmark[INDEX_TIP_ID].x * w
                    for lm in res.multi_hand_landmarks
                ]
                x_coords.sort()
                xL_hand, xR_hand = int(x_coords[0]), int(x_coords[1])

                # 2. Draw Yellow lines at REAL hand positions (Visual Feedback)
                cv2.line(frame, (xL_hand, 0), (xL_hand, h), (0, 255, 255), 2)
                cv2.line(frame, (xR_hand, 0), (xR_hand, h), (0, 255, 255), 2)

                # 3. Calculate Force-Centered Coordinates
                # We calculate the width of your hands, but ignore their position
                well_width = xR_hand - xL_hand
                center_screen = w / 2
                
                # Create boundaries centered on the screen
                centered_L = center_screen - (well_width / 2)
                centered_R = center_screen + (well_width / 2)

                params_to_display.append(f"Width: {well_width:4.0f} px")
                params_to_display.append(f"Status: Centered")

                # 4. Generate Potential (Centered)
                x_space = np.linspace(0, w, 400)
                pot_profile = np.ones_like(x_space)
                # Use centered_L/R instead of hand positions
                pot_profile[(x_space > centered_L) & (x_space < centered_R)] = 0

                """
                # 5. Visualize the Centered Potential (Red Line)
                display_pts = np.column_stack((
                    x_space, 
                    pot_profile * (h - 10) # simple scaling for viz
                )).astype(np.int32)
                cv2.polylines(frame, [display_pts], False, (0, 0, 255), 2)
                """
            # ----------------------------------------------------------
            # ONE HAND = PINCH PARABOLA (QHO)
            # ----------------------------------------------------------
            elif len(res.multi_hand_landmarks) == 1:
                mode_msg = "Mode: Pinch QHO"
                lm = res.multi_hand_landmarks[0]

                thumb = lm.landmark[THUMB_TIP_ID]
                index = lm.landmark[INDEX_TIP_ID]

                dx = index.x - thumb.x
                dy = index.y - thumb.y
                pinch_distance = math.sqrt(dx**2 + dy**2)

                # Compute curvature
                A = SLOPE * pinch_distance + INTERCEPT
                A = max(A_MIN, min(A_MAX, A))

                # This is already mathematically centered at 0
                x_space = np.linspace(-1, 1, 400)
                pot_profile = A * (x_space**2)

                # Fixed visual scale
                pot_profile = pot_profile / (PLOT_CEILING_A + EPS)
                pot_profile = np.clip(pot_profile, 0.0, 1.0)

                params_to_display.append(f"Pinch Dist: {pinch_distance:.4f}")
                params_to_display.append(f"A (curv): {A:.4f}")

                display_pts = np.column_stack((
                    (x_space + 1)/2 * w,
                    (1 - pot_profile) * h
                )).astype(np.int32)

                cv2.polylines(frame, [display_pts], False, (0, 0, 255), 2)

        # ==============================================================
        # STABILITY CHECK
        # ==============================================================
        if mode != 'wait':
            if current_landmarks_flat and prev_landmarks:
                if len(current_landmarks_flat) == len(prev_landmarks):
                    movement = np.mean(np.abs(
                        np.array(current_landmarks_flat)
                        - np.array(prev_landmarks)
                    ))
                    if movement < MOVEMENT_THRESHOLD:
                        stability_counter += 1
                    else:
                        stability_counter = 0
                else:
                    stability_counter = 0
            else:
                stability_counter = 0

            prev_landmarks = current_landmarks_flat

            # Show loading bar
            if stability_counter > 0:
                progress = stability_counter / REQUIRED_STABLE_FRAMES
                bar_width = int(w * progress)
                color = (0, 255*progress, 255*(1-progress))
                cv2.rectangle(frame, (0, 0), (bar_width, 20), color, -1)
                cv2.putText(frame, "HOLDING...", (10, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Finished
            if stability_counter >= REQUIRED_STABLE_FRAMES and pot_profile is not None:
                captured_V = pot_profile
                frame[:] = 255
                cv2.imshow("Quantum Potential Input", frame)
                cv2.waitKey(100)
                print("Stable capture triggered!")
                break

        # --------------------------------------------------------------
        # UI OVERLAY
        # --------------------------------------------------------------
        cv2.putText(frame, mode_msg, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display_params(frame, params_to_display)
        cv2.imshow("Quantum Potential Input", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -----------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    return captured_V

# Create a notebook-friendly version of the function
def capture_potential_notebook(tune, A_MIN, A_MAX, mode='wait'):
    import time
    from IPython.display import display, Image, clear_output

    
    # Copy relevant constants from the file for local scope
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    REQUIRED_STABLE_FRAMES = 45
    MOVEMENT_THRESHOLD = 0.015
    PLOT_CEILING_A = 10.0
    EPS = 1e-9
    
    D_MIN = 0.001
    D_MAX = 0.2
    D_RANGE = D_MAX - D_MIN
    A_RANGE = A_MAX - A_MIN
    SLOPE = -A_RANGE / D_RANGE
    INTERCEPT = A_MAX - SLOPE * D_MIN
    # End of copied constants
    
    cap = cv2.VideoCapture(0)
    captured_V = None
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Check permissions or camera index.")
        return None

    stability_counter = 0
    prev_landmarks = []
    
    start_time = time.time()
    MAX_RUN_TIME_SECONDS = 30 
    
    print("Controls: HOLD STILL to capture, or wait for the time limit to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pot_profile = None
        mode_msg = "No Hands"
        params_to_display = []
        current_landmarks_flat = []

        # --- LANDMARK AND POTENTIAL LOGIC (Skipped for brevity, assume this is correct) ---
        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    current_landmarks_flat.extend([lm.x, lm.y])
            for lm in res.multi_hand_landmarks:
                drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            # TWO HANDS (Square Well)
            if len(res.multi_hand_landmarks) >= 2:
                mode_msg = "Mode: Square Well (Auto-Centered)"
                x_coords = [lm.landmark[INDEX_TIP_ID].x * w for lm in res.multi_hand_landmarks]
                x_coords.sort()
                xL_hand, xR_hand = int(x_coords[0]), int(x_coords[1])
                cv2.line(frame, (xL_hand, 0), (xL_hand, h), (0, 255, 255), 2)
                cv2.line(frame, (xR_hand, 0), (xR_hand, h), (0, 255, 255), 2)
                well_width = xR_hand - xL_hand
                center_screen = w / 2
                centered_L = center_screen - (well_width / 2)
                centered_R = center_screen + (well_width / 2)
                params_to_display.append(f"Width: {well_width:4.0f} px")
                params_to_display.append(f"Status: Centered")
                x_space = np.linspace(0, w, 400)
                pot_profile = np.ones_like(x_space)
                pot_profile[(x_space > centered_L) & (x_space < centered_R)] = 0
            # ONE HAND (QHO)
            elif len(res.multi_hand_landmarks) == 1:
                mode_msg = "Mode: Pinch QHO"
                lm = res.multi_hand_landmarks[0]
                thumb = lm.landmark[THUMB_TIP_ID]
                index = lm.landmark[INDEX_TIP_ID]
                dx = index.x - thumb.x
                dy = index.y - thumb.y
                pinch_distance = math.sqrt(dx**2 + dy**2)
                A = SLOPE * pinch_distance + INTERCEPT
                A = max(A_MIN, min(A_MAX, A))
                x_space = np.linspace(-1, 1, 400)
                pot_profile = A * (x_space**2)
                pot_profile = pot_profile / (PLOT_CEILING_A + EPS)
                pot_profile = np.clip(pot_profile, 0.0, 1.0)
                params_to_display.append(f"Pinch Dist: {pinch_distance:.4f}")
                params_to_display.append(f"A (curv): {A:.4f}")
                display_pts = np.column_stack(((x_space + 1)/2 * w, (1 - pot_profile) * h)).astype(np.int32)
                cv2.polylines(frame, [display_pts], False, (0, 0, 255), 2)
        # --- END LANDMARK AND POTENTIAL LOGIC ---

        # STABILITY CHECK
        if mode != 'wait':
            if current_landmarks_flat and prev_landmarks and len(current_landmarks_flat) == len(prev_landmarks):
                movement = np.mean(np.abs(np.array(current_landmarks_flat) - np.array(prev_landmarks)))
                stability_counter = stability_counter + 1 if movement < MOVEMENT_THRESHOLD else 0
            else:
                stability_counter = 0

            prev_landmarks = current_landmarks_flat

            if stability_counter > 0:
                progress = stability_counter / REQUIRED_STABLE_FRAMES
                bar_width = int(w * progress)
                color = (0, 255*progress, 255*(1-progress))
                cv2.rectangle(frame, (0, 0), (bar_width, 20), color, -1)
                cv2.putText(frame, "HOLDING...", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Finished
            if stability_counter >= REQUIRED_STABLE_FRAMES and pot_profile is not None:
                captured_V = pot_profile
                cap.release()
                # --- LINE REMOVED HERE (was cv2.destroyAllWindows()) ---
                print("Stable capture triggered and video stream closed.")
                return captured_V

        # UI OVERLAY
        cv2.putText(frame, mode_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        display_params(frame, params_to_display)
        
        # NOTEBOOK DISPLAY
        clear_output(wait=True) 
        _, buffer = cv2.imencode('.jpeg', frame)
        display(Image(data=buffer.tobytes()))
        
        time.sleep(0.01)

        if time.time() - start_time > MAX_RUN_TIME_SECONDS:
            print(f"Time limit of {MAX_RUN_TIME_SECONDS} seconds reached.")
            break

    # -----------------------------------------------------------------
    cap.release()
    return captured_V





###
import qrcode
from IPython.display import display, Image

def show_QR(url):
    # The file name to save the QR code image
    file_name = "hand_wave_link_qrcode.png"

    # --- QR Code Generation ---
    # 1. Create a QR code object with specific settings
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # 2. Add the URL data to the object
    qr.add_data(url)
    qr.make(fit=True)

    # 3. Create the QR code image
    img = qr.make_image(fill_color="black", back_color="white")

    # 4. Save the image to the local directory
    img.save(file_name)

    # --- Display in Jupyter Notebook ---

    # 5. Display the saved image using IPython.display
    return display(Image(filename=file_name))
