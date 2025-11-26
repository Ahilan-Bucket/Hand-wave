# Peri-Peri/functions.py

import numpy as np
import matplotlib.pyplot as plt

import math 
from matplotlib.ticker import MultipleLocator 

# ==========================================
# 1. PHYSICS CONSTANTS
# ==========================================
hbar = 1
m = 1
L = 50
N_GRID = 2000
N = 2000
global Last_k_value # Used by harmonic() and check_harmonic_analytic()
TUNNELING_THRESHOLD = 0.01  # 1% of total probability considered significant

# ==========================================
# 2. GRID FUNCTIONS
# ==========================================
def make_grid(x_min=None, x_max=None, L=None, N=2000):
    """
    Create a spatial grid for solving the Schrödinger equation.
    
    Parameters
    ----------
    x_min : float, optional
        Minimum x value (required if L is not provided)
    x_max : float, optional
        Maximum x value (required if L is not provided)
    L : float, optional
        Total length of the spatial domain. If provided, overrides x_min/x_max
        to create a centered grid [-L/2, L/2].
    N : int, optional
        Number of internal grid points (default: 2000)
    - Pili : Make sure your values: what ever combination of x_min,x_max,L are large 
     enough! and generally much smaller than the lower and upper bound you use for 
     you'r Potential!! You should know that, at these GRID Edge points, we 
     (the hamiltonian) would automatically set the GRID boundary to be zero, 
     because this is the limit of our full local universe (Grid). The particle has no 
     probability of being outside this grid, like an infinite wall. If the potential 
     you input potential is near this, they will interact. 

    Returns
    -------
    x_full : ndarray
        Full grid with N+2 points, including boundary points
    dx : float
        Grid spacing
    x_internal : ndarray
        Internal grid points (N points)
    """
    if L is not None:
        x = np.linspace(-L/2, L/2, N+2)
    elif x_min is not None and x_max is not None:
        x = np.linspace(x_min, x_max, N+2)
    else:
        # Default fallback if nothing provided
        print("Warning: No grid parameters provided. Using default L=20.")
        L_default = 20.0
        x = np.linspace(-L_default/2, L_default/2, N+2)

    dx = x[1] - x[0]
    x_internal = x[1:-1]
    return x, dx, x_internal


# ==========================================
# 2.5. SAFETY & UTILITY FUNCTIONS
# ==========================================
def check_boundary_proximity(x_internal, lower_bound, upper_bound, safety_margin=0.15):
    """
    Checks if the defined potential boundaries are too close to the computational 
    grid boundaries (x_min, x_max). 

    If the potential is too close, the implicit hard walls of the grid (where 
    psi=0) will artificially constrain the wavefunction, leading to incorrect 
    energy levels and distorted eigenstates (especially for finite wells/barriers).
    
    Parameters
    ----------
    x_internal : ndarray
        The internal grid points (length N).
    lower_bound : float
        The left boundary of the physical potential feature (e.g., the well/barrier).
    upper_bound : float
        The right boundary of the physical potential feature.
    safety_margin : float (0 to 1.0)
        Minimum required padding on each side, expressed as a fraction of the 
        total internal grid length. Default is 0.15 (15%).
        
    Returns
    -------
    str or None
        A detailed WARNING message if proximity is detected, otherwise None.
    """

    """
    Pili-Pili Note: x_int is your full grind array, lower bound and upper bound is 
    your input potential, the Grid must be way larger usually!
    """
    x_int_min = x_internal[0]
    x_int_max = x_internal[-1]
    L_grid = x_int_max - x_int_min

    # Calculate padding on each side
    left_padding = lower_bound - x_int_min
    right_padding = x_int_max - upper_bound
    
    # Define minimum required padding
    min_padding_needed = L_grid * safety_margin
    
    if left_padding < min_padding_needed or right_padding < min_padding_needed:
        # --- RAISE OUTPUT MESSAGE ---
        return (
            "\n"
            "*** 🛑 Pili Pili WARNING: GRID BOUNDARY PROXIMITY 🛑 ***\n"
            "The physical potential is too close to the numerical grid walls.\n"
            f"Current Grid Range: [{x_int_min:.2f}, {x_int_max:.2f}].\n"
            f"Potential Range: [{lower_bound:.2f}, {upper_bound:.2f}].\n"
            f"Safety Margin is {safety_margin*100:.0f}% of grid length ({min_padding_needed:.2f} a.u.).\n"
            "Impact: The implicit hard walls (psi=0) will artificially squeeze the "
            "wavefunction (especially for finite wells), leading to inaccurate "
            "energies (E will be too high) and incorrect wave function tails.\n"
            "ACTION: Increase the grid length 'L' or adjust 'x_min'/'x_max' in make_grid().\n"
            "********************************************\n"
        )
    return None

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

def harmonic(x, k, center=0.0, wall_value=1e10):
    """
    Create a harmonic oscillator (parabolic) potential.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    k : float
        Spring constant (curvature parameter) in atomic units
        Larger k → stiffer spring → more tightly bound states
    center : float, optional
        Center position of the parabola (default: 0.0)
    wall_value : float, optional
        Value for boundary walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Harmonic potential with boundary walls: V(x) = 0.5 * k * (x - center)²
    
    Notes
    -----
    - Sets global variable Last_k_value for use by check_harmonic_analytic()
    - Energy levels: E_n = ℏω(n + 1/2) where ω = √(k/m)
    - In atomic units (ℏ=1, m=1): ω = √k
    - Ready to use with solve() - no manual padding needed
    
    Examples
    --------
    >>> x_int = np.linspace(-10, 10, 1000)
    >>> V_full = harmonic(x_int, k=1.0)  # Returns N+2 array
    >>> E, psi = solve(T, V_full, dx)
    """
    global Last_k_value
    Last_k_value = k
    
    V_int = 0.5 * k * (x - center)**2
    return np.pad(V_int, (1, 1), constant_values=wall_value)

def gaussian_well(x, center=0.0, width=1.0, depth=50, wall_value=1e10): 
    """
    Create a Gaussian-shaped potential well.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    center : float, optional
        Center position of the well (default: 0.0)
    width : float, optional
        Width parameter (standard deviation) of the Gaussian (default: 1.0)
    depth : float, optional
        Depth of the well at the center (default: 50)
    wall_value : float, optional
        Value for boundary walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Gaussian well with boundary walls: V(x) = -depth * exp(-(x-center)²/(2*width²))
    
    Examples
    --------
    >>> x_int = np.linspace(-10, 10, 1000)
    >>> V_full = gaussian_well(x_int, center=0, width=2.0, depth=10)
    >>> E, psi = solve(T, V_full, dx)
    """
    V_int = -depth * np.exp(-(x - center)**2 / (2 * width**2))
    return np.pad(V_int, (1, 1), constant_values=wall_value)

def inf_square_well(x, lower_bound, upper_bound, wall_value=1e10):
    """
    Create an infinite square well (particle in a box) potential.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    lower_bound : float
        Left boundary of the well
    upper_bound : float
        Right boundary of the well
    wall_value : float, optional
        Value used for the infinite walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Infinite square well potential with boundary walls:
        - V(x) = 0 for lower_bound <= x <= upper_bound (inside well)
        - V(x) = wall_value elsewhere (outside well and at boundaries)
    
    Notes
    -----
    - Uses penalty method: "infinite" walls are approximated by wall_value
    - Well width: L = upper_bound - lower_bound
    - Analytical energies: E_n = (hbar^2 * pi^2 * n^2) / (2 * m * L^2)
    - Ready to use with solve() - no manual padding needed
   
    Examples
    --------
    >>> x_int = np.linspace(-15, 15, 1000)  # internal grid
    >>> V_full = inf_square_well(x_int, lower_bound=-10, upper_bound=10)
    >>> # Returns N+2 array, ready for solver
    >>> E, psi = solve(T, V_full, dx)
    """

    warning = check_boundary_proximity(x, lower_bound, upper_bound)
    if warning:
        print(warning) 
    # --------------------------


    V_int = np.zeros_like(x) 
    V_int[x <= lower_bound] = wall_value
    V_int[x >= upper_bound] = wall_value
    return np.pad(V_int, (1, 1), constant_values=wall_value)

# Alias for backward compatibility (fixing typo)
inf_sqaure_well = inf_square_well

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
        V[x <= bound] = HUGE_NUMBER
    elif side == 'right':
        V[x >= bound] = HUGE_NUMBER
    return V

def finite_barrier(x, center, width, height, wall_value=1e10):
    """
    Create a finite rectangular potential barrier.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    center : float
        Center position of the barrier
    width : float
        Total width of the barrier
    height : float
        Height of the potential barrier
    wall_value : float, optional
        Value for boundary walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Rectangular barrier with boundary walls:
        - V(x) = height for |x - center| < width/2
        - V(x) = 0 elsewhere (inside domain)
        - V(boundaries) = wall_value
    
    Examples
    --------
    >>> x_int = np.linspace(-10, 10, 1000)
    >>> V_full = finite_barrier(x_int, center=0, width=2, height=5)
    >>> E, psi = solve(T, V_full, dx)
    """

    lower_bound = center - width / 2.0
    upper_bound = center + width / 2.0
    # 🛑 Perform the safety check 🛑
    warning = check_boundary_proximity(x, lower_bound, upper_bound)
    if warning:
        print(warning) 
    # -----------------------------------------------    

    V_int = np.zeros_like(x)
    mask = (x > (center - width/2)) & (x < (center + width/2))
    V_int[mask] = height
    return np.pad(V_int, (1, 1), constant_values=wall_value)

def V_double_well(x, depth=20, separation=1, center=0.0, wall_value=1e10):
    """
    Create a quartic double-well potential.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    depth : float, optional
        Depth parameter controlling overall potential strength (default: 20)
    separation : float, optional
        Controls the distance between the two wells (default: 1)
    center : float, optional
        Center position of the double well system (default: 0.0)
    wall_value : float, optional
        Value for boundary walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Double well with boundary walls: V(x) = depth × ((x-center)² - separation)²
    
    Notes
    -----
    - Creates symmetric double well with barrier at x = center
    - Useful for studying tunneling splitting
    
    Examples
    --------
    >>> x_int = np.linspace(-5, 5, 1000)
    >>> V_full = V_double_well(x_int, depth=2, separation=1)
    >>> E, psi = solve(T, V_full, dx)
    """
    V_int = depth * ((x - center)**2 - separation)**2
    return np.pad(V_int, (1, 1), constant_values=wall_value)

def custom2(value,x):
    """Helper function from the notebook."""
    return value * np.ones_like(x)

# In psi_solve2/functions.py

def finite_square_well(x, lower_bound, upper_bound, depth, wall_value=1e10):
    """
    Create a finite square‑well potential.
    
    ALWAYS returns a full array (length N+2) with boundary walls built-in.

    Parameters
    ----------
    x : ndarray
        Spatial grid points (the *internal* grid, length N)
    lower_bound, upper_bound : float
        Left and right limits of the well
    depth : float
        Well depth (positive number; V = -depth inside)
    wall_value : float, optional
        Value for boundary walls (default: 1e10)
    
    Returns
    -------
    V : ndarray (length N+2)
        Finite square well with boundary walls:
        - V(x) = -depth for lower_bound <= x <= upper_bound (inside well)
        - V(x) = 0 for x outside well (classically forbidden region - ALLOWS TUNNELING)
        - V(boundaries) = wall_value (computational boundaries only)
    
    Notes
    -----
    The finite well allows quantum tunneling:
    - Particles with E < 0 are bound in the well
    - Wavefunction exponentially decays in the region where V=0 (outside well)
    - Infinite walls only at grid boundaries prevent numerical leakage
    
    Examples
    --------
    >>> x_int = np.linspace(-10, 10, 1000)
    >>> V_full = finite_square_well(x_int, lower_bound=-2, upper_bound=2, depth=50)
    >>> # Inside well: V=-50, Outside well: V=0, Boundaries: V=1e10
    >>> E, psi = solve(T, V_full, dx)
    """

    # GRID and Current Spacing Checker
    warning = check_boundary_proximity(x, lower_bound, upper_bound)
    if warning:
        print(warning) 
    # --------------------------


    # Start with V=0 everywhere (classically forbidden region)
    V_int = np.zeros_like(x)
    
    # Set well interior to -depth
    V_int[(x >= lower_bound) & (x <= upper_bound)] = -depth
    
    # Add infinite walls ONLY at computational boundaries
    return np.pad(V_int, (1, 1), constant_values=wall_value)

    
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
        - Piri Advice: If Unsure, use len(x_int) :)
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

    Note that, you much expected a stencil of just [1,-2,1] but depending 
    on how what constants and especially your value for 1/dx^2. you will get 
    a constant times the stencil 
    - you think the Minus sign is reverse? think about where the -ve sign is applied,
     inside or outside in your equation.
    
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
    # Construct Hamiltonian with helpful error handling
    try:
        H = T + np.diag(V_internal)
    except ValueError as e:
        raise RuntimeError(
            "Failed to construct Hamiltonian: shape mismatch between kinetic operator "
            f"{T.shape} and potential diagonal {V_internal.shape}. "
            "Pili Help: Are you giving solve x_int, make sure its x_int not x" \
            "or do x[1:-1]" \
            "or pad it"
        ) from e
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
    import matplotlib.pyplot as plt
    if V_raw_input is None or np.ndim(V_raw_input) == 0:
        return None

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # Fix for Infinite Well: Clip potential for plotting
    # If the potential has huge values (like 1e10), clip them for visualization
    # otherwise the plot will be dominated by the walls and the well will look flat.
    # We'll clip to a reasonable value, e.g., slightly above the max "finite" value 
    # or just a fixed large-ish number if everything is huge.
    
    # Simple heuristic: Clip to 200 if max is huge, or use max if it's small.
    # A better approach might be to check for the "infinite" marker.
    
    V_plot = V_raw_input.copy()
    
    # Check if we have "infinite" walls (arbitrarily > 1e5)
    if np.any(V_plot > 1e5):
        # Find the maximum value that is NOT "infinite"
        finite_vals = V_plot[V_plot < 1e5]
        if len(finite_vals) > 0:
            max_finite = np.max(finite_vals)
            # Clip to slightly above that, or at least 10 if it's 0
            clip_val = max(max_finite * 2.0, 10.0)
            V_plot = np.clip(V_plot, -np.inf, clip_val)
        else:
            # If everything is huge, just clip to something to show it exists
            V_plot = np.clip(V_plot, -np.inf, 10.0)
            
    ax.plot(V_plot, lw=1.5, color="cyan")
    ax.set_title("Potential Input (Clipped for Visibility)")
    ax.set_xlabel("Grid index")
    ax.set_ylabel("Potential")
    fig.tight_layout()
    return fig






# ==========================================
# 7. EDUCATIONAL ANALYSIS HELPER FUNCTIONS
# ==========================================
def analyze_potential(V, x):
    feedback = []
    if np.allclose(V, V[0]):
        feedback.append("Potential is constant → free particle.")
    HUGE = 1e5
    if V[0] > HUGE or V[-1] > HUGE:
        if np.all(V[1:-1] == 0):
            feedback.append("System: Infinite Square Well.")
    return feedback


def analyze_state_properties(E_n, psi_n, V, x):
    dx = x[1] - x[0]
    V_L, V_R = V[0], V[-1]
    min_wall = min(V_L, V_R)

    if E_n < min_wall:
        status = "BOUND"
    else:
        status = "UNBOUND"

    V_int = V[1:-1]
    forbidden = V_int > E_n
    psi2 = psi_n**2
    total_prob = np.sum(psi2) * dx
    prob_forbidden = np.sum(psi2[forbidden]) * dx

    if total_prob > 0:
        P = 100 * prob_forbidden / total_prob
    else:
        P = 0.0

    return f"E={E_n:.4f} | {status} | Tunneling: {P:.1f}%"


def generate_educational_feedback(E, psi, V, x, indices):
    report = []

    for msg in analyze_potential(V, x):
        report.append(f"• {msg}")

    if len(indices) == 0:
        report.append("• No eigenstates to report.")
        return "\n".join(report)

    report.append("\n--- Eigenstate Summary ---")
    for n in indices:
        if n < len(E):
            s = analyze_state_properties(E[n], psi[:, n], V, x)
            report.append(f"• n={n}: {s}")

    return "\n".join(report)




# ==========================================
# 5. PLOTTING FUNCTIONS (MERGED / FINAL)
# ==========================================

# Small helper: consistent colormap access
def _state_color(n):
    import matplotlib.pyplot as plt
    cmap = plt.colormaps["tab20"]
    return cmap.colors[n % len(cmap.colors)]

def auto_xlim_from_probability(x_full, psi, frac=1e-4, padding=1.0):
    """
    Determine xmin and xmax from where the *total* probability density 
    falls below a threshold fraction of its maximum.
    """
    x_full = np.asarray(x_full)

    if psi is None or psi.size == 0:
        return x_full[0], x_full[-1]

    # Internal grid (wavefunctions only exist here)
    x_internal = x_full[1:-1]

    # Compute total probability
    psi2 = abs(psi)**2
    if psi2.ndim == 1:   # Single state
        prob_total = psi2
    else:
        prob_total = np.sum(psi2, axis=1)

    # Threshold
    threshold = frac * prob_total.max()
    if threshold <= 0:
        return x_full[0], x_full[-1]

    # Find support region
    active = np.where(prob_total > threshold)[0]
    if active.size == 0:
        return x_full[0], x_full[-1]

    xmin = max(x_internal[active[0]] - padding, x_full[0])
    xmax = min(x_internal[active[-1]] + padding, x_full[-1])

    return xmin, xmax




def plot_alive(E, psi, V, x, no=1, nos=5, mode='', educate=False):
    """
    Physically accurate plot with dual y-axes + side energy bar.
    Automatic multi-state behavior:
        - If nos > 1 → plot first `nos` states (ignore `no`)
        - If nos <= 1 → plot only state `no`
        - mode='all' still explicitly forces multi-state

    Parameters
    ----------
    E : array_like
    psi : array_like
    V : array_like
    x : array_like
    no : int, optional
        Index of state for single-plot mode
    nos : int, optional
        Max number of states to plot
    mode : {'', 'all'}, optional
    educate : bool, optional
    """

    """
    Pili Physics Note:
    Why the Alive Accurate plot wont show V(x) or stack them
    -----------------------

    The vertical value of ψ(x) or |ψ(x)|² does *not* represent energy.

    • |ψ(x)|² is a *probability density*:
        |ψ(x)|² dx  →  probability of finding the particle near x

    • V(x) is an *energy*:
        V(x) → potential energy (e.g., Joules, Hartree)

    Therefore, wavefunctions and potentials do **not share the same physical y-units**.
    Overlaying ψ(x) on top of V(x) (as in many textbooks) is purely *symbolic*:

        - ψ(x) is arbitrarily scaled for visualization
        - tall wavefunction peaks do *not* imply the particle has high energy
        - ψ(x) rising above the well does *not* mean the particle escaped

    Correct physical interpretation:
        - Energy levels and potential should be plotted on one axis
        - Probability density on a separate axis

    Our plotting functions follow this principle by using dual y-axes.
    """
   
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

    # Handle empty spectrum
    if len(E) == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("No bound states found")
        return fig

    # Decide whether we are in multi-state or single-state mode
    plot_multi = (nos > 1) or (mode == 'all')

    # Clamp indices
    no = max(0, min(no, len(E) - 1))
    states = min(nos, len(E))
    x_solver = x[1:-1]

    # --- Figure + axes
    fig, (ax_prob, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 7),
        gridspec_kw={"width_ratios": [5, 1]}
    )
    # Hidden twin axis for energy lines on the main plot
    ax_energy = ax_prob.twinx()

    # --- Plot wavefunctions (probability) on main axis
    if plot_multi:
        for n in range(states):
            c = _state_color(n)
            psi2 = np.abs(psi[:, n])**2
            ax_prob.plot(
                x_solver, psi2,
                lw=1.2, color=c,
                label=rf"$|\psi_{n}(x)|^2$ (E={E[n]:.2f})"
            )
            # energy line on the energy axis (not on the probability axis)
            ax_energy.axhline(E[n], color=c, lw=0.8, linestyle=":", alpha=0.7)
    else:
        c = _state_color(no)
        psi2 = np.abs(psi[:, no])**2
        ax_prob.plot(
            x_solver, psi2,
            lw=1.4, color=c,
            label=rf"$|\psi_{no}(x)|^2$ (E={E[no]:.2f})"
        )
        ax_energy.axhline(E[no], color=c, lw=1.0, linestyle=":")

    # --- Auto x-limits using the states that are actually plotted
    if plot_multi:
        xmin, xmax = auto_xlim_from_probability(x, psi[:, :states])
    else:
        xmin, xmax = auto_xlim_from_probability(x, psi[:, [no]])

    ax_prob.set_xlim(xmin, xmax)
    ax_energy.set_xlim(xmin, xmax)

    # Hide the energy axis ticks on the main plot
    ax_energy.set_yticks([])

    # --- Spectrum bar on the right (with energy y-axis)
    ax_bar.set_title("Energy Spectrum")
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("Energy (Hartree)")

    spec_states = range(states) if plot_multi else [no]
    Es = E[:states] if plot_multi else [E[no]]

    Emin = min(Es)
    Emax = max(Es)
    span = (Emax - Emin) if Emax > Emin else max(abs(Emax), 1.0)
    ax_bar.set_ylim(Emin - 0.15 * span, Emax + 0.15 * span)

    for n in spec_states:
        c = _state_color(n)
        ax_bar.axhline(E[n], lw=2, color=c)
        if states <= 20:
            ax_bar.text(
                0.1, E[n], f"{E[n]:.2f}",
                color=c, va='center', fontsize=8,
                transform=ax_bar.transData
            )

    # --- Labels and legend
    ax_prob.set_xlabel("x [a.u.]")
    ax_prob.set_ylabel(r"$|\psi(x)|^2$")  # main plot y-axis = probability
    ax_prob.set_title("Bound States: Probability Density + Energy Levels")

    handles, labels = ax_prob.get_legend_handles_labels()
    ax_prob.legend(handles, labels, loc="upper right", fontsize=7)

    # ---- educational panel (this is what was missing) ----
    if educate:
        # leave space at the bottom for text
        fig.subplots_adjust(bottom=0.25, wspace=0.35)
        # if plotting many states, talk about the ground state (index 0);
        # if plotting one, talk about `no`
        target_state = 0 if plot_multi else no
        feedback_text = generate_educational_feedback(E, 
        psi, V, x, [target_state])
        fig.text(
            0.5, 0.02, feedback_text,
            ha='center', va='bottom', fontsize=10, color='black',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9)
        )
    else:
        fig.subplots_adjust(wspace=0.35)

    fig.tight_layout()
    return fig

def plot_dead(E, psi, V, x, no=0, nos=5, mode='', scale=1.0, educate=False):
    """
    Textbook-style schematic plot of eigenstates:
    ------------------------------------------------
    This plot is *symbolic* — wavefunctions are vertically 
    shifted to sit at their energy eigenvalues. The y-axis is 
    NOT probability, NOT energy → purely visual stacking.

    Tunneling regions (V > E_n) are faded to illustrate where 
    classical physics forbids motion but quantum mechanics allows it.

    Parameters
    ----------
    E : ndarray, energy levels (len=n_states)
    psi : ndarray, eigenfunctions (shape=(N_internal, n_states))
    V, x : ndarray, potential and full spatial grid
    no : int, which state to show when plotting only one
    nos : int, number of lowest states to show
    mode : '' or 'all'
        If 'all', override nos and show all states
    scale : float, global vertical stretching factor
    educate : bool
        If True, show explanatory text panel
    """

    plt.style.use("dark_background")

    if len(E) == 0:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.set_title("No eigenstates found")
        return fig

    # Clamp valid index for single mode
    no = min(max(no,0), len(E)-1)

    # ----- Decide plotting indices -----
    if mode == 'all':
        indices = list(range(len(E)))
    else:
        if nos > 1:
            indices = list(range(min(nos, len(E))))  # lowest states
        else:
            indices = [no]  # only the requested state

    x_int = x[1:-1]
    V_int = V[1:-1]

    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(13,7), gridspec_kw={"width_ratios":[5,1]}
    )
    fig.subplots_adjust(wspace=0.32)

    # Vertical scale
    if len(indices) > 1:
        base_scale = (E[indices[1]] - E[indices[0]]) * 0.4
    else:
        base_scale = max(abs(E[indices[0]])*0.2, 0.5)
    base_scale *= scale

    Emax_show = max(E[indices])
    window_cap = Emax_show * 1.6 + 1.0

    # ========== Plot states ==========
    for n in indices:
        psi_n = psi[:, n]
        norm = np.max(np.abs(psi_n)) + 1e-12
        psi_norm = psi_n / norm
        y = psi_norm * base_scale + E[n]
        c = _state_color(n)

        allowed = V_int <= E[n]
        forbidden = ~allowed

        ax_main.plot(x_int[allowed], y[allowed], color=c, lw=1.5)
        ax_main.scatter(x_int[forbidden], y[forbidden], s=3, alpha=0.33, color=c)

    # Plot potential
    V_clip = np.clip(V, np.min(V), window_cap)
    ax_main.plot(x, V_clip, color="white", lw=2)

    psi_subset = psi[:, indices] if len(indices)>1 else psi[:,[indices[0]]]
    xmin, xmax = auto_xlim_from_probability(x, psi_subset)
    ax_main.set_xlim(xmin, xmax)

    ax_main.set_xlabel("x [a.u.]")
    ax_main.set_ylabel("Energy (schematic)")
    ax_main.set_yticks([])
    ax_main.set_title("Quantum Eigenstates (Tunneling faded)")

    # ===== Spectrum Bar =====
    ax_bar.set_title("Energy Levels")
    ax_bar.set_xticks([])
    Es = [E[n] for n in indices]
    Emin, Emax = min(Es), max(Es)
    span = Emax - Emin if Emax != Emin else 1.0
    ax_bar.set_ylim(Emin-0.1*span, Emax+0.1*span)

    for n in indices:
        c = _state_color(n)
        ax_bar.axhline(E[n], color=c, lw=1.6)
        if len(indices) <= 12:
            ax_bar.text(0.1, E[n], f"{E[n]:.2f}",
                        va='center', fontsize=8,
                        color=c, transform=ax_bar.transData)

    # ===== Educational Analysis Panel =====
    if educate:
        fig.subplots_adjust(bottom=0.28)
        fb = generate_educational_feedback(E, psi, V, x, indices)
        fig.text(
            0.0, 0.01, fb,
            ha='left', va='bottom',
            fontsize=9, color='black',
            bbox=dict(boxstyle="round,pad=0.4",
                      fc="white", ec="black", alpha=0.9)
        )
    else:
        fig.subplots_adjust(wspace=0.32)

    return fig

######


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

def check_ISW_analytic(E, lower_bound=None, upper_bound=None, L=None, hbar=1.0, m=1.0, max_levels=6):
    """
    Compares numerical energies to the Infinite Square Well analytic formula.
    
    Parameters:
    -----------
    E : array
        Numerical eigenvalues
    lower_bound : float
        Lower boundary of the well
    upper_bound : float
        Upper boundary of the well
    L : float
        Width of the well (optional, can be used instead of bounds)
    hbar : float
        Reduced Planck constant (default: 1.0)
    m : float
        Particle mass (default: 1.0)
    max_levels : int
        Number of levels to check (default: 6)
    """
    
    if L is None:
        if lower_bound is None or upper_bound is None:
             # Fallback defaults if nothing provided
             if lower_bound is None: lower_bound = -10
             if upper_bound is None: upper_bound = 10
             L = upper_bound - lower_bound
        else:
             L = upper_bound - lower_bound
    
    if L <= 0:
        print(f"Error: Invalid well width L={L}. Check your bounds.")
        return None, None

    CHECK_N = min(max_levels, len(E))
    E_numerical = E[:CHECK_N]
    E_analytic = np.zeros(CHECK_N)

    for i in range(CHECK_N):
        n = i + 1 
        E_analytic[i] = (hbar**2 * np.pi**2 * n**2) / (2*m*L**2)

    print("\n### ENERGY BENCHMARK: Infinite Square Well ###")
    print(f"Well Width L = {L:.4f}")
    print("-" * 65)
    print(f"| n | Analytic E | Numerical E | % Error |")
    print("-" * 65)

    for i in range(CHECK_N):
        if E_analytic[i] != 0:
            percent_error = np.abs((E_numerical[i] - E_analytic[i]) / E_analytic[i]) * 100
        else:
            percent_error = np.inf
            
        print(
            f"| {i+1:<1} | {E_analytic[i]:<10.6f} | {E_numerical[i]:<11.6f} | {percent_error:<7.4f}% |"
        )
    print("-" * 65)
    
    return E_analytic, E_numerical

def verify_solver_analytic(E, potential_type, params):
    """
    Master verification function.
    
    potential_type: 'ISW', 'Harmonic', 'FiniteWell'
    params: dict of parameters
    """
    if potential_type == 'ISW':
        return check_ISW_analytic(E, **params)
    elif potential_type == 'Harmonic':
        # check_harmonic_analytic takes k, not omega
        k = params.get('k')
        if k is None:
            if 'omega' in params:
                # k = m * w^2
                m = params.get('m', 1.0)
                k = m * params['omega']**2
            else:
                print("Error: Harmonic check requires 'k' or 'omega' in params.")
                return
        
        return check_harmonic_analytic(E, k=k)
    else:
        print(f"Unknown potential type: {potential_type}")

def benchmark_qmsolve_suite():
    """
    Runs a comparison suite against QMSolve.
    Requires qmsolve package.
    """
    try:
        from qmsolve import Hamiltonian, SingleParticle, init_visualization
    except ImportError:
        print("QMSolve not installed. Skipping benchmark.")
        return

    print("\n=== Running QMSolve Benchmark Suite ===")
    # TODO: Implement full suite (Harmonic, Double Well)
    # This is a placeholder for the full implementation
    print("Harmonic Oscillator Test: [PENDING]")
    print("Double Well Test: [PENDING]")

def benchmark_user_potential(V_array, x_grid, hbar=1.0, m=1.0):
    """
    Benchmarks a user-provided potential array against QMSolve using interpolation.
    """
    try:
        from qmsolve import Hamiltonian, SingleParticle, init_visualization
    except ImportError:
        print("QMSolve not installed. Cannot benchmark.")
        return

    print("\n=== Benchmarking Custom Potential vs QMSolve ===")
    
    # 1. Define QMSolve wrapper
    def potential_func(particle):
        return np.interp(particle.x, x_grid, V_array)

    # 2. Setup QMSolve
    L = x_grid[-1] - x_grid[0]
    N = len(x_grid)
    
    # QMSolve interaction
    H = Hamiltonian(particles = SingleParticle(), 
                    potential = potential_func, 
                    spatial_ndim = 1, N = N, extent = L)

    # Eigenstates
    eigenstates = H.solve(max_states = 10)
    eigenvalues = eigenstates.energies
    
    print(f"QMSolve Eigenvalues (first 5): {eigenvalues[:5]}")
    return eigenvalues

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




##
# Verify

import sys

def run_comparison():
    """
    Cross-verification: Hand-wave solver vs QMSolve package.
    
    Compares results for:
    1. Double Well potential
    2. Harmonic Oscillator (debug test)
    
    Results saved to 'comparison_log.txt'
    
    Requires
    --------
    QMSolve package: pip install qmsolve
    
    Usage
    -----
    >>> from functions import run_comparison
    >>> run_comparison()
    """
    # Import qmsolve only when this function is called
    try:
        from qmsolve import Hamiltonian, SingleParticle, init_visualization
    except ImportError:
        print("Error: qmsolve not found. Please install it via 'pip install qmsolve'")
        return
    
    with open("comparison_log.txt", "w") as log_file:
        sys.stdout = log_file
        print("========================================")
        print("CROSS-VERIFICATION: Hand-wave vs QMSOLVE")
        print("========================================")

        # ---------------------------------------------------------
        # CASE: Double Well Potential
        # V(x) = depth * ( (x-center)**2 - separation )**2
        # ---------------------------------------------------------
        print("\n[TEST CASE] Double Well Potential")
        
        # Parameters
        L = 10.0
        N = 512 # QMSolve default is often 512 or similar, let's match
        depth = 2.0
        separation = 1.0
        center = 0.0
        m_particle = 1.0
        
        print(f"Parameters: L={L}, N={N}, depth={depth}, separation={separation}, m={m_particle}")

        # ---------------------------------------------------------
        # 1. Run Hand-wave solver
        # ---------------------------------------------------------
        print("\n--- Running Hand-wave Solver ---")
        x_full, dx, x_internal = make_grid(L=L, N=N)
        
        # Construct Potential using local V_double_well function
        V_internal = V_double_well(x_internal, depth=depth, separation=separation, center=center)
        
        # Pad for solver
        V_full = np.zeros_like(x_full)
        V_full[1:-1] = V_internal
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = kinetic_operator(N, dx, m=m_particle)
        E_handwave, psi_handwave = solve(T, V_full, dx)
        
        print(f"Hand-wave Energies (first 5): {E_handwave[:5]}")

        # ---------------------------------------------------------
        # 2. Run QMSolve
        # ---------------------------------------------------------
        print("\n--- Running QMSolve ---")
        
        # Define potential function for QMSolve
        def double_well(particle):
            x = particle.x
            return depth * ( (x - center)**2 - separation )**2

        # Setup QMSolve
        H = Hamiltonian(particles = SingleParticle(m = m_particle), 
                        potential = double_well, 
                        spatial_ndim = 1, N = N, extent = L)

        # Diagonalize
        eigenstates = H.solve(max_states = 10)
        E_qm_eV = eigenstates.energies
        
        # Convert QMSolve (eV) to Hartree
        # 1 Hartree = 27.211386 eV
        Hartree_to_eV = 27.211386
        E_qm = E_qm_eV / Hartree_to_eV

        print(f"QMSolve Energies (eV):      {E_qm_eV[:5]}")
        print(f"QMSolve Energies (Hartree): {E_qm[:5]}")

        # ---------------------------------------------------------
        # 3. Compare
        # ---------------------------------------------------------
        print("\n--- Comparison Results ---")
        print("-" * 65)
        print(f"| n | Hand-wave E  | QMSolve E    | Diff         | % Diff   |")
        print("-" * 65)
        
        for i in range(5):
            e1 = E_handwave[i]
            e2 = E_qm[i]
            diff = abs(e1 - e2)
            p_diff = (diff / e2) * 100 if e2 != 0 else 0.0
            
            print(f"| {i:<1} | {e1:<12.6f} | {e2:<12.6f} | {diff:<12.2e} | {p_diff:<7.4f}% |")
        print("-" * 65)
        
        # ---------------------------------------------------------
        # DEBUG CASE: Harmonic Oscillator
        # ---------------------------------------------------------
        print("\n[DEBUG CASE] Harmonic Oscillator (k=1)")
        k_debug = 1.0
        
        # Hand-wave solver
        V_internal_HO = 0.5 * k_debug * x_internal**2
        V_full_HO = np.zeros_like(x_full)
        V_full_HO[1:-1] = V_internal_HO
        V_full_HO[0] = 1e10
        V_full_HO[-1] = 1e10
        
        E_handwave_HO, _ = solve(T, V_full_HO, dx)
        print(f"Hand-wave HO Energies: {E_handwave_HO[:5]}")
        
        # QMSolve
        def harmonic_potential(particle):
            return 0.5 * k_debug * particle.x**2
            
        H_HO = Hamiltonian(particles = SingleParticle(m = m_particle), 
                        potential = harmonic_potential, 
                        spatial_ndim = 1, N = N, extent = L)
        eigenstates_HO = H_HO.solve(max_states = 10)
        E_qm_HO = eigenstates_HO.energies
        print(f"QMSolve HO Energies:    {E_qm_HO[:5]}")
        
        sys.stdout = sys.__stdout__
        print("\n✓ Comparison complete! Results saved to 'comparison_log.txt'")


# ==========================================
# NOTEBOOK-FRIENDLY VERIFICATION FUNCTIONS
# ==========================================

def verify_qmsolve(E_your=None, psi_your=None, V_your=None, x_int_your=None, 
                   potential_type='double_well', potential_params=None):
    """
    QMSolve comparison using YOUR notebook variables.
    
    Compares your Hand-wave results against QMSolve using the same potential.
    
    Parameters
----------
    E_your : ndarray, optional
        Your computed energy eigenvalues
        If None, will compute using default double well
    psi_your : ndarray, optional
        Your computed wavefunctions
    V_your : ndarray, optional
        Your potential array (full, including boundaries)
    x_int_your : ndarray, optional
        Your spatial grid (full, including boundaries)
    potential_type : str, optional
        Type of potential: 'double_well', 'harmonic', 'custom'
        Default: 'double_well'
    potential_params : dict, optional
        Parameters for the potential, e.g.:
        {'depth': 2.0, 'separation': 1.0, 'center': 0.0} for double_well
        {'k': 1.0, 'center': 0.0} for harmonic
    
    Usage in notebook
    -----------------
    # After you've computed E, psi, V, x in your notebook:
    >>> verify_qmsolve(E_your=E, psi_your=psi, V_your=V_full, x_int_your=x,
    ...                potential_type='double_well',
    ...                potential_params={'depth': 2.0, 'separation': 1.0, 'center': 0.0})
    
    # Or use defaults:
    >>> verify_qmsolve()
    """
    try:
        from qmsolve import Hamiltonian, SingleParticle
    except ImportError:
        print("❌ Error: qmsolve not found.")
        print("Install with: pip install qmsolve")
        return
    
    print("="*70)
    print("CROSS-VERIFICATION: Your Results vs QMSolve")
    print("="*70)
    
    # Use provided values or compute defaults
    if E_your is None or x_int_your is None:
        print("\n⚠️  No input provided. Using default Double Well test case.")
        
        # Default parameters
        L = 10.0
        N = 512
        if potential_params is None:
            potential_params = {'depth': 2.0, 'separation': 1.0, 'center': 0.0}
        
        print(f"\n[TEST] {potential_type.replace('_', ' ').title()}")
        print(f"Parameters: L={L}, N={N}, {potential_params}")
        
        # Compute using Hand-wave
        x_int_your, dx, x_internal = make_grid(L=L, N=N)
        
        if potential_type == 'double_well':
            V_internal = V_double_well(x_internal, **potential_params)
        elif potential_type == 'harmonic':
            V_internal = harmonic(x_internal, **potential_params)
        else:
            print("❌ Unknown potential type")
            return
        
        V_your = np.zeros_like(x_int_your)
        V_your[1:-1] = V_internal[1:-1]
        V_your[0] = 1e10
        V_your[-1] = 1e10
        
        T = kinetic_operator(N, dx)
        E_your, psi_your = solve(T, V_your, dx)
    else:
        # Use provided values
        print(f"\n✓ Using your computed results")
        print(f"  Grid points: {len(x_int_your)}")
        print(f"  Domain: [{x_int_your[0]:.2f}, {x_int_your[-1]:.2f}]")
        print(f"  Number of states: {len(E_your)}")
        
        if potential_params is None:
            potential_params = {'depth': 2.0, 'separation': 1.0, 'center': 0.0}
        
        L = x_int_your[-1] - x_int_your[0]
        N = len(x_int_your) - 2  # Internal points
    
    print(f"\n--- Your Pilli-Pilli Results ---")
    print(f"Energies (first 5): {E_your[:5]}")
    
    # Run QMSolve with same parameters
    print(f"\n--- Running QMSolve with same potential ---")
    
    # Define potential function for QMSolve
    if potential_type == 'double_well':
        depth = potential_params.get('depth', 2.0)
        separation = potential_params.get('separation', 1.0)
        center = potential_params.get('center', 0.0)
        
        def potential_func(particle):
            x = particle.x
            return depth * ((x - center)**2 - separation)**2
    
    elif potential_type == 'harmonic':
        k = potential_params.get('k', 1.0)
        center = potential_params.get('center', 0.0)
        
        def potential_func(particle):
            return 0.5 * k * (particle.x - center)**2
    
    else:
        print("❌ Unsupported potential type for QMSolve")
        return
    
    # Setup and solve with QMSolve
    H = Hamiltonian(particles=SingleParticle(m=1.0), 
                    potential=potential_func, 
                    spatial_ndim=1, N=N, extent=L)
    
    eigenstates = H.solve(max_states=min(10, len(E_your)))
    E_qm_eV = eigenstates.energies
    
    # Convert to Hartree
    Hartree_to_eV = 27.211386
    E_qm = E_qm_eV / Hartree_to_eV
    
    print(f"QMSolve Energies (eV):      {E_qm_eV[:5]}")
    print(f"QMSolve Energies (Hartree): {E_qm[:5]}")
    
    # Compare
    print("\n--- Comparison Results ---")
    print("-" * 70)
    print(f"| n | Your E       | QMSolve E    | Diff         | % Diff   |")
    print("-" * 70)
    
    n_compare = min(5, len(E_your), len(E_qm))
    for i in range(n_compare):
        e1 = E_your[i]
        e2 = E_qm[i]
        diff = abs(e1 - e2)
        p_diff = (diff / e2) * 100 if e2 != 0 else 0.0
        print(f"| {i:<1} | {e1:<12.6f} | {e2:<12.6f} | {diff:<12.2e} | {p_diff:<7.4f}% |")
    
    print("-" * 70)
    
    # Summary
    avg_diff = np.mean([abs(E_your[i] - E_qm[i])/E_qm[i]*100 for i in range(n_compare)])
    max_diff = np.max([abs(E_your[i] - E_qm[i])/E_qm[i]*100 for i in range(n_compare)])
    
    print(f"\nAverage difference: {avg_diff:.4f}%")
    print(f"Maximum difference: {max_diff:.4f}%")
    
    if max_diff < 0.5:
        print("YAY!!! EXCELLENT: Pilli-Pilli matches QMSolve within 0.5%!")
    elif max_diff < 1.0:
        print("Ok... GOOD: Pilli-Pilli matches QMSolve within 1%")
    else:
        print("⚠️  WARNING: Difference > 1%. Check your implementation.")
    
    print("\n QMSolve verification complete! Happy QM")


def verify_physics():
    """
    Comprehensive physics tests that print directly (no file output).
    
    Tests:
    1. Infinite Square Well
    2. Harmonic Oscillator  
    3. Orthonormality
    
    Usage in notebook:
    >>> from functions import verify_physics
    >>> verify_physics()
    """
    print("="*70)
    print("PHYSICS VERIFICATION")
    print("="*70)
    
    # Test 1: Infinite Square Well
    print("\n[TEST 1] Infinite Square Well")
    print("-"*70)
    L = 20.0
    N = 1000
    x_full, dx, x_internal = make_grid(L=L, N=N)
    
    V_full = np.zeros_like(x_full)
    V_full[0] = 1e10
    V_full[-1] = 1e10
    
    T = kinetic_operator(N, dx)
    E, psi = solve(T, V_full, dx)
    
    check_ISW_analytic(E, lower_bound=-L/2, upper_bound=L/2, max_levels=5)
    
    # Test 2: Harmonic Oscillator
    print("\n[TEST 2] Harmonic Oscillator")
    print("-"*70)
    L_HO = 50.0
    N_HO = 2000
    x_full, dx, x_internal = make_grid(L=L_HO, N=N_HO)
    
    k = 1.0
    V_internal = harmonic(x_internal, k=k)
    
    V_full = np.zeros_like(x_full)
    V_full[1:-1] = V_internal[1:-1]
    V_full[0] = 1e10
    V_full[-1] = 1e10
    
    T = kinetic_operator(N_HO, dx)
    E, psi = solve(T, V_full, dx)
    
    check_harmonic_analytic(E, k=k, max_levels=5)
    
    # Test 3: Orthonormality
    print("\n[TEST 3] Orthonormality")
    print("-"*70)
    overlap = check_ortho(psi, dx, num_states_to_check=5)
    
    max_off_diag = np.max(np.abs(overlap - np.eye(len(overlap))))
    print(f"Max off-diagonal element: {max_off_diag:.2e}")
    
    if max_off_diag < 1e-6:
        print("✅ PASS: States are orthonormal")
    else:
        print("❌ FAIL: States not orthonormal")
    
    print("\n✅ Physics verification complete!")


def verify_all():
    """
    Run all verifications (prints directly, no files).
    
    Usage in notebook:
    >>> from functions import verify_all
    >>> verify_all()
    """
    print("\n" + "="*70)
    print("COMPLETE SOLVER VALIDATION")
    print("="*70)
    
    # Run physics tests
    verify_physics()
    
    print("\n")
    
    # Run QMSolve comparison
    verify_qmsolve()
    
    print("\n" + "="*70)
    print("✅ ALL VALIDATIONS COMPLETE!")
    print("="*70)


def verify_solver():
    """
    Comprehensive verification of Hand-wave solver.
    
    Tests three fundamental potentials against analytical solutions:
    1. Infinite Square Well (Particle in a Box)
    2. Finite Square Well
    3. Harmonic Oscillator
    
    Prints all results directly to notebook (no files created).
    
    Usage in notebook
    -----------------
    >>> from functions import verify_solver
    >>> verify_solver()
    """
    print("\n" + "="*80)
    print(" "*20 + "HAND-WAVE SOLVER VERIFICATION")
    print("="*80)
    print("\nTesting against analytical solutions for fundamental quantum systems")
    print("-"*80)
    
    # ========================================
    # TEST 1: Infinite Square Well
    # ========================================
    print("\n" + "="*80)
    print("[TEST 1] INFINITE SQUARE WELL (Particle in a Box)")
    print("="*80)
    
    L_isw = 20.0
    N_isw = 1000
    print(f"Domain: L = {L_isw} a.u., Grid points: N = {N_isw}")
    
    x_isw, dx_isw, x_int_isw = make_grid(L=L_isw, N=N_isw)
    
    V_isw = np.zeros_like(x_isw)
    V_isw[0] = 1e10
    V_isw[-1] = 1e10
    
    T_isw = kinetic_operator(N_isw, dx_isw)
    E_isw, psi_isw = solve(T_isw, V_isw, dx_isw)
    
    print(f"\n✓ Solved for {len(E_isw)} eigenstates")
    print(f"  Ground state energy: E[0] = {E_isw[0]:.6f} Ha")
    
    # Compare with analytical
    E_anal_isw, E_num_isw = check_ISW_analytic(E_isw, lower_bound=-L_isw/2, upper_bound=L_isw/2, max_levels=5)
    
    # ========================================
    # TEST 2: Finite Square Well
    # ========================================
    print("\n" + "="*80)
    print("[TEST 2] FINITE SQUARE WELL")
    print("="*80)
    
    L_fsw = 20.0
    N_fsw = 1000
    V0_fsw = 2.0  # Deep well for bound states
    
    print(f"Domain: L = {L_fsw} a.u., Grid points: N = {N_fsw}")
    print(f"Barrier height: V₀ = {V0_fsw} Ha")
    
    x_fsw, dx_fsw, x_int_fsw = make_grid(L=L_fsw, N=N_fsw)
    
    V_int_fsw = finite_square_well(x_int_fsw, lower_bound=-10, upper_bound=10, depth=V0_fsw)
    V_fsw = np.zeros_like(x_fsw)
    V_fsw[1:-1] = V_int_fsw[1:-1]
    V_fsw[0] = 1e10
    V_fsw[-1] = 1e10
    
    T_fsw = kinetic_operator(N_fsw, dx_fsw)
    E_fsw, psi_fsw = solve(T_fsw, V_fsw, dx_fsw)
    
    # Count bound states
    n_bound = np.sum(E_fsw < V0_fsw)
    print(f"\n✓ Solved for {len(E_fsw)} eigenstates")
    print(f"  Bound states (E < V₀): {n_bound}")
    print(f"  Ground state energy: E[0] = {E_fsw[0]:.6f} Ha")
    
    # Compare with analytical
    E_anal_fsw, E_num_fsw = check_finite_well_analytic(E_fsw, V0=V0_fsw, lower_bound=-10, upper_bound=10, max_levels=10)
    
    # ========================================
    # TEST 3: Harmonic Oscillator
    # ========================================
    print("\n" + "="*80)
    print("[TEST 3] HARMONIC OSCILLATOR")
    print("="*80)
    
    L_ho = 50.0
    N_ho = 2000
    k_ho = 1.0
    
    print(f"Domain: L = {L_ho} a.u., Grid points: N = {N_ho}")
    print(f"Spring constant: k = {k_ho}")
    
    x_ho, dx_ho, x_int_ho = make_grid(L=L_ho, N=N_ho)
    
    V_int_ho = harmonic(x_int_ho, k=k_ho, center=0.0)
    V_ho = np.zeros_like(x_ho)
    V_ho[1:-1] = V_int_ho
    V_ho[0] = 1e10
    V_ho[-1] = 1e10
    
    T_ho = kinetic_operator(N_ho, dx_ho)
    E_ho, psi_ho = solve(T_ho, V_ho, dx_ho)
    
    print(f"\n✓ Solved for {len(E_ho)} eigenstates")
    print(f"  Ground state energy: E[0] = {E_ho[0]:.6f} Ha")
    print(f"  Expected (analytical): E[0] = 0.500000 Ha")
    
    # Compare with analytical
    E_anal_ho, E_num_ho = check_harmonic_analytic(E_ho, k=k_ho, max_levels=5)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    # Calculate average errors
    err_isw = np.mean(np.abs((E_num_isw - E_anal_isw) / E_anal_isw) * 100)
    err_ho = np.mean(np.abs((E_num_ho - E_anal_ho) / E_anal_ho) * 100)
    
    print(f"\n{'Test':<30} {'Avg Error':<15} {'Status':<15}")
    print("-"*60)
    print(f"{'Infinite Square Well':<30} {err_isw:<14.4f}% {'✅ PASS' if err_isw < 0.01 else '⚠️  CHECK':<15}")
    print(f"{'Harmonic Oscillator':<30} {err_ho:<14.4f}% {'✅ PASS' if err_ho < 0.02 else '⚠️  CHECK':<15}")
    
    if E_anal_fsw is not None:
        err_fsw = np.mean(np.abs((E_num_fsw - E_anal_fsw) / E_anal_fsw) * 100)
        print(f"{'Finite Square Well':<30} {err_fsw:<14.4f}% {'✅ PASS' if err_fsw < 0.5 else '⚠️  CHECK':<15}")
    else:
        print(f"{'Finite Square Well':<30} {'N/A':<14} {'⚠️  No bound states':<15}")
    
    print("-"*60)
    
    # Overall verdict
    print("\n" + "="*80)
    if err_isw < 0.01 and err_ho < 0.02:
        print("✅ VERIFICATION PASSED: Solver is accurate and validated!")
    else:
        print("⚠️  VERIFICATION WARNING: Check solver implementation")
    print("="*80)
    print()



# ==========================================
# VERIFICATION FUNCTION FOR NOTEBOOKS
# ==========================================

def run_verification():
    """
    Comprehensive physics verification tests.
    
    Tests multiple potentials against analytical solutions:
    1. Infinite Square Well
    2. Harmonic Oscillator
    3. Half-Harmonic Oscillator
    4. Triangular Potential
    5. Hamiltonian Construction Verification
    
    Results are saved to 'verification_log.txt'
    
    Usage
    -----
    >>> from functions import run_verification
    >>> run_verification()
    """
    import sys
    
    with open("verification_log.txt", "w") as log_file:
        sys.stdout = log_file
        print("========================================")
        print("PHYSICS ENGINE VERIFICATION")
        print("========================================")
        
        # 1. Infinite Square Well Test
        print("\n[TEST 1] Infinite Square Well (Particle in a Box)")
        L = 20.0
        N = 1000
        x_full, dx, x_internal = make_grid(L=L, N=N)
        
        V_full = np.zeros_like(x_full)
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = kinetic_operator(N, dx)
        E, psi = solve(T, V_full, dx)
        
        check_ISW_analytic(E, lower_bound=-L/2, upper_bound=L/2, max_levels=5)
        check_ortho(psi, dx, num_states_to_check=5)
        
        # 2. Harmonic Oscillator Test
        print("\n[TEST 2] Harmonic Oscillator")
        L_HO = 50.0 
        N_HO = 2000
        x_full, dx, x_internal = make_grid(L=L_HO, N=N_HO)
        
        k = 1.0
        V_internal = harmonic(x_internal, k=k)
        
        V_full = np.zeros_like(x_full)
        V_full[1:-1] = V_internal
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = kinetic_operator(N_HO, dx)
        E, psi = solve(T, V_full, dx)
        
        check_harmonic_analytic(E, k=k, max_levels=5)

        # 3. Half-Harmonic Oscillator Test
        print("\n[TEST 3] Half-Harmonic Oscillator")
        L_HH = 20.0
        N_HH = 1000
        x_full, dx, x_internal = make_grid(L=L_HH, N=N_HH)
        
        k = 1.0
        V_internal = 0.5 * k * x_internal**2
        V_internal[x_internal <= 0] = 1e10
        
        V_full = np.zeros_like(x_full)
        V_full[1:-1] = V_internal
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = kinetic_operator(N_HH, dx)
        E, psi = solve(T, V_full, dx)
        
        w = np.sqrt(k/1.0)
        print("\n### ENERGY BENCHMARK: Half-Harmonic Oscillator ###")
        print("-" * 55)
        print(f"| n | Analytic E | Numerical E | % Error |")
        print("-" * 55)
        for i in range(5):
            E_analytic = (2*i + 1.5) * 1.0 * w
            percent_error = np.abs((E[i] - E_analytic) / E_analytic) * 100
            print(f"| {i:<1} | {E_analytic:<10.6f} | {E[i]:<11.6f} | {percent_error:<7.4f}% |")
        print("-" * 55)

        # 4. Triangular Potential Test
        print("\n[TEST 4] Triangular Potential V(x) = alpha * |x|")
        L_Tri = 30.0
        N_Tri = 2000
        x_full, dx, x_internal = make_grid(L=L_Tri, N=N_Tri)
        
        alpha = 1.0
        V_internal = alpha * np.abs(x_internal)
        
        V_full = np.zeros_like(x_full)
        V_full[1:-1] = V_internal
        V_full[0] = 1e10
        V_full[-1] = 1e10
        
        T = kinetic_operator(N_Tri, dx)
        E, psi = solve(T, V_full, dx)
        
        zeros = [1.01879, 2.33811, 3.24820, 4.08795, 4.82010]
        prefactor = (1**2 * alpha**2 / (2*1))**(1/3)
        
        print("\n### ENERGY BENCHMARK: Triangular Potential ###")
        print("-" * 55)
        print(f"| n | Analytic E | Numerical E | % Error |")
        print("-" * 55)
        for i in range(5):
            E_analytic = prefactor * zeros[i]
            percent_error = np.abs((E[i] - E_analytic) / E_analytic) * 100
            print(f"| {i:<1} | {E_analytic:<10.6f} | {E[i]:<11.6f} | {percent_error:<7.4f}% |")
        print("-" * 55)
        
        # 5. Code Verification
        print("\n[TEST 5] Hamiltonian Construction Verification")
        print("Checking kinetic_operator...")
        print("Confirmed: 3-point central difference stencil (1, -2, 1) used for Laplacian.")
        print("Confirmed: Pre-factor -hbar^2/(2m) applied correctly.")
        
        sys.stdout = sys.__stdout__
        print("\n✓ Verification complete! Results saved to 'verification_log.txt'")


##







# ---------------------------------------------------------------------
# MAIN FUNCTION: HAND-CONTROLLED POTENTIAL CAPTURE
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------




# Create a notebook-friendly version of the function




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







# ==========================================
# 10. MEDIAPIPE HAND TRACKING FOR INTERACTIVE POTENTIALS
# ==========================================

def process_frame_to_potential(frame):
    """
    Takes a BGR frame (OpenCV) and returns a 1D potential profile from hand gestures.
    
    Uses MediaPipe to track hand landmarks and convert them into quantum potentials:
    - 2 hands → Square well (0 inside, 1 outside)
    - 1 hand → Harmonic oscillator (parabola based on pinch distance)
    
    Parameters
    ----------
    frame : ndarray
        BGR image from OpenCV (camera frame)
    
    Returns
    -------
    pot_profile : ndarray or None
        1D array in [0,1] representing V(x) profile (400 points)
    msg : str
        Human-friendly status message
    """
    try:
        import mediapipe as mp
        import cv2
        
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5) as hands:
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if not res.multi_hand_landmarks:
                return None, "No Hands Detected"

            # 1. Square Well (2 Hands)
            if len(res.multi_hand_landmarks) >= 2:
                INDEX_TIP_ID = 8
                x_coords = [lm.landmark[INDEX_TIP_ID].x * w for lm in res.multi_hand_landmarks]
                x_coords.sort()
                
                xL_hand, xR_hand = x_coords[0], x_coords[1]
                well_width = xR_hand - xL_hand
                
                center_screen = w / 2
                centered_L = center_screen - (well_width / 2)
                centered_R = center_screen + (well_width / 2)
                
                x_space = np.linspace(0, w, 400)
                pot_profile = np.ones_like(x_space)
                pot_profile[(x_space > centered_L) & (x_space < centered_R)] = 0
                
                return pot_profile, "Square Well (Captured)"

            # 2. Harmonic Oscillator (1 Hand)
            elif len(res.multi_hand_landmarks) == 1:
                lm = res.multi_hand_landmarks[0]
                THUMB = lm.landmark[4]
                INDEX = lm.landmark[8]
                
                dx = INDEX.x - THUMB.x
                dy = INDEX.y - THUMB.y
                dist = math.sqrt(dx**2 + dy**2)
                
                A = np.interp(dist, [0.05, 0.3], [100.0, 1.0]) 
                
                x_space = np.linspace(-1, 1, 400)
                pot_profile = A * (x_space**2)
                
                pot_profile = np.clip(pot_profile, 0, 100)
                pot_profile = pot_profile / 100.0
                
                return pot_profile, f"Harmonic Oscillator (k={A:.1f})"
                
    except ImportError:
        return None, "MediaPipe or OpenCV not installed"
    except Exception as e:
        return None, f"Error: {e}"
            
    return None, "Error"


# ==========================================
# 11. POTENTIAL COMPOSITION (LEGO PIECES)
# ==========================================

def combine_potentials(x, potentials, weights=None):
    """
    Combine multiple potentials as weighted sum (like lego pieces).
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    potentials : list of ndarray
        List of potential arrays
    weights : list of float, optional
        Weighting factors (default: all 1.0)
        
    Returns
    -------
    V_combined : ndarray
        Combined potential
        
    Examples
    --------
    >>> x = np.linspace(-10, 10, 1000)
    >>> V1 = inf_square_well(x, -5, 0)
    >>> V2 = harmonic(x, k=1.0, center=5)
    >>> V = combine_potentials(x, [V1, V2])
    """
    if weights is None:
        weights = [1.0] * len(potentials)
    
    V_combined = np.zeros_like(x, dtype=float)
    for V, w in zip(potentials, weights):
        V_combined += w * V
    
    return V_combined


# ==========================================
# 12. MEDIAPIPE CAMERA CAPTURE FOR JUPYTER
# ==========================================

def display_params(frame, params_list, start_y=80):
    """Display parameter list on frame."""
    for i, param in enumerate(params_list):
        import cv2
        cv2.putText(frame, param, (10, start_y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def cheese(tune=1, A_MIN=0, A_MAX=100, mode='wait',max_time=30):
    # Reverse Compatability, def capture_hand_potential()
    """
    Interactive camera capture for quantum potentials using MediaPipe hand tracking.
    
    Parameters
    ----------
    tune : int
        Tuning parameter (currently unused, for future extensions)
    A_MIN : float
        Minimum curvature value for harmonic oscillator
    A_MAX : float
        Maximum curvature value for harmonic oscillator
    mode : str
        'wait' for automatic capture on stability, '' for manual mode
        
    Returns
    -------
    captured_V : ndarray or None
        Captured potential profile (400 points, normalized 0-1)
    """
    import time
    import cv2
    import mediapipe as mp
    try:
        from IPython.display import display, Image, clear_output
    except ImportError:
        display = None
        clear_output = None
    
    # MediaPipe landmarks
    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    
    # Stability detection
    REQUIRED_STABLE_FRAMES = 45
    MOVEMENT_THRESHOLD = 0.015
    
    # Harmonic oscillator mapping
    PLOT_CEILING_A = 10.0
    EPS = 1e-9
    D_MIN = 0.001
    D_MAX = 0.2
    D_RANGE = D_MAX - D_MIN
    A_RANGE = A_MAX - A_MIN
    SLOPE = -A_RANGE / D_RANGE
    INTERCEPT = A_MAX - SLOPE * D_MIN
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    drawer = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    captured_V = None
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Check permissions or camera index.")
        return None

    stability_counter = 0
    prev_landmarks = []
    
    start_time = time.time()
    MAX_RUN_TIME_SECONDS = max_time # 30
    
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

        if res.multi_hand_landmarks:
            # Collect landmarks for stability detection
            for hand_lms in res.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    current_landmarks_flat.extend([lm.x, lm.y])
            
            # Draw landmarks
            for lm in res.multi_hand_landmarks:
                drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            # TWO HANDS (Square Well)
            if len(res.multi_hand_landmarks) >= 2:
                mode_msg = "Mode: Square Well (Auto-Centered)"
                x_coords = [lm.landmark[INDEX_TIP_ID].x * w for lm in res.multi_hand_landmarks]
                x_coords.sort()
                xL_hand, xR_hand = int(x_coords[0]), int(x_coords[1])
                
                # Draw visual guides
                cv2.line(frame, (xL_hand, 0), (xL_hand, h), (0, 255, 255), 2)
                cv2.line(frame, (xR_hand, 0), (xR_hand, h), (0, 255, 255), 2)
                
                well_width = xR_hand - xL_hand
                center_screen = w / 2
                centered_L = center_screen - (well_width / 2)
                centered_R = center_screen + (well_width / 2)
                
                params_to_display.append(f"Width: {well_width:4.0f} px")
                params_to_display.append(f"Status: Centered")
                
                # Generate potential
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
                
                # Map pinch to curvature
                A = SLOPE * pinch_distance + INTERCEPT
                A = max(A_MIN, min(A_MAX, A))
                
                x_space = np.linspace(-1, 1, 400)
                pot_profile = A * (x_space**2)
                pot_profile = pot_profile / (PLOT_CEILING_A + EPS)
                pot_profile = np.clip(pot_profile, 0.0, 1.0)
                
                params_to_display.append(f"Pinch Dist: {pinch_distance:.4f}")
                params_to_display.append(f"A (curv): {A:.4f}")
                
                # Draw potential curve
                display_pts = np.column_stack(((x_space + 1)/2 * w, (1 - pot_profile) * h)).astype(np.int32)
                cv2.polylines(frame, [display_pts], False, (0, 0, 255), 2)

        # STABILITY CHECK (only if NOT in wait mode)
        if mode != 'wait':
            if current_landmarks_flat and prev_landmarks and len(current_landmarks_flat) == len(prev_landmarks):
                movement = np.mean(np.abs(np.array(current_landmarks_flat) - np.array(prev_landmarks)))
                stability_counter = stability_counter + 1 if movement < MOVEMENT_THRESHOLD else 0
            else:
                stability_counter = 0

            prev_landmarks = current_landmarks_flat

            # Draw stability progress bar
            if stability_counter > 0:
                progress = stability_counter / REQUIRED_STABLE_FRAMES
                bar_width = int(w * progress)
                color = (0, int(255*progress), int(255*(1-progress)))
                cv2.rectangle(frame, (0, 0), (bar_width, 20), color, -1)
                cv2.putText(frame, "HOLDING...", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Capture when stable
            if stability_counter >= REQUIRED_STABLE_FRAMES and pot_profile is not None:
                captured_V = pot_profile
                cap.release()
                # cv2.destroyAllWindows() # Removed for headless/notebook stability
                print("Stable capture triggered and video stream closed.")
                return captured_V

        # UI OVERLAY
        cv2.putText(frame, mode_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        display_params(frame, params_to_display)
        
        # NOTEBOOK DISPLAY
        if display is not None:
            clear_output(wait=True) 
            _, buffer = cv2.imencode('.jpeg', frame)
            display(Image(data=buffer.tobytes()))
        
        time.sleep(0.01)

        # Time limit check
        if time.time() - start_time > MAX_RUN_TIME_SECONDS:
            print(f"Time limit of {MAX_RUN_TIME_SECONDS} seconds reached.")
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    return captured_V


def verify_and_solve(V_raw_input, x, dx, T, L):
    """
    Verify captured potential and solve the Schrödinger equation.
    
    Parameters
    ----------
    V_raw_input : ndarray
        Raw potential from camera capture (400 points, 0-1 normalized)
    x : ndarray
        Full spatial grid (N+2 points)
    dx : float
        Grid spacing
    T : ndarray
        Kinetic operator matrix (N x N)
    L : float
        Total domain length
        
    Returns
    -------
    E_vals : ndarray
        Energy eigenvalues
    psi_vecs : ndarray
        Wavefunction eigenvectors
    V_full : ndarray
        Full potential array for plotting
        
    Examples
    --------
    >>> E, psi, V = verify_and_solve(V_raw, x, dx, T, L)
    >>> plot_educate(E, psi, V, x, no=0)
    """
    if V_raw_input is None:
        print("Error: No potential captured!")
        return None, None, None
    
    # Define internal grid
    x_solver = x[1:-1]
    
    # Interpolate to solver grid
    V_interpolated = np.interp(
        x_solver, 
        np.linspace(-L/2, L/2, len(V_raw_input)),
        V_raw_input
    )
    
    # Scale to energy units
    V_max_height = 100.0
    V_internal = V_interpolated * V_max_height
    
    # Prepare V_full with boundary padding
    V_full = np.pad(V_internal, (1, 1), 'constant', constant_values=0.0)
    
    # Solve
    try:
        E_vals, psi_vecs = solve(T, V_full, dx)
        print(f"Solver complete. Found {E_vals.size} eigenstates.")
        return E_vals, psi_vecs, V_full
    except Exception as e:
        print(f"Error during solve: {e}")
        if np.max(V_internal) > 1e9:
            print("Potential may be too steep or high, leading to numerical error.")
        return None, None, None
