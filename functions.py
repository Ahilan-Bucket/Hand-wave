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
    Returns:
        x_full: N+2 points from -L/2 to L/2 (including 'walls')
        dx: grid spacing
        x_internal: internal N points (where we solve)
    """
    x = np.linspace(-L/2, L/2, N+2)
    dx = x[1] - x[0]
    x_internal = x[1:-1]
    return x, dx, x_internal

# ==========================================
# 3. POTENTIAL GENERATORS (V(x))
# ==========================================
def constant(x,c):
    """Adds a flat baseline potential."""
    return np.ones_like(x)*c

def harmonic(x,k,center=0.0):
    """A Parabola, setting the global k-value."""
    global Last_k_value
    Last_k_value = k
    
    constant_factor = 1 
    potential = 0.5*k*(x - center)**2
    return constant_factor * potential

def gaussian_well(x, center=0.0, width=1.0, depth=50): 
    """A dip in the potential (finite well)."""
    return -depth * np.exp(-(x - center)**2 / (2 * width**2))

def inf_sqaure_well(x,lower_bound,upper_bound):
    """Gives you an Infinite square well of Length L"""
    HUGE_NUMBER = 1e10
    V = np.zeros_like(x) 
    V[x<lower_bound] = HUGE_NUMBER
    V[x>upper_bound] = HUGE_NUMBER
    return V

def inf_wall(x,side,bound):
    """Places an 'Infinite' wall (Penalty Method)."""
    V = np.zeros_like(x)
    HUGE_NUMBER = 9e10 
    side = side.strip(', . ').lower() 

    if side =='left':
        V[x<bound] = HUGE_NUMBER
    elif side =='right':
        V[x>bound] = HUGE_NUMBER
    return V

def finite_barrier(x, center, width, height):
    """A square block in the middle."""
    V = np.zeros_like(x)
    mask = (x > (center - width/2)) & (x < (center + width/2))
    V[mask] = height
    return V

def V_double_well(x, depth=20, separation=1,center=0.0):
    """Quartic double well potential."""
    V = depth * ( (x-center)**2 - separation )**2
    return V

def custom2(value,x):
    """Helper function from the notebook."""
    return value * np.ones_like(x)

# In psi_solve2/functions.py

def finite_square_well(x, lower_bound, upper_bound, depth_V):
    """A finite square well of a specific depth_V (height of the walls)."""
    
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
def kinetic_operator(N, dx, hbar=hbar,m=m):
    """Builds the Kinetic Energy Matrix (N x N) using finite difference."""
    main_diagonal = (1/dx**2)*np.diag(-2*np.ones(N))
    off_diagonal1 = (1/dx**2)*np.diag(np.ones(N-1),-1)
    off_diagonal2 = (1/dx**2)*np.diag(np.ones(N-1),1)
    D2 = (main_diagonal + off_diagonal1 + off_diagonal2)

    T = (-(hbar**2/(2*m) ) * D2 )
    return T

def solve(T,V_full,dx):
    """Solves H psi = E psi (Eigenvalue problem)."""
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
    """Returns a figure showing a 1D potential profile."""
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


def plot_alive(E, psi, V, x, no = 1,nos=5,mode=''):
    """
    Physically accurate plot:
    - |psi|^2 has its own scale on right y-axis
    - Potential & energy use left y-axis
    
    UPDATED: Colors synchronized between probability density and energy line.
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

def check_ISW_analytic(E,L,hbar=1.0, m=1.0, max_levels=6):
    """Compares numerical energies to the Infinite Square Well analytic formula."""
    CHECK_N = max_levels
    E_numerical = E[:CHECK_N]
    E_analytic = np.zeros(CHECK_N) # Changed to CHECK_N size for correct indexing

    for i in range(CHECK_N):
        n = i + 1 
        E_analytic[i] = (hbar**2 * np.pi**2 * n**2 ) / (2*m*L**2)

    print("\n### ENERGY BENCHMARK: Infinite Square Well ###")
    print("-" * 55)
    print(f"| n | Analytic E | Numerical E | % Error |")
    print("-" * 55)

    for i in range(CHECK_N):
        percent_error = np.abs((E_numerical[i] - E_analytic[i]) / E_analytic[i]) * 100
        print(
            f"| {i+1:<1} | {E_analytic[i]:<10.6f} | {E_numerical[i]:<11.6f} | {percent_error:<7.4f}% |"
        )
    print("-" * 55)

def check_harmonic_analytic(E, hbar=1.0, m=1.0, max_levels=6):
    """Compares numerical energies to the Harmonic Oscillator analytic formula."""
    CHECK_N = max_levels
    try: 
        k = Last_k_value 
        if k is None:
            print("ERROR: k is not set. Run a harmonic potential first.")
            return
        
        w = np.sqrt(k/m)
        E_numerical = E[:CHECK_N]
        E_analytic = np.zeros(CHECK_N)

        for i in range(CHECK_N):
            n_quantum = i 
            E_analytic[i] = (n_quantum + 0.5 ) * hbar * w

        print("\n### ENERGY BENCHMARK: Harmonic Oscillator ###")
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

    except Exception as e:
        print(f"I don't think this is a Harmonic Oscillator, or an error occurred: {e}")






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

