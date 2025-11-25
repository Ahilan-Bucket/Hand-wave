"""
Camera capture functions for Hand-Wave quantum solver.
This module provides functions to capture hand gestures via webcam
and convert them into quantum potentials.
"""

import time
import cv2
import numpy as np
import math
try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("Warning: MediaPipe not available. Camera capture will not work.")

try:
    from IPython.display import display, Image, clear_output
except ImportError:
    display = None
    clear_output = None
    print("Warning: IPython display not available. Running in non-notebook mode.")


# Constants for hand detection
THUMB_TIP_ID = 4
INDEX_TIP_ID = 8
REQUIRED_STABLE_FRAMES = 45
MOVEMENT_THRESHOLD = 0.015
PLOT_CEILING_A = 10.0
EPS = 1e-9

# Pinch distance to potential strength mapping
D_MIN = 0.001
D_MAX = 0.2
D_RANGE = D_MAX - D_MIN


def display_params(frame, params_list):
    """
    Display parameter text on video frame.
    
    Parameters
    ----------
    frame : ndarray
        Video frame to draw on
    params_list : list of str
        List of parameter strings to display
    """
    y_offset = 80
    for param in params_list:
        cv2.putText(frame, param, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25


def capture_hand_potential(A_MIN=0, A_MAX=100, mode='capture', max_time=30):
    """
    Capture hand gestures and convert to quantum potential.
    
    Uses MediaPipe to detect hands and create potential profiles:
    - ONE HAND (pinch): Quantum Harmonic Oscillator (QHO)
    - TWO HANDS (spread): Square Well
    
    Parameters
    ----------
    A_MIN : float
        Minimum potential strength
    A_MAX : float
        Maximum potential strength  
    mode : str
        'capture' to require stability, 'wait' to skip
    max_time : int
        Maximum capture time in seconds
        
    Returns
    -------
    V_profile : ndarray or None
        Captured potential profile (normalized 0-1)
        Returns None if capture failed or timed out
        
    Examples
    --------
    >>> V = capture_hand_potential(A_MIN=0, A_MAX=100)
    >>> if V is not None:
    ...     plot_V(V)
    """
    if mp is None:
        print("Error: MediaPipe not installed. Cannot capture from camera.")
        return None
        
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    drawer = mp.solutions.drawing_utils
    
    # Potential mapping parameters
    A_RANGE = A_MAX - A_MIN
    SLOPE = -A_RANGE / D_RANGE
    INTERCEPT = A_MAX - SLOPE * D_MIN
    
    cap = cv2.VideoCapture(0)
    captured_V = None
    
    if not cap.isOpened():
        print("Error: Could not open video stream. Check camera permissions.")
        return None
    
    stability_counter = 0
    prev_landmarks = []
    start_time = time.time()
    
    print("Controls: HOLD STILL to capture, or wait for timeout.")
    print(f"ONE HAND (pinch) = QHO, TWO HANDS (spread) = Square Well")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        pot_profile = None
        mode_msg = "No Hands Detected"
        params_to_display = []
        current_landmarks_flat = []
        
        # Process hand landmarks
        if res.multi_hand_landmarks:
            # Flatten landmarks for stability check
            for hand_lms in res.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    current_landmarks_flat.extend([lm.x, lm.y])
                    
            # Draw hand landmarks
            for lm in res.multi_hand_landmarks:
                drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            # TWO HANDS: Square Well
            if len(res.multi_hand_landmarks) >= 2:
                mode_msg = "Mode: Square Well (Auto-Centered)"
                x_coords = [lm.landmark[INDEX_TIP_ID].x * w 
                           for lm in res.multi_hand_landmarks]
                x_coords.sort()
                xL_hand, xR_hand = int(x_coords[0]), int(x_coords[1])
                
                # Draw well boundaries
                cv2.line(frame, (xL_hand, 0), (xL_hand, h), (0, 255, 255), 2)
                cv2.line(frame, (xR_hand, 0), (xR_hand, h), (0, 255, 255), 2)
                
                well_width = xR_hand - xL_hand
                center_screen = w / 2
                centered_L = center_screen - (well_width / 2)
                centered_R = center_screen + (well_width / 2)
                
                params_to_display.append(f"Width: {well_width:4.0f} px")
                params_to_display.append(f"Status: Centered")
                
                # Create square well profile
                x_space = np.linspace(0, w, 400)
                pot_profile = np.ones_like(x_space)
                pot_profile[(x_space > centered_L) & (x_space < centered_R)] = 0
                
            # ONE HAND: Quantum Harmonic Oscillator
            elif len(res.multi_hand_landmarks) == 1:
                mode_msg = "Mode: Pinch QHO"
                lm = res.multi_hand_landmarks[0]
                thumb = lm.landmark[THUMB_TIP_ID]
                index = lm.landmark[INDEX_TIP_ID]
                
                # Calculate pinch distance
                dx = index.x - thumb.x
                dy = index.y - thumb.y
                pinch_distance = math.sqrt(dx**2 + dy**2)
                
                # Map to potential strength
                A = SLOPE * pinch_distance + INTERCEPT
                A = max(A_MIN, min(A_MAX, A))
                
                # Create parabolic profile
                x_space = np.linspace(-1, 1, 400)
                pot_profile = A * (x_space**2)
                pot_profile = pot_profile / (PLOT_CEILING_A + EPS)
                pot_profile = np.clip(pot_profile, 0.0, 1.0)
                
                params_to_display.append(f"Pinch Dist: {pinch_distance:.4f}")
                params_to_display.append(f"A (curv): {A:.4f}")
                
                # Draw potential curve on frame
                display_pts = np.column_stack((
                    (x_space + 1)/2 * w, 
                    (1 - pot_profile) * h
                )).astype(np.int32)
                cv2.polylines(frame, [display_pts], False, (0, 0, 255), 2)
        
        # Stability check (only if mode == 'capture')
        if mode == 'capture':
            if (current_landmarks_flat and prev_landmarks and 
                len(current_landmarks_flat) == len(prev_landmarks)):
                movement = np.mean(np.abs(
                    np.array(current_landmarks_flat) - np.array(prev_landmarks)
                ))
                if movement < MOVEMENT_THRESHOLD:
                    stability_counter += 1
                else:
                    stability_counter = 0
            else:
                stability_counter = 0
                
            prev_landmarks = current_landmarks_flat
            
            # Draw stability progress bar
            if stability_counter > 0:
                progress = stability_counter / REQUIRED_STABLE_FRAMES
                bar_width = int(w * progress)
                color = (0, int(255*progress), int(255*(1-progress)))
                cv2.rectangle(frame, (0, 0), (bar_width, 20), color, -1)
                cv2.putText(frame, "HOLDING...", (10, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Capture complete
            if stability_counter >= REQUIRED_STABLE_FRAMES and pot_profile is not None:
                captured_V = pot_profile
                cap.release()
                cv2.destroyAllWindows()
                print("Stable capture triggered. Video stream closed.")
                return captured_V
        
        # Display UI
        cv2.putText(frame, mode_msg, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        display_params(frame, params_to_display)
        
        # Show frame (notebook or window)
        if display is not None and clear_output is not None:
            # Jupyter notebook display
            clear_output(wait=True)
            _, buffer = cv2.imencode('.jpeg', frame)
            display(Image(data=buffer.tobytes()))
        else:
            # Standard OpenCV window
            cv2.imshow('Hand-Wave Capture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        time.sleep(0.01)
        
        # Timeout check
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_V


def verify_and_plot_capture(V_raw_input, x, dx, T, L=50):
    """
    Verify captured potential and solve quantum system.
    
    Takes raw camera input, interpolates to solver grid,
    solves the Schrödinger equation, and returns results.
    
    Parameters
    ----------
    V_raw_input : ndarray
        Raw potential from camera capture
    x : ndarray
        Full spatial grid
    dx : float
        Grid spacing
    T : ndarray
        Kinetic energy operator
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
    >>> V_raw = capture_hand_potential()
    >>> E, psi, V = verify_and_plot_capture(V_raw, x, dx, T)
    >>> plot_educational(E, psi, V, x, no=0)
    """
    # Import solve function (avoid circular import)
    from functions import solve
    
    if V_raw_input is None:
        print("Error: No potential captured!")
        return None, None, None
    
    print("Potential capture successful.")
    
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
            print("Potential may be too steep or high.")
        return None, None, None
