import numpy as np
from functions import analyze_state

def verify_fix():
    print("Verifying analyze_state fix...")
    
    # Setup dummy data
    N = 10
    x = np.linspace(0, 1, N+2)
    V = np.zeros(N+2)
    V[0] = 100
    V[-1] = 100
    
    E_n = 0.5
    psi_n = np.ones(N) # Size N (internal)
    
    try:
        feedback = analyze_state(E_n, psi_n, V, x)
        print("Success! analyze_state ran without IndexError.")
        for msg in feedback:
            print(f"- {msg}")
    except IndexError as e:
        print(f"FAILED with IndexError: {e}")
    except Exception as e:
        print(f"FAILED with {type(e).__name__}: {e}")

if __name__ == "__main__":
    verify_fix()
