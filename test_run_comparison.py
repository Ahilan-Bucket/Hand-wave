"""
Quick test of run_comparison function
"""

from functions import run_comparison

print("Testing run_comparison()...")
print("="*70)

try:
    run_comparison()
    print("\n✓ Function executed successfully!")
    print("\nCheck 'comparison_log.txt' for results")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
