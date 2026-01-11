# test_tm_score.py
import numpy as np
from utils.tm_score import tm_score

print("="*60)
print("TM-Score Implementation Validation")
print("="*60)

# Test 1: Identical structures → TM ≈ 1.0
np.random.seed(42)
coords = np.random.rand(20, 3) * 50.0  # Scale to realistic Angstrom range
tm_identical = tm_score(coords, coords)
print(f"\nTest 1: Identical structures")
print(f"  TM-score: {tm_identical:.4f}")
print(f"  Expected: ~1.0000")
print(f"  Status: {'PASS' if tm_identical > 0.99 else 'FAIL'}")

# Test 2: Slight perturbation → TM ≈ 0.7–0.9
perturbed = coords + np.random.normal(0, 0.5, coords.shape)
tm_perturbed = tm_score(perturbed, coords)
print(f"\nTest 2: Perturbed structures (0.5 Å noise)")
print(f"  TM-score: {tm_perturbed:.4f}")
print(f"  Expected: 0.70–0.95")
print(f"  Status: {'PASS' if 0.7 <= tm_perturbed <= 0.95 else 'FAIL'}")

# Test 3: Random structures → TM ≈ 0.1–0.2
random = np.random.rand(20, 3) * 50.0
tm_random = tm_score(random, coords)
print(f"\nTest 3: Random structures")
print(f"  TM-score: {tm_random:.4f}")
print(f"  Expected: 0.05–0.30")
print(f"  Status: {'PASS' if 0.05 <= tm_random <= 0.30 else 'FAIL'}")

# Test 4: Rotated structures (should align perfectly) → TM ≈ 1.0
angle = np.pi / 4
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle), np.cos(angle), 0],
    [0, 0, 1]
])
rotated = coords @ rotation_matrix.T + np.array([10, 20, 30])  # Rotate + translate
tm_rotated = tm_score(rotated, coords)
print(f"\nTest 4: Rotated + translated structures")
print(f"  TM-score: {tm_rotated:.4f}")
print(f"  Expected: ~1.0000 (Kabsch should align perfectly)")
print(f"  Status: {'PASS' if tm_rotated > 0.99 else 'FAIL'}")

# Summary
print(f"\n{'='*60}")
all_pass = (
    tm_identical > 0.99 and
    0.7 <= tm_perturbed <= 0.95 and
    0.05 <= tm_random <= 0.30 and
    tm_rotated > 0.99
)
if all_pass:
    print("All tests PASSED")
    print("TM-score implementation is correct")
else:
    print("Some tests FAILED")
    print("Check implementation")
print("="*60)
