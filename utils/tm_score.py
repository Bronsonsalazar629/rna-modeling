# utils/tm_score.py
import numpy as np

def kabsch(P, Q):
    """
    Robust Kabsch algorithm with proper reflection handling.

    Args:
        P: (L, 3) predicted coordinates
        Q: (L, 3) true coordinates

    Returns:
        P_aligned: (L, 3) P optimally aligned to Q
    """
    # Center both structures at origin
    P_center = P - P.mean(axis=0)
    Q_center = Q - Q.mean(axis=0)

    # Compute covariance matrix
    C = np.dot(P_center.T, Q_center)

    # Singular Value Decomposition
    V, _, Wt = np.linalg.svd(C)

    # Handle reflection properly
    d = np.sign(np.linalg.det(Wt.T @ V.T))
    if d < 0:
        Wt[-1, :] *= -1

    # Optimal rotation matrix
    U = V @ Wt

    # Apply rotation + translation to P
    P_aligned = (P_center @ U) + Q.mean(axis=0)

    return P_aligned

def tm_score(pred, true, debug=False):
    """
    Compute TM-score between predicted and true RNA structures.

    TM-score ranges from 0 (no similarity) to 1 (identical folds).
    Uses length-normalized distance threshold per Zhang & Skolnick (2005).

    Args:
        pred: (L, 3) predicted C1' coordinates
        true: (L, 3) true C1' coordinates
        debug: If True, print diagnostic information

    Returns:
        float: TM-score (0.0 to 1.0)
    """
    L = len(pred)
    if L < 5:
        return 0.0

    # Length-dependent distance scale (Zhang & Skolnick, 2005)
    d0 = 1.24 * (L - 15) ** (1/3) - 1.8
    if d0 <= 0:
        d0 = 1.0

    if debug:
        pred_centered = pred - pred.mean(axis=0)
        true_centered = true - true.mean(axis=0)
        print(f"DEBUG: L={L}, d0={d0:.3f}")
        print(f"DEBUG: Pred RMSD from center = {np.sqrt(np.mean(np.sum(pred_centered**2, axis=1))):.2f}")
        print(f"DEBUG: True RMSD from center = {np.sqrt(np.mean(np.sum(true_centered**2, axis=1))):.2f}")

    # Optimal superposition
    pred_aligned = kabsch(pred, true)

    # Per-residue distances after alignment
    d = np.sqrt(np.sum((pred_aligned - true) ** 2, axis=1))

    if debug:
        print(f"DEBUG: Mean distance after alignment = {d.mean():.2f}")
        print(f"DEBUG: Min/Max distance = {d.min():.2f}/{d.max():.2f}")

    # TM-score formula
    tm = np.mean(1 / (1 + (d / d0) ** 2))

    return float(tm)
