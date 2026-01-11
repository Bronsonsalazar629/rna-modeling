"""
Loss functions for RNA structure prediction training.
Handles multiple experimental conformations and physics-based regularization.
"""

import jax
import jax.numpy as jnp
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from physics.energy import rna_energy


def build_local_frames(coords: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Build local coordinate frames from backbone C1' positions.

    Args:
        coords: (L, 3) C1' coordinates
        epsilon: Small constant for numerical stability

    Returns:
        frames: (L, 3, 3) rotation matrices for local frames
    """
    L = coords.shape[0]

    # Build x-axis from neighboring residues
    # x-axis points along backbone direction
    # For interior residues: use i-1 to i+1
    # For terminal residues: use available neighbors

    x_axes = jnp.zeros((L, 3))

    # First residue: 0 to 1
    x_axes = x_axes.at[0].set(coords[1] - coords[0])

    # Interior residues: average of forward and backward
    if L > 2:
        interior = (coords[2:] - coords[:-2]) / 2.0  # (L-2, 3)
        x_axes = x_axes.at[1:-1].set(interior)

    # Last residue: L-2 to L-1
    x_axes = x_axes.at[-1].set(coords[-1] - coords[-2])

    # Normalize x-axes
    x_norms = jnp.linalg.norm(x_axes, axis=1, keepdims=True) + epsilon
    x_axes = x_axes / x_norms

    # Build y-axis orthogonal to x using Gram-Schmidt
    up = jnp.array([0.0, 0.0, 1.0])
    up_broadcast = jnp.tile(up[None, :], (L, 1))

    dot_product = jnp.sum(up_broadcast * x_axes, axis=1, keepdims=True)
    y_axes = up_broadcast - dot_product * x_axes
    y_norms = jnp.linalg.norm(y_axes, axis=1, keepdims=True) + epsilon
    y_axes = y_axes / y_norms

    # z-axis = x × y
    z_axes = jnp.cross(x_axes, y_axes)

    # Stack into rotation matrices (L, 3, 3)
    frames = jnp.stack([x_axes, y_axes, z_axes], axis=-1)

    return frames


def fape_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    clamp_distance: float = 10.0,
    epsilon: float = 1e-8,
    use_local_frames: bool = True
) -> jnp.ndarray:
    """
    Frame Aligned Point Error (FAPE) loss from AlphaFold2.

    Args:
        pred_coords: Predicted coordinates (L, 3)
        true_coords: True coordinates (L, 3)
        clamp_distance: Maximum distance for clamping
        epsilon: Small constant for numerical stability
        use_local_frames: If True, use local residue frames (proper FAPE)

    Returns:
        Scalar FAPE loss
    """
    L = pred_coords.shape[0]

    if use_local_frames and L > 2:
        # Build local frames from ground truth
        true_frames = build_local_frames(true_coords, epsilon)

        # For each frame i, transform ALL coordinates into that frame
        errors_list = []
        for i in range(L):
            # Translate to residue i origin (use GROUND TRUTH origin)
            pred_translated = pred_coords - true_coords[i:i+1, :]
            true_translated = true_coords - true_coords[i:i+1, :]

            # Rotate into local frame i
            R = true_frames[i]  # (3, 3) rotation matrix
            pred_in_frame = pred_translated @ R  # (L, 3)
            true_in_frame = true_translated @ R  # (L, 3)

            # Compute error for all residues in this frame
            frame_error = jnp.sqrt(jnp.sum((pred_in_frame - true_in_frame) ** 2, axis=-1) + epsilon)  # (L,)
            errors_list.append(frame_error)

        # Stack all frame errors: (L frames, L residues)
        all_errors = jnp.stack(errors_list, axis=0)  # (L, L)
        clamped_errors = jnp.minimum(all_errors, clamp_distance)

        # Mean over all frame-residue pairs
        fape = jnp.mean(clamped_errors)

    else:
        # Fallback: pairwise distance-based loss
        pred_dists = pred_coords[:, None, :] - pred_coords[None, :, :]
        true_dists = true_coords[:, None, :] - true_coords[None, :, :]
        diff = pred_dists - true_dists
        dist_error = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + epsilon)
        dist_error_clamped = jnp.minimum(dist_error, clamp_distance)
        fape = jnp.mean(dist_error_clamped)

    return fape


def rmsd_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    align: bool = True
) -> jnp.ndarray:
    """
    Root Mean Square Deviation loss.

    Args:
        pred_coords: Predicted coordinates (L, 3)
        true_coords: True coordinates (L, 3)
        align: Whether to align structures first (recommended)

    Returns:
        Scalar RMSD
    """
    if align:
        # Center both structures
        pred_centered = pred_coords - jnp.mean(pred_coords, axis=0, keepdims=True)
        true_centered = true_coords - jnp.mean(true_coords, axis=0, keepdims=True)

        # Kabsch alignment (simplified - full version would compute optimal rotation)
        # For now, just use centered coordinates
        pred_aligned = pred_centered
        true_aligned = true_centered
    else:
        pred_aligned = pred_coords
        true_aligned = true_coords

    # RMSD
    diff = pred_aligned - true_aligned
    squared_dist = jnp.sum(diff ** 2, axis=-1)
    rmsd = jnp.sqrt(jnp.mean(squared_dist))

    return rmsd


def distogram_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    num_bins: int = 64,
    max_distance: float = 50.0
) -> jnp.ndarray:
    """
    Loss on predicted distance distributions.

    Args:
        pred_coords: Predicted coordinates (L, 3)
        true_coords: True coordinates (L, 3)
        num_bins: Number of distance bins
        max_distance: Maximum distance in Angstroms

    Returns:
        Scalar distogram loss
    """
    # Compute distance matrices
    pred_dists = jnp.sqrt(jnp.sum(
        (pred_coords[:, None, :] - pred_coords[None, :, :]) ** 2,
        axis=-1
    ))
    true_dists = jnp.sqrt(jnp.sum(
        (true_coords[:, None, :] - true_coords[None, :, :]) ** 2,
        axis=-1
    ))

    # Bin edges
    bin_edges = jnp.linspace(0, max_distance, num_bins + 1)

    # Create soft histograms using differentiable binning
    def soft_histogram(dists):
        # Gaussian kernel around each bin center
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        sigma = (max_distance / num_bins) / 2

        # Compute probabilities for each distance falling in each bin
        probs = jax.nn.softmax(
            -((dists[..., None] - bin_centers) ** 2) / (2 * sigma ** 2),
            axis=-1
        )
        return probs

    pred_hist = soft_histogram(pred_dists)
    true_hist = soft_histogram(true_dists)

    # Cross-entropy loss
    loss = -jnp.sum(true_hist * jnp.log(pred_hist + 1e-8)) / (pred_dists.size)

    return loss


def torsion_angle_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray
) -> jnp.ndarray:
    """
    Loss on backbone torsion angles (pseudo-torsions from C1' atoms).

    Args:
        pred_coords: Predicted coordinates (L, 3)
        true_coords: True coordinates (L, 3)

    Returns:
        Scalar torsion loss
    """
    L = pred_coords.shape[0]

    if L < 4:
        return jnp.array(0.0)

    def compute_torsions(coords):
        """Compute torsion angles from coordinates."""
        torsions = []
        for i in range(L - 3):
            p1, p2, p3, p4 = coords[i:i+4]

            # Vectors
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3

            # Normal vectors
            n1 = jnp.cross(b1, b2)
            n2 = jnp.cross(b2, b3)

            # Normalize
            n1 = n1 / (jnp.linalg.norm(n1) + 1e-8)
            n2 = n2 / (jnp.linalg.norm(n2) + 1e-8)

            # Angle
            m1 = jnp.cross(n1, b2 / (jnp.linalg.norm(b2) + 1e-8))
            x = jnp.dot(n1, n2)
            y = jnp.dot(m1, n2)

            torsion = jnp.arctan2(y, x)
            torsions.append(torsion)

        return jnp.stack(torsions)

    pred_torsions = compute_torsions(pred_coords)
    true_torsions = compute_torsions(true_coords)

    # Circular distance (torsions wrap at ±π)
    angular_diff = pred_torsions - true_torsions
    angular_diff = jnp.arctan2(jnp.sin(angular_diff), jnp.cos(angular_diff))

    # Mean squared angular error
    loss = jnp.mean(angular_diff ** 2)

    return loss


def physics_regularization(
    coords: jnp.ndarray,
    sequence: Optional[str] = None,
    energy_threshold: float = 100.0
) -> jnp.ndarray:
    """
    Physics-based regularization from energy function.

    Args:
        coords: Predicted coordinates (L, 3)
        sequence: RNA sequence
        energy_threshold: Energy threshold (penalize above this)

    Returns:
        Scalar regularization term
    """
    energy_dict = rna_energy(coords, sequence)
    total_energy = energy_dict['total']

    # Only penalize if energy is too high
    penalty = jnp.maximum(total_energy - energy_threshold, 0.0)

    return penalty


def multi_structure_loss(
    pred_coords: jnp.ndarray,
    true_coords_list: List[jnp.ndarray],
    sequence: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, jnp.ndarray]:
    """
    Compute loss against multiple experimental conformations.

    Args:
        pred_coords: Predicted coordinates (L, 3)
        true_coords_list: List of ground truth coordinate arrays
        sequence: RNA sequence
        weights: Optional dict to override loss weights

    Returns:
        Dictionary with total loss and components
    """
    # Default weights
    default_weights = {
        'fape': 1.0,
        'rmsd': 0.5,
        'distogram': 0.5,
        'torsion': 0.3,
        'physics': 0.1,
    }

    if weights is not None:
        default_weights.update(weights)

    # Compute loss against each ground truth
    fape_losses = []
    rmsd_losses = []
    distogram_losses = []
    torsion_losses = []

    for true_coords in true_coords_list:
        fape_losses.append(fape_loss(pred_coords, true_coords))
        rmsd_losses.append(rmsd_loss(pred_coords, true_coords))
        distogram_losses.append(distogram_loss(pred_coords, true_coords))
        torsion_losses.append(torsion_angle_loss(pred_coords, true_coords))

    # Take minimum loss (or soft minimum) across conformations
    # This allows model to match any of the experimental structures
    fape_losses = jnp.stack(fape_losses)
    rmsd_losses = jnp.stack(rmsd_losses)
    distogram_losses = jnp.stack(distogram_losses)
    torsion_losses = jnp.stack(torsion_losses)

    # Soft minimum (differentiable)
    temperature = 0.1
    def soft_min(x):
        weights = jax.nn.softmax(-x / temperature)
        return jnp.sum(x * weights)

    best_fape = soft_min(fape_losses)
    best_rmsd = soft_min(rmsd_losses)
    best_distogram = soft_min(distogram_losses)
    best_torsion = soft_min(torsion_losses)

    # Physics regularization
    physics_reg = physics_regularization(pred_coords, sequence)

    # Weighted sum
    total_loss = (
        default_weights['fape'] * best_fape +
        default_weights['rmsd'] * best_rmsd +
        default_weights['distogram'] * best_distogram +
        default_weights['torsion'] * best_torsion +
        default_weights['physics'] * physics_reg
    )

    return {
        'total': total_loss,
        'fape': best_fape,
        'rmsd': best_rmsd,
        'distogram': best_distogram,
        'torsion': best_torsion,
        'physics': physics_reg,
    }


def diversity_loss(predictions: List[jnp.ndarray], margin: float = 5.0) -> jnp.ndarray:
    """
    Encourage diversity among ensemble predictions (repulsion loss).

    Args:
        predictions: List of predicted coordinate arrays (L, 3)
        margin: Minimum desired RMSD between predictions

    Returns:
        Scalar diversity loss (negative reward for diversity)
    """
    if len(predictions) < 2:
        return jnp.array(0.0)

    # Compute pairwise RMSDs
    diversity_losses = []

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            rmsd_ij = rmsd_loss(predictions[i], predictions[j], align=True)

            # Penalize if RMSD is below margin
            penalty = jnp.maximum(margin - rmsd_ij, 0.0)
            diversity_losses.append(penalty)

    return jnp.mean(jnp.stack(diversity_losses))


@jax.jit
def compute_loss_jitted(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    sequence_onehot: jnp.ndarray
) -> jnp.ndarray:
    """
    JIT-compiled loss function for training.

    Args:
        pred_coords: Predicted coordinates
        true_coords: True coordinates
        sequence_onehot: One-hot encoded sequence

    Returns:
        Scalar loss
    """
    # Simple FAPE loss for efficiency
    return fape_loss(pred_coords, true_coords)


if __name__ == "__main__":
    print("="* 70)
    print("Loss Functions for RNA Structure Prediction")
    print("="* 70)

    # Test with dummy data
    L = 20
    rng = jax.random.PRNGKey(42)

    pred_coords = jax.random.normal(rng, (L, 3)) * 10
    true_coords = jax.random.normal(rng, (L, 3)) * 10
    true_coords_2 = true_coords + jax.random.normal(rng, (L, 3)) * 2

    print("\nTest structures:")
    print(f"  Length: {L} residues")
    print(f"  Num conformations: 2")

    # Test individual losses
    print("\nIndividual losses:")
    print(f"  FAPE: {fape_loss(pred_coords, true_coords):.4f}")
    print(f"  RMSD: {rmsd_loss(pred_coords, true_coords):.4f}")
    print(f"  Distogram: {distogram_loss(pred_coords, true_coords):.4f}")
    print(f"  Torsion: {torsion_angle_loss(pred_coords, true_coords):.4f}")
    print(f"  Physics: {physics_regularization(pred_coords):.4f}")

    # Test multi-structure loss
    print("\nMulti-structure loss:")
    loss_dict = multi_structure_loss(
        pred_coords,
        [true_coords, true_coords_2],
        sequence='A' * L
    )

    for key, value in loss_dict.items():
        print(f"  {key}: {float(value):.4f}")

    # Test diversity loss
    print("\nDiversity loss:")
    pred_ensemble = [pred_coords, pred_coords + jax.random.normal(rng, (L, 3))]
    div_loss = diversity_loss(pred_ensemble, margin=5.0)
    print(f"  Diversity: {float(div_loss):.4f}")

    # Test JIT compilation
    print("\nTesting JIT compilation...")
    sequence_onehot = jax.random.normal(rng, (L, 5))

    # Compile
    loss_jit = compute_loss_jitted(pred_coords, true_coords, sequence_onehot)
    print(f"  JIT loss: {float(loss_jit):.4f}")

    # Test gradient flow
    print("\nTesting gradient flow...")
    grad_fn = jax.grad(lambda p: compute_loss_jitted(p, true_coords, sequence_onehot))
    grads = grad_fn(pred_coords)
    print(f"  Gradient norm: {jnp.linalg.norm(grads):.6f}")
    print(f"  Gradient shape: {grads.shape}")

    print("\n✓ All loss functions ready")
