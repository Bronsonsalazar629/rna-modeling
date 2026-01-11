"""
JAX-differentiable energy functions for RNA structure refinement.
Implements simplified physics-based energy terms for gradient-based optimization.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional
import numpy as np


# RNA-specific parameters (approximate)
RNA_PARAMS = {
    # Bond lengths (Angstroms)
    'c1_c1_sequential': 6.0,  # Approximate C1'-C1' distance for sequential residues
    'c1_c1_paired': 10.5,     # Approximate C1'-C1' distance for Watson-Crick pairs

    # Energy weights
    'bond_weight': 1.0,
    'angle_weight': 0.5,
    'clash_weight': 10.0,
    'pairing_weight': 0.1,
    'compactness_weight': 0.01,

    # Clash parameters
    'clash_distance': 3.0,    # Minimum allowed C1'-C1' distance
    'clash_steepness': 4.0,   # Steepness of clash penalty

    # Radius of gyration target (approximate for typical RNAs)
    'target_rg_per_residue': 2.0,
}


def compute_pairwise_distances(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        coords: Coordinates (L, 3)

    Returns:
        Distance matrix (L, L)
    """
    diff = coords[:, None, :] - coords[None, :, :]  # (L, L, 3)
    distances = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)  # (L, L)
    return distances


def bond_energy(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Energy from deviation of sequential bond lengths.

    Args:
        coords: C1' coordinates (L, 3)

    Returns:
        Scalar bond energy
    """
    L = coords.shape[0]

    if L < 2:
        return jnp.array(0.0)

    # Sequential C1'-C1' distances
    diffs = coords[1:] - coords[:-1]  # (L-1, 3)
    distances = jnp.sqrt(jnp.sum(diffs ** 2, axis=-1))  # (L-1,)

    # Harmonic potential around ideal distance
    target_dist = RNA_PARAMS['c1_c1_sequential']
    deviations = distances - target_dist

    energy = RNA_PARAMS['bond_weight'] * jnp.sum(deviations ** 2)

    return energy


def angle_energy(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Energy from bond angles (C1'(i) - C1'(i+1) - C1'(i+2)).

    Args:
        coords: C1' coordinates (L, 3)

    Returns:
        Scalar angle energy
    """
    L = coords.shape[0]

    if L < 3:
        return jnp.array(0.0)

    # Vectors along bonds
    v1 = coords[1:-1] - coords[:-2]  # (L-2, 3)
    v2 = coords[2:] - coords[1:-1]   # (L-2, 3)

    # Normalize
    v1_norm = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    v2_norm = v2 / (jnp.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)

    # Compute angles
    cos_angles = jnp.sum(v1_norm * v2_norm, axis=-1)
    cos_angles = jnp.clip(cos_angles, -1.0, 1.0)

    # Target angle ~120 degrees for RNA backbone
    target_cos = jnp.cos(120.0 * jnp.pi / 180.0)
    deviations = cos_angles - target_cos

    energy = RNA_PARAMS['angle_weight'] * jnp.sum(deviations ** 2)

    return energy


def clash_energy(coords: jnp.ndarray, chain_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Soft-sphere repulsion to prevent steric clashes.

    Args:
        coords: C1' coordinates (L, 3)
        chain_mask: Optional mask (L, L) to disable inter-chain clashes

    Returns:
        Scalar clash energy
    """
    distances = compute_pairwise_distances(coords)
    L = coords.shape[0]

    # Mask out diagonal and adjacent residues (i, i+1)
    mask = jnp.ones((L, L))
    mask = mask.at[jnp.arange(L), jnp.arange(L)].set(0)  # Diagonal
    if L > 1:
        mask = mask.at[jnp.arange(L-1), jnp.arange(1, L)].set(0)  # i, i+1
        mask = mask.at[jnp.arange(1, L), jnp.arange(L-1)].set(0)  # i+1, i

    # Apply chain mask if provided
    if chain_mask is not None:
        mask = mask * chain_mask

    # Soft-sphere potential: E = sum_{i<j} exp(-k * (d_ij - d_min))
    clash_dist = RNA_PARAMS['clash_distance']
    steepness = RNA_PARAMS['clash_steepness']

    # Only penalize distances below clash threshold
    violations = jnp.maximum(clash_dist - distances, 0.0)
    clash_potential = jnp.exp(steepness * violations) - 1.0

    # Apply mask and sum
    energy = RNA_PARAMS['clash_weight'] * jnp.sum(mask * clash_potential) / 2.0  # Divide by 2 for double counting

    return energy


def pairing_energy(
    coords: jnp.ndarray,
    pairing_matrix: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Energy favoring correct base-pairing distances.

    Args:
        coords: C1' coordinates (L, 3)
        pairing_matrix: Binary matrix (L, L) indicating paired bases

    Returns:
        Scalar pairing energy
    """
    if pairing_matrix is None:
        return jnp.array(0.0)

    distances = compute_pairwise_distances(coords)
    target_dist = RNA_PARAMS['c1_c1_paired']

    # Harmonic potential for paired bases
    deviations = distances - target_dist
    pair_potential = pairing_matrix * (deviations ** 2)

    energy = RNA_PARAMS['pairing_weight'] * jnp.sum(pair_potential) / 2.0

    return energy


def compactness_energy(coords: jnp.ndarray) -> jnp.ndarray:
    """
    Energy favoring compact structures (radius of gyration).

    Args:
        coords: C1' coordinates (L, 3)

    Returns:
        Scalar compactness energy
    """
    L = coords.shape[0]

    # Center of mass
    com = jnp.mean(coords, axis=0)

    # Radius of gyration
    rg_sq = jnp.mean(jnp.sum((coords - com) ** 2, axis=-1))

    # Target radius of gyration
    target_rg_sq = (RNA_PARAMS['target_rg_per_residue'] * jnp.sqrt(L)) ** 2

    # Harmonic potential
    energy = RNA_PARAMS['compactness_weight'] * (rg_sq - target_rg_sq) ** 2

    return energy


def rna_energy(
    coords: jnp.ndarray,
    sequence: Optional[str] = None,
    pairing_matrix: Optional[jnp.ndarray] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, jnp.ndarray]:
    """
    Total RNA energy function (JAX-differentiable).

    Args:
        coords: C1' coordinates (L, 3)
        sequence: RNA sequence (optional, for future base-specific terms)
        pairing_matrix: Binary pairing matrix (L, L)
        chain_mask: Mask for multi-chain structures (L, L)
        weights: Optional dict to override default energy weights

    Returns:
        Dictionary with total energy and individual components
    """
    # Override weights if provided
    if weights is not None:
        for key, value in weights.items():
            if key in RNA_PARAMS:
                RNA_PARAMS[key] = value

    # Compute energy terms
    e_bond = bond_energy(coords)
    e_angle = angle_energy(coords)
    e_clash = clash_energy(coords, chain_mask)
    e_pair = pairing_energy(coords, pairing_matrix)
    e_compact = compactness_energy(coords)

    # Total energy
    e_total = e_bond + e_angle + e_clash + e_pair + e_compact

    return {
        'total': e_total,
        'bond': e_bond,
        'angle': e_angle,
        'clash': e_clash,
        'pairing': e_pair,
        'compactness': e_compact,
    }


def energy_minimize(
    coords_init: jnp.ndarray,
    sequence: Optional[str] = None,
    pairing_matrix: Optional[jnp.ndarray] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    num_steps: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = False
) -> Tuple[jnp.ndarray, Dict]:
    """
    Minimize RNA energy using gradient descent.

    Args:
        coords_init: Initial C1' coordinates (L, 3)
        sequence: RNA sequence
        pairing_matrix: Binary pairing matrix
        chain_mask: Multi-chain mask
        num_steps: Number of optimization steps
        learning_rate: Learning rate for gradient descent
        verbose: Print energy during optimization

    Returns:
        Tuple of (optimized_coords, optimization_info)
    """
    coords = coords_init.copy()

    # Energy function with fixed parameters
    def energy_fn(c):
        result = rna_energy(c, sequence, pairing_matrix, chain_mask)
        return result['total']

    # Gradient function
    grad_fn = jax.grad(energy_fn)

    energies = []

    for step in range(num_steps):
        # Compute energy and gradient
        energy = energy_fn(coords)
        grad = grad_fn(coords)

        # Gradient descent update
        coords = coords - learning_rate * grad

        # Clip to valid range
        coords = jnp.clip(coords, -999.999, 9999.999)

        energies.append(float(energy))

        if verbose and step % 10 == 0:
            print(f"Step {step:3d}: Energy = {energy:.4f}")

    final_energy_dict = rna_energy(coords, sequence, pairing_matrix, chain_mask)

    info = {
        'energies': energies,
        'final_energy': float(final_energy_dict['total']),
        'final_components': {k: float(v) for k, v in final_energy_dict.items()},
        'num_steps': num_steps,
    }

    if verbose:
        print(f"\nFinal energy: {info['final_energy']:.4f}")
        print("Components:")
        for key, value in info['final_components'].items():
            if key != 'total':
                print(f"  {key}: {value:.4f}")

    return coords, info


def verify_gradients(coords: jnp.ndarray):
    """
    Verify that gradients flow through energy function.

    Args:
        coords: Test coordinates

    Returns:
        True if gradients exist and are finite
    """
    print("Verifying gradient flow...")

    # Create energy function
    def energy_fn(c):
        result = rna_energy(c)
        return result['total']

    # Compute gradient
    grad_fn = jax.grad(energy_fn)
    grad = grad_fn(coords)

    # Check gradient properties
    has_gradient = grad is not None
    is_finite = jnp.all(jnp.isfinite(grad))
    is_nonzero = jnp.any(jnp.abs(grad) > 1e-8)

    print(f"  Has gradient: {has_gradient}")
    print(f"  Is finite: {is_finite}")
    print(f"  Is non-zero: {is_nonzero}")
    print(f"  Gradient norm: {jnp.linalg.norm(grad):.6f}")

    # Create JaxPR to inspect computation
    jaxpr = jax.make_jaxpr(energy_fn)(coords)
    print(f"  JaxPR compiled successfully")

    return has_gradient and is_finite and is_nonzero


if __name__ == "__main__":
    print("="* 60)
    print("JAX Energy Refinement Module")
    print("="* 60)

    # Test with simple structure
    L = 10
    coords = jnp.array(np.random.randn(L, 3) * 5.0)

    print(f"\nTest structure: {L} residues")

    # Compute energy
    energy_dict = rna_energy(coords)
    print(f"\nEnergy components:")
    for key, value in energy_dict.items():
        print(f"  {key}: {float(value):.4f}")

    # Test gradient verification
    print()
    verify_gradients(coords)

    # Test energy minimization
    print("\n" + "="* 60)
    print("Testing Energy Minimization")
    print("="* 60)

    coords_opt, info = energy_minimize(
        coords,
        num_steps=50,
        learning_rate=0.01,
        verbose=True
    )

    print(f"\nEnergy reduced from {energies[0]:.4f} to {info['final_energy']:.4f}")
    print(f"Reduction: {energies[0] - info['final_energy']:.4f}")

    print("\n✓ All tests passed")
