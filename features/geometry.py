"""
Geometric feature extraction for RNA structures.
Handles torsion angles, local coordinate frames, and distance matrices.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# RNA backbone torsion angle definitions
TORSION_ANGLES = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']


def compute_torsion_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate torsion angle defined by four points.

    Args:
        p1, p2, p3, p4: 3D coordinates of four atoms

    Returns:
        Torsion angle in radians [-π, π]
    """
    # Vectors along bonds
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)

    # Calculate angle
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)


def extract_backbone_torsions(
    c1_prime_coords: np.ndarray,
    sequence: str
) -> Dict[str, np.ndarray]:
    """
    Extract idealized backbone torsion angles from C1' coordinates.

    Since we only have C1' atoms, we compute pseudo-torsions based on C1' positions.
    This provides a coarse geometric representation.

    Args:
        c1_prime_coords: C1' coordinates (L, 3)
        sequence: RNA sequence

    Returns:
        Dictionary with torsion features:
            - pseudo_torsions: (L, 4) array of pseudo-torsions
            - bond_angles: (L,) array of C1'-C1'-C1' angles
    """
    L = len(c1_prime_coords)

    # Pseudo-torsions: consecutive quadruplets of C1' atoms
    pseudo_torsions = np.zeros((L, 4), dtype=np.float32)

    for i in range(L - 3):
        p1 = c1_prime_coords[i]
        p2 = c1_prime_coords[i + 1]
        p3 = c1_prime_coords[i + 2]
        p4 = c1_prime_coords[i + 3]

        torsion = compute_torsion_angle(p1, p2, p3, p4)
        pseudo_torsions[i, 0] = torsion

    # Shift to get 4 consecutive pseudo-torsion angles
    for offset in range(1, 4):
        for i in range(L - 3 - offset):
            p1 = c1_prime_coords[i + offset]
            p2 = c1_prime_coords[i + offset + 1]
            p3 = c1_prime_coords[i + offset + 2]
            p4 = c1_prime_coords[i + offset + 3]

            torsion = compute_torsion_angle(p1, p2, p3, p4)
            pseudo_torsions[i, offset] = torsion

    # Bond angles: C1'(i) - C1'(i+1) - C1'(i+2)
    bond_angles = np.zeros(L, dtype=np.float32)

    for i in range(L - 2):
        v1 = c1_prime_coords[i] - c1_prime_coords[i + 1]
        v2 = c1_prime_coords[i + 2] - c1_prime_coords[i + 1]

        # Normalize
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)

        # Angle
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        bond_angles[i + 1] = np.arccos(cos_angle)

    return {
        'pseudo_torsions': pseudo_torsions,
        'bond_angles': bond_angles,
    }


def create_local_frames(c1_prime_coords: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create local coordinate frames at each residue position.

    Args:
        c1_prime_coords: C1' coordinates (L, 3)

    Returns:
        Dictionary with:
            - origins: Frame origins (L, 3)
            - rotations: Frame rotation matrices (L, 3, 3)
    """
    L = len(c1_prime_coords)

    origins = c1_prime_coords.copy()
    rotations = np.zeros((L, 3, 3), dtype=np.float32)

    for i in range(L):
        # Define local frame based on neighboring C1' atoms
        if i == 0:
            # First residue: use next two residues
            if L > 2:
                v1 = c1_prime_coords[i + 1] - c1_prime_coords[i]
                v2 = c1_prime_coords[i + 2] - c1_prime_coords[i]
            else:
                v1 = np.array([1.0, 0.0, 0.0])
                v2 = np.array([0.0, 1.0, 0.0])
        elif i == L - 1:
            # Last residue: use previous two residues
            v1 = c1_prime_coords[i] - c1_prime_coords[i - 1]
            v2 = c1_prime_coords[i - 1] - c1_prime_coords[i - 2] if i > 1 else np.array([0.0, 1.0, 0.0])
        else:
            # Middle residue: use neighbors
            v1 = c1_prime_coords[i + 1] - c1_prime_coords[i]
            v2 = c1_prime_coords[i] - c1_prime_coords[i - 1]

        # Gram-Schmidt orthogonalization
        e1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_orth = v2 - np.dot(v2, e1) * e1
        e2 = v2_orth / (np.linalg.norm(v2_orth) + 1e-8)
        e3 = np.cross(e1, e2)

        # Stack as rotation matrix
        rotations[i] = np.stack([e1, e2, e3], axis=1)

    return {
        'origins': origins,
        'rotations': rotations,
    }


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        coords: Coordinates (L, 3)

    Returns:
        Distance matrix (L, L)
    """
    L = coords.shape[0]
    dist_matrix = np.zeros((L, L), dtype=np.float32)

    for i in range(L):
        for j in range(i, L):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def compute_contact_map(
    coords: np.ndarray,
    threshold: float = 8.0
) -> np.ndarray:
    """
    Compute binary contact map based on distance threshold.

    Args:
        coords: Coordinates (L, 3)
        threshold: Distance threshold in Angstroms

    Returns:
        Binary contact map (L, L)
    """
    dist_matrix = compute_distance_matrix(coords)
    return (dist_matrix < threshold).astype(np.float32)


def extract_idealized_backbone_geometry(sequence: str, length: int) -> Dict[str, np.ndarray]:
    """
    Generate idealized A-form RNA backbone geometry as initial structure.

    Args:
        sequence: RNA sequence
        length: Sequence length

    Returns:
        Dictionary with idealized coordinates and torsions
    """
    # A-form RNA parameters (approximate)
    rise_per_residue = 2.8  # Angstroms
    rotation_per_residue = 32.7 * np.pi / 180.0  # radians
    helix_radius = 10.0  # Angstroms

    # Generate C1' positions in idealized helix
    coords = np.zeros((length, 3), dtype=np.float32)

    for i in range(length):
        theta = i * rotation_per_residue
        z = i * rise_per_residue

        x = helix_radius * np.cos(theta)
        y = helix_radius * np.sin(theta)

        coords[i] = [x, y, z]

    # Center coordinates
    coords -= coords.mean(axis=0)

    # Compute torsions from idealized structure
    torsions = extract_backbone_torsions(coords, sequence)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(coords)

    return {
        'idealized_coords': coords,
        'pseudo_torsions': torsions['pseudo_torsions'],
        'bond_angles': torsions['bond_angles'],
        'distance_matrix': dist_matrix,
    }


def compute_pairwise_features(coords: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute pairwise geometric features between all residues.

    Args:
        coords: C1' coordinates (L, 3)

    Returns:
        Dictionary with:
            - distances: (L, L) distance matrix
            - displacement_vectors: (L, L, 3) displacement vectors
            - relative_positions: (L, L, 3) normalized displacements
    """
    L = coords.shape[0]

    # Distance matrix
    distances = compute_distance_matrix(coords)

    # Displacement vectors
    displacement_vectors = np.zeros((L, L, 3), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            displacement_vectors[i, j] = coords[j] - coords[i]

    # Normalized relative positions
    relative_positions = np.zeros_like(displacement_vectors)
    for i in range(L):
        for j in range(L):
            norm = distances[i, j] + 1e-8
            relative_positions[i, j] = displacement_vectors[i, j] / norm

    return {
        'distances': distances,
        'displacement_vectors': displacement_vectors,
        'relative_positions': relative_positions,
    }


def extract_per_chain_geometry(
    coords: np.ndarray,
    chain_boundaries: List[Tuple[int, int, str]]
) -> Dict[str, np.ndarray]:
    """
    Extract geometric features per chain with chain boundary awareness.

    Args:
        coords: Full concatenated coordinates (L, 3)
        chain_boundaries: List of (start, end, chain_id)

    Returns:
        Dictionary with chain-aware geometric features
    """
    L = coords.shape[0]

    # Compute global features
    global_dist = compute_distance_matrix(coords)
    global_torsions = extract_backbone_torsions(coords, 'N' * L)

    # Mask inter-chain distances (set to large value)
    masked_dist = global_dist.copy()
    for i, (start_i, end_i, chain_i) in enumerate(chain_boundaries):
        for j, (start_j, end_j, chain_j) in enumerate(chain_boundaries):
            if chain_i != chain_j:
                masked_dist[start_i:end_i, start_j:end_j] = 999.0

    # Compute per-chain local frames
    frames = create_local_frames(coords)

    return {
        'distance_matrix': global_dist,
        'masked_distance_matrix': masked_dist,
        'pseudo_torsions': global_torsions['pseudo_torsions'],
        'bond_angles': global_torsions['bond_angles'],
        'frame_origins': frames['origins'],
        'frame_rotations': frames['rotations'],
    }


if __name__ == "__main__":
    # Test with idealized structure
    test_seq = "AUGCAUGC"
    ideal_geom = extract_idealized_backbone_geometry(test_seq, len(test_seq))

    print(f"Sequence: {test_seq}")
    print(f"Idealized coords shape: {ideal_geom['idealized_coords'].shape}")
    print(f"Pseudo-torsions shape: {ideal_geom['pseudo_torsions'].shape}")
    print(f"Bond angles shape: {ideal_geom['bond_angles'].shape}")

    # Test local frames
    frames = create_local_frames(ideal_geom['idealized_coords'])
    print(f"\nFrame origins shape: {frames['origins'].shape}")
    print(f"Frame rotations shape: {frames['rotations'].shape}")

    # Test pairwise features
    pairwise = compute_pairwise_features(ideal_geom['idealized_coords'])
    print(f"\nDistance matrix shape: {pairwise['distances'].shape}")
    print(f"Displacement vectors shape: {pairwise['displacement_vectors'].shape}")
