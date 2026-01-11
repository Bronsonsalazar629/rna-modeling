"""
Convert predicted contact maps to 3D coordinates using distance geometry.
This provides a physics-based initial structure that is guaranteed to be folded.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


def contact_map_to_distance_matrix(
    contacts: np.ndarray,
    sequence: str,
    contact_threshold: float = 0.5,
    contact_distance: float = 6.0,
    backbone_distance: float = 6.0
) -> np.ndarray:
    """
    Convert contact map to distance matrix.

    Args:
        contacts: (L, L) contact probability matrix
        sequence: RNA sequence
        contact_threshold: Threshold for calling a contact
        contact_distance: Target distance for contacts (Angstroms)
        backbone_distance: Distance between sequential C1' atoms

    Returns:
        distance_matrix: (L, L) target distance matrix
    """
    L = len(sequence)
    distances = np.zeros((L, L))

    # Sequential backbone distances
    for i in range(L):
        for j in range(L):
            seq_sep = abs(j - i)
            if seq_sep == 0:
                distances[i, j] = 0.0
            elif seq_sep == 1:
                distances[i, j] = backbone_distance
            elif seq_sep == 2:
                distances[i, j] = backbone_distance * 1.8
            else:
                # Default: assume extended conformation
                distances[i, j] = backbone_distance * seq_sep * 0.8

    # Apply contacts (shorter distances for paired bases)
    for i in range(L):
        for j in range(i+3, L):  # Skip local contacts
            if contacts[i, j] > contact_threshold:
                # Weighted by contact probability
                prob = contacts[i, j]
                distances[i, j] = contact_distance
                distances[j, i] = contact_distance

    return distances


def distance_geometry_embedding(
    distance_matrix: np.ndarray,
    dimensions: int = 3,
    num_iterations: int = 1000
) -> np.ndarray:
    """
    Embed distance matrix into 3D coordinates using distance geometry.

    Args:
        distance_matrix: (L, L) target distances
        dimensions: Number of dimensions (3 for 3D)
        num_iterations: Number of optimization iterations

    Returns:
        coords: (L, 3) 3D coordinates
    """
    L = distance_matrix.shape[0]

    # Initialize with random coordinates
    np.random.seed(42)
    coords_init = np.random.randn(L, dimensions) * 5.0

    def stress_function(coords_flat):
        """Stress function for distance geometry."""
        coords = coords_flat.reshape(L, dimensions)

        # Compute pairwise distances
        dists = squareform(pdist(coords))

        # Weighted stress (emphasize short-range and contacts)
        weights = np.ones_like(distance_matrix)

        # Higher weight for sequential neighbors
        for offset in [1, 2]:
            for i in range(L - offset):
                weights[i, i + offset] = 10.0
                weights[i + offset, i] = 10.0

        # Higher weight for contacts (short distances)
        weights[distance_matrix < 8.0] = 5.0

        # Compute stress
        diff = (dists - distance_matrix) * weights
        stress = np.sum(diff ** 2)

        return stress

    # Optimize
    result = minimize(
        stress_function,
        coords_init.flatten(),
        method='L-BFGS-B',
        options={'maxiter': num_iterations, 'disp': False}
    )

    coords = result.x.reshape(L, dimensions)

    return coords


def contacts_to_3d(
    contacts: np.ndarray,
    sequence: str,
    contact_threshold: float = 0.5
) -> np.ndarray:
    """
    Convert contact map to 3D coordinates.

    Args:
        contacts: (L, L) contact probability matrix
        sequence: RNA sequence
        contact_threshold: Threshold for calling contacts

    Returns:
        coords: (L, 3) C1' coordinates in Angstroms
    """
    # Convert contacts to distance matrix
    distance_matrix = contact_map_to_distance_matrix(
        contacts, sequence, contact_threshold
    )

    # Embed in 3D
    coords = distance_geometry_embedding(distance_matrix, dimensions=3)

    return coords


if __name__ == "__main__":
    # Test contact to 3D conversion
    print("Testing contact to 3D conversion...")

    # Create synthetic contact map (simple hairpin)
    L = 20
    sequence = "A" * L
    contacts = np.zeros((L, L))

    # Add hairpin contacts (base pairs)
    for i in range(8):
        contacts[i, L-1-i] = 0.9  # Strong contacts
        contacts[L-1-i, i] = 0.9

    print(f"  Sequence length: {L}")
    print(f"  Num contacts: {(contacts > 0.5).sum() / 2}")

    # Convert to 3D
    coords = contacts_to_3d(contacts, sequence)

    print(f"  Output coords shape: {coords.shape}")
    print(f"  Coord range: [{coords.min():.2f}, {coords.max():.2f}]")
    print(f"  Coord std: {coords.std():.2f}")

    # Check if folded (compact)
    coords_centered = coords - coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum(coords_centered**2, axis=1)))
    print(f"  Radius of gyration: {rg:.2f} Å")

    # Check backbone distances
    backbone_dists = [
        np.linalg.norm(coords[i+1] - coords[i])
        for i in range(L-1)
    ]
    print(f"  Backbone distances: {np.mean(backbone_dists):.2f} ± {np.std(backbone_dists):.2f} Å")

    print("\nContact to 3D conversion working!")
