"""
Visualize ground truth structures to verify they are properly folded.
"""
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.pdb_writer import write_pdb


def visualize_ground_truth(data_dir: str = "data", num_samples: int = 5, max_length: int = 40):
    """
    Extract and save ground truth structures as PDB files for visualization.

    Args:
        data_dir: Data directory
        num_samples: Number of structures to visualize
        max_length: Maximum sequence length to include
    """
    # Load data
    dataset_path = Path(data_dir) / "processed" / "train_structures.pkl"

    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found")
        return

    with open(dataset_path, 'rb') as f:
        all_data = pickle.load(f)

    print(f"Loaded {len(all_data)} structures")

    # Filter to easy targets
    filtered = [
        item for item in all_data
        if item['sequence_length'] <= max_length
        and item['num_structures'] >= 1
    ]

    print(f"Found {len(filtered)} structures with length <= {max_length}")

    # Create output directory
    output_dir = Path(data_dir) / "ground_truth_pdbs"
    output_dir.mkdir(exist_ok=True)

    # Save first N structures
    for i, item in enumerate(filtered[:num_samples]):
        target_id = item['target_id']
        sequence = item['sequence']
        coords = item['structures'][0]  # First conformation

        # Compute statistics
        coords_centered = coords - coords.mean(axis=0)
        radius_gyration = np.sqrt(np.mean(np.sum(coords_centered**2, axis=1)))
        coord_std = coords.std()

        print(f"\nStructure {i+1}/{num_samples}: {target_id}")
        print(f"  Length: {len(sequence)}")
        print(f"  Sequence: {sequence}")
        print(f"  Coord range: [{coords.min():.2f}, {coords.max():.2f}]")
        print(f"  Coord std: {coord_std:.2f} Å")
        print(f"  Radius of gyration: {radius_gyration:.2f} Å")

        # Check if structure looks linear vs folded
        # Linear structures have Rg ≈ L * (backbone spacing)
        # Folded structures have Rg << L
        expected_linear_rg = len(sequence) * 2.8 / np.sqrt(12)  # For uniform distribution
        compactness = radius_gyration / expected_linear_rg

        if compactness < 0.5:
            fold_type = "COMPACT/FOLDED"
        elif compactness < 0.8:
            fold_type = "PARTIALLY FOLDED"
        else:
            fold_type = "LINEAR/EXTENDED"

        print(f"  Compactness ratio: {compactness:.2f} ({fold_type})")

        # Save PDB
        pdb_path = output_dir / f"{target_id}_ground_truth.pdb"
        write_pdb(coords, str(pdb_path), sequence)

    print(f"\n{'='*70}")
    print(f"Saved {num_samples} PDB files to {output_dir}")
    print(f"Open in ChimeraX to visualize:")
    print(f"  chimera {output_dir}/*.pdb")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize ground truth structures")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of structures to visualize")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum sequence length")

    args = parser.parse_args()

    visualize_ground_truth(args.data_dir, args.num_samples, args.max_length)
