"""
Prepare training labels: Convert train_labels.csv to multi-structure format.
Handles multiple experimental conformations per target.
"""

import polars as pl
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from features.sequence import expand_sequence_with_stoichiometry


def extract_target_structures(
    labels_df: pl.DataFrame,
    target_id: str
) -> List[np.ndarray]:
    """
    Extract all experimental structures for a target.

    Args:
        labels_df: Full labels DataFrame
        target_id: Target identifier

    Returns:
        List of coordinate arrays (L, 3), one per conformation
    """
    # Filter to this target
    target_rows = labels_df.filter(
        pl.col('ID').str.starts_with(f"{target_id}_")
    ).sort('resid')

    if len(target_rows) == 0:
        return []

    L = len(target_rows)
    structures = []

    # Check for up to 5 conformations
    for conf_idx in range(1, 6):
        x_col = f'x_{conf_idx}'
        y_col = f'y_{conf_idx}'
        z_col = f'z_{conf_idx}'

        if x_col not in target_rows.columns:
            break

        # Check if this conformation has data
        x_vals = target_rows[x_col].to_numpy()
        y_vals = target_rows[y_col].to_numpy()
        z_vals = target_rows[z_col].to_numpy()

        # Skip if all null
        if np.all(np.isnan(x_vals)):
            continue

        # Check for missing residues
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(z_vals))
        if not np.all(valid_mask):
            print(f"  Warning: {target_id} conf {conf_idx} has {(~valid_mask).sum()} missing residues")
            continue

        # Stack coordinates
        coords = np.stack([x_vals, y_vals, z_vals], axis=1).astype(np.float32)
        structures.append(coords)

    return structures


def prepare_training_dataset(
    sequences_path: Path,
    labels_path: Path,
    output_path: Path,
    temporal_cutoff: str = "2022-01-01"
) -> Dict:
    """
    Prepare complete training dataset with multi-structure labels.

    Args:
        sequences_path: Path to test_sequences.csv or train_sequences.csv
        labels_path: Path to train_labels.csv
        output_path: Output parquet path
        temporal_cutoff: Date cutoff for train/val split

    Returns:
        Dictionary with dataset statistics
    """
    print("="* 70)
    print("Preparing Training Dataset")
    print("="* 70)

    # Load sequences
    print(f"\n1. Loading sequences from {sequences_path}...")
    sequences_df = pl.read_csv(sequences_path)
    print(f"   Found {len(sequences_df)} targets")

    # Load labels
    print(f"\n2. Loading labels from {labels_path}...")
    labels_df = pl.read_csv(labels_path)
    print(f"   Found {len(labels_df)} residue records")

    # Extract unique target IDs from labels
    unique_targets = set(
        labels_df['ID'].str.extract(r'^(.+?)_\d+$', 1).unique().to_list()
    )
    print(f"   Unique targets in labels: {len(unique_targets)}")

    # Process each target
    print(f"\n3. Extracting structures per target...")
    training_data = []

    for i, row in enumerate(sequences_df.iter_rows(named=True)):
        target_id = row['target_id']

        if target_id not in unique_targets:
            continue

        # Extract structures
        structures = extract_target_structures(labels_df, target_id)

        if len(structures) == 0:
            continue

        # Get sequence info
        sequence = row['sequence']
        stoich = row.get('stoichiometry', '{A:1}')
        all_seqs = row.get('all_sequences', f">A\n{sequence}")
        temporal_cutoff_date = row.get('temporal_cutoff', '2025-01-01')

        # Expand multi-chain sequence if needed
        try:
            full_seq, boundaries = expand_sequence_with_stoichiometry(stoich, all_seqs)
        except:
            full_seq = sequence
            boundaries = [(0, len(sequence), 'A')]

        # Determine split (train vs val based on temporal cutoff)
        is_validation = temporal_cutoff_date >= temporal_cutoff

        training_data.append({
            'target_id': target_id,
            'sequence': full_seq,
            'sequence_length': len(full_seq),
            'num_structures': len(structures),
            'structures': structures,  # List of numpy arrays
            'chain_boundaries': boundaries,
            'stoichiometry': stoich,
            'is_validation': is_validation,
            'temporal_cutoff': temporal_cutoff_date,
            'description': row.get('description', ''),
        })

        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(sequences_df)} targets...")

    print(f"\n   Total targets with structures: {len(training_data)}")

    # Statistics
    num_train = sum(1 for d in training_data if not d['is_validation'])
    num_val = len(training_data) - num_train
    num_multi_conf = sum(1 for d in training_data if d['num_structures'] > 1)

    stats = {
        'total_targets': len(training_data),
        'num_train': num_train,
        'num_val': num_val,
        'num_multi_conformation': num_multi_conf,
        'avg_length': np.mean([d['sequence_length'] for d in training_data]),
        'avg_structures_per_target': np.mean([d['num_structures'] for d in training_data]),
    }

    print(f"\n4. Dataset Statistics:")
    print(f"   Total targets: {stats['total_targets']}")
    print(f"   Training set: {stats['num_train']}")
    print(f"   Validation set: {stats['num_val']}")
    print(f"   Multi-conformation targets: {stats['num_multi_conformation']}")
    print(f"   Avg sequence length: {stats['avg_length']:.1f}")
    print(f"   Avg structures per target: {stats['avg_structures_per_target']:.2f}")

    # Save as pickle (parquet doesn't handle nested arrays well)
    print(f"\n5. Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(training_data, f)

    print(f"   Saved {len(training_data)} targets")

    # Also save stats
    stats_path = output_path.parent / f"{output_path.stem}_stats.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Dataset preparation complete")

    return stats


def create_difficulty_splits(dataset_path: Path) -> Dict[str, List]:
    """
    Split dataset by difficulty (easy/medium/hard).

    Args:
        dataset_path: Path to prepared dataset pickle

    Returns:
        Dictionary mapping difficulty -> list of target_ids
    """
    print("\nCreating difficulty splits...")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    easy = []
    medium = []
    hard = []

    for item in data:
        target_id = item['target_id']
        length = item['sequence_length']
        # Note: would need pseudoknot info from secondary structure

        if length <= 50:
            easy.append(target_id)
        elif length <= 150:
            medium.append(target_id)
        else:
            hard.append(target_id)

    splits = {
        'easy': easy,
        'medium': medium,
        'hard': hard,
    }

    print(f"  Easy (≤50 nt): {len(easy)}")
    print(f"  Medium (50-150 nt): {len(medium)}")
    print(f"  Hard (>150 nt): {len(hard)}")

    return splits


if __name__ == "__main__":
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    sequences_path = output_dir / "train_sequences.csv"  # Use reconstructed sequences
    labels_path = data_dir / "raw" / "train_labels.csv"
    output_path = output_dir / "train_structures.pkl"

    # Prepare dataset
    stats = prepare_training_dataset(
        sequences_path,
        labels_path,
        output_path,
        temporal_cutoff="2022-01-01"
    )

    # Create difficulty splits
    splits = create_difficulty_splits(output_path)

    # Save splits
    splits_path = output_dir / "difficulty_splits.pkl"
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)

    print(f"\n✓ All data prepared and ready for training")
