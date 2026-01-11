"""
Align training sequences with labels.
Filters sequences to only those present in train_labels.csv.
"""

import polars as pl
from pathlib import Path


def align_training_data(raw_dir: Path, processed_dir: Path):
    """
    Create aligned training dataset.

    Args:
        raw_dir: Directory with raw CSV files
        processed_dir: Output directory
    """
    print("="*70)
    print("Aligning Training Data")
    print("="*70)

    # Load files
    print("\n1. Loading raw data...")
    sequences_path = raw_dir / "test_sequences.csv"  # Contains some training sequences
    labels_path = raw_dir / "train_labels.csv"

    sequences = pl.read_csv(sequences_path)
    labels = pl.read_csv(labels_path)

    print(f"   Sequences: {len(sequences)} targets")
    print(f"   Labels: {len(labels)} residue records")

    # Extract unique target_ids from labels
    print("\n2. Extracting target IDs from labels...")
    label_target_ids = (
        labels
        .select(pl.col("ID").str.split("_").list.first().alias("target_id"))
        .unique()
        .sort("target_id")
    )

    print(f"   Unique targets in labels: {len(label_target_ids)}")

    # Filter sequences to match labels
    print("\n3. Filtering sequences...")
    train_sequences_filtered = sequences.filter(
        pl.col("target_id").is_in(label_target_ids["target_id"])
    )

    print(f"   Matched targets: {len(train_sequences_filtered)}")
    print(f"   Unmatched: {len(sequences) - len(train_sequences_filtered)}")

    # Save aligned data
    print("\n4. Saving aligned data...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    aligned_seq_path = processed_dir / "train_sequences_aligned.csv"
    aligned_labels_path = processed_dir / "train_labels.parquet"

    train_sequences_filtered.write_csv(aligned_seq_path)
    labels.write_parquet(aligned_labels_path)

    print(f"   Sequences: {aligned_seq_path}")
    print(f"   Labels: {aligned_labels_path}")

    # Statistics
    print("\n5. Alignment Statistics:")
    print(f"   Total training targets: {len(train_sequences_filtered)}")

    # Length distribution
    lengths = train_sequences_filtered["sequence"].str.lengths()
    print(f"   Sequence length: {lengths.min()}-{lengths.max()} nt")
    print(f"   Median length: {lengths.median():.0f} nt")

    # Count labels per target
    labels_per_target = (
        labels
        .with_columns(pl.col("ID").str.split("_").list.first().alias("target_id"))
        .group_by("target_id")
        .agg(pl.count().alias("num_residues"))
    )

    # Join with sequences
    merged = train_sequences_filtered.join(
        labels_per_target,
        on="target_id",
        how="left"
    )

    # Check for mismatches
    seq_lengths = merged["sequence"].str.lengths()
    label_counts = merged["num_residues"]

    mismatches = (seq_lengths != label_counts).sum()

    if mismatches > 0:
        print(f"\n   [WARNING] {mismatches} targets have sequence/label length mismatch")
        print("   This may indicate:")
        print("     - Multi-chain targets (sequence is concatenated)")
        print("     - Missing residues in structures")

        # Show examples
        mismatch_examples = merged.filter(seq_lengths != label_counts).head(5)
        print("\n   Examples:")
        for row in mismatch_examples.iter_rows(named=True):
            print(f"     {row['target_id']}: seq_len={len(row['sequence'])}, labels={row['num_residues']}")
    else:
        print(f"   [PASS] All targets have matching sequence/label lengths")

    print("\n[DONE] Training data aligned")
    print(f"\nNext step:")
    print(f"  python train/prepare_labels.py")

    return aligned_seq_path, aligned_labels_path


if __name__ == "__main__":
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"

    align_training_data(raw_dir, processed_dir)
