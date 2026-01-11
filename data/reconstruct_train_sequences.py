"""
Reconstruct train_sequences.csv from train_labels.csv.
Extract sequence from resname column and group by target_id.
"""

import polars as pl
from pathlib import Path


def reconstruct_sequences(labels_path: Path, output_path: Path):
    """
    Reconstruct sequences from labels.

    Args:
        labels_path: Path to train_labels.csv
        output_path: Output path for train_sequences.csv
    """
    print("="*70)
    print("Reconstructing Training Sequences from Labels")
    print("="*70)

    print("\n1. Loading train_labels.csv...")
    labels = pl.read_csv(labels_path)
    print(f"   Total residue records: {len(labels):,}")

    # Extract target_id from ID column (format: "157D_1" -> "157D")
    print("\n2. Extracting sequences...")
    sequences_df = (
        labels
        .with_columns([
            pl.col("ID").str.split("_").list.first().alias("target_id")
        ])
        .sort(["target_id", "resid"])
        .group_by("target_id")
        .agg([
            pl.col("resname").str.concat("").alias("sequence"),
            pl.col("resid").count().alias("sequence_length"),
            pl.col("chain").first().alias("chain"),
            pl.col("copy").max().alias("num_copies"),
        ])
    )

    print(f"   Unique targets: {len(sequences_df)}")

    # Add metadata placeholders
    sequences_df = sequences_df.with_columns([
        pl.lit("").alias("temporal_cutoff"),  # Unknown, will need to set
        pl.lit("").alias("description"),
        pl.lit("{A:1}").alias("stoichiometry"),  # Default single chain
        pl.lit("").alias("all_sequences"),
        pl.lit("").alias("ligand_ids"),
        pl.lit("").alias("ligand_SMILES"),
    ])

    # Build all_sequences in FASTA format
    sequences_df = sequences_df.with_columns([
        (pl.lit(">") + pl.col("target_id") + pl.lit("_1|Chain ") + pl.col("chain") +
         pl.lit("|RNA|\n") + pl.col("sequence")).alias("all_sequences")
    ])

    # Update stoichiometry for multi-copy targets
    sequences_df = sequences_df.with_columns([
        pl.when(pl.col("num_copies") > 1)
        .then(pl.lit("{") + pl.col("chain") + pl.lit(":") + pl.col("num_copies").cast(str) + pl.lit("}"))
        .otherwise(pl.lit("{") + pl.col("chain") + pl.lit(":1}"))
        .alias("stoichiometry")
    ])

    # Reorder columns to match expected format
    sequences_df = sequences_df.select([
        "target_id",
        "sequence",
        "temporal_cutoff",
        "description",
        "stoichiometry",
        "all_sequences",
        "ligand_ids",
        "ligand_SMILES",
    ])

    # Save
    print(f"\n3. Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sequences_df.write_csv(output_path)

    print(f"   Saved {len(sequences_df)} targets")

    # Statistics
    print("\n4. Statistics:")
    lengths = sequences_df["sequence"].str.len_chars()
    print(f"   Sequence length range: {lengths.min()}-{lengths.max()} nt")
    print(f"   Median length: {lengths.median():.0f} nt")
    print(f"   Mean length: {lengths.mean():.0f} nt")

    # Multi-copy targets
    multi_copy = sequences_df.filter(pl.col("num_copies") > 1)
    print(f"   Multi-copy targets: {len(multi_copy)} ({len(multi_copy)/len(sequences_df)*100:.1f}%)")

    # Show sample
    print("\n5. Sample entries:")
    print(sequences_df.select(["target_id", "sequence", "sequence_length", "stoichiometry"]).head(5))

    print("\n[DONE] Training sequences reconstructed")
    print(f"\nNext step:")
    print(f"  python train/prepare_labels.py")

    return sequences_df


if __name__ == "__main__":
    labels_path = Path(__file__).parent / "raw" / "train_labels.csv"
    output_path = Path(__file__).parent / "processed" / "train_sequences.csv"

    reconstruct_sequences(labels_path, output_path)
