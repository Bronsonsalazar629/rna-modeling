"""
Polars-based lazy ETL pipeline for RNA 3D structure data.
Memory-efficient streaming processing with <64GB RAM requirement.
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from features.sequence import (
    parse_stoichiometry,
    parse_fasta_sequences,
    expand_sequence_with_stoichiometry,
    sequence_to_one_hot,
    create_chain_features,
    create_chain_attention_masks,
)
from features.secondary import (
    predict_per_chain_structure,
    dotbracket_to_pairing,
)
from features.geometry import (
    extract_per_chain_geometry,
    extract_idealized_backbone_geometry,
)


def parse_fasta_from_csv(all_sequences: str) -> Dict[str, Tuple[str, List[str]]]:
    """
    Parse FASTA content from CSV all_sequences column.
    Compatible with parse_fasta_py.py format.

    Args:
        all_sequences: FASTA-formatted string from CSV

    Returns:
        Dictionary mapping auth chain_id to (sequence, list_of_auth_chain_ids)
    """
    result = {}
    lines = all_sequences.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith(">"):
            # Parse header: >104D_1|Chains A[auth A], B[auth B]|...|
            parts = line.split("|")
            if len(parts) < 2:
                auth_chain_ids = []
                chains_part = ""
            else:
                chains_part = parts[1].strip()

                # Extract auth chain IDs
                auth_chain_ids = []
                replaced_chains_part = re.sub(r"^Chains? ", "", chains_part)
                chains = replaced_chains_part.split(",")
                for chain in chains:
                    auth_match = re.search(r"\[auth ([^\]]+)\]", chain)
                    if auth_match:
                        auth_chain_ids.append(auth_match.group(1).strip())
                    else:
                        c = chain.strip()
                        if c:
                            auth_chain_ids.append(c)

            primary_auth_chain = auth_chain_ids[0] if auth_chain_ids else None

            # Read sequence
            sequence = ""
            while (i + 1) < len(lines) and not lines[i + 1].startswith(">"):
                sequence += lines[i + 1].strip()
                i += 1

            if primary_auth_chain:
                result[primary_auth_chain] = (sequence, auth_chain_ids)

        i += 1

    return result


def process_sequence_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process sequence features with multi-chain awareness.

    Args:
        df: DataFrame with columns [target_id, sequence, stoichiometry, all_sequences]

    Returns:
        DataFrame with added sequence feature columns
    """
    def _expand_sequence(row):
        """Expand sequence according to stoichiometry."""
        stoich = row['stoichiometry']
        all_seqs = row['all_sequences']

        if stoich and all_seqs:
            full_seq, boundaries = expand_sequence_with_stoichiometry(stoich, all_seqs)
            return full_seq, boundaries
        else:
            # Fallback: single chain
            return row['sequence'], [(0, len(row['sequence']), 'A')]

    def _compute_features(row):
        """Compute sequence features."""
        sequence, boundaries = row['expanded_sequence'], row['chain_boundaries']

        # One-hot encoding
        one_hot = sequence_to_one_hot(sequence)

        # Chain features
        chain_feats = create_chain_features(sequence, boundaries)

        # Chain attention mask
        attn_mask = create_chain_attention_masks(boundaries, allow_interchain=False)

        return {
            'one_hot': one_hot.tolist(),
            'chain_features': chain_feats.tolist(),
            'attention_mask': attn_mask.tolist(),
            'sequence_length': len(sequence),
        }

    # Apply expansions
    expanded = []
    for row in df.iter_rows(named=True):
        seq, bounds = _expand_sequence(row)
        expanded.append({
            **row,
            'expanded_sequence': seq,
            'chain_boundaries': bounds,
        })

    df_expanded = pl.DataFrame(expanded)

    # Compute features
    features = []
    for row in df_expanded.iter_rows(named=True):
        feats = _compute_features(row)
        features.append(feats)

    df_features = pl.DataFrame(features)

    # Combine
    return pl.concat([df_expanded, df_features], how='horizontal')


def process_secondary_structure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Predict secondary structure per chain.

    Args:
        df: DataFrame with expanded_sequence and chain_boundaries

    Returns:
        DataFrame with secondary structure features
    """
    def _predict_structure(row):
        """Predict secondary structure."""
        sequence = row['expanded_sequence']
        boundaries = row['chain_boundaries']

        struct_result = predict_per_chain_structure(sequence, boundaries)

        # Convert to serializable format
        return {
            'structure_dotbracket': struct_result['structure'],
            'bpp_matrix': struct_result['bpp_matrix'].tolist(),
            'is_paired': struct_result['is_paired'].tolist(),
            'has_pseudoknots': struct_result['has_pseudoknots'],
        }

    structures = []
    for row in df.iter_rows(named=True):
        struct = _predict_structure(row)
        structures.append(struct)

    df_struct = pl.DataFrame(structures)

    return pl.concat([df, df_struct], how='horizontal')


def process_geometry_features(df: pl.DataFrame, use_labels: bool = True) -> pl.DataFrame:
    """
    Extract geometric features from coordinates or idealized structure.

    Args:
        df: DataFrame with sequence and chain info
        use_labels: If True, use training labels; otherwise use idealized geometry

    Returns:
        DataFrame with geometry features
    """
    def _extract_geometry(row):
        """Extract geometry features."""
        sequence = row['expanded_sequence']
        boundaries = row['chain_boundaries']
        seq_len = len(sequence)

        if use_labels and 'x_1' in row and row['x_1'] is not None:
            # Use actual coordinates from labels
            # Note: This assumes coordinates are already in the DataFrame
            coords = np.array([
                [row.get(f'x_{i}', 0.0), row.get(f'y_{i}', 0.0), row.get(f'z_{i}', 0.0)]
                for i in range(1, seq_len + 1)
            ], dtype=np.float32)
        else:
            # Use idealized A-form geometry
            ideal_geom = extract_idealized_backbone_geometry(sequence, seq_len)
            coords = ideal_geom['idealized_coords']

        # Extract geometric features
        geom_features = extract_per_chain_geometry(coords, boundaries)

        return {
            'c1_prime_coords': coords.tolist(),
            'distance_matrix': geom_features['distance_matrix'].tolist(),
            'pseudo_torsions': geom_features['pseudo_torsions'].tolist(),
            'bond_angles': geom_features['bond_angles'].tolist(),
        }

    geometries = []
    for row in df.iter_rows(named=True):
        geom = _extract_geometry(row)
        geometries.append(geom)

    df_geom = pl.DataFrame(geometries)

    return pl.concat([df, df_geom], how='horizontal')


def load_training_labels(labels_path: Path) -> pl.LazyFrame:
    """
    Load training labels with lazy evaluation.

    Args:
        labels_path: Path to train_labels.csv

    Returns:
        Lazy DataFrame with training labels
    """
    return pl.scan_csv(
        labels_path,
        schema_overrides={
            'ID': pl.Utf8,
            'resname': pl.Utf8,
            'resid': pl.Int32,
            'x_1': pl.Float32,
            'y_1': pl.Float32,
            'z_1': pl.Float32,
            'chain': pl.Utf8,
            'copy': pl.Int32,
        }
    )


def pivot_coordinates_to_wide(labels_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Pivot coordinate columns from long to wide format.

    Args:
        labels_df: Long-format labels with one row per residue

    Returns:
        Wide-format with one row per target_id
    """
    # Extract target_id from ID column (format: {target_id}_{resid})
    return (
        labels_df
        .with_columns([
            pl.col('ID').str.extract(r'^(.+)_\d+$', 1).alias('target_id')
        ])
        .group_by('target_id')
        .agg([
            pl.col('resid').alias('resids'),
            pl.col('x_1').alias('x_coords'),
            pl.col('y_1').alias('y_coords'),
            pl.col('z_1').alias('z_coords'),
            pl.col('chain').alias('chains'),
            pl.col('copy').alias('copies'),
        ])
    )


def build_training_dataset(
    sequences_path: Path,
    labels_path: Path,
    output_path: Path,
    streaming: bool = True
) -> None:
    """
    Build complete training dataset with all features.

    Args:
        sequences_path: Path to test_sequences.csv (or train_sequences.csv if available)
        labels_path: Path to train_labels.csv
        output_path: Output parquet file path
        streaming: Use streaming mode for memory efficiency
    """
    print("Loading sequence data...")
    sequences_df = pl.read_csv(sequences_path)

    print(f"Loaded {len(sequences_df)} sequences")

    # Process in chunks to manage memory
    chunk_size = 100
    all_chunks = []

    for i in range(0, len(sequences_df), chunk_size):
        chunk = sequences_df[i:i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}/{(len(sequences_df) + chunk_size - 1) // chunk_size}...")

        # Process sequence features
        print("  - Extracting sequence features...")
        chunk = process_sequence_features(chunk)

        # Process secondary structure
        print("  - Predicting secondary structure...")
        chunk = process_secondary_structure(chunk)

        # Process geometry (without labels for test sequences)
        print("  - Extracting geometry features...")
        chunk = process_geometry_features(chunk, use_labels=False)

        all_chunks.append(chunk)

        # Free memory
        if i % (chunk_size * 5) == 0 and i > 0:
            print(f"  - Writing intermediate results...")
            temp_df = pl.concat(all_chunks)
            all_chunks = [temp_df]

    # Combine all chunks
    print("Combining all chunks...")
    final_df = pl.concat(all_chunks)

    # Write to parquet
    print(f"Writing to {output_path}...")
    final_df.write_parquet(
        output_path,
        compression='snappy',
        use_pyarrow=False,
    )

    print(f"Dataset saved: {output_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")


def build_rna_dataset_lazy(
    data_dir: Path,
    output_dir: Path,
    max_resolution: float = 3.0
) -> pl.LazyFrame:
    """
    Build RNA dataset with lazy evaluation for memory efficiency.

    Args:
        data_dir: Directory containing raw data files
        output_dir: Output directory for processed parquet files
        max_resolution: Maximum resolution filter for PDB structures

    Returns:
        Lazy DataFrame ready for streaming collection
    """
    sequences_path = data_dir / "test_sequences.csv"

    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequence file not found: {sequences_path}")

    # Build lazy pipeline
    return (
        pl.scan_csv(sequences_path)
        .filter(
            # Add any filters here (e.g., sequence length, resolution if available)
            pl.col('sequence').str.lengths() >= 10  # Minimum length
        )
    )


def validate_dataset(parquet_path: Path) -> Dict[str, any]:
    """
    Validate processed dataset.

    Args:
        parquet_path: Path to processed parquet file

    Returns:
        Dictionary with validation statistics
    """
    df = pl.read_parquet(parquet_path)

    stats = {
        'num_samples': len(df),
        'num_columns': len(df.columns),
        'columns': df.columns,
        'sequence_lengths': {
            'min': df['sequence_length'].min(),
            'max': df['sequence_length'].max(),
            'mean': df['sequence_length'].mean(),
        },
        'memory_usage_mb': df.estimated_size('mb'),
    }

    return stats


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent / "raw"
    output_dir = Path(__file__).parent / "processed"
    output_dir.mkdir(exist_ok=True)

    sequences_path = data_dir / "test_sequences.csv"
    labels_path = data_dir / "train_labels.csv"
    output_path = output_dir / "rna_dataset.parquet"

    print("="* 60)
    print("RNA 3D Folding Dataset Pipeline")
    print("="* 60)

    if sequences_path.exists():
        build_training_dataset(
            sequences_path=sequences_path,
            labels_path=labels_path,
            output_path=output_path,
            streaming=True
        )

        print("\n" + "="* 60)
        print("Validation Statistics")
        print("="* 60)

        stats = validate_dataset(output_path)
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print(f"Error: {sequences_path} not found")
        print("Please ensure CSV files are in data/raw/")
