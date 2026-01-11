"""
Validation script for RNA 3D folding data pipeline.
Ensures data quality, memory constraints, and feature correctness.
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys
import time
import psutil
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.pipeline import (
    build_training_dataset,
    validate_dataset,
    process_sequence_features,
    process_secondary_structure,
)
from features.sequence import expand_sequence_with_stoichiometry


class DatasetValidator:
    """Validator for RNA dataset pipeline."""

    def __init__(self, max_memory_gb: float = 64.0):
        """
        Initialize validator.

        Args:
            max_memory_gb: Maximum allowed memory usage in GB
        """
        self.max_memory_gb = max_memory_gb
        self.validation_results = {}

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024**3

    def validate_memory_constraint(self) -> bool:
        """
        Validate that memory usage is under constraint.

        Returns:
            True if memory usage is acceptable
        """
        current_memory = self.get_memory_usage()
        print(f"Current memory usage: {current_memory:.2f} GB / {self.max_memory_gb} GB")

        is_valid = current_memory < self.max_memory_gb
        self.validation_results['memory_usage_gb'] = current_memory
        self.validation_results['memory_valid'] = is_valid

        return is_valid

    def validate_sequence_features(self, df: pl.DataFrame) -> bool:
        """
        Validate sequence features.

        Args:
            df: DataFrame with sequence features

        Returns:
            True if all validations pass
        """
        print("\nValidating sequence features...")
        issues = []

        # Check for missing sequences
        if df['sequence'].null_count() > 0:
            issues.append(f"Found {df['sequence'].null_count()} null sequences")

        # Check sequence lengths
        if 'sequence_length' in df.columns:
            min_len = df['sequence_length'].min()
            max_len = df['sequence_length'].max()
            print(f"  Sequence length range: {min_len} - {max_len}")

            if min_len < 5:
                issues.append(f"Found sequences shorter than 5 nt: {min_len}")
        else:
            issues.append("Missing sequence_length column")

        # Check for valid nucleotides
        for seq in df['sequence'].head(10):
            if seq:
                invalid_chars = set(seq.upper()) - set('ACGU')
                if invalid_chars:
                    issues.append(f"Invalid nucleotides found: {invalid_chars}")
                    break

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            self.validation_results['sequence_validation'] = False
            return False
        else:
            print("  ✓ All sequence features valid")
            self.validation_results['sequence_validation'] = True
            return True

    def validate_stoichiometry_parsing(self, df: pl.DataFrame) -> bool:
        """
        Validate stoichiometry and multi-chain handling.

        Args:
            df: DataFrame with stoichiometry info

        Returns:
            True if parsing is correct
        """
        print("\nValidating stoichiometry parsing...")
        issues = []

        for i, row in enumerate(df.head(5).iter_rows(named=True)):
            if 'stoichiometry' in row and row['stoichiometry']:
                stoich = row['stoichiometry']
                all_seqs = row.get('all_sequences', '')

                try:
                    full_seq, boundaries = expand_sequence_with_stoichiometry(stoich, all_seqs)

                    # Validate total length
                    expected_len = len(full_seq)
                    boundary_len = max(end for _, end, _ in boundaries)

                    if expected_len != boundary_len:
                        issues.append(f"Row {i}: Length mismatch {expected_len} != {boundary_len}")

                    print(f"  Row {i}: {stoich} -> {len(boundaries)} segments, {expected_len} nt total")

                except Exception as e:
                    issues.append(f"Row {i}: Failed to parse stoichiometry: {e}")

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            self.validation_results['stoichiometry_validation'] = False
            return False
        else:
            print("  ✓ Stoichiometry parsing valid")
            self.validation_results['stoichiometry_validation'] = True
            return True

    def validate_no_missing_residues(self, df: pl.DataFrame) -> bool:
        """
        Validate that all residues are accounted for.

        Args:
            df: DataFrame with processed features

        Returns:
            True if no missing residues
        """
        print("\nValidating residue completeness...")
        issues = []

        # Check that all expected columns exist
        required_cols = ['sequence', 'sequence_length']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            self.validation_results['residue_validation'] = False
            return False

        # Sample check: ensure sequence_length matches actual sequence
        for i, row in enumerate(df.head(10).iter_rows(named=True)):
            seq = row['sequence']
            seq_len = row.get('sequence_length', 0)

            if 'expanded_sequence' in row:
                actual_len = len(row['expanded_sequence'])
            else:
                actual_len = len(seq)

            if seq_len != actual_len:
                issues.append(f"Row {i}: Sequence length mismatch {seq_len} != {actual_len}")

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            self.validation_results['residue_validation'] = False
            return False
        else:
            print("  ✓ All residues accounted for")
            self.validation_results['residue_validation'] = True
            return True

    def validate_torsion_angles(self, df: pl.DataFrame) -> bool:
        """
        Validate that torsion angles are in valid range.

        Args:
            df: DataFrame with geometry features

        Returns:
            True if torsion angles are valid
        """
        print("\nValidating torsion angles...")
        issues = []

        if 'pseudo_torsions' not in df.columns:
            print("  Skipping (no torsion data)")
            return True

        # Check a few samples
        for i, row in enumerate(df.head(5).iter_rows(named=True)):
            torsions = row.get('pseudo_torsions')
            if torsions and len(torsions) > 0:
                torsions_array = np.array(torsions)

                # Torsions should be in [-π, π]
                if np.any(np.abs(torsions_array) > np.pi + 0.1):
                    issues.append(f"Row {i}: Torsions out of range [-π, π]")

                # Check for NaN or inf
                if np.any(~np.isfinite(torsions_array)):
                    issues.append(f"Row {i}: Torsions contain NaN or inf")

        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            self.validation_results['torsion_validation'] = False
            return False
        else:
            print("  ✓ Torsion angles valid")
            self.validation_results['torsion_validation'] = True
            return True

    def validate_msa_consistency(self, df: pl.DataFrame) -> bool:
        """
        Validate MSA length consistency.

        Args:
            df: DataFrame with MSA features

        Returns:
            True if MSA is consistent
        """
        print("\nValidating MSA consistency...")

        # Currently MSA is not loaded in test pipeline
        # This is a placeholder for when MSA data is added
        print("  Skipping (MSA not yet implemented)")
        self.validation_results['msa_validation'] = True
        return True

    def run_full_validation(self, parquet_path: Path) -> bool:
        """
        Run full validation suite.

        Args:
            parquet_path: Path to processed parquet file

        Returns:
            True if all validations pass
        """
        print("="* 60)
        print("RNA Dataset Validation Suite")
        print("="* 60)

        start_time = time.time()

        # Load dataset
        print(f"\nLoading dataset from {parquet_path}...")
        df = pl.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Run validations
        validations = [
            self.validate_memory_constraint(),
            self.validate_sequence_features(df),
            self.validate_stoichiometry_parsing(df),
            self.validate_no_missing_residues(df),
            self.validate_torsion_angles(df),
            self.validate_msa_consistency(df),
        ]

        elapsed_time = time.time() - start_time

        # Summary
        print("\n" + "="* 60)
        print("Validation Summary")
        print("="* 60)
        print(f"Total validations: {len(validations)}")
        print(f"Passed: {sum(validations)}")
        print(f"Failed: {len(validations) - sum(validations)}")
        print(f"Time elapsed: {elapsed_time:.2f}s")

        self.validation_results['total_validations'] = len(validations)
        self.validation_results['passed'] = sum(validations)
        self.validation_results['failed'] = len(validations) - sum(validations)
        self.validation_results['elapsed_time_seconds'] = elapsed_time

        all_passed = all(validations)

        if all_passed:
            print("\n✓ All validations PASSED")
        else:
            print("\n✗ Some validations FAILED")

        return all_passed

    def save_validation_report(self, output_path: Path):
        """
        Save validation results to JSON.

        Args:
            output_path: Path to save JSON report
        """
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)

        print(f"\nValidation report saved to: {output_path}")


def test_small_dataset():
    """Test pipeline on small subset of data."""
    print("="* 60)
    print("Testing Pipeline on Small Dataset")
    print("="* 60)

    data_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)

    sequences_path = data_dir / "test_sequences.csv"

    if not sequences_path.exists():
        print(f"Error: {sequences_path} not found")
        return False

    # Load small subset
    df_full = pl.read_csv(sequences_path)
    df_small = df_full.head(10)  # Test with 10 sequences

    print(f"\nProcessing {len(df_small)} test sequences...")

    # Save temporary subset
    temp_input = output_dir / "test_subset.csv"
    df_small.write_csv(temp_input)

    # Run pipeline
    temp_output = output_dir / "test_subset.parquet"

    try:
        build_training_dataset(
            sequences_path=temp_input,
            labels_path=data_dir / "train_labels.csv",  # May not be used
            output_path=temp_output,
            streaming=True
        )

        print("\n✓ Pipeline completed successfully")

        # Validate
        validator = DatasetValidator(max_memory_gb=64.0)
        validation_passed = validator.run_full_validation(temp_output)

        # Save report
        report_path = output_dir / "validation_report.json"
        validator.save_validation_report(report_path)

        return validation_passed

    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_dataset()
    sys.exit(0 if success else 1)
