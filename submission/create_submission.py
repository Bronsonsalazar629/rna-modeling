"""
Create Kaggle submission using template-based folding.
Generates 5 diverse predictions per target.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))
from inference.template_based_folding import predict_structure_template_based


def create_submission(
    test_sequences_path: str,
    template_library_path: str,
    output_path: str,
    num_predictions: int = 5
):
    """
    Create submission file for Kaggle competition.

    Args:
        test_sequences_path: Path to test_sequences.csv
        template_library_path: Path to template library pickle
        output_path: Path to save submission.csv
        num_predictions: Number of predictions per target (must be 5)
    """
    print("="*70)
    print("Creating Kaggle Submission - Template-Based Folding")
    print("="*70)

    # Load test sequences
    print(f"\nLoading test sequences from {test_sequences_path}...")
    test_df = pd.read_csv(test_sequences_path)
    print(f"  Found {len(test_df)} test targets")
    print(f"  Columns: {list(test_df.columns)}")

    # Load template library
    print(f"\nLoading template library from {template_library_path}...")
    with open(template_library_path, 'rb') as f:
        template_library = pickle.load(f)
    print(f"  Loaded {len(template_library)} templates")

    # Prepare submission rows
    submission_rows = []

    print(f"\nGenerating predictions...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        target_id = row['target_id']  # Changed from 'ID' to 'target_id'
        sequence = row['sequence']

        # Predict structures
        predictions = predict_structure_template_based(
            sequence,
            template_library,
            num_predictions=num_predictions
        )

        # Convert to submission format
        for res_idx, nucleotide in enumerate(sequence):
            resid = res_idx + 1
            row_id = f"{target_id}_{resid}"

            # Collect coordinates from all 5 predictions
            coords_dict = {'ID': row_id, 'resname': nucleotide, 'resid': resid}

            for pred_num in range(num_predictions):
                pred_coords = predictions[pred_num][res_idx]
                coords_dict[f'x_{pred_num+1}'] = pred_coords[0]
                coords_dict[f'y_{pred_num+1}'] = pred_coords[1]
                coords_dict[f'z_{pred_num+1}'] = pred_coords[2]

            submission_rows.append(coords_dict)

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order matches expected format
    columns = ['ID', 'resname', 'resid']
    for i in range(1, num_predictions + 1):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    submission_df = submission_df[columns]

    # Save submission
    print(f"\nSaving submission to {output_path}...")
    submission_df.to_csv(output_path, index=False)

    print(f"\n{'='*70}")
    print("Submission created successfully!")
    print(f"{'='*70}")
    print(f"  Total rows: {len(submission_df):,}")
    print(f"  Targets: {len(test_df)}")
    print(f"  Predictions per target: {num_predictions}")
    print(f"\nSubmission file: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Kaggle submission using template-based folding"
    )
    parser.add_argument(
        "--test_sequences",
        type=str,
        default="data/test_sequences.csv",
        help="Path to test_sequences.csv"
    )
    parser.add_argument(
        "--template_library",
        type=str,
        default="data/processed/train_structures.pkl",
        help="Path to template library pickle"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output submission file path"
    )
    parser.add_argument(
        "--num_predictions",
        type=int,
        default=5,
        help="Number of predictions per target"
    )

    args = parser.parse_args()

    create_submission(
        args.test_sequences,
        args.template_library,
        args.output,
        args.num_predictions
    )
