"""
End-to-end pipeline for one RNA target.
Tests: Model → JAX Refinement → OpenMM Validation → Submission Format

This validates all components work together before scaling to full dataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.rnafold_se3 import create_model, RNAFoldConfig
from physics.energy import energy_minimize, rna_energy
from validation.openmm_check import run_openmm_validation, validate_structure_batch
from features.sequence import (
    sequence_to_one_hot,
    expand_sequence_with_stoichiometry,
    parse_fasta_sequences,
)
from features.secondary import predict_per_chain_structure


class OneTargetPipeline:
    """Complete pipeline for single RNA target."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        config: Optional[RNAFoldConfig] = None
    ):
        """
        Initialize pipeline.

        Args:
            data_dir: Directory with input CSV files
            output_dir: Directory for outputs
            config: Model configuration
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config if config else RNAFoldConfig()
        self.model = None
        self.params = None

    def load_target_data(self, target_id: str) -> Dict:
        """
        Load data for specific target.

        Args:
            target_id: Target identifier

        Returns:
            Dictionary with target data
        """
        print(f"Loading data for target: {target_id}")

        # Load sequences
        sequences_path = self.data_dir / "test_sequences.csv"
        if not sequences_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {sequences_path}")

        df = pl.read_csv(sequences_path)
        target_row = df.filter(pl.col('target_id') == target_id)

        if len(target_row) == 0:
            raise ValueError(f"Target {target_id} not found in sequences")

        row = target_row.row(0, named=True)

        # Parse sequence and stoichiometry
        stoich = row.get('stoichiometry', '{A:1}')
        all_seqs = row.get('all_sequences', f">A\n{row['sequence']}")

        full_seq, boundaries = expand_sequence_with_stoichiometry(stoich, all_seqs)

        # Predict secondary structure
        struct_result = predict_per_chain_structure(full_seq, boundaries)

        # Load MSA if available
        msa_path = self.data_dir / "MSA" / f"{target_id}.MSA.fasta"
        msa_sequences = None

        if msa_path.exists():
            print(f"  Loading MSA from {msa_path}")
            with open(msa_path, 'r') as f:
                msa_content = f.read()

            # Parse MSA (simplified - would need per-chain parsing)
            msa_lines = [line for line in msa_content.split('\n') if not line.startswith('>')]
            msa_sequences = [seq.replace('-', '') for seq in msa_lines if seq.strip()]
            print(f"  Loaded {len(msa_sequences)} MSA sequences")
        else:
            print(f"  No MSA file found, using sequence only")

        return {
            'target_id': target_id,
            'sequence': full_seq,
            'chain_boundaries': boundaries,
            'stoichiometry': stoich,
            'structure': struct_result,
            'msa': msa_sequences,
            'description': row.get('description', ''),
        }

    def initialize_model(self, sequence_length: int):
        """
        Initialize model with random weights.

        Args:
            sequence_length: Length of target sequence
        """
        print("Initializing model...")

        self.model = create_model(self.config)

        # Create dummy input for initialization
        rng = jax.random.PRNGKey(42)
        dummy_sequence = jax.random.normal(rng, (sequence_length, self.config.vocab_size))

        # Initialize parameters
        rng, key = jax.random.split(rng)
        self.params = self.model.init(key, dummy_sequence)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"  Model initialized with {num_params:,} parameters")

    def generate_predictions(
        self,
        sequence: str,
        msa: Optional[List[str]] = None,
        num_predictions: int = 5,
        temperatures: Optional[List[float]] = None
    ) -> List[np.ndarray]:
        """
        Generate multiple structure predictions.

        Args:
            sequence: RNA sequence
            msa: MSA sequences
            num_predictions: Number of predictions to generate
            temperatures: Temperature values for diversity

        Returns:
            List of coordinate arrays
        """
        if temperatures is None:
            temperatures = [0.5, 0.8, 1.0, 1.2, 1.5][:num_predictions]

        print(f"\nGenerating {num_predictions} predictions...")

        # Convert sequence to one-hot
        one_hot = sequence_to_one_hot(sequence)
        one_hot_jax = jnp.array(one_hot)

        # Convert MSA if provided
        msa_jax = None
        if msa is not None and len(msa) > 0:
            # Sample MSA to max depth
            msa_sampled = msa[:self.config.max_msa_depth]

            # Pad sequences to same length
            max_len = max(len(s) for s in msa_sampled)
            msa_one_hot = []

            for seq in msa_sampled:
                oh = sequence_to_one_hot(seq)
                if len(oh) < max_len:
                    padding = np.zeros((max_len - len(oh), oh.shape[1]))
                    oh = np.vstack([oh, padding])
                msa_one_hot.append(oh)

            msa_jax = jnp.array(msa_one_hot)

        # Generate predictions with different temperatures
        predictions = []
        rng = jax.random.PRNGKey(0)

        for i, temp in enumerate(temperatures):
            print(f"  Prediction {i + 1}/{num_predictions} (temp={temp:.2f})...")

            rng, key = jax.random.split(rng)

            # Forward pass
            coords = self.model.apply(
                self.params,
                key,
                one_hot_jax,
                msa_jax,
                temp
            )

            predictions.append(np.array(coords))

        return predictions

    def refine_with_jax(
        self,
        coords: np.ndarray,
        sequence: str,
        pairing_matrix: Optional[np.ndarray] = None,
        num_steps: int = 100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Refine structure with JAX energy minimization.

        Args:
            coords: Initial coordinates
            sequence: RNA sequence
            pairing_matrix: Base-pairing matrix
            num_steps: Optimization steps

        Returns:
            (refined_coords, optimization_info)
        """
        print(f"    JAX refinement ({num_steps} steps)...")

        coords_jax = jnp.array(coords)

        refined_coords, info = energy_minimize(
            coords_jax,
            sequence=sequence,
            pairing_matrix=pairing_matrix,
            num_steps=num_steps,
            learning_rate=0.01,
            verbose=False
        )

        print(f"      Energy: {info['energies'][0]:.2f} → {info['final_energy']:.2f}")

        return np.array(refined_coords), info

    def validate_with_openmm(
        self,
        coords: np.ndarray,
        sequence: str,
        simulation_steps: int = 100
    ) -> Dict:
        """
        Validate structure with OpenMM.

        Args:
            coords: Coordinates to validate
            sequence: RNA sequence
            simulation_steps: MD simulation steps

        Returns:
            Validation result dictionary
        """
        print(f"    OpenMM validation...")

        result = run_openmm_validation(
            coords,
            sequence,
            simulation_steps=simulation_steps,
            rmsd_threshold=2.0
        )

        status = "✓ PASS" if result['is_valid'] else "✗ FAIL"
        print(f"      {status} - RMSD drift: {result['rmsd_drift']:.3f} Å")

        return result

    def format_submission(
        self,
        target_id: str,
        predictions: List[np.ndarray],
        sequence: str,
        output_path: Path
    ):
        """
        Format predictions in Kaggle submission format.

        Args:
            target_id: Target identifier
            predictions: List of coordinate arrays (up to 5)
            sequence: RNA sequence
            output_path: Output CSV path
        """
        print(f"\nFormatting submission for {target_id}...")

        L = len(sequence)
        num_preds = min(len(predictions), 5)

        # Clip coordinates
        predictions_clipped = [
            np.clip(pred, -999.999, 9999.999) for pred in predictions[:num_preds]
        ]

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['ID', 'resname', 'resid']
            for i in range(1, num_preds + 1):
                header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
            writer.writerow(header)

            # Rows (one per residue)
            for resid in range(1, L + 1):
                row_id = f"{target_id}_{resid}"
                resname = sequence[resid - 1]

                row = [row_id, resname, resid]

                for pred in predictions_clipped:
                    coord = pred[resid - 1]
                    row.extend([f"{coord[0]:.3f}", f"{coord[1]:.3f}", f"{coord[2]:.3f}"])

                writer.writerow(row)

        print(f"  Saved to {output_path}")
        print(f"  {L} residues × {num_preds} predictions")

    def run_full_pipeline(self, target_id: str) -> Dict:
        """
        Run complete pipeline for one target.

        Args:
            target_id: Target identifier

        Returns:
            Dictionary with results
        """
        print("="* 70)
        print(f"ONE-TARGET PIPELINE: {target_id}")
        print("="* 70)

        # 1. Load data
        target_data = self.load_target_data(target_id)
        sequence = target_data['sequence']
        L = len(sequence)

        print(f"\nTarget info:")
        print(f"  Sequence length: {L}")
        print(f"  Stoichiometry: {target_data['stoichiometry']}")
        print(f"  Chains: {len(target_data['chain_boundaries'])}")
        print(f"  Has MSA: {target_data['msa'] is not None}")

        # 2. Initialize model
        self.initialize_model(L)

        # 3. Generate predictions
        raw_predictions = self.generate_predictions(
            sequence,
            msa=target_data['msa'],
            num_predictions=5
        )

        # 4. Refine and validate each prediction
        print(f"\nRefining and validating predictions...")

        refined_predictions = []
        validation_results = []

        pairing_matrix = target_data['structure']['bpp_matrix']
        pairing_matrix = np.array(pairing_matrix) if pairing_matrix is not None else None

        for i, coords in enumerate(raw_predictions):
            print(f"\n  Prediction {i + 1}/5:")

            # JAX refinement
            refined_coords, opt_info = self.refine_with_jax(
                coords,
                sequence,
                pairing_matrix=pairing_matrix,
                num_steps=100
            )

            # OpenMM validation
            val_result = self.validate_with_openmm(
                refined_coords,
                sequence,
                simulation_steps=100
            )

            refined_predictions.append(refined_coords)
            validation_results.append(val_result)

        # 5. Summary
        num_valid = sum(1 for r in validation_results if r['is_valid'])

        print(f"\n" + "="* 70)
        print(f"VALIDATION SUMMARY")
        print(f"="* 70)
        print(f"Predictions generated: 5")
        print(f"Predictions passed: {num_valid}/5")

        if num_valid < 5:
            print(f"\n⚠ Warning: {5 - num_valid} predictions failed validation")
            for i, r in enumerate(validation_results):
                if not r['is_valid']:
                    print(f"  Prediction {i + 1}: {r['message']}")

        # 6. Format output
        output_path = self.output_dir / f"{target_id}_submission.csv"
        self.format_submission(
            target_id,
            refined_predictions,
            sequence,
            output_path
        )

        print(f"\n✓ Pipeline complete for {target_id}")

        return {
            'target_id': target_id,
            'sequence_length': L,
            'num_predictions': 5,
            'num_valid': num_valid,
            'validation_results': validation_results,
            'output_path': str(output_path),
        }


def main():
    """Main entry point."""
    print("="* 70)
    print("RNA 3D FOLDING - ONE TARGET PROTOTYPE")
    print("="* 70)

    # Setup paths
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir = Path(__file__).parent.parent / "data" / "prototype_output"

    # Use first target from test sequences
    sequences_path = data_dir / "test_sequences.csv"

    if not sequences_path.exists():
        print(f"Error: {sequences_path} not found")
        print("Please ensure test_sequences.csv is in data/raw/")
        return

    # Get first target
    df = pl.read_csv(sequences_path)
    target_id = df['target_id'][0]

    print(f"\nUsing target: {target_id}")

    # Create pipeline
    pipeline = OneTargetPipeline(data_dir, output_dir)

    # Run
    try:
        result = pipeline.run_full_pipeline(target_id)

        print(f"\n" + "="* 70)
        print("SUCCESS")
        print("="* 70)
        print(f"Output saved to: {result['output_path']}")
        print(f"Valid predictions: {result['num_valid']}/5")

        return result

    except Exception as e:
        print(f"\n" + "="* 70)
        print("ERROR")
        print("="* 70)
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Enable JAX debugging
    import os
    os.environ['JAX_DEBUG_NANS'] = '1'

    result = main()

    if result and result['num_valid'] == 5:
        print("\n🎉 All components working! Ready to scale.")
        sys.exit(0)
    else:
        print("\n⚠ Some validations failed. Review and fix before scaling.")
        sys.exit(1)
