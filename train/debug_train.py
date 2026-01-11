"""
Debug training script: Train on 10 easy targets to verify learning.
Critical first test before scaling to full curriculum.
"""

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, List
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from model.rnafold_se3_full import create_full_model, FullRNAFoldConfig
from train.loss import fape_loss, distogram_loss, multi_structure_loss
from features.sequence import sequence_to_one_hot
from validation.openmm_check import run_openmm_validation
from utils.tm_score import tm_score


# Removed approximate TM-score - now using proper Kabsch-aligned tm_score from utils


class DebugTrainer:
    """Minimal trainer for debugging."""

    def __init__(
        self,
        config: FullRNAFoldConfig,
        learning_rate: float = 1e-3,
        log_every: int = 1
    ):
        self.config = config
        self.learning_rate = learning_rate
        self.log_every = log_every

        # Create model
        self.model = create_full_model(config)

        # Create optimizer
        self.optimizer = optax.adam(learning_rate)

        self.params = None
        self.opt_state = None

    def initialize(self, sample_sequence_length: int):
        """Initialize model parameters."""
        print(f"\nInitializing model...")

        rng = jax.random.PRNGKey(42)
        dummy_seq = jax.random.normal(rng, (sample_sequence_length, self.config.vocab_size))

        # Init params
        rng, key = jax.random.split(rng)
        self.params = self.model.init(key, dummy_seq)

        # Init optimizer state
        self.opt_state = self.optimizer.init(self.params)

        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"  Parameters: {num_params:,}")

    def train_step(
        self,
        params,
        opt_state,
        rng_key,
        sequence,
        true_coords
    ):
        """Single training step."""
        # Create JIT-compiled step function (without self)
        def _step(p, opt_st, rng, seq, true_c):
            def loss_fn(params_inner):
                # Forward pass
                pred_coords = self.model.apply(params_inner, rng, seq)

                # Compute loss (simple FAPE for speed)
                loss = fape_loss(pred_coords, true_c)

                return loss, pred_coords

            # Compute loss and gradients
            (loss, pred_coords), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)

            # Update parameters
            updates, new_opt_state = self.optimizer.update(grads, opt_st)
            new_params = optax.apply_updates(p, updates)

            return new_params, new_opt_state, loss, pred_coords

        # Call the function (not JIT for now to avoid compilation issues)
        return _step(params, opt_state, rng_key, sequence, true_coords)

    def train_epoch(
        self,
        training_data: List[Dict],
        epoch: int,
        rng_key
    ) -> Dict:
        """Train for one epoch."""
        epoch_losses = []
        first_pred = None
        first_true = None

        for i, item in enumerate(training_data):
            # Prepare input
            sequence_str = item['sequence']
            one_hot = sequence_to_one_hot(sequence_str)
            one_hot_jax = jnp.array(one_hot)

            # Get first ground truth structure
            true_coords_np = item['structures'][0]
            true_coords = jnp.array(true_coords_np)

            # Training step
            rng_key, step_key = jax.random.split(rng_key)

            self.params, self.opt_state, loss, pred_coords = self.train_step(
                self.params,
                self.opt_state,
                step_key,
                one_hot_jax,
                true_coords
            )

            epoch_losses.append(float(loss))

            # Store first prediction for diagnostics
            if i == 0 and epoch == 1:
                first_pred = np.array(pred_coords)
                first_true = true_coords_np

        metrics = {
            'epoch': epoch,
            'train_loss': np.mean(epoch_losses),
            'train_loss_std': np.std(epoch_losses),
            'first_pred': first_pred,
            'first_true': first_true,
        }

        return metrics

    def validate(
        self,
        validation_data: List[Dict],
        rng_key
    ) -> Dict:
        """Validate on validation set."""
        val_losses = []
        tm_scores = []
        openmm_passes = []

        for item in validation_data[:5]:  # Validate on first 5 only
            sequence_str = item['sequence']
            one_hot = sequence_to_one_hot(sequence_str)
            one_hot_jax = jnp.array(one_hot)

            true_coords_np = item['structures'][0]
            true_coords = jnp.array(true_coords_np)

            # Forward pass
            rng_key, step_key = jax.random.split(rng_key)
            pred_coords = self.model.apply(self.params, step_key, one_hot_jax)

            # Compute metrics
            loss = fape_loss(pred_coords, true_coords)
            val_losses.append(float(loss))

            # TM-score with proper Kabsch alignment
            tm = tm_score(
                np.array(pred_coords),
                true_coords_np
            )
            tm_scores.append(tm)

            # OpenMM validation (on first target only to save time)
            if len(openmm_passes) == 0:
                openmm_result = run_openmm_validation(
                    np.array(pred_coords),
                    sequence_str,
                    simulation_steps=50  # Reduced for speed
                )
                openmm_passes.append(1 if openmm_result['is_valid'] else 0)

        metrics = {
            'val_loss': np.mean(val_losses),
            'val_tm_score': np.mean(tm_scores),
            'openmm_pass_rate': np.mean(openmm_passes) if openmm_passes else 0.0,
        }

        return metrics


def load_debug_dataset(dataset_path: Path, num_targets: int = 10, max_length: int = 40):
    """Load small subset for debugging."""
    print(f"\nLoading debug dataset...")
    print(f"  Max targets: {num_targets}")
    print(f"  Max length: {max_length}")

    with open(dataset_path, 'rb') as f:
        all_data = pickle.load(f)

    # Filter to easy targets
    filtered = [
        item for item in all_data
        if item['sequence_length'] <= max_length
        and item['num_structures'] >= 1
    ]

    # Split into train/val (80/20 split)
    num_val = max(3, len(filtered) // 5)  # At least 3 for validation
    debug_data = filtered[:num_targets]
    val_data = filtered[num_targets:num_targets+num_val]

    print(f"  Training targets: {len(debug_data)}")
    print(f"  Validation targets: {len(val_data)}")
    print(f"  Avg length: {np.mean([d['sequence_length'] for d in debug_data]):.1f}")

    return debug_data, val_data


def main(args):
    """Main training loop."""
    print("="* 70)
    print("DEBUG TRAINING - 10 Easy Targets")
    print("="* 70)

    # Configure model (reduced for debugging)
    config = FullRNAFoldConfig()
    config.num_evoformer_blocks = args.evoformer_blocks
    config.num_ipa_blocks = 2  # Reduced
    config.max_msa_depth = 64  # Reduced
    config.use_bfloat16 = True

    print(f"\nConfiguration:")
    print(f"  Evoformer blocks: {config.num_evoformer_blocks}")
    print(f"  IPA blocks: {config.num_ipa_blocks}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Load data
    dataset_path = Path(args.data_dir) / "processed" / "train_structures.pkl"

    if not dataset_path.exists():
        print(f"\nError: {dataset_path} not found")
        print("Run: python train/prepare_labels.py first")
        return

    train_data, val_data = load_debug_dataset(
        dataset_path,
        num_targets=args.num_targets,
        max_length=args.max_length
    )

    # Initialize trainer
    trainer = DebugTrainer(
        config,
        learning_rate=args.lr,
        log_every=args.log_every
    )

    # Initialize with sample length
    sample_length = train_data[0]['sequence_length']
    trainer.initialize(sample_length)

    # Training loop
    print(f"\n{'='*70}")
    print("Training Loop")
    print(f"{'='*70}")

    rng = jax.random.PRNGKey(42)
    best_tm = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_data, epoch, rng)
        rng, _ = jax.random.split(rng)

        # Diagnostic output after epoch 1
        if epoch == 1 and train_metrics.get('first_pred') is not None:
            pred = train_metrics['first_pred']
            true = train_metrics['first_true']
            print(f"\n{'='*70}")
            print("EPOCH 1 DIAGNOSTICS")
            print(f"{'='*70}")
            print(f"Prediction shape: {pred.shape}")
            print(f"Ground truth shape: {true.shape}")
            print(f"\nPrediction stats:")
            print(f"  Min: {pred.min():.3f}, Max: {pred.max():.3f}")
            print(f"  Mean: {pred.mean():.3f}, Std: {pred.std():.3f}")
            print(f"\nGround truth stats:")
            print(f"  Min: {true.min():.3f}, Max: {true.max():.3f}")
            print(f"  Mean: {true.mean():.3f}, Std: {true.std():.3f}")
            print(f"\nFirst 5 predicted coords:")
            print(pred[:5])
            print(f"\nFirst 5 true coords:")
            print(true[:5])
            print(f"{'='*70}\n")

        # Validate
        val_metrics = trainer.validate(val_data, rng)
        rng, _ = jax.random.split(rng)

        epoch_time = time.time() - start_time

        # Log
        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train loss: {train_metrics['train_loss']:.4f} ± {train_metrics['train_loss_std']:.4f}")
            print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val TM-score: {val_metrics['val_tm_score']:.4f}")
            print(f"  OpenMM pass: {val_metrics['openmm_pass_rate']:.1%}")

        # Track best
        if val_metrics['val_tm_score'] > best_tm:
            best_tm = val_metrics['val_tm_score']
            print(f"  ✓ New best TM-score: {best_tm:.4f}")

            # Save checkpoint
            if args.save_checkpoint:
                checkpoint_dir = Path(args.data_dir) / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = checkpoint_dir / "debug_best.pkl"

                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(trainer.params, f)

                print(f"  Saved checkpoint: {checkpoint_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"Best TM-score: {best_tm:.4f}")

    # Success criteria check
    success = best_tm >= 0.60

    if success:
        print(f"\n✓ SUCCESS: TM-score ≥ 0.60 achieved!")
        print(f"  → Ready to scale to full Stage 1 training")
    else:
        print(f"\n✗ FAILED: TM-score < 0.60")
        print(f"  → Debug required before scaling")
        print(f"\nTroubleshooting:")
        print(f"  - Check if loss is decreasing")
        print(f"  - Verify gradients are flowing (not NaN)")
        print(f"  - Try higher learning rate (2e-3)")
        print(f"  - Increase epochs to 20")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug training on 10 easy targets")

    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--num_targets", type=int, default=10,
                        help="Number of training targets")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum sequence length")
    parser.add_argument("--evoformer_blocks", type=int, default=8,
                        help="Number of Evoformer blocks")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (currently processes one at a time)")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Learning rate")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Log every N epochs")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save best checkpoint")

    args = parser.parse_args()

    # Run
    success = main(args)

    sys.exit(0 if success else 1)
