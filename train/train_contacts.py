"""
Training script for RNA contact prediction.
This is a simpler 2D problem that provides foundation for 3D folding.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import time
from pathlib import Path
from typing import Dict, List
import argparse
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.contact_prediction import create_contact_model, FullRNAFoldConfig
from features.sequence import sequence_to_one_hot
from physics.contact_to_3d import contacts_to_3d
from utils.tm_score import tm_score


def extract_contact_map_from_coords(coords: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    """
    Extract contact map from 3D coordinates.

    Args:
        coords: (L, 3) C1' coordinates
        threshold: Distance threshold for contacts (Angstroms)

    Returns:
        contacts: (L, L) binary contact matrix
    """
    L = len(coords)
    contacts = np.zeros((L, L))

    for i in range(L):
        for j in range(i+3, L):  # Skip local contacts
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contacts[i, j] = 1.0
                contacts[j, i] = 1.0

    return contacts


class ContactTrainer:
    """Trainer for contact prediction model."""

    def __init__(self, config: FullRNAFoldConfig, learning_rate: float = 1e-3):
        self.config = config
        self.model = create_contact_model(config)

        # Optimizer
        self.optimizer = optax.adam(learning_rate)

        # Initialize
        self.params = None
        self.opt_state = None

    def initialize(self, sequence_length: int, rng_key):
        """Initialize model parameters."""
        dummy_seq = jnp.zeros((sequence_length, 5))
        self.params = self.model.init(rng_key, dummy_seq)
        self.opt_state = self.optimizer.init(self.params)

        # Count parameters
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"  Parameters: {num_params:,}")

    def focal_loss(self, pred_logits, target, alpha=0.75, gamma=2.0):
        """
        Focal loss for highly imbalanced contact prediction.

        Args:
            pred_logits: Raw logits before sigmoid
            target: Ground truth contacts (0 or 1)
            alpha: Weight for positive class (contacts)
            gamma: Focusing parameter (hard example mining)

        Returns:
            Focal loss value
        """
        pred_sigmoid = jax.nn.sigmoid(pred_logits)

        # Positive examples (contacts)
        pos_loss = -alpha * (1 - pred_sigmoid) ** gamma * jnp.log(pred_sigmoid + 1e-8)
        # Negative examples (non-contacts)
        neg_loss = -(1 - alpha) * pred_sigmoid ** gamma * jnp.log(1 - pred_sigmoid + 1e-8)

        loss = jnp.where(target == 1, pos_loss, neg_loss)

        # Weight by sequence separation (long-range contacts are harder)
        L = target.shape[0]
        i_idx = jnp.arange(L)[:, None]
        j_idx = jnp.arange(L)[None, :]
        seq_sep = jnp.abs(i_idx - j_idx)
        sep_weight = jnp.where(seq_sep > 10, 2.0, 1.0)  # Long-range gets 2x weight

        weighted_loss = loss * sep_weight

        return jnp.mean(weighted_loss)

    def train_step(self, params, opt_state, rng_key, sequence, true_contacts):
        """Single training step."""
        def _step(p, opt_st, rng, seq, true_c):
            def loss_fn(params_inner):
                pred_contacts = self.model.apply(params_inner, rng, seq)

                # Convert probabilities back to logits for focal loss
                epsilon = 1e-7
                pred_clipped = jnp.clip(pred_contacts, epsilon, 1.0 - epsilon)
                pred_logits = jnp.log(pred_clipped / (1 - pred_clipped))

                # Focal loss
                loss = self.focal_loss(pred_logits, true_c, alpha=0.75, gamma=2.0)

                return loss, pred_contacts

            (loss, pred_contacts), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, new_opt_state = self.optimizer.update(grads, opt_st)
            new_params = optax.apply_updates(p, updates)

            return new_params, new_opt_state, loss, pred_contacts

        return _step(params, opt_state, rng_key, sequence, true_contacts)

    def train_epoch(self, training_data: List[Dict], epoch: int, rng_key) -> Dict:
        """Train for one epoch."""
        epoch_losses = []
        epoch_precisions = []
        epoch_recalls = []

        for i, item in enumerate(training_data):
            # Prepare input
            sequence_str = item['sequence']
            one_hot = sequence_to_one_hot(sequence_str)
            one_hot_jax = jnp.array(one_hot)

            # Extract ground truth contacts from 3D coords
            true_coords = item['structures'][0]
            true_contacts_np = extract_contact_map_from_coords(true_coords, threshold=8.0)
            true_contacts = jnp.array(true_contacts_np)

            # Training step
            rng_key, step_key = jax.random.split(rng_key)

            self.params, self.opt_state, loss, pred_contacts = self.train_step(
                self.params,
                self.opt_state,
                step_key,
                one_hot_jax,
                true_contacts
            )

            epoch_losses.append(float(loss))

            # Compute precision/recall
            pred_binary = (np.array(pred_contacts) > 0.5).astype(float)
            true_binary = true_contacts_np

            tp = np.sum((pred_binary == 1) & (true_binary == 1))
            fp = np.sum((pred_binary == 1) & (true_binary == 0))
            fn = np.sum((pred_binary == 0) & (true_binary == 1))

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)

            epoch_precisions.append(precision)
            epoch_recalls.append(recall)

        metrics = {
            'epoch': epoch,
            'train_loss': np.mean(epoch_losses),
            'train_precision': np.mean(epoch_precisions),
            'train_recall': np.mean(epoch_recalls),
        }

        return metrics

    def validate(self, validation_data: List[Dict], rng_key) -> Dict:
        """Validate model."""
        val_losses = []
        val_precisions = []
        val_recalls = []
        tm_scores = []

        for item in validation_data:
            # Prepare input
            sequence_str = item['sequence']
            one_hot = sequence_to_one_hot(sequence_str)
            one_hot_jax = jnp.array(one_hot)

            # Ground truth
            true_coords = item['structures'][0]
            true_contacts_np = extract_contact_map_from_coords(true_coords, threshold=8.0)
            true_contacts = jnp.array(true_contacts_np)

            # Forward pass
            rng_key, step_key = jax.random.split(rng_key)
            pred_contacts = self.model.apply(self.params, step_key, one_hot_jax)

            # Loss
            epsilon = 1e-7
            pred_clipped = jnp.clip(pred_contacts, epsilon, 1.0 - epsilon)
            bce = -(true_contacts * jnp.log(pred_clipped) + (1 - true_contacts) * jnp.log(1 - pred_clipped))
            loss = float(jnp.mean(bce))
            val_losses.append(loss)

            # Metrics
            pred_binary = (np.array(pred_contacts) > 0.5).astype(float)
            true_binary = true_contacts_np

            tp = np.sum((pred_binary == 1) & (true_binary == 1))
            fp = np.sum((pred_binary == 1) & (true_binary == 0))
            fn = np.sum((pred_binary == 0) & (true_binary == 1))

            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)

            val_precisions.append(precision)
            val_recalls.append(recall)

            # Convert contacts to 3D and compute TM-score
            try:
                pred_coords = contacts_to_3d(np.array(pred_contacts), sequence_str)
                tm = tm_score(pred_coords, true_coords)
                tm_scores.append(tm)
            except:
                tm_scores.append(0.0)

        metrics = {
            'val_loss': np.mean(val_losses),
            'val_precision': np.mean(val_precisions),
            'val_recall': np.mean(val_recalls),
            'val_tm_score': np.mean(tm_scores),
        }

        return metrics


def load_dataset(dataset_path: Path, num_targets: int = 50, max_length: int = 40):
    """Load dataset for contact prediction."""
    print(f"\nLoading dataset...")

    with open(dataset_path, 'rb') as f:
        all_data = pickle.load(f)

    # Filter to easy targets
    filtered = [
        item for item in all_data
        if item['sequence_length'] <= max_length
        and item['num_structures'] >= 1
    ]

    # Split train/val
    train_data = filtered[:num_targets]
    val_data = filtered[num_targets:num_targets+10]

    print(f"  Training targets: {len(train_data)}")
    print(f"  Validation targets: {len(val_data)}")
    print(f"  Avg length: {np.mean([d['sequence_length'] for d in train_data]):.1f}")

    return train_data, val_data


def main(args):
    """Main training loop."""
    print("="* 70)
    print("CONTACT PREDICTION TRAINING")
    print("="* 70)

    # Configure model
    config = FullRNAFoldConfig()
    config.num_evoformer_blocks = 8  # Lighter model for contact prediction
    config.use_bfloat16 = False  # Use float32 for better precision

    print(f"\nConfiguration:")
    print(f"  Evoformer blocks: {config.num_evoformer_blocks}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")

    # Load data
    dataset_path = Path(args.data_dir) / "processed" / "train_structures.pkl"

    if not dataset_path.exists():
        print(f"\nError: {dataset_path} not found")
        return False

    train_data, val_data = load_dataset(
        dataset_path,
        num_targets=args.num_targets,
        max_length=args.max_length
    )

    # Initialize trainer
    print(f"\nInitializing model...")
    trainer = ContactTrainer(config, learning_rate=args.lr)

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    trainer.initialize(args.max_length, init_rng)

    # Training loop
    print(f"\n{'='*70}")
    print("Training Loop")
    print("="* 70)

    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_data, epoch, rng)
        rng, _ = jax.random.split(rng)

        # Validate
        val_metrics = trainer.validate(val_data, rng)
        rng, _ = jax.random.split(rng)

        epoch_time = time.time() - start_time

        # Compute F1 scores
        train_f1 = 2 * (train_metrics['train_precision'] * train_metrics['train_recall']) / \
                   (train_metrics['train_precision'] + train_metrics['train_recall'] + 1e-7)
        val_f1 = 2 * (val_metrics['val_precision'] * val_metrics['val_recall']) / \
                 (val_metrics['val_precision'] + val_metrics['val_recall'] + 1e-7)

        # Log
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_metrics['train_loss']:.4f}, "
              f"P={train_metrics['train_precision']:.3f}, "
              f"R={train_metrics['train_recall']:.3f}, "
              f"F1={train_f1:.3f}")
        print(f"  Val:   Loss={val_metrics['val_loss']:.4f}, "
              f"P={val_metrics['val_precision']:.3f}, "
              f"R={val_metrics['val_recall']:.3f}, "
              f"F1={val_f1:.3f}, "
              f"TM={val_metrics['val_tm_score']:.4f}")

        # Track best
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"  New best F1: {best_f1:.3f}")

            if args.save_checkpoint:
                checkpoint_dir = Path(args.data_dir) / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_path = checkpoint_dir / "contact_best.pkl"

                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(trainer.params, f)

                print(f"  Saved checkpoint: {checkpoint_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("Training Complete")
    print(f"{'='*70}")
    print(f"Best F1 score: {best_f1:.3f}")

    success = best_f1 >= 0.5
    if success:
        print(f"\nSUCCESS: Contact prediction F1 >= 0.50")
    else:
        print(f"\nNeed improvement: F1 < 0.50")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNA contact prediction model")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--num_targets", type=int, default=50,
                        help="Number of training targets")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Save best checkpoint")

    args = parser.parse_args()

    success = main(args)
    sys.exit(0 if success else 1)
