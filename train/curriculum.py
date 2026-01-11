"""
Curriculum learning scheduler for RNA structure prediction.
Stages: Easy (short, no pseudoknots) → Medium → Hard (long, complex)
"""

import polars as pl
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class CurriculumStage:
    """Configuration for one curriculum stage."""
    stage_id: int
    name: str
    max_length: int
    min_length: int = 5
    allow_pseudoknots: bool = True
    max_samples: Optional[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 8


# Define curriculum stages
CURRICULUM_STAGES = [
    CurriculumStage(
        stage_id=1,
        name="Easy",
        max_length=50,
        min_length=10,
        allow_pseudoknots=False,
        max_samples=1000,
        learning_rate=1e-3,
        batch_size=16,
    ),
    CurriculumStage(
        stage_id=2,
        name="Medium",
        max_length=150,
        min_length=30,
        allow_pseudoknots=True,
        max_samples=5000,
        learning_rate=5e-4,
        batch_size=8,
    ),
    CurriculumStage(
        stage_id=3,
        name="Hard",
        max_length=500,
        min_length=100,
        allow_pseudoknots=True,
        max_samples=None,  # Full dataset
        learning_rate=1e-4,
        batch_size=4,
    ),
    CurriculumStage(
        stage_id=4,
        name="Full",
        max_length=2048,
        min_length=5,
        allow_pseudoknots=True,
        max_samples=None,
        learning_rate=5e-5,
        batch_size=2,
    ),
]


def filter_by_length_and_complexity(
    df: pl.DataFrame,
    max_len: int,
    min_len: int = 5,
    allow_pseudoknots: bool = True
) -> pl.DataFrame:
    """
    Filter dataset by sequence length and complexity.

    Args:
        df: DataFrame with processed data
        max_len: Maximum sequence length
        min_len: Minimum sequence length
        allow_pseudoknots: Whether to allow pseudoknots

    Returns:
        Filtered DataFrame
    """
    # Filter by length
    filtered = df.filter(
        (pl.col('sequence_length') >= min_len) &
        (pl.col('sequence_length') <= max_len)
    )

    # Filter by pseudoknots if not allowed
    if not allow_pseudoknots and 'has_pseudoknots' in df.columns:
        filtered = filtered.filter(~pl.col('has_pseudoknots'))

    return filtered


def get_curriculum_dataloader(
    dataset_path: Path,
    stage: int,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[pl.DataFrame, CurriculumStage]:
    """
    Get dataloader for specific curriculum stage.

    Args:
        dataset_path: Path to processed parquet file
        stage: Stage ID (1-4)
        shuffle: Whether to shuffle data
        seed: Random seed

    Returns:
        (filtered_dataframe, stage_config)
    """
    if stage < 1 or stage > len(CURRICULUM_STAGES):
        raise ValueError(f"Invalid stage {stage}. Must be 1-{len(CURRICULUM_STAGES)}")

    stage_config = CURRICULUM_STAGES[stage - 1]

    print(f"\n{'='*70}")
    print(f"Curriculum Stage {stage}: {stage_config.name}")
    print(f"{'='*70}")
    print(f"  Length range: {stage_config.min_length}-{stage_config.max_length} nt")
    print(f"  Pseudoknots: {'Allowed' if stage_config.allow_pseudoknots else 'Excluded'}")
    print(f"  Batch size: {stage_config.batch_size}")
    print(f"  Learning rate: {stage_config.learning_rate}")

    # Load full dataset
    print(f"\nLoading dataset from {dataset_path}...")
    df = pl.read_parquet(dataset_path)
    print(f"  Total samples: {len(df)}")

    # Filter
    filtered_df = filter_by_length_and_complexity(
        df,
        max_len=stage_config.max_length,
        min_len=stage_config.min_length,
        allow_pseudoknots=stage_config.allow_pseudoknots
    )
    print(f"  Filtered samples: {len(filtered_df)}")

    # Sample if needed
    if stage_config.max_samples and len(filtered_df) > stage_config.max_samples:
        if shuffle:
            filtered_df = filtered_df.sample(
                n=stage_config.max_samples,
                seed=seed,
                shuffle=True
            )
        else:
            filtered_df = filtered_df.head(stage_config.max_samples)
        print(f"  Sampled to: {len(filtered_df)}")
    elif shuffle:
        filtered_df = filtered_df.sample(
            fraction=1.0,
            seed=seed,
            shuffle=True
        )

    return filtered_df, stage_config


def create_batches_by_length(
    df: pl.DataFrame,
    batch_size: int,
    max_length_in_batch: Optional[int] = None,
    pad_to_multiple: int = 32
) -> List[pl.DataFrame]:
    """
    Create batches grouped by similar sequence lengths for efficiency.

    Args:
        df: DataFrame with sequences
        batch_size: Target batch size
        max_length_in_batch: Maximum sequence length variation in batch
        pad_to_multiple: Pad sequences to multiple of this value

    Returns:
        List of batch DataFrames
    """
    # Sort by length
    df_sorted = df.sort('sequence_length')

    batches = []
    current_batch = []
    current_max_len = 0

    for row in df_sorted.iter_rows(named=True):
        seq_len = row['sequence_length']

        # Pad length to multiple
        padded_len = ((seq_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple

        # Check if should start new batch
        if len(current_batch) > 0:
            length_diff = abs(padded_len - current_max_len)

            should_start_new = (
                len(current_batch) >= batch_size or
                (max_length_in_batch and length_diff > max_length_in_batch)
            )

            if should_start_new:
                batches.append(pl.DataFrame(current_batch))
                current_batch = []
                current_max_len = 0

        current_batch.append(row)
        current_max_len = max(current_max_len, padded_len)

    # Add last batch
    if current_batch:
        batches.append(pl.DataFrame(current_batch))

    print(f"\nCreated {len(batches)} batches:")
    print(f"  Avg batch size: {np.mean([len(b) for b in batches]):.1f}")
    print(f"  Min/Max batch size: {min(len(b) for b in batches)}/{max(len(b) for b in batches)}")

    return batches


def get_training_schedule() -> Dict[int, Dict]:
    """
    Get complete training schedule for all stages.

    Returns:
        Dictionary mapping stage_id to training parameters
    """
    schedule = {}

    for stage_config in CURRICULUM_STAGES:
        schedule[stage_config.stage_id] = {
            'name': stage_config.name,
            'max_length': stage_config.max_length,
            'learning_rate': stage_config.learning_rate,
            'batch_size': stage_config.batch_size,
            'num_epochs': {
                1: 20,  # Easy: more epochs
                2: 15,  # Medium
                3: 10,  # Hard
                4: 5,   # Full: few epochs
            }[stage_config.stage_id],
        }

    return schedule


if __name__ == "__main__":
    print("="* 70)
    print("Curriculum Learning Scheduler")
    print("="* 70)

    # Display schedule
    schedule = get_training_schedule()

    print("\nTraining Schedule:")
    print("-"* 70)
    print(f"{'Stage':<10} {'Length':<15} {'LR':<12} {'Batch':<8} {'Epochs':<8}")
    print("-"* 70)

    for stage_id, params in schedule.items():
        print(f"{params['name']:<10} "
              f"≤{params['max_length']:<14} "
              f"{params['learning_rate']:<12.0e} "
              f"{params['batch_size']:<8} "
              f"{params['num_epochs']:<8}")

    print("\n" + "="* 70)
    print("Curriculum Philosophy")
    print("="* 70)
    print("""
1. Start with short, simple RNAs (easy to fold correctly)
2. Gradually increase length and complexity
3. Reduce learning rate as task difficulty increases
4. Reduce batch size for longer sequences (memory constraint)
5. Final stage uses full dataset for fine-tuning

Expected training time: ~100 GPU hours for full schedule
    """)

    # Test with dummy data
    print("\nTesting with sample data...")

    # Create dummy DataFrame
    n_samples = 100
    dummy_data = {
        'target_id': [f'RNA_{i}' for i in range(n_samples)],
        'sequence': ['A' * (10 + i) for i in range(n_samples)],
        'sequence_length': list(range(10, 10 + n_samples)),
        'has_pseudoknots': [i % 3 == 0 for i in range(n_samples)],
    }
    df = pl.DataFrame(dummy_data)

    # Test stage 1
    filtered, config = get_curriculum_dataloader(
        dataset_path=Path("dummy.parquet"),  # Not used for in-memory df
        stage=1,
        shuffle=False
    )

    # Override with dummy data
    filtered = filter_by_length_and_complexity(
        df,
        max_len=config.max_length,
        min_len=config.min_length,
        allow_pseudoknots=config.allow_pseudoknots
    )

    print(f"\nStage 1 filtered: {len(filtered)} samples")

    # Create batches
    batches = create_batches_by_length(filtered, batch_size=8, pad_to_multiple=32)
    print(f"Created {len(batches)} batches")

    print("\n✓ Curriculum scheduler ready")
