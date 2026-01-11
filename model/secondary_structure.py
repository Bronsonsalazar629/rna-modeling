"""
Secondary structure prediction for RNA.
Predicts helix/loop/stem regions, which can be converted to 3D using ModeRNA.
This is a much simpler problem with stronger signal than direct 3D prediction.
"""
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from model.rnafold_se3_full import FullRNAFoldConfig, EvoformerBlock, RelativePositionalEncoding


# Secondary structure labels
SS_LABELS = {
    'H': 0,  # Helix/Stem (paired)
    'E': 1,  # External loop (unpaired)
    'I': 2,  # Internal loop
    'M': 3,  # Multi-loop
    'B': 4,  # Bulge
}


class SecondaryStructureHead(hk.Module):
    """Predict per-residue secondary structure type."""

    def __init__(self, num_classes: int = 5, name: Optional[str] = None):
        super().__init__(name=name)
        self.num_classes = num_classes

    def __call__(self, node_features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict secondary structure class for each residue.

        Args:
            node_features: (L, node_dim) per-residue features

        Returns:
            ss_logits: (L, num_classes) class logits
        """
        # Project to hidden
        hidden = hk.Linear(128, name='ss_hidden')(node_features)
        hidden = jax.nn.relu(hidden)
        hidden = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(hidden)

        # Output logits
        ss_logits = hk.Linear(self.num_classes, name='ss_logits')(hidden)

        return ss_logits


class BasePairHead(hk.Module):
    """Predict base pairing probabilities."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, pair_features: jnp.ndarray) -> jnp.ndarray:
        """
        Predict base pairing matrix.

        Args:
            pair_features: (L, L, pair_dim)

        Returns:
            pairing_probs: (L, L) pairing probabilities
        """
        L = pair_features.shape[0]

        # Project pair features
        hidden = hk.Linear(64, name='bp_hidden')(pair_features)
        hidden = jax.nn.relu(hidden)

        # Pairing logits
        logits = hk.Linear(1, name='bp_logits')(hidden)
        logits = logits.squeeze(-1)  # (L, L)

        # Symmetrize
        logits = (logits + logits.T) / 2.0

        # Mask diagonal and close neighbors
        mask = jnp.ones((L, L))
        for offset in [0, 1, 2, 3]:
            mask = mask.at[jnp.arange(L-offset), jnp.arange(offset, L)].set(0.0)
            if offset > 0:
                mask = mask.at[jnp.arange(offset, L), jnp.arange(L-offset)].set(0.0)

        logits = logits * mask

        # Convert to probabilities
        pairing_probs = jax.nn.sigmoid(logits)

        return pairing_probs


class RNASecondaryStructureModel(hk.Module):
    """RNA secondary structure prediction model."""

    def __init__(self, config: FullRNAFoldConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self,
        sequence: jnp.ndarray,
        msa: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict secondary structure and base pairs.

        Args:
            sequence: One-hot sequence (L, vocab_size)
            msa: MSA tensor (N, L, vocab_size) or None

        Returns:
            ss_logits: (L, 5) secondary structure class logits
            pairing_probs: (L, L) base pairing probabilities
        """
        L = sequence.shape[0]

        # Embed sequence
        node_features = hk.Linear(self.config.node_embedding_dim, name='seq_embed')(sequence)

        # Initialize pair representation
        rel_pos_encoding = RelativePositionalEncoding(
            self.config.pair_embedding_dim, name='rel_pos'
        )(L)
        pair_features = rel_pos_encoding

        # Process MSA if provided (lightweight - only 4 blocks)
        if msa is not None:
            N = msa.shape[0]
            msa_features = hk.Linear(self.config.msa_embedding_dim, name='msa_embed')(msa)
            msa_mask = jnp.ones((N, L))

            for i in range(4):  # Lightweight processing
                msa_features, pair_features = EvoformerBlock(
                    self.config, name=f'evoformer_{i}'
                )(msa_features, pair_features, msa_mask)

            # Update node features from MSA
            single_repr = msa_features[0]
            node_features = node_features + hk.Linear(
                self.config.node_embedding_dim, name='msa_to_node'
            )(single_repr)

        # Predict secondary structure
        ss_logits = SecondaryStructureHead(num_classes=5, name='ss_head')(node_features)

        # Predict base pairs
        pairing_probs = BasePairHead(name='bp_head')(pair_features)

        return ss_logits, pairing_probs


def create_secondary_structure_model(config: Optional[FullRNAFoldConfig] = None):
    """
    Create secondary structure prediction model.

    Args:
        config: Model configuration

    Returns:
        Haiku-transformed model
    """
    if config is None:
        config = FullRNAFoldConfig()

    def forward(sequence, msa=None):
        model = RNASecondaryStructureModel(config)
        return model(sequence, msa)

    model_transformed = hk.transform(forward)
    return model_transformed


if __name__ == "__main__":
    # Test secondary structure model
    print("Testing secondary structure prediction model...")

    config = FullRNAFoldConfig()
    config.num_evoformer_blocks = 4

    model = create_secondary_structure_model(config)

    # Test input
    rng = jax.random.PRNGKey(42)
    L = 20
    sequence = jax.random.normal(rng, (L, 5))

    # Initialize
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sequence)

    # Forward pass
    rng, apply_rng = jax.random.split(rng)
    ss_logits, pairing_probs = model.apply(params, apply_rng, sequence)

    print(f"  Input shape: {sequence.shape}")
    print(f"  SS logits shape: {ss_logits.shape}")
    print(f"  Pairing probs shape: {pairing_probs.shape}")
    print(f"  Pairing range: [{pairing_probs.min():.3f}, {pairing_probs.max():.3f}]")
    print(f"  Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    print("\nSecondary structure model working!")
