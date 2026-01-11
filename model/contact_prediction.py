"""
Contact map prediction model for RNA secondary structure.
This is a simpler 2D problem that provides a foundation for 3D folding.
"""
import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from model.rnafold_se3_full import FullRNAFoldConfig, EvoformerBlock, RelativePositionalEncoding


class ContactPredictionHead(hk.Module):
    """Predict RNA contact map from pair representation."""

    def __init__(self, config: FullRNAFoldConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config

    def __call__(self, pair_repr: jnp.ndarray) -> jnp.ndarray:
        """
        Predict contact probabilities.

        Args:
            pair_repr: (L, L, pair_dim) pair representation

        Returns:
            contact_probs: (L, L) contact probability matrix
        """
        L = pair_repr.shape[0]

        # Project pair features
        hidden = hk.Linear(128, name='contact_hidden_1')(pair_repr)
        hidden = jax.nn.relu(hidden)
        hidden = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(hidden)

        hidden = hk.Linear(64, name='contact_hidden_2')(hidden)
        hidden = jax.nn.relu(hidden)

        # Final contact logits
        logits = hk.Linear(1, name='contact_logits')(hidden)
        logits = logits.squeeze(-1)  # (L, L)

        # Symmetrize (contacts are symmetric)
        logits = (logits + logits.T) / 2.0

        # Zero out diagonal and close neighbors (i, i+1, i+2)
        mask = jnp.ones((L, L))
        for offset in [0, 1, 2]:
            mask = mask.at[jnp.arange(L-offset), jnp.arange(offset, L)].set(0.0)
            if offset > 0:
                mask = mask.at[jnp.arange(offset, L), jnp.arange(L-offset)].set(0.0)

        logits = logits * mask

        # Convert to probabilities
        contact_probs = jax.nn.sigmoid(logits)

        return contact_probs


class ContactPredictionModel(hk.Module):
    """RNA contact prediction model using Evoformer."""

    def __init__(self, config: FullRNAFoldConfig, name: Optional[str] = None):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self,
        sequence: jnp.ndarray,
        msa: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Predict contact map.

        Args:
            sequence: One-hot sequence (L, vocab_size)
            msa: MSA tensor (N, L, vocab_size) or None

        Returns:
            contact_probs: (L, L) contact probability matrix
        """
        L = sequence.shape[0]

        # Embed sequence
        node_features = hk.Linear(self.config.node_embedding_dim, name='seq_embed')(sequence)

        # Initialize pair representation
        rel_pos_encoding = RelativePositionalEncoding(
            self.config.pair_embedding_dim, name='rel_pos'
        )(L)
        pair_features = rel_pos_encoding

        # Process MSA if provided
        if msa is not None:
            N = msa.shape[0]
            msa_features = hk.Linear(self.config.msa_embedding_dim, name='msa_embed')(msa)
            msa_mask = jnp.ones((N, L))

            # Simplified Evoformer (fewer blocks for contact prediction)
            num_blocks = min(8, self.config.num_evoformer_blocks)
            for i in range(num_blocks):
                msa_features, pair_features = EvoformerBlock(
                    self.config, name=f'evoformer_{i}'
                )(msa_features, pair_features, msa_mask)

        # Contact prediction head
        contact_probs = ContactPredictionHead(self.config, name='contact_head')(pair_features)

        return contact_probs


def create_contact_model(config: Optional[FullRNAFoldConfig] = None):
    """
    Create contact prediction model as Haiku transform.

    Args:
        config: Model configuration

    Returns:
        Haiku-transformed model
    """
    if config is None:
        config = FullRNAFoldConfig()

    def forward(sequence, msa=None):
        model = ContactPredictionModel(config)
        return model(sequence, msa)

    model_transformed = hk.transform(forward)
    return model_transformed


if __name__ == "__main__":
    # Test contact model
    print("Testing contact prediction model...")

    config = FullRNAFoldConfig()
    config.num_evoformer_blocks = 8

    model = create_contact_model(config)

    # Test input
    rng = jax.random.PRNGKey(42)
    L = 20
    sequence = jax.random.normal(rng, (L, 5))

    # Initialize
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, sequence)

    # Forward pass
    rng, apply_rng = jax.random.split(rng)
    contacts = model.apply(params, apply_rng, sequence)

    print(f"  Input shape: {sequence.shape}")
    print(f"  Output shape: {contacts.shape}")
    print(f"  Contact range: [{contacts.min():.3f}, {contacts.max():.3f}]")
    print(f"  Num predicted contacts (p>0.5): {(contacts > 0.5).sum()}")
    print(f"  Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")
    print("\nContact prediction model working!")
