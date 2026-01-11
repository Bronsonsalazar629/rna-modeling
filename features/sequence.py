"""
Sequence feature extraction for multi-chain RNA structures.
Handles stoichiometry parsing, per-chain one-hot encoding, and chain-aware MSA processing.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
import re


# Canonical RNA nucleotides only (modifications pre-mapped in input)
NUCLEOTIDE_VOCAB = {
    'A': 0,
    'C': 1,
    'G': 2,
    'U': 3,
    'PAD': 4,  # Padding token
}

VOCAB_SIZE = len(NUCLEOTIDE_VOCAB)


def parse_stoichiometry(stoich_string: str) -> List[str]:
    """
    Parse stoichiometry string to chain list.

    Args:
        stoich_string: e.g., "{A:2};{B:1}" or "{A:1}"

    Returns:
        List of chain IDs in order, e.g., ['A', 'A', 'B']

    Examples:
        "{A:2};{B:1}" -> ['A', 'A', 'B']
        "{A:1}" -> ['A']
    """
    chain_list = []

    # Parse pattern {CHAIN:COUNT}
    pattern = r'\{([A-Z]+):(\d+)\}'
    matches = re.findall(pattern, stoich_string)

    for chain_id, count in matches:
        chain_list.extend([chain_id] * int(count))

    return chain_list


def parse_fasta_sequences(all_sequences: str) -> Dict[str, str]:
    """
    Parse FASTA format all_sequences field to extract per-chain sequences.

    Args:
        all_sequences: FASTA-formatted string with >chain_id headers

    Returns:
        Dictionary mapping chain_id -> sequence

    Example:
        ">A\\nAUGC\\n>B\\nGCUA" -> {'A': 'AUGC', 'B': 'GCUA'}
    """
    chain_sequences = {}

    # Split by > to get each record
    records = all_sequences.strip().split('>')
    for record in records:
        if not record.strip():
            continue

        lines = record.strip().split('\n')
        chain_id = lines[0].strip()
        sequence = ''.join(lines[1:]).strip().upper()

        chain_sequences[chain_id] = sequence

    return chain_sequences


def expand_sequence_with_stoichiometry(
    stoichiometry: str,
    all_sequences: str
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Expand multi-chain sequence according to stoichiometry.

    Args:
        stoichiometry: Stoichiometry string like "{A:2};{B:1}"
        all_sequences: FASTA-formatted sequences

    Returns:
        - Concatenated sequence string (all chains/copies)
        - List of (start_idx, end_idx, chain_id) tuples for each segment
    """
    chain_list = parse_stoichiometry(stoichiometry)
    chain_sequences = parse_fasta_sequences(all_sequences)

    full_sequence = []
    chain_boundaries = []

    current_idx = 0
    for chain_id in chain_list:
        seq = chain_sequences[chain_id]
        seq_len = len(seq)

        full_sequence.append(seq)
        chain_boundaries.append((current_idx, current_idx + seq_len, chain_id))
        current_idx += seq_len

    return ''.join(full_sequence), chain_boundaries


def sequence_to_one_hot(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence to one-hot encoding (4 canonical bases only).

    Args:
        sequence: RNA sequence string (A/C/G/U only)

    Returns:
        One-hot encoded array of shape (L, vocab_size)
    """
    sequence = sequence.upper()

    # Map sequence to indices
    indices = []
    for nt in sequence:
        if nt in NUCLEOTIDE_VOCAB:
            indices.append(NUCLEOTIDE_VOCAB[nt])
        else:
            # Should not happen if input is properly canonicalized
            # Default to A if unknown
            indices.append(NUCLEOTIDE_VOCAB['A'])

    # Create one-hot encoding
    one_hot = np.zeros((len(indices), VOCAB_SIZE), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1.0

    return one_hot


def create_chain_features(
    sequence: str,
    chain_boundaries: List[Tuple[int, int, str]]
) -> np.ndarray:
    """
    Create chain ID features and relative position encoding.

    Args:
        sequence: Full concatenated sequence
        chain_boundaries: List of (start, end, chain_id) tuples

    Returns:
        Feature array (L, 4) with:
            - chain_id_encoding (one-hot over max 26 chains)
            - relative_position_in_chain (0 to 1)
            - absolute_position (0 to 1)
            - is_chain_start (binary flag)
    """
    seq_len = len(sequence)

    # Chain ID encoding (support up to 26 chains A-Z)
    chain_id_feature = np.zeros(seq_len, dtype=np.int32)
    relative_pos = np.zeros(seq_len, dtype=np.float32)
    is_chain_start = np.zeros(seq_len, dtype=np.float32)

    for start, end, chain_id in chain_boundaries:
        # Convert chain ID to integer (A=0, B=1, etc.)
        chain_idx = ord(chain_id[0]) - ord('A')
        chain_id_feature[start:end] = chain_idx

        # Relative position within chain
        chain_len = end - start
        relative_pos[start:end] = np.linspace(0, 1, chain_len, dtype=np.float32)

        # Mark chain start
        is_chain_start[start] = 1.0

    # Absolute position
    absolute_pos = np.linspace(0, 1, seq_len, dtype=np.float32)

    # Stack features
    features = np.stack([
        chain_id_feature.astype(np.float32) / 26.0,  # Normalize to [0, 1]
        relative_pos,
        absolute_pos,
        is_chain_start
    ], axis=1)

    return features


def parse_chain_msa(
    msa_fasta: str,
    chain_boundaries: List[Tuple[int, int, str]]
) -> Dict[str, List[str]]:
    """
    Parse per-chain MSA from multi-chain MSA FASTA file.

    Args:
        msa_fasta: Full MSA FASTA content with chain annotations
        chain_boundaries: Chain boundary information

    Returns:
        Dictionary mapping chain_id -> list of aligned sequences
    """
    chain_msas = {chain_id: [] for _, _, chain_id in set((s, e, c) for s, e, c in chain_boundaries)}

    # Parse FASTA records
    records = msa_fasta.strip().split('>')
    for record in records:
        if not record.strip():
            continue

        lines = record.strip().split('\n')
        header = lines[0]
        sequence = ''.join(lines[1:]).strip()

        # Extract chain ID from header (format: >homolog|chain=A)
        chain_match = re.search(r'chain=([A-Z]+)', header)
        if chain_match:
            chain_id = chain_match.group(1)
            if chain_id in chain_msas:
                # Remove gaps from sequence
                seq_no_gaps = sequence.replace('-', '')
                chain_msas[chain_id].append(seq_no_gaps)

    return chain_msas


def build_chain_aware_msa(
    chain_msas: Dict[str, List[str]],
    chain_boundaries: List[Tuple[int, int, str]],
    max_seqs: int = 512
) -> Dict[str, np.ndarray]:
    """
    Build chain-aware MSA tensor with proper gap masking.

    Args:
        chain_msas: Per-chain MSA sequences
        chain_boundaries: Chain boundary information
        max_seqs: Maximum MSA depth

    Returns:
        Dictionary with MSA features
    """
    # Get total sequence length
    total_len = max(end for _, end, _ in chain_boundaries)

    # Initialize MSA array (max_seqs, total_len, vocab_size)
    msa_array = np.zeros((max_seqs, total_len, VOCAB_SIZE), dtype=np.float32)
    msa_mask = np.zeros(max_seqs, dtype=np.float32)

    # Build query sequence (first row)
    query_row = np.zeros((total_len, VOCAB_SIZE), dtype=np.float32)
    for start, end, chain_id in chain_boundaries:
        if chain_id in chain_msas and len(chain_msas[chain_id]) > 0:
            chain_seq = chain_msas[chain_id][0]  # First is query
            one_hot = sequence_to_one_hot(chain_seq)
            query_row[start:start + len(chain_seq)] = one_hot

    msa_array[0] = query_row
    msa_mask[0] = 1.0

    # Fill with homologs (interleave across chains for diversity)
    msa_idx = 1
    max_homologs = max_seqs - 1

    # Collect all homologs per chain (excluding query)
    chain_homologs = {
        chain_id: seqs[1:] if len(seqs) > 1 else []
        for chain_id, seqs in chain_msas.items()
    }

    # Sample homologs round-robin across chains
    homolog_indices = {chain_id: 0 for chain_id in chain_homologs}
    chains_with_data = [c for c in chain_homologs if len(chain_homologs[c]) > 0]

    while msa_idx < max_seqs and chains_with_data:
        for chain_id in chains_with_data[:]:
            if msa_idx >= max_seqs:
                break

            homologs = chain_homologs[chain_id]
            idx = homolog_indices[chain_id]

            if idx < len(homologs):
                # Add this homolog
                row = np.zeros((total_len, VOCAB_SIZE), dtype=np.float32)

                # Fill only this chain's segment
                for start, end, seg_chain_id in chain_boundaries:
                    if seg_chain_id == chain_id:
                        seq = homologs[idx]
                        one_hot = sequence_to_one_hot(seq)
                        row[start:start + min(len(seq), end - start)] = one_hot[:end - start]

                msa_array[msa_idx] = row
                msa_mask[msa_idx] = 1.0
                msa_idx += 1

                homolog_indices[chain_id] += 1
            else:
                # This chain exhausted
                chains_with_data.remove(chain_id)

    return {
        'msa': msa_array,
        'msa_mask': msa_mask,
        'num_alignments': int(msa_mask.sum()),
    }


def create_chain_attention_masks(
    chain_boundaries: List[Tuple[int, int, str]],
    allow_interchain: bool = False
) -> np.ndarray:
    """
    Create attention mask that respects chain boundaries.

    Args:
        chain_boundaries: List of (start, end, chain_id)
        allow_interchain: If False, mask out cross-chain attention

    Returns:
        Attention mask (L, L) where 1=attend, 0=mask
    """
    total_len = max(end for _, end, _ in chain_boundaries)
    mask = np.ones((total_len, total_len), dtype=np.float32)

    if not allow_interchain:
        # Block cross-chain attention
        for i, (start_i, end_i, chain_i) in enumerate(chain_boundaries):
            for j, (start_j, end_j, chain_j) in enumerate(chain_boundaries):
                if chain_i != chain_j:
                    mask[start_i:end_i, start_j:end_j] = 0.0

    return mask


if __name__ == "__main__":
    # Test stoichiometry parsing
    stoich = "{A:2};{B:1}"
    chain_list = parse_stoichiometry(stoich)
    print(f"Stoichiometry: {stoich}")
    print(f"Chain list: {chain_list}")

    # Test FASTA parsing
    fasta = ">A\\nAUGC\\n>B\\nGCUA"
    chain_seqs = parse_fasta_sequences(fasta)
    print(f"\\nChain sequences: {chain_seqs}")

    # Test sequence expansion
    full_seq, boundaries = expand_sequence_with_stoichiometry(stoich, fasta)
    print(f"\\nFull sequence: {full_seq}")
    print(f"Boundaries: {boundaries}")
    print(f"Total length: {len(full_seq)}")

    # Test chain features
    chain_feats = create_chain_features(full_seq, boundaries)
    print(f"\\nChain features shape: {chain_feats.shape}")

    # Test attention mask
    attn_mask = create_chain_attention_masks(boundaries, allow_interchain=False)
    print(f"Attention mask shape: {attn_mask.shape}")
    print(f"Intra-chain only: {attn_mask.sum()} / {attn_mask.size}")
