"""
Secondary structure prediction and feature extraction for RNA.
Uses ViennaRNA for structure prediction and base-pairing probability matrices.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional


try:
    import RNA  # ViennaRNA Python bindings
    VIENNA_AVAILABLE = True
except ImportError:
    VIENNA_AVAILABLE = False
    print("Warning: ViennaRNA not available. Install with: conda install -c bioconda viennarna")


def predict_secondary_structure(
    sequence: str,
    temperature: float = 37.0,
    constraint: Optional[str] = None
) -> Dict[str, any]:
    """
    Predict secondary structure using ViennaRNA RNAfold.

    Args:
        sequence: RNA sequence (A/C/G/U)
        temperature: Temperature in Celsius for folding
        constraint: Optional structure constraint in dot-bracket notation

    Returns:
        Dictionary containing:
            - structure: Dot-bracket notation string
            - mfe: Minimum free energy (kcal/mol)
            - ensemble_energy: Free energy of ensemble
            - probability: Base-pairing probability matrix (L, L)
    """
    if not VIENNA_AVAILABLE:
        # Return dummy structure if ViennaRNA not available
        L = len(sequence)
        return {
            'structure': '.' * L,
            'mfe': 0.0,
            'ensemble_energy': 0.0,
            'probability': np.zeros((L, L), dtype=np.float32),
        }

    # Create fold compound
    fc = RNA.fold_compound(sequence)

    # Set temperature
    if temperature != 37.0:
        fc.params.temperature = temperature

    # Apply constraint if provided
    if constraint is not None:
        fc.constraints_add(constraint, RNA.CONSTRAINT_DB | RNA.CONSTRAINT_DB_ENFORCE_BP)

    # Compute MFE structure
    structure, mfe = fc.mfe()

    # Compute partition function for probabilities
    ensemble_energy = fc.pf()

    # Get base-pairing probability matrix
    bpp_matrix = np.zeros((len(sequence), len(sequence)), dtype=np.float32)

    try:
        bpp = fc.bpp()  # Returns base-pairing probabilities
        # ViennaRNA bpp is 1-indexed
        for i in range(1, len(sequence) + 1):
            for j in range(i + 1, len(sequence) + 1):
                prob = bpp[i][j]
                if prob > 0:
                    bpp_matrix[i - 1, j - 1] = prob
                    bpp_matrix[j - 1, i - 1] = prob  # Symmetric
    except:
        # Fallback if bpp extraction fails
        pass

    return {
        'structure': structure,
        'mfe': mfe,
        'ensemble_energy': ensemble_energy[0] if isinstance(ensemble_energy, tuple) else ensemble_energy,
        'probability': bpp_matrix,
    }


def dotbracket_to_pairing(structure: str) -> np.ndarray:
    """
    Convert dot-bracket notation to base-pairing matrix.

    Args:
        structure: Dot-bracket string (e.g., "((...))")

    Returns:
        Binary pairing matrix (L, L) where 1 indicates paired bases
    """
    L = len(structure)
    pairing = np.zeros((L, L), dtype=np.int32)

    # Stack to track opening brackets
    stack = []
    pseudoknot_stack = []

    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairing[i, j] = 1
                pairing[j, i] = 1
        elif char == '[':  # Pseudoknot notation
            pseudoknot_stack.append(i)
        elif char == ']':
            if pseudoknot_stack:
                j = pseudoknot_stack.pop()
                pairing[i, j] = 1
                pairing[j, i] = 1
        # '.' indicates unpaired

    return pairing


def detect_pseudoknots(structure: str) -> bool:
    """
    Detect if structure contains pseudoknots.

    Args:
        structure: Dot-bracket notation

    Returns:
        True if pseudoknots detected (contains [ or ])
    """
    return '[' in structure or ']' in structure


def extract_stem_features(structure: str) -> Dict[str, np.ndarray]:
    """
    Extract structural features from secondary structure.

    Args:
        structure: Dot-bracket notation

    Returns:
        Dictionary with per-residue features:
            - is_paired: Binary indicator
            - stem_id: Stem identifier (-1 for unpaired)
            - loop_type: 0=unpaired, 1=hairpin, 2=bulge, 3=internal, 4=multi
    """
    L = len(structure)
    is_paired = np.zeros(L, dtype=np.int32)
    stem_id = np.full(L, -1, dtype=np.int32)
    loop_type = np.zeros(L, dtype=np.int32)

    # Mark paired positions
    stack = []
    current_stem = 0

    for i, char in enumerate(structure):
        if char in '([':
            stack.append((i, current_stem))
            is_paired[i] = 1
        elif char in ')]':
            if stack:
                j, stem = stack.pop()
                is_paired[i] = 1
                stem_id[i] = stem
                stem_id[j] = stem

                # Check if continuing stem
                if i > 0 and structure[i - 1] in ')]':
                    pass  # Continue current stem
                else:
                    current_stem += 1

    # Identify loop types (simplified heuristic)
    for i in range(L):
        if structure[i] == '.':
            # Check context
            left_paired = (i > 0 and is_paired[i - 1])
            right_paired = (i < L - 1 and is_paired[i + 1])

            if left_paired and right_paired:
                loop_type[i] = 2  # Bulge/internal
            elif left_paired or right_paired:
                loop_type[i] = 1  # Hairpin
            else:
                loop_type[i] = 0  # Unpaired

    return {
        'is_paired': is_paired,
        'stem_id': stem_id,
        'loop_type': loop_type,
    }


def compute_structure_contact_map(
    bpp_matrix: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Convert base-pairing probability matrix to binary contact map.

    Args:
        bpp_matrix: Base-pairing probability matrix (L, L)
        threshold: Probability threshold for calling contact

    Returns:
        Binary contact map (L, L)
    """
    return (bpp_matrix > threshold).astype(np.float32)


def predict_per_chain_structure(
    sequence: str,
    chain_boundaries: List[Tuple[int, int, str]]
) -> Dict[str, any]:
    """
    Predict secondary structure for each chain independently.

    Args:
        sequence: Full concatenated sequence
        chain_boundaries: Chain boundary information

    Returns:
        Dictionary with per-chain structures and combined features
    """
    full_length = len(sequence)

    # Initialize full arrays
    full_structure = ['.' for _ in range(full_length)]
    full_bpp = np.zeros((full_length, full_length), dtype=np.float32)
    full_is_paired = np.zeros(full_length, dtype=np.int32)

    chain_structures = {}

    # Predict for each unique chain
    unique_chains = {}
    for start, end, chain_id in chain_boundaries:
        if chain_id not in unique_chains:
            chain_seq = sequence[start:end]
            struct_result = predict_secondary_structure(chain_seq)
            unique_chains[chain_id] = struct_result

    # Map back to full sequence
    for start, end, chain_id in chain_boundaries:
        chain_result = unique_chains[chain_id]
        chain_len = end - start

        # Copy structure
        struct = chain_result['structure']
        for i, char in enumerate(struct[:chain_len]):
            full_structure[start + i] = char

        # Copy BPP matrix
        bpp = chain_result['probability']
        full_bpp[start:end, start:end] = bpp[:chain_len, :chain_len]

        # Mark paired positions
        pairing = dotbracket_to_pairing(struct)
        full_is_paired[start:end] = pairing.diagonal()

    # Combine into full structure string
    full_structure_str = ''.join(full_structure)

    # Extract stem features
    stem_features = extract_stem_features(full_structure_str)

    return {
        'structure': full_structure_str,
        'bpp_matrix': full_bpp,
        'is_paired': full_is_paired,
        'stem_features': stem_features,
        'has_pseudoknots': detect_pseudoknots(full_structure_str),
        'chain_structures': unique_chains,
    }


def create_distance_bins(bpp_matrix: np.ndarray, num_bins: int = 64) -> np.ndarray:
    """
    Create distance histogram features for structure prediction (distogram).

    Args:
        bpp_matrix: Base-pairing probability matrix
        num_bins: Number of distance bins

    Returns:
        Distance bin features (L, L, num_bins)
    """
    L = bpp_matrix.shape[0]
    distogram = np.zeros((L, L, num_bins), dtype=np.float32)

    # Distance bins (in Angstroms, typical RNA C1'-C1' distances)
    # Paired bases: ~10-15 Å
    # Sequential: ~6 Å
    # Long-range: 15-50 Å
    distance_edges = np.linspace(0, 50, num_bins + 1)

    for i in range(L):
        for j in range(L):
            if bpp_matrix[i, j] > 0:
                # Estimate distance based on pairing probability
                # Paired bases are ~10 Å apart
                dist = 10.0 if bpp_matrix[i, j] > 0.5 else 20.0

                # Find bin
                bin_idx = np.digitize(dist, distance_edges) - 1
                bin_idx = np.clip(bin_idx, 0, num_bins - 1)

                distogram[i, j, bin_idx] = bpp_matrix[i, j]

    return distogram


if __name__ == "__main__":
    # Test structure prediction
    test_seq = "GGGAAACCC"
    result = predict_secondary_structure(test_seq)
    print(f"Sequence: {test_seq}")
    print(f"Structure: {result['structure']}")
    print(f"MFE: {result['mfe']:.2f} kcal/mol")
    print(f"BPP matrix shape: {result['probability'].shape}")

    # Test pairing matrix
    pairing = dotbracket_to_pairing(result['structure'])
    print(f"\nPairing matrix:\n{pairing}")

    # Test stem features
    stem_feats = extract_stem_features(result['structure'])
    print(f"\nIs paired: {stem_feats['is_paired']}")
    print(f"Loop types: {stem_feats['loop_type']}")
