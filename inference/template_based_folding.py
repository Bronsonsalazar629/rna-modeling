"""
Template-based RNA folding using homology modeling.
This approach finds similar sequences in the training set and uses their structures.
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.tm_score import kabsch


def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity between two RNA sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Identity fraction (0.0 to 1.0)
    """
    if len(seq1) != len(seq2):
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def needleman_wunsch_align(seq1: str, seq2: str, match_score: int = 2,
                            mismatch_penalty: int = -1, gap_penalty: int = -2) -> Tuple[str, str, float]:
    """
    Global sequence alignment using Needleman-Wunsch algorithm.

    Args:
        seq1: Query sequence
        seq2: Template sequence
        match_score: Score for matching bases
        mismatch_penalty: Penalty for mismatches
        gap_penalty: Penalty for gaps

    Returns:
        aligned_seq1: Aligned query
        aligned_seq2: Aligned template
        score: Alignment score
    """
    m, n = len(seq1), len(seq2)

    # Initialize scoring matrix
    score_matrix = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        score_matrix[i][0] = gap_penalty * i
    for j in range(n + 1):
        score_matrix[0][j] = gap_penalty * j

    # Fill scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            delete = score_matrix[i-1][j] + gap_penalty
            insert = score_matrix[i][j-1] + gap_penalty
            score_matrix[i][j] = max(match, delete, insert)

    # Traceback
    aligned_seq1, aligned_seq2 = '', ''
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i][j] == score_matrix[i-1][j-1] + \
           (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + gap_penalty:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1

    return aligned_seq1, aligned_seq2, score_matrix[m][n]


def find_best_template(query_seq: str, template_library: List[Dict],
                       top_k: int = 5) -> List[Dict]:
    """
    Find best template structures for a query sequence.

    Args:
        query_seq: Query RNA sequence
        template_library: List of template structures
        top_k: Number of top templates to return

    Returns:
        List of top templates with alignment info
    """
    candidates = []

    for template in template_library:
        template_seq = template['sequence']

        # Skip empty templates
        if len(template_seq) == 0:
            continue

        # Quick filter: length difference
        length_ratio = len(query_seq) / len(template_seq)
        if length_ratio < 0.5 or length_ratio > 2.0:
            continue

        # Align sequences
        aligned_query, aligned_template, align_score = needleman_wunsch_align(
            query_seq, template_seq
        )

        # Compute identity
        identity = sum(1 for a, b in zip(aligned_query, aligned_template) if a == b and a != '-')
        identity /= len(aligned_query)

        candidates.append({
            'template': template,
            'aligned_query': aligned_query,
            'aligned_template': aligned_template,
            'identity': identity,
            'align_score': align_score,
        })

    # Sort by identity (descending)
    candidates.sort(key=lambda x: x['identity'], reverse=True)

    return candidates[:top_k]


def thread_sequence_on_template(query_seq: str, aligned_query: str,
                                aligned_template: str, template_coords: np.ndarray) -> np.ndarray:
    """
    Thread query sequence onto template structure.

    Args:
        query_seq: Query sequence
        aligned_query: Aligned query (with gaps)
        aligned_template: Aligned template (with gaps)
        template_coords: Template C1' coordinates

    Returns:
        Threaded coordinates for query
    """
    # Build mapping: query position -> template position
    query_idx = 0
    template_idx = 0
    mapping = {}

    for aq, at in zip(aligned_query, aligned_template):
        if aq != '-' and at != '-':
            # Both aligned
            mapping[query_idx] = template_idx
            query_idx += 1
            template_idx += 1
        elif aq != '-':
            # Gap in template - will need to model this position
            query_idx += 1
        elif at != '-':
            # Gap in query
            template_idx += 1

    # Thread coordinates
    query_coords = np.zeros((len(query_seq), 3))

    for q_pos in range(len(query_seq)):
        if q_pos in mapping:
            # Direct copy from template
            t_pos = mapping[q_pos]
            query_coords[q_pos] = template_coords[t_pos]
        else:
            # Model missing position by interpolation
            # Find nearest mapped positions
            prev_mapped = [p for p in mapping.keys() if p < q_pos]
            next_mapped = [p for p in mapping.keys() if p > q_pos]

            if prev_mapped and next_mapped:
                # Interpolate
                p_prev = max(prev_mapped)
                p_next = min(next_mapped)
                t_prev = mapping[p_prev]
                t_next = mapping[p_next]

                alpha = (q_pos - p_prev) / (p_next - p_prev)
                query_coords[q_pos] = (1 - alpha) * template_coords[t_prev] + \
                                     alpha * template_coords[t_next]
            elif prev_mapped:
                # Extend from previous
                p_prev = max(prev_mapped)
                t_prev = mapping[p_prev]
                # Extend along previous direction
                if t_prev > 0:
                    direction = template_coords[t_prev] - template_coords[t_prev - 1]
                else:
                    direction = np.array([0.0, 0.0, 6.0])  # Default backbone spacing
                query_coords[q_pos] = query_coords[p_prev] + direction
            elif next_mapped:
                # Extend from next
                p_next = min(next_mapped)
                t_next = mapping[p_next]
                if t_next < len(template_coords) - 1:
                    direction = template_coords[t_next + 1] - template_coords[t_next]
                else:
                    direction = np.array([0.0, 0.0, 6.0])
                query_coords[q_pos] = query_coords[p_next] - direction

    return query_coords


def generate_ensemble_from_template(query_seq: str, template_info: Dict,
                                   num_variants: int = 5) -> List[np.ndarray]:
    """
    Generate ensemble of structures from a template by perturbation.

    Args:
        query_seq: Query sequence
        template_info: Template information with alignment
        num_variants: Number of variants to generate

    Returns:
        List of coordinate arrays
    """
    # Base structure from threading
    base_coords = thread_sequence_on_template(
        query_seq,
        template_info['aligned_query'],
        template_info['aligned_template'],
        template_info['template']['structures'][0]
    )

    ensemble = [base_coords]

    # Generate variants by adding noise
    for i in range(num_variants - 1):
        variant = base_coords.copy()

        # Add Gaussian noise scaled by distance from template
        noise_scale = 0.5 * (1.0 + i * 0.3)  # Increasing noise for diversity
        noise = np.random.randn(*variant.shape) * noise_scale

        # Apply noise more to loop regions (high uncertainty)
        # For simplicity, uniform noise here
        variant += noise

        ensemble.append(variant)

    return ensemble


def predict_structure_template_based(
    query_seq: str,
    template_library: List[Dict],
    num_predictions: int = 5
) -> List[np.ndarray]:
    """
    Predict RNA structure using template-based modeling.

    Args:
        query_seq: Query RNA sequence
        template_library: Library of template structures
        num_predictions: Number of predictions to generate

    Returns:
        List of predicted coordinate arrays
    """
    # Find best templates
    top_templates = find_best_template(query_seq, template_library, top_k=3)

    if not top_templates:
        # No good template - return random extended structure
        print(f"Warning: No template found for {query_seq[:20]}...")
        L = len(query_seq)
        predictions = []
        for i in range(num_predictions):
            coords = np.zeros((L, 3))
            coords[:, 2] = np.arange(L) * 6.0  # Extended along z-axis
            coords += np.random.randn(L, 3) * 2.0  # Add noise
            predictions.append(coords)
        return predictions

    # Use best template
    best_template = top_templates[0]

    print(f"  Best template: {best_template['template']['target_id']}, "
          f"identity={best_template['identity']:.2%}")

    # Generate ensemble
    predictions = generate_ensemble_from_template(
        query_seq, best_template, num_variants=num_predictions
    )

    return predictions


if __name__ == "__main__":
    # Test template-based folding
    print("Testing template-based folding...")

    # Load template library
    data_dir = Path("data")
    dataset_path = data_dir / "processed" / "train_structures.pkl"

    with open(dataset_path, 'rb') as f:
        template_library = pickle.load(f)

    print(f"Loaded {len(template_library)} templates")

    # Test on a query
    query_seq = "AUGCAUGCAUGC"
    predictions = predict_structure_template_based(
        query_seq, template_library, num_predictions=5
    )

    print(f"\nGenerated {len(predictions)} predictions for query: {query_seq}")
    for i, pred in enumerate(predictions):
        print(f"  Prediction {i+1}: shape={pred.shape}, "
              f"range=[{pred.min():.2f}, {pred.max():.2f}]")

    print("\nTemplate-based folding working!")
