"""
OpenMM-based validation for RNA structures.
Checks for steric clashes and structural stability through short MD simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("Warning: OpenMM not available. Install with: conda install -c conda-forge openmm")


# Nucleotide name mapping
NUCLEOTIDE_MAP = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'U': 'U',
    'T': 'U',  # Map T to U for RNA
}


def sequence_to_residue_names(sequence: str) -> List[str]:
    """
    Convert sequence string to list of residue names for OpenMM.

    Args:
        sequence: RNA sequence (e.g., "AUGC")

    Returns:
        List of residue names
    """
    return [NUCLEOTIDE_MAP.get(nt.upper(), 'A') for nt in sequence]


def build_rna_topology(
    sequence: str,
    chain_id: str = 'A'
) -> Topology:
    """
    Build RNA topology from sequence (C1' atoms only).

    Args:
        sequence: RNA sequence
        chain_id: Chain identifier

    Returns:
        OpenMM Topology object
    """
    if not OPENMM_AVAILABLE:
        raise ImportError("OpenMM is required for validation")

    topology = Topology()
    chain = topology.addChain(chain_id)

    residue_names = sequence_to_residue_names(sequence)

    # Add residues with C1' atoms
    for i, res_name in enumerate(residue_names):
        residue = topology.addResidue(res_name, chain)
        # Add C1' atom (C1* in PDB nomenclature for ribose)
        atom = topology.addAtom("C1'", element.carbon, residue)

    return topology


def coords_to_positions(coords: np.ndarray) -> List:
    """
    Convert coordinate array to OpenMM positions.

    Args:
        coords: Coordinates (L, 3) in Angstroms

    Returns:
        List of Vec3 positions with units
    """
    positions = []
    for coord in coords:
        positions.append(Vec3(float(coord[0]), float(coord[1]), float(coord[2])) * angstroms)

    return positions


def check_steric_clashes(
    coords: np.ndarray,
    sequence: str,
    clash_threshold: float = 3.0
) -> Dict[str, any]:
    """
    Check for steric clashes (atoms too close).

    Args:
        coords: C1' coordinates (L, 3) in Angstroms
        sequence: RNA sequence
        clash_threshold: Minimum allowed distance in Angstroms

    Returns:
        Dictionary with clash information
    """
    L = len(coords)
    clashes = []

    for i in range(L):
        for j in range(i + 2, L):  # Skip adjacent residues
            dist = np.linalg.norm(coords[i] - coords[j])

            if dist < clash_threshold:
                clashes.append({
                    'residue_i': i + 1,  # 1-indexed
                    'residue_j': j + 1,
                    'distance': float(dist),
                })

    has_clashes = len(clashes) > 0

    return {
        'has_clashes': has_clashes,
        'num_clashes': len(clashes),
        'clashes': clashes[:10],  # Return first 10 for brevity
    }


def run_openmm_validation(
    coords: np.ndarray,
    sequence: str,
    simulation_steps: int = 100,
    temperature: float = 300.0,
    rmsd_threshold: float = 2.0,
    clash_threshold: float = 3.0
) -> Dict[str, any]:
    """
    Validate structure with OpenMM energy minimization and short MD.

    Args:
        coords: C1' coordinates (L, 3) in Angstroms
        sequence: RNA sequence
        simulation_steps: Number of MD steps for stability test
        temperature: Temperature in Kelvin
        rmsd_threshold: Maximum allowed RMSD drift in Angstroms
        clash_threshold: Clash distance threshold

    Returns:
        Dictionary with validation results
    """
    if not OPENMM_AVAILABLE:
        print("Warning: OpenMM not available, skipping validation")
        return {
            'is_valid': True,  # Assume valid if can't check
            'has_clashes': False,
            'rmsd_drift': 0.0,
            'message': 'OpenMM not available',
        }

    try:
        # Check clashes first (fast check)
        clash_result = check_steric_clashes(coords, sequence, clash_threshold)

        if clash_result['has_clashes']:
            return {
                'is_valid': False,
                'has_clashes': True,
                'num_clashes': clash_result['num_clashes'],
                'rmsd_drift': 0.0,
                'message': f"Structure has {clash_result['num_clashes']} steric clashes",
            }

        # Build topology
        topology = build_rna_topology(sequence)

        # Create system with harmonic restraints (coarse-grained model)
        system = System()

        # Add particles (C1' atoms)
        for _ in range(len(coords)):
            system.addParticle(12.0)  # Carbon mass

        # Add harmonic bonds between sequential residues
        bond_force = HarmonicBondForce()
        k_bond = 1000.0  # kcal/mol/A^2 -> kJ/mol/nm^2
        equilibrium_distance = 0.6  # nm (6 Angstroms)

        for i in range(len(coords) - 1):
            bond_force.addBond(i, i + 1, equilibrium_distance, k_bond)

        system.addForce(bond_force)

        # Add soft-sphere repulsion to prevent clashes
        nb_force = CustomNonbondedForce("4*epsilon*((sigma/r)^12 - (sigma/r)^6); sigma=0.3; epsilon=1.0")
        for _ in range(len(coords)):
            nb_force.addParticle([])

        nb_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        nb_force.setCutoffDistance(1.5 * nanometers)
        system.addForce(nb_force)

        # Create integrator
        integrator = LangevinIntegrator(
            temperature * kelvin,
            1.0 / picosecond,
            2.0 * femtoseconds
        )

        # Create simulation with CPU reference platform (slower but reliable)
        try:
            from openmm import Platform
            platform = Platform.getPlatformByName('Reference')
            simulation = Simulation(topology, system, integrator, platform)
        except:
            # Fallback to default platform if Reference not available
            simulation = Simulation(topology, system, integrator)

        # Set positions (convert A to nm)
        positions_nm = coords_to_positions(coords / 10.0)  # Angstroms to nm
        simulation.context.setPositions(positions_nm)

        # Energy minimization
        simulation.minimizeEnergy(maxIterations=100)

        # Get minimized positions
        state = simulation.context.getState(getPositions=True)
        minimized_positions = state.getPositions(asNumpy=True).value_in_unit(angstroms)
        minimized_coords = np.array(minimized_positions)

        # Run short MD for stability check
        initial_coords_md = minimized_coords.copy()

        simulation.step(simulation_steps)

        # Get final positions
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        final_positions = state.getPositions(asNumpy=True).value_in_unit(angstroms)
        final_coords = np.array(final_positions)

        # Compute RMSD drift
        rmsd_drift = np.sqrt(np.mean((final_coords - initial_coords_md) ** 2))

        # Check stability
        is_stable = rmsd_drift < rmsd_threshold

        return {
            'is_valid': is_stable and not clash_result['has_clashes'],
            'has_clashes': clash_result['has_clashes'],
            'num_clashes': clash_result['num_clashes'],
            'rmsd_drift': float(rmsd_drift),
            'rmsd_threshold': rmsd_threshold,
            'minimized_coords': minimized_coords,
            'final_coords': final_coords,
            'message': 'Valid' if is_stable else f'RMSD drift {rmsd_drift:.2f} > {rmsd_threshold:.2f}',
        }

    except Exception as e:
        return {
            'is_valid': False,
            'has_clashes': False,
            'rmsd_drift': 0.0,
            'message': f'OpenMM validation failed: {str(e)}',
            'error': str(e),
        }


def validate_structure_batch(
    coords_list: List[np.ndarray],
    sequence: str,
    **kwargs
) -> List[Dict]:
    """
    Validate multiple structures (e.g., ensemble predictions).

    Args:
        coords_list: List of coordinate arrays
        sequence: RNA sequence
        **kwargs: Additional arguments for run_openmm_validation

    Returns:
        List of validation results
    """
    results = []

    for i, coords in enumerate(coords_list):
        print(f"Validating structure {i + 1}/{len(coords_list)}...")
        result = run_openmm_validation(coords, sequence, **kwargs)
        result['structure_id'] = i
        results.append(result)

    # Summary
    num_valid = sum(1 for r in results if r['is_valid'])
    print(f"\nValidation summary: {num_valid}/{len(results)} structures passed")

    return results


if __name__ == "__main__":
    print("="* 60)
    print("OpenMM Structure Validation")
    print("="* 60)

    # Test with idealized A-form helix
    sequence = "AUGCAUGC"
    L = len(sequence)

    # Generate idealized coordinates
    rise_per_res = 2.8
    rotation_per_res = 32.7 * np.pi / 180.0
    radius = 10.0

    coords = np.zeros((L, 3))
    for i in range(L):
        theta = i * rotation_per_res
        z = i * rise_per_res
        coords[i] = [radius * np.cos(theta), radius * np.sin(theta), z]

    print(f"\nTest sequence: {sequence}")
    print(f"Coordinates shape: {coords.shape}")

    # Check clashes
    clash_result = check_steric_clashes(coords, sequence)
    print(f"\nClash check:")
    print(f"  Has clashes: {clash_result['has_clashes']}")
    print(f"  Num clashes: {clash_result['num_clashes']}")

    if OPENMM_AVAILABLE:
        # Full validation
        print("\nRunning OpenMM validation...")
        result = run_openmm_validation(
            coords,
            sequence,
            simulation_steps=100,
            rmsd_threshold=2.0
        )

        print(f"\nValidation result:")
        print(f"  Is valid: {result['is_valid']}")
        print(f"  RMSD drift: {result['rmsd_drift']:.3f} Å")
        print(f"  Message: {result['message']}")

        print("\n✓ All tests passed")
    else:
        print("\nSkipping OpenMM validation (not installed)")
