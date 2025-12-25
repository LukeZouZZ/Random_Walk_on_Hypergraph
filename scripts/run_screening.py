#!/usr/bin/env python
"""
Command-line interface for Hypergraph-MCMC molecular screening.

Usage:
    python run_screening.py --data molecules.csv --query 42 --steps 200
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import numpy as np


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'default.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Molecular screening via MCMC random walk on hypergraphs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to molecule data file (CSV, SDF, or SMILES)'
    )
    parser.add_argument(
        '--query', '-q',
        type=int,
        required=True,
        help='Index of query molecule'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--steps', '-n',
        type=int,
        default=200,
        help='Number of MCMC steps'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['mh', 'sa', 'pt'],
        default='mh',
        help='Sampling algorithm: mh (Metropolis-Hastings), sa (Simulated Annealing), pt (Parallel Tempering)'
    )
    
    # Temperature settings
    parser.add_argument(
        '--initial-temp',
        type=float,
        default=1.0,
        help='Initial temperature'
    )
    parser.add_argument(
        '--final-temp',
        type=float,
        default=0.01,
        help='Final temperature'
    )
    parser.add_argument(
        '--cooling',
        type=str,
        choices=['exponential', 'linear', 'logarithmic', 'adaptive'],
        default='exponential',
        help='Cooling schedule'
    )
    
    # Hypergraph settings
    parser.add_argument(
        '--weighting',
        type=str,
        choices=['none', 'frequency', 'tfidf', 'bm25'],
        default='tfidf',
        help='Hyperedge weighting scheme'
    )
    parser.add_argument(
        '--min-hyperedge-size',
        type=int,
        default=2,
        help='Minimum hyperedge size'
    )
    
    # Fingerprint settings
    parser.add_argument(
        '--fp-type',
        type=str,
        choices=['morgan', 'rdkit', 'maccs'],
        default='morgan',
        help='Fingerprint type'
    )
    parser.add_argument(
        '--fp-radius',
        type=int,
        default=2,
        help='Fingerprint radius (for Morgan)'
    )
    
    # Similarity settings
    parser.add_argument(
        '--similarity',
        type=str,
        choices=['tanimoto', 'dice', 'cosine'],
        default='tanimoto',
        help='Similarity metric'
    )
    parser.add_argument(
        '--cache-size',
        type=int,
        default=10000,
        help='Similarity cache size'
    )
    
    # Output settings
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top results to return'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (JSON)'
    )
    parser.add_argument(
        '--save-trajectory',
        action='store_true',
        help='Save full MCMC trajectory'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default='mcmc_results.png',
        help='Output path for plots'
    )
    
    # Other settings
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--smiles-col',
        type=str,
        default='smiles',
        help='Name of SMILES column in CSV'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(verbose=not args.quiet)
    
    logger = logging.getLogger(__name__)
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
    
    logger.info("Starting Hypergraph-MCMC Molecular Screening")
    logger.info(f"Data file: {args.data}")
    logger.info(f"Query molecule: {args.query}")
    
    # Import here to avoid slow startup for --help
    from src.data import MoleculeLoader, FingerprintExtractor
    from src.hypergraph import HypergraphBuilder, get_weighting_scheme
    from src.sampler import (
        HypergraphRandomWalk, 
        MetropolisHastingsSampler,
        SimulatedAnnealingSampler,
        ParallelTemperingSampler
    )
    from src.similarity import LazySimilarityCache
    
    # Load molecules
    logger.info("Loading molecules...")
    loader = MoleculeLoader(args.data, smiles_col=args.smiles_col)
    molecules = loader.load()
    logger.info(f"Loaded {len(molecules)} molecules")
    
    # Validate query index
    if args.query < 0 or args.query >= len(molecules):
        logger.error(f"Invalid query index: {args.query}. Must be in [0, {len(molecules)-1}]")
        sys.exit(1)
    
    # Extract fingerprints
    logger.info(f"Extracting {args.fp_type} fingerprints (radius={args.fp_radius})...")
    fp_extractor = FingerprintExtractor(
        fp_type=args.fp_type,
        radius=args.fp_radius
    )
    fingerprints = fp_extractor.extract(molecules)
    logger.info(f"Extracted {fingerprints.n_features} unique features")
    
    # Build hypergraph
    logger.info("Building hypergraph...")
    builder = HypergraphBuilder(min_hyperedge_size=args.min_hyperedge_size)
    hypergraph = builder.build(fingerprints, precompute_transition=False)
    
    # Apply weighting
    if args.weighting != 'none':
        logger.info(f"Applying {args.weighting} weighting...")
        weighting = get_weighting_scheme(args.weighting)
        hypergraph = weighting.apply(hypergraph)
    
    # Compute transition matrix
    logger.info("Computing transition matrix...")
    hypergraph.compute_transition_matrix()
    
    # Initialize components
    logger.info("Initializing sampler components...")
    random_walk = HypergraphRandomWalk(hypergraph, seed=args.seed)
    
    similarity_cache = LazySimilarityCache(
        molecules,
        metric=args.similarity,
        max_size=args.cache_size,
        fp_type='topological',
        precompute_fps=True
    )
    
    # Create sampler
    if args.algorithm == 'mh':
        sampler = MetropolisHastingsSampler(
            random_walk=random_walk,
            similarity_cache=similarity_cache,
            initial_temp=args.initial_temp,
            final_temp=args.final_temp,
            cooling_schedule=args.cooling,
            seed=args.seed
        )
    elif args.algorithm == 'sa':
        sampler = SimulatedAnnealingSampler(
            random_walk=random_walk,
            similarity_cache=similarity_cache,
            initial_temp=args.initial_temp,
            final_temp=args.final_temp,
            cooling_schedule=args.cooling,
            seed=args.seed
        )
    else:  # pt
        sampler = ParallelTemperingSampler(
            random_walk=random_walk,
            similarity_cache=similarity_cache,
            temp_min=args.final_temp,
            temp_max=args.initial_temp,
            seed=args.seed
        )
    
    # Run sampling
    logger.info(f"Running {args.algorithm.upper()} sampling for {args.steps} steps...")
    result = sampler.sample(
        query_idx=args.query,
        n_steps=args.steps,
        n_results=args.top_k,
        save_trajectory=args.save_trajectory,
        progress_bar=not args.quiet
    )
    
    # Print results
    print("\n" + "="*60)
    print(result.summary())
    print("="*60)
    
    # Get ground truth (for comparison)
    logger.info("Computing ground truth (exhaustive search)...")
    ground_truth = similarity_cache.get_top_similar(
        args.query, 
        k=args.top_k,
        exclude_query=True
    )
    
    print("\nGround Truth (Exhaustive Search):")
    for idx, sim in ground_truth:
        print(f"  Molecule {idx}: {sim:.4f}")
    
    # Check if we found the best
    best_true = ground_truth[0] if ground_truth else (None, 0)
    if result.best_molecule == best_true[0]:
        print(f"\n✓ Found optimal molecule!")
    else:
        print(f"\n✗ Best found: {result.best_similarity:.4f}, Optimal: {best_true[1]:.4f}")
        print(f"  Gap: {best_true[1] - result.best_similarity:.4f}")
    
    # Cache stats
    cache_stats = similarity_cache.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
                f"hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Save output
    if args.output:
        output_data = {
            'query_idx': int(args.query),
            'n_steps': int(args.steps),
            'algorithm': args.algorithm,
            'best_molecule': int(result.best_molecule),
            'best_similarity': float(result.best_similarity),
            'n_unique_visited': int(result.n_unique_visited),
            'total_time': float(result.total_time),
            'top_molecules': [(int(idx), float(sim)) for idx, sim in result.top_molecules],
            'ground_truth': [(int(idx), float(sim)) for idx, sim in ground_truth],
            'cache_stats': {k: int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v for k, v in cache_stats.items()},
            'parameters': {
                'weighting': args.weighting,
                'fp_type': args.fp_type,
                'fp_radius': int(args.fp_radius),
                'initial_temp': float(args.initial_temp),
                'final_temp': float(args.final_temp),
                'cooling': args.cooling,
            }
        }
        
        if args.save_trajectory:
            output_data['trajectory'] = [int(x) for x in result.trajectory]
            output_data['similarities'] = [float(x) for x in result.similarities]
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Generate plots
    if args.plot:
        from src.utils.visualization import create_summary_plot
        import matplotlib.pyplot as plt
        
        fig = create_summary_plot(result)
        fig.savefig(args.plot_output, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {args.plot_output}")
        plt.close(fig)
    
    logger.info("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
