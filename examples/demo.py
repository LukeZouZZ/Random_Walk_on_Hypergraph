#!/usr/bin/env python
"""
Demo script showing how to use Hypergraph-MCMC for molecular screening.

This example downloads a sample dataset and demonstrates the complete workflow.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run demo."""
    # Import modules
    from src.data import MoleculeLoader, FingerprintExtractor, load_molecules_from_url
    from src.hypergraph import HypergraphBuilder, TFIDFWeighting
    from src.sampler import HypergraphRandomWalk, MetropolisHastingsSampler
    from src.similarity import LazySimilarityCache
    
    # =========================================================================
    # Step 1: Load molecules
    # =========================================================================
    logger.info("Step 1: Loading molecules...")
    
    # Download Platinum dataset (a curated set of drug-like molecules)
    url = 'https://raw.githubusercontent.com/onecoinbuybus/Database_chemoinformatics/master/platinum_dataset.csv'
    
    try:
        molecules = load_molecules_from_url(url, smiles_col='smiles')
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("Using a small synthetic dataset instead...")
        
        # Create a small synthetic dataset
        from rdkit import Chem
        smiles_list = [
            'CCO', 'CCCO', 'CCCCO', 'CC(C)O', 'CC(C)CO',
            'c1ccccc1', 'c1ccccc1O', 'c1ccccc1CO', 'c1ccccc1CCO',
            'CC(=O)O', 'CCC(=O)O', 'CC(=O)OC', 'CC(=O)Oc1ccccc1',
            'CN', 'CCN', 'CCCN', 'c1ccccc1N', 'c1ccccc1CN',
            'CC(C)N', 'CC(C)(C)N', 'C1CCCCC1', 'C1CCNCC1', 'c1ccc2ccccc2c1'
        ]
        molecules = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    logger.info(f"Loaded {len(molecules)} molecules")
    
    # =========================================================================
    # Step 2: Extract fingerprints
    # =========================================================================
    logger.info("Step 2: Extracting fingerprints...")
    
    fp_extractor = FingerprintExtractor(
        fp_type='morgan',
        radius=2,  # ECFP4
        n_bits=None  # Unhashed for hypergraph construction
    )
    fingerprints = fp_extractor.extract(molecules)
    
    logger.info(f"Extracted {fingerprints.n_features} unique substructure features")
    
    # =========================================================================
    # Step 3: Build hypergraph
    # =========================================================================
    logger.info("Step 3: Building hypergraph...")
    
    builder = HypergraphBuilder(min_hyperedge_size=2)
    hypergraph = builder.build(fingerprints, precompute_transition=False)
    
    logger.info(f"Hypergraph: {hypergraph.n_vertices} vertices, {hypergraph.n_hyperedges} hyperedges")
    
    # =========================================================================
    # Step 4: Apply TF-IDF weighting
    # =========================================================================
    logger.info("Step 4: Applying TF-IDF weighting...")
    
    weighting = TFIDFWeighting(smooth=True)
    hypergraph = weighting.apply(hypergraph)
    
    # Compute transition matrix
    hypergraph.compute_transition_matrix()
    
    # =========================================================================
    # Step 5: Initialize components
    # =========================================================================
    logger.info("Step 5: Initializing random walk and similarity cache...")
    
    random_walk = HypergraphRandomWalk(hypergraph, seed=42)
    
    similarity_cache = LazySimilarityCache(
        molecules,
        metric='tanimoto',
        max_size=10000,
        precompute_fps=True
    )
    
    # =========================================================================
    # Step 6: Run MCMC sampling
    # =========================================================================
    logger.info("Step 6: Running Metropolis-Hastings MCMC...")
    
    # Pick a random query molecule
    np.random.seed(42)
    query_idx = np.random.randint(0, len(molecules))
    logger.info(f"Query molecule index: {query_idx}")
    
    sampler = MetropolisHastingsSampler(
        random_walk=random_walk,
        similarity_cache=similarity_cache,
        initial_temp=1.0,
        final_temp=0.01,
        cooling_schedule='exponential',
        seed=42
    )
    
    result = sampler.sample(
        query_idx=query_idx,
        n_steps=200,
        n_results=10,
        save_trajectory=True,
        progress_bar=True
    )
    
    # =========================================================================
    # Step 7: Analyze results
    # =========================================================================
    logger.info("Step 7: Analyzing results...")
    
    print("\n" + "="*60)
    print("MCMC SAMPLING RESULTS")
    print("="*60)
    print(result.summary())
    
    # Compare with exhaustive search
    print("\n" + "-"*60)
    print("COMPARISON WITH EXHAUSTIVE SEARCH")
    print("-"*60)
    
    ground_truth = similarity_cache.get_top_similar(query_idx, k=10)
    
    print(f"\nGround truth (optimal):")
    for rank, (idx, sim) in enumerate(ground_truth, 1):
        print(f"  {rank}. Molecule {idx}: similarity = {sim:.4f}")
    
    # Check accuracy
    mcmc_best = result.best_molecule
    true_best = ground_truth[0][0]
    
    print(f"\nMCMC found best: {'✓ YES' if mcmc_best == true_best else '✗ NO'}")
    print(f"MCMC best similarity: {result.best_similarity:.4f}")
    print(f"True best similarity: {ground_truth[0][1]:.4f}")
    
    # Efficiency metrics
    print(f"\nEfficiency:")
    print(f"  Molecules evaluated: {result.n_unique_visited}")
    print(f"  Total molecules: {len(molecules)}")
    print(f"  Reduction: {100 * (1 - result.n_unique_visited / len(molecules)):.1f}%")
    print(f"  Time: {result.total_time:.3f}s")
    
    # Cache stats
    cache_stats = similarity_cache.get_cache_stats()
    print(f"\nCache performance:")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
    
    # =========================================================================
    # Step 8: Visualize results (optional)
    # =========================================================================
    try:
        from src.utils.visualization import create_summary_plot
        import matplotlib.pyplot as plt
        
        logger.info("Step 8: Creating visualization...")
        
        fig = create_summary_plot(result)
        fig.savefig('demo_results.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: demo_results.png")
        plt.close(fig)
        
    except ImportError:
        logger.info("Matplotlib not available, skipping visualization")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
