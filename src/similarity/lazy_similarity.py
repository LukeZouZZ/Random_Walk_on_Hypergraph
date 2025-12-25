"""
Lazy similarity computation with caching.

Computes molecular similarities on-demand with LRU caching to avoid
redundant calculations for large datasets.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from functools import lru_cache
from collections import OrderedDict
import logging

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class SimilarityMetric(ABC):
    """Abstract base class for similarity metrics."""
    
    @abstractmethod
    def compute(self, fp1, fp2) -> float:
        """
        Compute similarity between two fingerprints.
        
        Parameters
        ----------
        fp1 : fingerprint
            First fingerprint.
        fp2 : fingerprint
            Second fingerprint.
        
        Returns
        -------
        float
            Similarity score in [0, 1].
        """
        pass
    
    @abstractmethod
    def compute_bulk(self, fp1, fp_list) -> np.ndarray:
        """
        Compute similarity between one fingerprint and a list.
        
        Parameters
        ----------
        fp1 : fingerprint
            Query fingerprint.
        fp_list : list
            List of fingerprints.
        
        Returns
        -------
        np.ndarray
            Array of similarities.
        """
        pass


class TanimotoSimilarity(SimilarityMetric):
    """
    Tanimoto (Jaccard) similarity.
    
    Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
    """
    
    def compute(self, fp1, fp2) -> float:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def compute_bulk(self, fp1, fp_list) -> np.ndarray:
        return np.array(DataStructs.BulkTanimotoSimilarity(fp1, fp_list))


class DiceSimilarity(SimilarityMetric):
    """
    Dice similarity.
    
    Dice(A, B) = 2|A ∩ B| / (|A| + |B|)
    """
    
    def compute(self, fp1, fp2) -> float:
        return DataStructs.DiceSimilarity(fp1, fp2)
    
    def compute_bulk(self, fp1, fp_list) -> np.ndarray:
        return np.array(DataStructs.BulkDiceSimilarity(fp1, fp_list))


class CosineSimilarity(SimilarityMetric):
    """
    Cosine similarity.
    
    Cosine(A, B) = |A ∩ B| / sqrt(|A| * |B|)
    """
    
    def compute(self, fp1, fp2) -> float:
        return DataStructs.CosineSimilarity(fp1, fp2)
    
    def compute_bulk(self, fp1, fp_list) -> np.ndarray:
        return np.array(DataStructs.BulkCosineSimilarity(fp1, fp_list))


class TverskySimilarity(SimilarityMetric):
    """
    Tversky similarity (asymmetric).
    
    Tversky(A, B) = |A ∩ B| / (|A ∩ B| + α|A - B| + β|B - A|)
    
    α = β = 1 gives Tanimoto
    α = β = 0.5 gives Dice
    
    Parameters
    ----------
    alpha : float
        Weight for features in A but not B.
    beta : float
        Weight for features in B but not A.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
    
    def compute(self, fp1, fp2) -> float:
        return DataStructs.TverskySimilarity(fp1, fp2, self.alpha, self.beta)
    
    def compute_bulk(self, fp1, fp_list) -> np.ndarray:
        return np.array(DataStructs.BulkTverskySimilarity(
            fp1, fp_list, self.alpha, self.beta
        ))


def get_similarity_metric(name: str, **kwargs) -> SimilarityMetric:
    """
    Factory function to get similarity metric by name.
    
    Parameters
    ----------
    name : str
        Name of the metric: 'tanimoto', 'dice', 'cosine', or 'tversky'.
    **kwargs
        Additional arguments for the metric.
    
    Returns
    -------
    SimilarityMetric
        Instantiated similarity metric.
    """
    metrics = {
        'tanimoto': TanimotoSimilarity,
        'dice': DiceSimilarity,
        'cosine': CosineSimilarity,
        'tversky': TverskySimilarity,
    }
    
    name = name.lower()
    if name not in metrics:
        raise ValueError(
            f"Unknown similarity metric: {name}. "
            f"Available: {list(metrics.keys())}"
        )
    
    return metrics[name](**kwargs)


class LRUCache:
    """
    Simple LRU cache implementation.
    
    Parameters
    ----------
    max_size : int
        Maximum number of items to cache.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache, moving it to end (most recent)."""
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        """Put item in cache, evicting oldest if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __len__(self):
        return len(self.cache)


class LazySimilarityCache:
    """
    Lazy similarity computation with LRU caching.
    
    Computes similarities on-demand and caches results to avoid
    redundant calculations. Essential for scaling to large datasets.
    
    Parameters
    ----------
    molecules : List[Chem.Mol]
        List of RDKit molecule objects.
    metric : str or SimilarityMetric
        Similarity metric to use.
    max_size : int
        Maximum cache size.
    fp_type : str
        Fingerprint type: 'morgan', 'rdkit', or 'topological'.
    fp_radius : int
        Radius for Morgan fingerprints.
    precompute_fps : bool
        Whether to precompute all fingerprints upfront.
    
    Examples
    --------
    >>> cache = LazySimilarityCache(molecules, metric='tanimoto')
    >>> sim = cache.get_similarity(0, 42)  # Computed on-demand
    >>> sim = cache.get_similarity(0, 42)  # Retrieved from cache
    """
    
    def __init__(
        self,
        molecules: List[Chem.Mol],
        metric: Union[str, SimilarityMetric] = 'tanimoto',
        max_size: int = 10000,
        fp_type: str = 'topological',
        fp_radius: int = 2,
        precompute_fps: bool = True
    ):
        self.molecules = molecules
        self.n_molecules = len(molecules)
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        
        if isinstance(metric, str):
            self.metric = get_similarity_metric(metric)
        else:
            self.metric = metric
        
        # Similarity cache
        self.cache = LRUCache(max_size)
        
        # Fingerprint cache
        self.fingerprints = {}
        
        if precompute_fps:
            logger.info("Precomputing fingerprints...")
            for i, mol in enumerate(molecules):
                if mol is not None:
                    self.fingerprints[i] = self._compute_fingerprint(mol)
            logger.info(f"Computed {len(self.fingerprints)} fingerprints")
    
    def _compute_fingerprint(self, mol: Chem.Mol):
        """Compute fingerprint for a molecule."""
        if self.fp_type == 'morgan':
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_radius, nBits=2048
            )
        elif self.fp_type == 'rdkit':
            return Chem.RDKFingerprint(mol)
        else:  # topological
            return FingerprintMols.FingerprintMol(mol)
    
    def _get_fingerprint(self, idx: int):
        """Get fingerprint for molecule index, computing if necessary."""
        if idx not in self.fingerprints:
            mol = self.molecules[idx]
            if mol is None:
                return None
            self.fingerprints[idx] = self._compute_fingerprint(mol)
        return self.fingerprints[idx]
    
    def get_similarity(self, idx1: int, idx2: int) -> float:
        """
        Get similarity between two molecules.
        
        Parameters
        ----------
        idx1 : int
            Index of first molecule.
        idx2 : int
            Index of second molecule.
        
        Returns
        -------
        float
            Similarity score.
        """
        if idx1 == idx2:
            return 1.0
        
        # Normalize key order for symmetric similarity
        key = (min(idx1, idx2), max(idx1, idx2))
        
        # Check cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Compute similarity
        fp1 = self._get_fingerprint(idx1)
        fp2 = self._get_fingerprint(idx2)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        similarity = self.metric.compute(fp1, fp2)
        
        # Cache result
        self.cache.put(key, similarity)
        
        return similarity
    
    def get_similarities_to_query(
        self, 
        query_idx: int,
        molecule_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get similarities of multiple molecules to a query.
        
        Parameters
        ----------
        query_idx : int
            Index of query molecule.
        molecule_indices : List[int], optional
            Indices of molecules to compare. If None, use all.
        
        Returns
        -------
        np.ndarray
            Array of similarities.
        """
        if molecule_indices is None:
            molecule_indices = range(self.n_molecules)
        
        similarities = np.zeros(len(molecule_indices))
        
        for i, idx in enumerate(molecule_indices):
            similarities[i] = self.get_similarity(query_idx, idx)
        
        return similarities
    
    def get_top_similar(
        self,
        query_idx: int,
        k: int = 10,
        exclude_query: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Get top-k most similar molecules to query.
    
        Note: This computes all pairwise similarities, which is expensive
        for large datasets. Use MCMC sampling instead for scalability.
        """
        # Use individual similarity computation to avoid length mismatch issues
        results = []
        for i in range(self.n_molecules):
            if exclude_query and i == query_idx:
                continue
            sim = self.get_similarity(query_idx, i)
            results.append((i, sim))
    
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
    
        return results[:k]
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.cache.max_size,
            'hits': self.cache.hits,
            'misses': self.cache.misses,
            'hit_rate': self.cache.hit_rate,
            'n_fingerprints': len(self.fingerprints),
        }
    
    def clear_cache(self):
        """Clear similarity cache (keeps fingerprints)."""
        self.cache.clear()
    
    def precompute_for_query(self, query_idx: int, neighbor_indices: List[int]):
        """
        Precompute similarities for a query's neighbors.
        
        Useful for warming up the cache before MCMC sampling.
        
        Parameters
        ----------
        query_idx : int
            Index of query molecule.
        neighbor_indices : List[int]
            Indices of potential neighbors to precompute.
        """
        query_fp = self._get_fingerprint(query_idx)
        if query_fp is None:
            return
        
        for idx in neighbor_indices:
            if idx != query_idx:
                self.get_similarity(query_idx, idx)
