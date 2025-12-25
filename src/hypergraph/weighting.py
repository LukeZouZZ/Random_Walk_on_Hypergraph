"""
Hyperedge weighting schemes.

Different weighting strategies to prioritize informative molecular substructures
over common ones.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import logging

import numpy as np

from .builder import Hypergraph

logger = logging.getLogger(__name__)


class WeightingScheme(ABC):
    """Abstract base class for hyperedge weighting schemes."""
    
    @abstractmethod
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        """
        Compute weights for all hyperedges.
        
        Parameters
        ----------
        hypergraph : Hypergraph
            The hypergraph to weight.
        
        Returns
        -------
        np.ndarray
            Weight for each hyperedge.
        """
        pass
    
    def apply(self, hypergraph: Hypergraph) -> Hypergraph:
        """
        Apply weighting scheme to hypergraph.
        
        Parameters
        ----------
        hypergraph : Hypergraph
            The hypergraph to weight.
        
        Returns
        -------
        Hypergraph
            Hypergraph with updated weights.
        """
        weights = self.compute_weights(hypergraph)
        hypergraph.hyperedge_weights = weights
        
        # Invalidate cached matrices
        hypergraph._transition_matrix = None
        hypergraph._weighted_adjacency = None
        
        return hypergraph


class UniformWeighting(WeightingScheme):
    """
    Uniform weighting (all hyperedges have equal weight).
    
    This is the default scheme where w_e = 1 for all hyperedges.
    """
    
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        weights = np.ones(hypergraph.n_hyperedges, dtype=np.float32)
        logger.info("Applied uniform weighting")
        return weights


class FrequencyWeighting(WeightingScheme):
    """
    Frequency-based weighting (inverse of hyperedge size).
    
    Less common substructures get higher weights:
    w_e = 1 / |e|
    
    Parameters
    ----------
    power : float
        Exponent for the inverse frequency. Default is 1.
        Higher values give more weight to rare features.
    """
    
    def __init__(self, power: float = 1.0):
        self.power = power
    
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        sizes = hypergraph.hyperedge_sizes.astype(np.float32)
        weights = 1.0 / np.power(sizes, self.power)
        
        # Normalize to have mean 1
        weights = weights / weights.mean()
        
        logger.info(f"Applied frequency weighting (power={self.power})")
        return weights


class TFIDFWeighting(WeightingScheme):
    """
    TF-IDF inspired weighting for hyperedges.
    
    Hyperedges (substructures) that appear in fewer molecules are more informative
    and receive higher weights:
    
    w_e = log(N / |e|)
    
    where N is the total number of vertices (molecules) and |e| is the number
    of vertices in hyperedge e.
    
    Parameters
    ----------
    smooth : bool
        Whether to use smoothed IDF: log((N + 1) / (|e| + 1)) + 1
    sublinear : bool
        Whether to use sublinear scaling.
    """
    
    def __init__(self, smooth: bool = True, sublinear: bool = False):
        self.smooth = smooth
        self.sublinear = sublinear
    
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        N = hypergraph.n_vertices
        sizes = hypergraph.hyperedge_sizes.astype(np.float32)
        
        if self.smooth:
            # Smoothed IDF to avoid log(0)
            weights = np.log((N + 1) / (sizes + 1)) + 1
        else:
            # Standard IDF
            weights = np.log(N / np.maximum(sizes, 1))
        
        if self.sublinear:
            weights = 1 + np.log(weights + 1)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        # Normalize to have mean 1
        if weights.mean() > 0:
            weights = weights / weights.mean()
        
        logger.info(
            f"Applied TF-IDF weighting (smooth={self.smooth}, sublinear={self.sublinear})"
        )
        return weights


class BM25Weighting(WeightingScheme):
    """
    BM25-inspired weighting for hyperedges.
    
    A more sophisticated weighting that balances between TF-IDF and frequency:
    
    w_e = IDF(e) * (k1 + 1) / (frequency_factor + k1)
    
    where:
    - IDF(e) = log((N - |e| + 0.5) / (|e| + 0.5))
    - frequency_factor accounts for hyperedge saturation
    
    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter. Default is 1.5.
    b : float
        Length normalization parameter. Default is 0.75.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
    
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        N = hypergraph.n_vertices
        sizes = hypergraph.hyperedge_sizes.astype(np.float32)
        
        # Average hyperedge size
        avg_size = sizes.mean()
        
        # IDF component
        idf = np.log((N - sizes + 0.5) / (sizes + 0.5) + 1)
        
        # Length normalization
        length_norm = 1 - self.b + self.b * (sizes / avg_size)
        
        # BM25-style weighting
        # Using a simplified version since we don't have "term frequency" per se
        weights = idf / (length_norm + 1e-8)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        # Normalize
        if weights.mean() > 0:
            weights = weights / weights.mean()
        
        logger.info(f"Applied BM25 weighting (k1={self.k1}, b={self.b})")
        return weights


class InformationGainWeighting(WeightingScheme):
    """
    Information gain based weighting.
    
    Weights hyperedges by their entropy reduction potential:
    
    w_e = H(V) - H(V|e)
    
    where H(V) is the entropy of the vertex distribution and H(V|e) is
    the conditional entropy given the hyperedge.
    
    This requires knowing which vertices are "positive" examples,
    so it's primarily useful when you have a target vertex set.
    """
    
    def __init__(self, target_vertices: Optional[np.ndarray] = None):
        self.target_vertices = target_vertices
    
    def compute_weights(self, hypergraph: Hypergraph) -> np.ndarray:
        N = hypergraph.n_vertices
        sizes = hypergraph.hyperedge_sizes.astype(np.float32)
        
        if self.target_vertices is None:
            # Without targets, fall back to entropy of hyperedge size
            p = sizes / N
            # Entropy-like measure: -p*log(p) - (1-p)*log(1-p)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            weights = -p * np.log(p) - (1 - p) * np.log(1 - p)
        else:
            # Compute information gain with respect to target
            H = hypergraph.incidence_matrix
            target_set = set(self.target_vertices)
            
            weights = np.zeros(hypergraph.n_hyperedges, dtype=np.float32)
            
            for e_idx in range(hypergraph.n_hyperedges):
                members = set(H.getcol(e_idx).nonzero()[0])
                
                # Count targets in and out of hyperedge
                targets_in = len(members & target_set)
                targets_out = len(target_set) - targets_in
                non_targets_in = len(members) - targets_in
                non_targets_out = N - len(members) - targets_out
                
                # Information gain calculation
                total = N
                p_target = len(target_set) / total
                
                # Entropy before split
                H_before = self._entropy(p_target)
                
                # Entropy after split
                size_in = len(members)
                size_out = total - size_in
                
                if size_in > 0:
                    p_in = targets_in / size_in
                    H_in = self._entropy(p_in)
                else:
                    H_in = 0
                
                if size_out > 0:
                    p_out = targets_out / size_out
                    H_out = self._entropy(p_out)
                else:
                    H_out = 0
                
                H_after = (size_in / total) * H_in + (size_out / total) * H_out
                
                weights[e_idx] = H_before - H_after
        
        # Normalize
        if weights.max() > 0:
            weights = weights / weights.max()
        
        logger.info("Applied information gain weighting")
        return weights
    
    @staticmethod
    def _entropy(p: float) -> float:
        """Compute binary entropy."""
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def get_weighting_scheme(name: str, **kwargs) -> WeightingScheme:
    """
    Factory function to get weighting scheme by name.
    
    Parameters
    ----------
    name : str
        Name of the weighting scheme: 'uniform', 'frequency', 'tfidf', or 'bm25'.
    **kwargs
        Additional arguments passed to the weighting scheme constructor.
    
    Returns
    -------
    WeightingScheme
        Instantiated weighting scheme.
    """
    schemes = {
        'none': UniformWeighting,
        'uniform': UniformWeighting,
        'frequency': FrequencyWeighting,
        'tfidf': TFIDFWeighting,
        'bm25': BM25Weighting,
        'info_gain': InformationGainWeighting,
    }
    
    name = name.lower()
    if name not in schemes:
        raise ValueError(
            f"Unknown weighting scheme: {name}. "
            f"Available: {list(schemes.keys())}"
        )
    
    return schemes[name](**kwargs)
