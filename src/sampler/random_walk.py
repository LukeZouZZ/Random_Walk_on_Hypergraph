"""
Random walk on hypergraphs.

Implements various random walk strategies on hypergraphs for molecular screening.
"""

from typing import List, Optional, Tuple
import logging

import numpy as np

from ..hypergraph.builder import Hypergraph

logger = logging.getLogger(__name__)


class HypergraphRandomWalk:
    """
    Random walk on a hypergraph.
    
    The random walk transition from vertex u to v considers all hyperedges
    containing both u and v, weighted by hyperedge weights and sizes.
    
    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph to walk on.
    lazy : float
        Probability of staying at current vertex (lazy random walk).
        Default is 0 (no laziness).
    seed : int, optional
        Random seed for reproducibility.
    
    Examples
    --------
    >>> rw = HypergraphRandomWalk(hypergraph)
    >>> path = rw.walk(start_vertex=0, n_steps=100)
    """
    
    def __init__(
        self,
        hypergraph: Hypergraph,
        lazy: float = 0.0,
        seed: Optional[int] = None
    ):
        self.hypergraph = hypergraph
        self.lazy = lazy
        self.rng = np.random.default_rng(seed)
        
        # Ensure transition matrix is computed
        self.hypergraph.compute_transition_matrix()
    
    def step(self, current_vertex: int) -> int:
        """
        Take one step of the random walk.
        
        Parameters
        ----------
        current_vertex : int
            Current vertex index.
        
        Returns
        -------
        int
            Next vertex index.
        """
        # Lazy random walk: stay with probability self.lazy
        if self.lazy > 0 and self.rng.random() < self.lazy:
            return current_vertex
        
        return self.hypergraph.sample_next_vertex(current_vertex, self.rng)
    
    def walk(
        self, 
        start_vertex: int, 
        n_steps: int,
        return_counts: bool = False
    ) -> np.ndarray:
        """
        Perform a random walk.
        
        Parameters
        ----------
        start_vertex : int
            Starting vertex index.
        n_steps : int
            Number of steps to take.
        return_counts : bool
            If True, return visit counts instead of path.
        
        Returns
        -------
        np.ndarray
            If return_counts is False: path of visited vertices.
            If return_counts is True: visit count for each vertex.
        """
        if return_counts:
            counts = np.zeros(self.hypergraph.n_vertices, dtype=np.int32)
            current = start_vertex
            
            for _ in range(n_steps):
                current = self.step(current)
                counts[current] += 1
            
            return counts
        else:
            path = np.zeros(n_steps + 1, dtype=np.int32)
            path[0] = start_vertex
            
            current = start_vertex
            for i in range(n_steps):
                current = self.step(current)
                path[i + 1] = current
            
            return path
    
    def get_transition_probability(self, u: int, v: int) -> float:
        """
        Get transition probability from u to v.
        
        Parameters
        ----------
        u : int
            Source vertex.
        v : int
            Target vertex.
        
        Returns
        -------
        float
            Transition probability P(u -> v).
        """
        neighbors, probs = self.hypergraph.get_transition_probabilities(u)
        
        idx = np.where(neighbors == v)[0]
        if len(idx) == 0:
            return 0.0
        
        return probs[idx[0]]
    
    def get_proposal_ratio(self, u: int, v: int) -> float:
        """
        Compute the proposal ratio P(v -> u) / P(u -> v).
        
        Used for Metropolis-Hastings acceptance ratio.
        
        Parameters
        ----------
        u : int
            Current vertex.
        v : int
            Proposed vertex.
        
        Returns
        -------
        float
            Proposal ratio.
        """
        p_uv = self.get_transition_probability(u, v)
        p_vu = self.get_transition_probability(v, u)
        
        if p_uv == 0:
            return 0.0
        
        return p_vu / p_uv
    
    def stationary_distribution(
        self, 
        n_steps: int = 10000,
        n_walks: int = 10
    ) -> np.ndarray:
        """
        Estimate stationary distribution via random walks.
        
        Parameters
        ----------
        n_steps : int
            Number of steps per walk.
        n_walks : int
            Number of independent walks.
        
        Returns
        -------
        np.ndarray
            Estimated stationary distribution.
        """
        total_counts = np.zeros(self.hypergraph.n_vertices, dtype=np.float64)
        
        for _ in range(n_walks):
            start = self.rng.integers(0, self.hypergraph.n_vertices)
            counts = self.walk(start, n_steps, return_counts=True)
            total_counts += counts
        
        # Normalize
        total_counts /= total_counts.sum()
        
        return total_counts
    
    def mixing_time_estimate(
        self,
        epsilon: float = 0.01,
        max_steps: int = 10000
    ) -> int:
        """
        Estimate mixing time of the random walk.
        
        Uses total variation distance to estimate when the walk
        has approximately mixed.
        
        Parameters
        ----------
        epsilon : float
            Target total variation distance.
        max_steps : int
            Maximum number of steps to try.
        
        Returns
        -------
        int
            Estimated mixing time.
        """
        n = self.hypergraph.n_vertices
        
        # Start from uniform distribution
        current_dist = np.ones(n) / n
        
        # Get transition matrix
        T = self.hypergraph._transition_matrix.toarray()
        
        for t in range(1, max_steps + 1):
            current_dist = current_dist @ T
            
            # Total variation to uniform
            tv_distance = 0.5 * np.abs(current_dist - 1/n).sum()
            
            if tv_distance < epsilon:
                return t
        
        logger.warning(
            f"Mixing time not reached within {max_steps} steps "
            f"(TV distance: {tv_distance:.4f})"
        )
        return max_steps


class BiasedRandomWalk(HypergraphRandomWalk):
    """
    Biased random walk that prefers vertices similar to a target.
    
    The transition probability is modified by a bias factor based on
    similarity to a target vertex.
    
    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph to walk on.
    similarity_func : callable
        Function that takes (vertex_idx, target_idx) and returns similarity.
    target_vertex : int
        Target vertex to bias towards.
    bias_strength : float
        Strength of the bias. Higher values give stronger preference
        to similar vertices.
    """
    
    def __init__(
        self,
        hypergraph: Hypergraph,
        similarity_func,
        target_vertex: int,
        bias_strength: float = 1.0,
        **kwargs
    ):
        super().__init__(hypergraph, **kwargs)
        self.similarity_func = similarity_func
        self.target_vertex = target_vertex
        self.bias_strength = bias_strength
        
        # Cache for biased probabilities
        self._biased_cache = {}
    
    def step(self, current_vertex: int) -> int:
        """Take one biased step."""
        neighbors, base_probs = self.hypergraph.get_transition_probabilities(current_vertex)
        
        if len(neighbors) == 0:
            return current_vertex
        
        # Compute similarity-based bias
        similarities = np.array([
            self.similarity_func(v, self.target_vertex)
            for v in neighbors
        ])
        
        # Apply bias: P'(v) ∝ P(v) * exp(β * sim(v, target))
        bias = np.exp(self.bias_strength * similarities)
        biased_probs = base_probs * bias
        
        # Normalize
        biased_probs /= biased_probs.sum()
        
        return self.rng.choice(neighbors, p=biased_probs)
