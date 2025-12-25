"""
Hypergraph construction from molecular fingerprints.

The hypergraph represents molecules as vertices and fingerprint features as hyperedges.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import numpy as np
from scipy import sparse

from ..data.fingerprint import FingerprintResult

logger = logging.getLogger(__name__)


@dataclass
class Hypergraph:
    """
    Hypergraph data structure for molecular screening.
    
    A hypergraph H = (V, E) where:
    - V: set of vertices (molecules)
    - E: set of hyperedges (fingerprint features)
    
    Attributes
    ----------
    incidence_matrix : scipy.sparse.csr_matrix
        Binary incidence matrix H where H[v,e] = 1 if vertex v is in hyperedge e.
        Shape: (n_vertices, n_hyperedges)
    hyperedge_weights : np.ndarray
        Weight for each hyperedge. Shape: (n_hyperedges,)
    vertex_degrees : np.ndarray
        Degree of each vertex (number of hyperedges containing it).
    hyperedge_sizes : np.ndarray
        Size of each hyperedge (number of vertices in it).
    feature_ids : List[int]
        Original feature identifiers.
    """
    incidence_matrix: sparse.csr_matrix
    hyperedge_weights: np.ndarray
    vertex_degrees: np.ndarray
    hyperedge_sizes: np.ndarray
    feature_ids: List[int]
    
    # Cached matrices for random walk
    _transition_matrix: Optional[sparse.csr_matrix] = field(default=None, repr=False)
    _weighted_adjacency: Optional[sparse.csr_matrix] = field(default=None, repr=False)
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices (molecules)."""
        return self.incidence_matrix.shape[0]
    
    @property
    def n_hyperedges(self) -> int:
        """Number of hyperedges (features)."""
        return self.incidence_matrix.shape[1]
    
    def get_hyperedge_members(self, hyperedge_idx: int) -> np.ndarray:
        """Get vertex indices belonging to a hyperedge."""
        return self.incidence_matrix.getcol(hyperedge_idx).nonzero()[0]
    
    def get_vertex_hyperedges(self, vertex_idx: int) -> np.ndarray:
        """Get hyperedge indices containing a vertex."""
        return self.incidence_matrix.getrow(vertex_idx).nonzero()[1]
    
    def get_neighbors(self, vertex_idx: int) -> np.ndarray:
        """
        Get all vertices connected to a vertex through any hyperedge.
        
        A vertex u is a neighbor of v if there exists a hyperedge e
        such that both u and v are in e.
        """
        # Get all hyperedges containing this vertex
        hyperedges = self.get_vertex_hyperedges(vertex_idx)
        
        # Get all vertices in those hyperedges
        neighbors = set()
        for e_idx in hyperedges:
            members = self.get_hyperedge_members(e_idx)
            neighbors.update(members)
        
        # Remove the vertex itself
        neighbors.discard(vertex_idx)
        
        return np.array(sorted(neighbors))
    
    def compute_transition_matrix(self) -> sparse.csr_matrix:
        """
        Compute the random walk transition matrix.
        
        The transition probability from u to v is:
        P(u -> v) = sum_{e: u,v in e} w_e / (|e| - 1)
                    / sum_{v' != u} sum_{e: u,v' in e} w_e / (|e| - 1)
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Row-stochastic transition matrix.
        """
        if self._transition_matrix is not None:
            return self._transition_matrix
        
        logger.info("Computing transition matrix...")
        
        H = self.incidence_matrix
        n = self.n_vertices
        
        # Compute weighted contributions from each hyperedge
        # Weight: w_e / (|e| - 1) for each pair in the hyperedge
        edge_contrib = self.hyperedge_weights / np.maximum(self.hyperedge_sizes - 1, 1)
        
        # Create diagonal matrix of edge contributions
        W_diag = sparse.diags(edge_contrib)
        
        # Weighted adjacency: A_w = H @ W @ H^T
        # A_w[u,v] = sum_e H[u,e] * w_e/(|e|-1) * H[v,e]
        weighted_adj = H @ W_diag @ H.T
        
        # Convert to dense for diagonal manipulation (for small matrices)
        # For large matrices, we handle this more efficiently
        if n < 10000:
            weighted_adj = weighted_adj.toarray()
            np.fill_diagonal(weighted_adj, 0)  # Remove self-loops
            
            # Normalize rows to get transition probabilities
            row_sums = weighted_adj.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_matrix = weighted_adj / row_sums
            transition_matrix = sparse.csr_matrix(transition_matrix)
        else:
            # For large matrices, handle sparsely
            weighted_adj = weighted_adj.tolil()
            for i in range(n):
                weighted_adj[i, i] = 0
            weighted_adj = weighted_adj.tocsr()
            
            row_sums = np.asarray(weighted_adj.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1
            
            # Normalize rows
            inv_row_sums = sparse.diags(1.0 / row_sums)
            transition_matrix = inv_row_sums @ weighted_adj
        
        self._transition_matrix = transition_matrix
        self._weighted_adjacency = sparse.csr_matrix(weighted_adj) if isinstance(weighted_adj, np.ndarray) else weighted_adj
        
        logger.info(f"Transition matrix computed: {transition_matrix.shape}")
        
        return self._transition_matrix
    
    def get_transition_probabilities(self, vertex_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get transition probabilities from a vertex.
        
        Parameters
        ----------
        vertex_idx : int
            Source vertex index.
        
        Returns
        -------
        neighbors : np.ndarray
            Indices of reachable vertices.
        probabilities : np.ndarray
            Transition probabilities (sum to 1).
        """
        if self._transition_matrix is None:
            self.compute_transition_matrix()
        
        row = self._transition_matrix.getrow(vertex_idx)
        neighbors = row.indices
        probabilities = row.data
        
        return neighbors, probabilities
    
    def sample_next_vertex(
        self, 
        current_vertex: int, 
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Sample next vertex according to transition probabilities.
        
        Parameters
        ----------
        current_vertex : int
            Current vertex index.
        rng : np.random.Generator, optional
            Random number generator.
        
        Returns
        -------
        int
            Sampled next vertex index.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        neighbors, probs = self.get_transition_probabilities(current_vertex)
        
        if len(neighbors) == 0:
            # No neighbors, return current vertex
            return current_vertex
        
        return rng.choice(neighbors, p=probs)


class HypergraphBuilder:
    """
    Build hypergraph from fingerprint data.
    
    Parameters
    ----------
    min_hyperedge_size : int
        Minimum number of vertices in a hyperedge.
    max_hyperedge_size : int, optional
        Maximum number of vertices in a hyperedge.
    
    Examples
    --------
    >>> builder = HypergraphBuilder(min_hyperedge_size=2)
    >>> hypergraph = builder.build(fingerprint_result)
    """
    
    def __init__(
        self,
        min_hyperedge_size: int = 2,
        max_hyperedge_size: Optional[int] = None
    ):
        self.min_hyperedge_size = min_hyperedge_size
        self.max_hyperedge_size = max_hyperedge_size
    
    def build(
        self, 
        fingerprints: FingerprintResult,
        precompute_transition: bool = True
    ) -> Hypergraph:
        """
        Build hypergraph from fingerprint results.
        
        Parameters
        ----------
        fingerprints : FingerprintResult
            Fingerprint extraction results.
        precompute_transition : bool
            Whether to precompute transition matrix.
        
        Returns
        -------
        Hypergraph
            Constructed hypergraph.
        """
        H = fingerprints.matrix
        
        # Compute hyperedge sizes
        hyperedge_sizes = np.asarray(H.sum(axis=0)).flatten()
        
        # Filter hyperedges by size
        mask = hyperedge_sizes >= self.min_hyperedge_size
        if self.max_hyperedge_size is not None:
            mask &= hyperedge_sizes <= self.max_hyperedge_size
        
        kept_indices = np.where(mask)[0]
        
        if len(kept_indices) < H.shape[1]:
            logger.info(
                f"Filtered hyperedges: {H.shape[1]} -> {len(kept_indices)} "
                f"(size in [{self.min_hyperedge_size}, {self.max_hyperedge_size or 'inf'}])"
            )
            H = H[:, kept_indices]
            hyperedge_sizes = hyperedge_sizes[kept_indices]
            feature_ids = [fingerprints.feature_ids[i] for i in kept_indices]
        else:
            feature_ids = fingerprints.feature_ids
        
        # Compute vertex degrees
        vertex_degrees = np.asarray(H.sum(axis=1)).flatten()
        
        # Initialize uniform weights
        hyperedge_weights = np.ones(len(hyperedge_sizes), dtype=np.float32)
        
        hypergraph = Hypergraph(
            incidence_matrix=H,
            hyperedge_weights=hyperedge_weights,
            vertex_degrees=vertex_degrees,
            hyperedge_sizes=hyperedge_sizes,
            feature_ids=feature_ids
        )
        
        if precompute_transition:
            hypergraph.compute_transition_matrix()
        
        logger.info(
            f"Built hypergraph: {hypergraph.n_vertices} vertices, "
            f"{hypergraph.n_hyperedges} hyperedges"
        )
        
        return hypergraph
    
    def build_from_adjacency(
        self,
        adjacency_lists: Dict[int, List[int]],
        n_vertices: int
    ) -> Hypergraph:
        """
        Build hypergraph from adjacency list representation.
        
        Parameters
        ----------
        adjacency_lists : Dict[int, List[int]]
            Mapping from hyperedge ID to list of vertex indices.
        n_vertices : int
            Total number of vertices.
        
        Returns
        -------
        Hypergraph
            Constructed hypergraph.
        """
        hyperedge_ids = sorted(adjacency_lists.keys())
        n_hyperedges = len(hyperedge_ids)
        
        rows = []
        cols = []
        
        for col_idx, he_id in enumerate(hyperedge_ids):
            for vertex_idx in adjacency_lists[he_id]:
                rows.append(vertex_idx)
                cols.append(col_idx)
        
        data = np.ones(len(rows), dtype=np.float32)
        H = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_vertices, n_hyperedges)
        )
        
        hyperedge_sizes = np.asarray(H.sum(axis=0)).flatten()
        vertex_degrees = np.asarray(H.sum(axis=1)).flatten()
        hyperedge_weights = np.ones(n_hyperedges, dtype=np.float32)
        
        return Hypergraph(
            incidence_matrix=H,
            hyperedge_weights=hyperedge_weights,
            vertex_degrees=vertex_degrees,
            hyperedge_sizes=hyperedge_sizes,
            feature_ids=hyperedge_ids
        )
