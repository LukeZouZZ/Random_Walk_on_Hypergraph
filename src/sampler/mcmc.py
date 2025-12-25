"""
MCMC samplers for molecular screening.

Implements Metropolis-Hastings and Simulated Annealing samplers
for finding molecules similar to a query.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import logging
import time

import numpy as np
from tqdm import tqdm

from .random_walk import HypergraphRandomWalk
from .scheduler import TemperatureScheduler, get_scheduler
from ..similarity import LazySimilarityCache

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """
    Results from MCMC sampling.
    
    Attributes
    ----------
    top_molecules : List[Tuple[int, float]]
        Top molecules as (index, similarity) pairs, sorted by similarity.
    trajectory : List[int]
        Full trajectory of visited molecules (if saved).
    similarities : List[float]
        Similarity at each step (if saved).
    acceptance_rates : List[float]
        Acceptance rate over time.
    temperatures : List[float]
        Temperature at each step.
    best_molecule : int
        Index of best molecule found.
    best_similarity : float
        Highest similarity achieved.
    n_unique_visited : int
        Number of unique molecules visited.
    total_time : float
        Total sampling time in seconds.
    """
    top_molecules: List[Tuple[int, float]]
    trajectory: List[int] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    best_molecule: int = -1
    best_similarity: float = 0.0
    n_unique_visited: int = 0
    total_time: float = 0.0
    
    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"MCMC Sampling Results",
            f"---------------------",
            f"Best molecule: {self.best_molecule} (similarity: {self.best_similarity:.4f})",
            f"Unique molecules visited: {self.n_unique_visited}",
            f"Total time: {self.total_time:.2f}s",
            f"",
            f"Top {len(self.top_molecules)} molecules:",
        ]
        for idx, sim in self.top_molecules[:10]:
            lines.append(f"  Molecule {idx}: {sim:.4f}")
        
        return "\n".join(lines)


class MCMCSampler(ABC):
    """Abstract base class for MCMC samplers."""
    
    @abstractmethod
    def sample(
        self,
        query_idx: int,
        n_steps: int,
        **kwargs
    ) -> SamplingResult:
        """
        Run MCMC sampling to find molecules similar to query.
        
        Parameters
        ----------
        query_idx : int
            Index of query molecule.
        n_steps : int
            Number of MCMC steps.
        
        Returns
        -------
        SamplingResult
            Sampling results.
        """
        pass


class MetropolisHastingsSampler(MCMCSampler):
    """
    Metropolis-Hastings MCMC sampler for molecular screening.
    
    Samples from a distribution proportional to similarity to the query:
    π(v) ∝ exp(sim(v, query) / T)
    
    Uses proper MH acceptance ratio for theoretical correctness.
    
    Parameters
    ----------
    random_walk : HypergraphRandomWalk
        Random walk on the hypergraph for proposals.
    similarity_cache : LazySimilarityCache
        Cache for lazy similarity computation.
    initial_temp : float
        Initial temperature.
    final_temp : float
        Final temperature.
    cooling_schedule : str
        Type of cooling schedule.
    use_log_prob : bool
        Use log probabilities for numerical stability.
    seed : int, optional
        Random seed.
    
    Examples
    --------
    >>> sampler = MetropolisHastingsSampler(
    ...     random_walk=rw,
    ...     similarity_cache=cache,
    ...     initial_temp=1.0,
    ...     final_temp=0.01
    ... )
    >>> results = sampler.sample(query_idx=42, n_steps=200)
    """
    
    def __init__(
        self,
        random_walk: HypergraphRandomWalk,
        similarity_cache: LazySimilarityCache,
        initial_temp: float = 1.0,
        final_temp: float = 0.01,
        cooling_schedule: str = "exponential",
        use_log_prob: bool = True,
        seed: Optional[int] = None,
        **scheduler_kwargs
    ):
        self.random_walk = random_walk
        self.similarity_cache = similarity_cache
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_schedule = cooling_schedule
        self.use_log_prob = use_log_prob
        self.scheduler_kwargs = scheduler_kwargs
        
        self.rng = np.random.default_rng(seed)
    
    def sample(
        self,
        query_idx: int,
        n_steps: int,
        n_results: int = 10,
        start_vertex: Optional[int] = None,
        burn_in: int = 10,
        save_trajectory: bool = False,
        progress_bar: bool = True
    ) -> SamplingResult:
        """
        Run Metropolis-Hastings sampling.
        
        Parameters
        ----------
        query_idx : int
            Index of query molecule.
        n_steps : int
            Number of MCMC steps.
        n_results : int
            Number of top results to return.
        start_vertex : int, optional
            Starting vertex. If None, random start.
        burn_in : int
            Number of burn-in steps to discard.
        save_trajectory : bool
            Whether to save full trajectory.
        progress_bar : bool
            Whether to show progress bar.
        
        Returns
        -------
        SamplingResult
            Sampling results.
        """
        start_time = time.time()
        
        # Initialize scheduler
        scheduler = get_scheduler(
            self.cooling_schedule,
            self.initial_temp,
            self.final_temp,
            n_steps,
            **self.scheduler_kwargs
        )
        
        # Initialize state
        n_vertices = self.random_walk.hypergraph.n_vertices
        if start_vertex is None:
            current = self.rng.integers(0, n_vertices)
        else:
            current = start_vertex
        
        # Avoid starting at query
        while current == query_idx:
            current = self.rng.integers(0, n_vertices)
        
        current_sim = self.similarity_cache.get_similarity(current, query_idx)
        
        # Tracking
        visited = {current: current_sim}
        trajectory = [current] if save_trajectory else []
        similarities = [current_sim] if save_trajectory else []
        temperatures = []
        
        n_accepted = 0
        acceptance_rates = []
        window_size = 50
        recent_accepts = []
        
        # Best found
        best_mol = current
        best_sim = current_sim
        
        # Run MCMC
        iterator = range(n_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="MCMC Sampling")
        
        for step in iterator:
            temp = scheduler.get_temperature(step)
            temperatures.append(temp)
            
            # Propose next vertex via random walk
            proposal = self.random_walk.step(current)
            
            # Skip if proposal is query itself
            if proposal == query_idx:
                recent_accepts.append(False)
                continue
            
            # Get similarity (lazy computation)
            proposal_sim = self.similarity_cache.get_similarity(proposal, query_idx)
            
            # Compute acceptance probability
            accept_prob = self._compute_acceptance(
                current_sim, proposal_sim, temp,
                current, proposal
            )
            
            # Accept or reject
            if self.rng.random() < accept_prob:
                current = proposal
                current_sim = proposal_sim
                n_accepted += 1
                recent_accepts.append(True)
                
                # Track visited
                if proposal not in visited or visited[proposal] < proposal_sim:
                    visited[proposal] = proposal_sim
                
                # Update best
                if proposal_sim > best_sim:
                    best_mol = proposal
                    best_sim = proposal_sim
            else:
                recent_accepts.append(False)
            
            # Track trajectory
            if save_trajectory:
                trajectory.append(current)
                similarities.append(current_sim)
            
            # Update acceptance rate
            if len(recent_accepts) > window_size:
                recent_accepts = recent_accepts[-window_size:]
            acceptance_rates.append(sum(recent_accepts) / len(recent_accepts))
            
            # Update adaptive scheduler if applicable
            if hasattr(scheduler, 'update'):
                scheduler.update(recent_accepts[-1])
        
        # Get top results
        sorted_visited = sorted(
            visited.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top_molecules = sorted_visited[:n_results]
        
        total_time = time.time() - start_time
        
        result = SamplingResult(
            top_molecules=top_molecules,
            trajectory=trajectory,
            similarities=similarities,
            acceptance_rates=acceptance_rates,
            temperatures=temperatures,
            best_molecule=best_mol,
            best_similarity=best_sim,
            n_unique_visited=len(visited),
            total_time=total_time
        )
        
        logger.info(
            f"MCMC completed: best similarity = {best_sim:.4f}, "
            f"visited {len(visited)} unique molecules in {total_time:.2f}s"
        )
        
        return result
    
    def _compute_acceptance(
        self,
        current_sim: float,
        proposal_sim: float,
        temperature: float,
        current: int,
        proposal: int
    ) -> float:
        """
        Compute Metropolis-Hastings acceptance probability.
        
        α = min(1, π(proposal) * P(proposal → current) / (π(current) * P(current → proposal)))
        
        where π(v) ∝ exp(sim(v) / T)
        """
        if self.use_log_prob:
            # Log-space computation for numerical stability
            log_target_ratio = (proposal_sim - current_sim) / temperature
            
            # Proposal ratio (for non-symmetric proposals)
            proposal_ratio = self.random_walk.get_proposal_ratio(current, proposal)
            
            if proposal_ratio == 0:
                return 0.0
            
            log_proposal_ratio = np.log(proposal_ratio)
            log_accept = log_target_ratio + log_proposal_ratio
            
            return min(1.0, np.exp(log_accept))
        else:
            # Direct computation
            target_ratio = np.exp((proposal_sim - current_sim) / temperature)
            proposal_ratio = self.random_walk.get_proposal_ratio(current, proposal)
            
            return min(1.0, target_ratio * proposal_ratio)


class SimulatedAnnealingSampler(MCMCSampler):
    """
    Simulated Annealing sampler for molecular screening.
    
    Unlike standard MH, this focuses on optimization rather than sampling,
    always moving to better solutions and probabilistically accepting worse ones.
    
    Parameters
    ----------
    random_walk : HypergraphRandomWalk
        Random walk on the hypergraph for proposals.
    similarity_cache : LazySimilarityCache
        Cache for lazy similarity computation.
    initial_temp : float
        Initial temperature.
    final_temp : float
        Final temperature.
    cooling_schedule : str
        Type of cooling schedule.
    n_restarts : int
        Number of random restarts.
    seed : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        random_walk: HypergraphRandomWalk,
        similarity_cache: LazySimilarityCache,
        initial_temp: float = 1.0,
        final_temp: float = 0.001,
        cooling_schedule: str = "exponential",
        n_restarts: int = 1,
        seed: Optional[int] = None,
        **scheduler_kwargs
    ):
        self.random_walk = random_walk
        self.similarity_cache = similarity_cache
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_schedule = cooling_schedule
        self.n_restarts = n_restarts
        self.scheduler_kwargs = scheduler_kwargs
        
        self.rng = np.random.default_rng(seed)
    
    def sample(
        self,
        query_idx: int,
        n_steps: int,
        n_results: int = 10,
        save_trajectory: bool = False,
        progress_bar: bool = True
    ) -> SamplingResult:
        """Run simulated annealing."""
        start_time = time.time()
        
        n_vertices = self.random_walk.hypergraph.n_vertices
        steps_per_restart = n_steps // self.n_restarts
        
        all_visited = {}
        all_trajectories = []
        all_similarities = []
        all_temperatures = []
        
        global_best_mol = -1
        global_best_sim = -1
        
        for restart in range(self.n_restarts):
            # Initialize scheduler
            scheduler = get_scheduler(
                self.cooling_schedule,
                self.initial_temp,
                self.final_temp,
                steps_per_restart,
                **self.scheduler_kwargs
            )
            
            # Random start
            current = self.rng.integers(0, n_vertices)
            while current == query_idx:
                current = self.rng.integers(0, n_vertices)
            
            current_sim = self.similarity_cache.get_similarity(current, query_idx)
            
            # Track best in this restart
            best_mol = current
            best_sim = current_sim
            
            iterator = range(steps_per_restart)
            if progress_bar:
                desc = f"SA Restart {restart + 1}/{self.n_restarts}"
                iterator = tqdm(iterator, desc=desc)
            
            for step in iterator:
                temp = scheduler.get_temperature(step)
                
                if save_trajectory:
                    all_temperatures.append(temp)
                
                # Propose
                proposal = self.random_walk.step(current)
                
                if proposal == query_idx:
                    continue
                
                proposal_sim = self.similarity_cache.get_similarity(proposal, query_idx)
                
                # SA acceptance
                delta = proposal_sim - current_sim
                
                if delta > 0:
                    # Always accept improvements
                    accept = True
                else:
                    # Probabilistically accept worse solutions
                    accept_prob = np.exp(delta / temp)
                    accept = self.rng.random() < accept_prob
                
                if accept:
                    current = proposal
                    current_sim = proposal_sim
                    
                    if proposal not in all_visited or all_visited[proposal] < proposal_sim:
                        all_visited[proposal] = proposal_sim
                    
                    if proposal_sim > best_sim:
                        best_mol = proposal
                        best_sim = proposal_sim
                
                if save_trajectory:
                    all_trajectories.append(current)
                    all_similarities.append(current_sim)
            
            # Update global best
            if best_sim > global_best_sim:
                global_best_mol = best_mol
                global_best_sim = best_sim
            
            logger.info(
                f"Restart {restart + 1}: best = {best_sim:.4f} at molecule {best_mol}"
            )
        
        # Get top results
        sorted_visited = sorted(
            all_visited.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_molecules = sorted_visited[:n_results]
        
        total_time = time.time() - start_time
        
        return SamplingResult(
            top_molecules=top_molecules,
            trajectory=all_trajectories,
            similarities=all_similarities,
            temperatures=all_temperatures,
            acceptance_rates=[],  # Not tracked for SA
            best_molecule=global_best_mol,
            best_similarity=global_best_sim,
            n_unique_visited=len(all_visited),
            total_time=total_time
        )


class ParallelTemperingSampler(MCMCSampler):
    """
    Parallel Tempering (Replica Exchange) MCMC.
    
    Runs multiple chains at different temperatures and periodically
    swaps states between adjacent chains. This helps escape local optima.
    
    Parameters
    ----------
    random_walk : HypergraphRandomWalk
        Random walk on the hypergraph.
    similarity_cache : LazySimilarityCache
        Cache for similarity computation.
    n_replicas : int
        Number of parallel chains.
    temp_min : float
        Minimum temperature.
    temp_max : float
        Maximum temperature.
    swap_interval : int
        Steps between swap attempts.
    seed : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        random_walk: HypergraphRandomWalk,
        similarity_cache: LazySimilarityCache,
        n_replicas: int = 4,
        temp_min: float = 0.01,
        temp_max: float = 1.0,
        swap_interval: int = 10,
        seed: Optional[int] = None
    ):
        self.random_walk = random_walk
        self.similarity_cache = similarity_cache
        self.n_replicas = n_replicas
        self.swap_interval = swap_interval
        
        # Geometric temperature ladder
        self.temperatures = np.geomspace(temp_min, temp_max, n_replicas)
        
        self.rng = np.random.default_rng(seed)
    
    def sample(
        self,
        query_idx: int,
        n_steps: int,
        n_results: int = 10,
        save_trajectory: bool = False,
        progress_bar: bool = True
    ) -> SamplingResult:
        """Run parallel tempering."""
        start_time = time.time()
        
        n_vertices = self.random_walk.hypergraph.n_vertices
        
        # Initialize replicas
        states = []
        sims = []
        for _ in range(self.n_replicas):
            v = self.rng.integers(0, n_vertices)
            while v == query_idx:
                v = self.rng.integers(0, n_vertices)
            states.append(v)
            sims.append(self.similarity_cache.get_similarity(v, query_idx))
        
        visited = {}
        for v, s in zip(states, sims):
            visited[v] = s
        
        trajectory = [] if save_trajectory else None
        
        iterator = range(n_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Parallel Tempering")
        
        n_swaps = 0
        
        for step in iterator:
            # Update each replica
            for i in range(self.n_replicas):
                proposal = self.random_walk.step(states[i])
                
                if proposal == query_idx:
                    continue
                
                proposal_sim = self.similarity_cache.get_similarity(proposal, query_idx)
                
                # MH acceptance at this temperature
                log_accept = (proposal_sim - sims[i]) / self.temperatures[i]
                
                if np.log(self.rng.random()) < log_accept:
                    states[i] = proposal
                    sims[i] = proposal_sim
                    
                    if proposal not in visited or visited[proposal] < proposal_sim:
                        visited[proposal] = proposal_sim
            
            # Swap attempts
            if step % self.swap_interval == 0:
                # Try swapping adjacent replicas
                for i in range(self.n_replicas - 1):
                    # Swap acceptance
                    delta = (1/self.temperatures[i] - 1/self.temperatures[i+1]) * \
                            (sims[i+1] - sims[i])
                    
                    if np.log(self.rng.random()) < delta:
                        states[i], states[i+1] = states[i+1], states[i]
                        sims[i], sims[i+1] = sims[i+1], sims[i]
                        n_swaps += 1
            
            if save_trajectory:
                trajectory.append(states[0])  # Track coldest chain
        
        # Get results from coldest chain's perspective
        sorted_visited = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        top_molecules = sorted_visited[:n_results]
        
        best_mol, best_sim = top_molecules[0] if top_molecules else (-1, 0)
        
        total_time = time.time() - start_time
        
        logger.info(
            f"Parallel tempering: {n_swaps} swaps, "
            f"best = {best_sim:.4f} at molecule {best_mol}"
        )
        
        return SamplingResult(
            top_molecules=top_molecules,
            trajectory=trajectory or [],
            similarities=[],
            temperatures=list(self.temperatures),
            acceptance_rates=[],
            best_molecule=best_mol,
            best_similarity=best_sim,
            n_unique_visited=len(visited),
            total_time=total_time
        )
