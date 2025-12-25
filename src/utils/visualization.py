"""
Visualization utilities for MCMC sampling results.
"""

from typing import List, Optional, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_sampling_trajectory(
    similarities: List[float],
    temperatures: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "MCMC Sampling Trajectory"
) -> Figure:
    """
    Plot similarity trajectory during MCMC sampling.
    
    Parameters
    ----------
    similarities : List[float]
        Similarity values at each step.
    temperatures : List[float], optional
        Temperature values at each step.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if temperatures is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    steps = np.arange(len(similarities))
    
    # Similarity plot
    ax1.plot(steps, similarities, 'b-', alpha=0.7, linewidth=0.5)
    ax1.fill_between(steps, 0, similarities, alpha=0.3)
    
    # Running maximum
    running_max = np.maximum.accumulate(similarities)
    ax1.plot(steps, running_max, 'r-', linewidth=2, label='Best so far')
    
    ax1.set_ylabel('Similarity')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    
    # Temperature plot
    if temperatures is not None:
        ax2.plot(steps, temperatures, 'g-', linewidth=1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Temperature')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Step')
    
    plt.tight_layout()
    return fig


def plot_temperature_schedule(
    scheduler,
    n_steps: int = 1000,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Temperature Schedule"
) -> Figure:
    """
    Plot temperature schedule.
    
    Parameters
    ----------
    scheduler : TemperatureScheduler
        The temperature scheduler.
    n_steps : int
        Number of steps to plot.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    temperatures = scheduler.get_schedule()[:n_steps]
    steps = np.arange(len(temperatures))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(steps, temperatures, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Mark key temperatures
    ax.axhline(y=temperatures[0], color='r', linestyle='--', alpha=0.5, label='Initial')
    ax.axhline(y=temperatures[-1], color='g', linestyle='--', alpha=0.5, label='Final')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_acceptance_rate(
    acceptance_rates: List[float],
    window_size: int = 50,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Acceptance Rate"
) -> Figure:
    """
    Plot acceptance rate over time.
    
    Parameters
    ----------
    acceptance_rates : List[float]
        Acceptance rate at each step.
    window_size : int
        Window size for smoothing.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    steps = np.arange(len(acceptance_rates))
    
    # Raw acceptance rate
    ax.plot(steps, acceptance_rates, 'b-', alpha=0.3, linewidth=0.5)
    
    # Smoothed
    if len(acceptance_rates) > window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(acceptance_rates, kernel, mode='valid')
        ax.plot(
            np.arange(len(smoothed)) + window_size // 2,
            smoothed,
            'r-',
            linewidth=2,
            label=f'Smoothed (window={window_size})'
        )
    
    # Target acceptance rate
    ax.axhline(y=0.44, color='g', linestyle='--', alpha=0.7, label='Target (0.44)')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Acceptance Rate')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_similarity_distribution(
    similarities: List[float],
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Similarity Distribution"
) -> Figure:
    """
    Plot histogram of similarities.
    
    Parameters
    ----------
    similarities : List[float]
        List of similarity values.
    bins : int
        Number of histogram bins.
    figsize : Tuple[int, int]
        Figure size.
    title : str
        Plot title.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(similarities, bins=bins, density=True, alpha=0.7, edgecolor='black')
    
    # Mark statistics
    mean_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    
    ax.axvline(x=mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    ax.axvline(x=max_sim, color='g', linestyle='--', label=f'Max: {max_sim:.3f}')
    
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_molecules(
    molecules,
    similarities: Optional[List[float]] = None,
    n_cols: int = 5,
    mol_size: Tuple[int, int] = (200, 200),
    title: str = "Top Similar Molecules"
) -> Optional[object]:
    """
    Visualize molecules using RDKit.
    
    Parameters
    ----------
    molecules : List[Chem.Mol]
        List of RDKit molecule objects.
    similarities : List[float], optional
        Similarity values for labels.
    n_cols : int
        Number of columns in grid.
    mol_size : Tuple[int, int]
        Size of each molecule image.
    title : str
        Title for the image.
    
    Returns
    -------
    PIL.Image or None
        Image of molecules, or None if visualization fails.
    """
    try:
        from rdkit.Chem import Draw
        
        if similarities is not None:
            legends = [f"Sim: {s:.3f}" for s in similarities]
        else:
            legends = [f"Mol {i}" for i in range(len(molecules))]
        
        img = Draw.MolsToGridImage(
            molecules,
            molsPerRow=n_cols,
            subImgSize=mol_size,
            legends=legends
        )
        
        return img
        
    except Exception as e:
        logger.warning(f"Could not visualize molecules: {e}")
        return None


def create_summary_plot(
    result,
    figsize: Tuple[int, int] = (14, 10)
) -> Figure:
    """
    Create a summary plot with multiple panels.
    
    Parameters
    ----------
    result : SamplingResult
        MCMC sampling result.
    figsize : Tuple[int, int]
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        The summary figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Similarity trajectory
    if result.similarities:
        ax = axes[0, 0]
        steps = np.arange(len(result.similarities))
        ax.plot(steps, result.similarities, 'b-', alpha=0.5, linewidth=0.5)
        running_max = np.maximum.accumulate(result.similarities)
        ax.plot(steps, running_max, 'r-', linewidth=2, label='Best')
        ax.set_xlabel('Step')
        ax.set_ylabel('Similarity')
        ax.set_title('Sampling Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Temperature
    if result.temperatures:
        ax = axes[0, 1]
        ax.plot(result.temperatures, 'g-', linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Temperature')
        ax.set_yscale('log')
        ax.set_title('Temperature Schedule')
        ax.grid(True, alpha=0.3)
    
    # Acceptance rate
    if result.acceptance_rates:
        ax = axes[1, 0]
        ax.plot(result.acceptance_rates, 'b-', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.44, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Acceptance Rate')
        ax.set_title('Acceptance Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # Top molecules bar chart
    ax = axes[1, 1]
    if result.top_molecules:
        indices = [str(m[0]) for m in result.top_molecules[:10]]
        sims = [m[1] for m in result.top_molecules[:10]]
        bars = ax.bar(indices, sims)
        ax.set_xlabel('Molecule Index')
        ax.set_ylabel('Similarity')
        ax.set_title(f'Top {len(indices)} Molecules')
        ax.set_ylim(0, 1)
        
        # Color bars by similarity
        for bar, sim in zip(bars, sims):
            bar.set_color(plt.cm.RdYlGn(sim))
    
    plt.suptitle(
        f'MCMC Summary | Best: {result.best_similarity:.4f} | '
        f'Visited: {result.n_unique_visited} | Time: {result.total_time:.2f}s',
        fontsize=12
    )
    
    plt.tight_layout()
    return fig
