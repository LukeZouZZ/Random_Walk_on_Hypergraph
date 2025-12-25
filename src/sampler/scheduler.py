"""
Temperature scheduling for simulated annealing.

Various cooling schedules to control exploration-exploitation tradeoff.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import logging

import numpy as np

logger = logging.getLogger(__name__)


class TemperatureScheduler(ABC):
    """
    Abstract base class for temperature schedulers.
    
    Temperature controls the acceptance probability in MCMC:
    - High temperature: more exploration, accept worse solutions
    - Low temperature: more exploitation, focus on good solutions
    """
    
    def __init__(self, initial_temp: float, final_temp: float, n_steps: int):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.current_step = 0
    
    @abstractmethod
    def get_temperature(self, step: Optional[int] = None) -> float:
        """
        Get temperature at a given step.
        
        Parameters
        ----------
        step : int, optional
            Step number. If None, use current step.
        
        Returns
        -------
        float
            Temperature at the given step.
        """
        pass
    
    def step(self) -> float:
        """Advance to next step and return new temperature."""
        self.current_step += 1
        return self.get_temperature()
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
    
    def get_schedule(self) -> np.ndarray:
        """Get full temperature schedule."""
        return np.array([self.get_temperature(i) for i in range(self.n_steps)])


class ExponentialScheduler(TemperatureScheduler):
    """
    Exponential cooling schedule.
    
    T(t) = T_0 * γ^t
    
    where γ is the cooling rate (automatically computed from T_0, T_f, n_steps).
    
    Parameters
    ----------
    initial_temp : float
        Starting temperature.
    final_temp : float
        Final temperature.
    n_steps : int
        Total number of steps.
    cooling_rate : float, optional
        Explicit cooling rate. If None, computed automatically.
    """
    
    def __init__(
        self,
        initial_temp: float,
        final_temp: float,
        n_steps: int,
        cooling_rate: Optional[float] = None
    ):
        super().__init__(initial_temp, final_temp, n_steps)
        
        if cooling_rate is not None:
            self.cooling_rate = cooling_rate
        else:
            # Compute rate to reach final_temp at n_steps
            # T_f = T_0 * γ^n => γ = (T_f / T_0)^(1/n)
            self.cooling_rate = (final_temp / initial_temp) ** (1.0 / max(n_steps - 1, 1))
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        temp = self.initial_temp * (self.cooling_rate ** step)
        return max(temp, self.final_temp)


class LinearScheduler(TemperatureScheduler):
    """
    Linear cooling schedule.
    
    T(t) = T_0 - (T_0 - T_f) * t / n_steps
    """
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        progress = step / max(self.n_steps - 1, 1)
        temp = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        return max(temp, self.final_temp)


class LogarithmicScheduler(TemperatureScheduler):
    """
    Logarithmic cooling schedule.
    
    T(t) = T_0 / log(1 + t + c)
    
    where c is chosen to achieve T_f at n_steps.
    
    This schedule cools slowly, providing theoretical convergence guarantees
    for simulated annealing.
    """
    
    def __init__(self, initial_temp: float, final_temp: float, n_steps: int):
        super().__init__(initial_temp, final_temp, n_steps)
        
        # Find c such that T_0 / log(1 + n + c) = T_f
        # log(1 + n + c) = T_0 / T_f
        # c = exp(T_0 / T_f) - 1 - n
        self.c = max(np.exp(initial_temp / max(final_temp, 1e-10)) - 1 - n_steps, 1)
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        temp = self.initial_temp / np.log(1 + step + self.c)
        return max(temp, self.final_temp)


class AdaptiveScheduler(TemperatureScheduler):
    """
    Adaptive temperature scheduler.
    
    Adjusts temperature based on acceptance rate:
    - High acceptance rate → cool faster
    - Low acceptance rate → cool slower
    
    Parameters
    ----------
    initial_temp : float
        Starting temperature.
    final_temp : float
        Final temperature.
    n_steps : int
        Total number of steps.
    target_acceptance : float
        Target acceptance rate (e.g., 0.44 for optimal MCMC).
    window_size : int
        Window size for computing acceptance rate.
    increase_factor : float
        Factor to increase temperature when acceptance is too low.
    decrease_factor : float
        Factor to decrease temperature when acceptance is too high.
    """
    
    def __init__(
        self,
        initial_temp: float,
        final_temp: float,
        n_steps: int,
        target_acceptance: float = 0.44,
        window_size: int = 50,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.9
    ):
        super().__init__(initial_temp, final_temp, n_steps)
        
        self.target_acceptance = target_acceptance
        self.window_size = window_size
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        
        self.current_temp = initial_temp
        self.acceptance_history: List[bool] = []
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        return max(self.current_temp, self.final_temp)
    
    def update(self, accepted: bool):
        """
        Update temperature based on acceptance.
        
        Parameters
        ----------
        accepted : bool
            Whether the last proposal was accepted.
        """
        self.acceptance_history.append(accepted)
        
        # Only adapt after window is full
        if len(self.acceptance_history) >= self.window_size:
            recent = self.acceptance_history[-self.window_size:]
            acceptance_rate = sum(recent) / len(recent)
            
            if acceptance_rate < self.target_acceptance - 0.05:
                # Too few acceptances, increase temperature
                self.current_temp *= self.increase_factor
            elif acceptance_rate > self.target_acceptance + 0.05:
                # Too many acceptances, decrease temperature
                self.current_temp *= self.decrease_factor
            
            # Ensure temperature stays in bounds
            self.current_temp = np.clip(
                self.current_temp,
                self.final_temp,
                self.initial_temp * 2
            )
    
    def step(self) -> float:
        self.current_step += 1
        
        # Gradual cooling overlay
        progress = self.current_step / max(self.n_steps - 1, 1)
        max_temp = self.initial_temp * (1 - 0.5 * progress)
        self.current_temp = min(self.current_temp, max_temp)
        
        return self.get_temperature()
    
    def reset(self):
        super().reset()
        self.current_temp = self.initial_temp
        self.acceptance_history = []


class CyclicScheduler(TemperatureScheduler):
    """
    Cyclic temperature scheduler with restarts.
    
    Periodically restarts from high temperature to escape local optima.
    
    Parameters
    ----------
    initial_temp : float
        Starting temperature.
    final_temp : float
        Final temperature.
    n_steps : int
        Total number of steps.
    cycle_length : int
        Number of steps per cycle.
    decay : float
        Factor to decay initial temperature each cycle.
    """
    
    def __init__(
        self,
        initial_temp: float,
        final_temp: float,
        n_steps: int,
        cycle_length: int = 100,
        decay: float = 0.9
    ):
        super().__init__(initial_temp, final_temp, n_steps)
        self.cycle_length = cycle_length
        self.decay = decay
    
    def get_temperature(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        
        # Which cycle are we in?
        cycle = step // self.cycle_length
        step_in_cycle = step % self.cycle_length
        
        # Initial temp for this cycle
        cycle_initial = self.initial_temp * (self.decay ** cycle)
        
        # Linear cooling within cycle
        progress = step_in_cycle / max(self.cycle_length - 1, 1)
        temp = cycle_initial * (1 - progress) + self.final_temp * progress
        
        return max(temp, self.final_temp)


def get_scheduler(
    name: str,
    initial_temp: float,
    final_temp: float,
    n_steps: int,
    **kwargs
) -> TemperatureScheduler:
    """
    Factory function to get scheduler by name.
    
    Parameters
    ----------
    name : str
        Name of the scheduler: 'exponential', 'linear', 'logarithmic',
        'adaptive', or 'cyclic'.
    initial_temp : float
        Starting temperature.
    final_temp : float
        Final temperature.
    n_steps : int
        Total number of steps.
    **kwargs
        Additional arguments for the scheduler.
    
    Returns
    -------
    TemperatureScheduler
        Instantiated scheduler.
    """
    schedulers = {
        'exponential': ExponentialScheduler,
        'linear': LinearScheduler,
        'logarithmic': LogarithmicScheduler,
        'adaptive': AdaptiveScheduler,
        'cyclic': CyclicScheduler,
    }
    
    name = name.lower()
    if name not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Available: {list(schedulers.keys())}"
        )
    
    return schedulers[name](initial_temp, final_temp, n_steps, **kwargs)
