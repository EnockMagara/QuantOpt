"""
Models module for portfolio optimization
"""

from .baseline import MPTOptimizer

# Import optional modules if available
try:
    from .deep_rl import DeepRLPortfolio
    DEEP_RL_AVAILABLE = True
except ImportError:
    DEEP_RL_AVAILABLE = False
    DeepRLPortfolio = None

try:
    from .monte_carlo import MonteCarloSimulator
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    MonteCarloSimulator = None

__all__ = ["MPTOptimizer"]

if DEEP_RL_AVAILABLE:
    __all__.append("DeepRLPortfolio")

if MONTE_CARLO_AVAILABLE:
    __all__.append("MonteCarloSimulator")
