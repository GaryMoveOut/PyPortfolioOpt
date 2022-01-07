"""
The ``efficient_frontier`` module houses the EfficientFrontier class and its descendants,
which generate optimal portfolios for various possible objective functions and parameters.
"""

from .efficient_frontier import EfficientFrontier
from .efficient_cvar import EfficientCVaR
from .efficient_semivariance import EfficientSemivariance
from .efficient_cdar import EfficientCDaR
from .genetic_algorithm_utils import Constraints, Interval, SingleRoundParams, GeneticAlgorithmParams, \
    force_portfolio_into_constraints, compute_portfolio_returns, compute_standard_deviation, compute_sharpe_ratio, \
    TargetFunction

__all__ = [
    "EfficientFrontier",
    "EfficientCVaR",
    "EfficientSemivariance",
    "EfficientCDaR",
    "Constraints",
    "Interval",
    "SingleRoundParams",
    "GeneticAlgorithmParams",
    "TargetFunction",
    "compute_standard_deviation",
    "compute_portfolio_returns",
    "compute_sharpe_ratio",
    "force_portfolio_into_constraints",  # TODO: Remove this export, after all manual testing is done
]
