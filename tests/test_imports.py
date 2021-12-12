def test_import_modules():
    from pypfopt_fork import (
        base_optimizer,
        black_litterman,
        cla,
        discrete_allocation,
        exceptions,
        expected_returns,
        hierarchical_portfolio,
        objective_functions,
        plotting,
        risk_models,
    )


def test_explicit_import():
    from pypfopt_fork.black_litterman import (
        market_implied_prior_returns,
        market_implied_risk_aversion,
        BlackLittermanModel,
    )
    from pypfopt_fork.cla import CLA
    from pypfopt_fork.discrete_allocation import get_latest_prices, DiscreteAllocation
    from pypfopt_fork.efficient_frontier import (
        EfficientFrontier,
        EfficientSemivariance,
        EfficientCVaR,
    )
    from pypfopt_fork.hierarchical_portfolio import HRPOpt
    from pypfopt_fork.risk_models import CovarianceShrinkage


def test_import_toplevel():
    from pypfopt_fork import (
        market_implied_prior_returns,
        market_implied_risk_aversion,
        BlackLittermanModel,
        CLA,
        get_latest_prices,
        DiscreteAllocation,
        EfficientFrontier,
        EfficientSemivariance,
        EfficientCVaR,
        HRPOpt,
        CovarianceShrinkage,
    )
