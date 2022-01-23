import copy
import random
from typing import NewType, Callable

import numpy as np
from dataclasses import dataclass

# If two numbers are closer than EPSILON from each other,
# some parts of this code will consider them to be equal:
EPSILON = 0.000001


@dataclass
class Interval:
    min: float
    max: float

    def __post_init__(self):
        assert self.min <= self.max


@dataclass
class Constraints:
    allowed_allocation_per_index: Interval
    allowed_number_of_indexes: Interval

    def __post_init__(self):
        assert self.allowed_allocation_per_index.min == 0 or self.allowed_allocation_per_index.min >= EPSILON
        assert self.allowed_allocation_per_index.max <= 1
        assert self.allowed_number_of_indexes.min >= 0

        min_total_allocation = self.allowed_allocation_per_index.min * self.allowed_number_of_indexes.min
        assert min_total_allocation <= 1

        max_total_allocation = self.allowed_allocation_per_index.max * self.allowed_number_of_indexes.max
        assert max_total_allocation >= 1


@dataclass
class SingleRoundParams:
    n_crossovers1: int
    n_crossovers2: int
    n_mutations1: int
    n_mutations2: int
    max_population_size: int


@dataclass
class GeneticAlgorithmParams:
    seed: int
    # How many times the algorithm should start from scratch (from random portfolios):
    n_runs: int

    # See min_improvement_threshold comment
    n_rounds: int

    # The algorithm will stop when it fails to improve by more than min_improvement_threshold, over n_rounds
    min_improvement_threshold: float

    single_round_params: SingleRoundParams

    show_logs: bool

###############################################################################
# Target functions:
###############################################################################

Portfolio = np.array
ExpectedReturns = np.array
CovMatrix = np.array
RiskFreeRate = float

TargetFunction = NewType('TargetFunction',
                         Callable[
                             [Portfolio, ExpectedReturns, CovMatrix, RiskFreeRate],
                             float
                         ])


def compute_standard_deviation(portfolio, expected_returns, cov_matrix, risk_free_rate):
    variance = 0
    for i in range(0, len(portfolio)):
        for j in range(0, len(portfolio)):
            variance += portfolio[i] * portfolio[j] * cov_matrix[i][j]
    return variance ** 0.5


def compute_portfolio_returns(portfolio, expected_returns, cov_matrix, risk_free_rate):
    result = 0
    for i in range(0, len(portfolio)):
        result += expected_returns[i] * portfolio[i]
    return result


def compute_sharpe_ratio(portfolio, expected_returns, cov_matrix, risk_free_rate):
    standard_deviation = compute_standard_deviation(
        portfolio=portfolio,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate)
    returns = compute_portfolio_returns(
        portfolio=portfolio,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate)
    return (returns - risk_free_rate) / standard_deviation


###############################################################################
# Crossovers and mutations:
###############################################################################

def force_portfolio_into_constraints(portfolio, constraints: Constraints, recursion_level=0):
    if recursion_level > 1000:
        raise Exception('Recursion too deep, possibly infinite')

    result = copy.deepcopy(portfolio)

    # Check if number of allowed positions is within allowed interval:
    number_of_indexes = np.count_nonzero(result)
    if number_of_indexes < constraints.allowed_number_of_indexes.min:
        number_of_indexes_to_add = constraints.allowed_number_of_indexes.min - number_of_indexes
        available_indexes = np.argwhere(result == 0).ravel()
        random.shuffle(available_indexes)
        indexes_to_add = available_indexes[:number_of_indexes_to_add]
        # Just mark the indexes that are to be added. We'll handle min allocation later:
        for i in indexes_to_add:
            result[i] = EPSILON
        # Ensure that everything sums to 1 at all times:
        result = result / sum(result)
    elif number_of_indexes > constraints.allowed_number_of_indexes.max:
        number_of_indexes_to_remove = number_of_indexes - constraints.allowed_number_of_indexes.max
        occupied_indexes = np.argwhere(result > 0).ravel()
        random.shuffle(occupied_indexes)
        indexes_to_remove = occupied_indexes[:number_of_indexes_to_remove]
        # Just mark the indexes that are to be removed. We'll handle min/max allocation later:
        for i in indexes_to_remove:
            result[i] = 0
        # Ensure that everything sums to 1 at all times:
        result = result / sum(result)

    # Check if min allocation of each index is within allowed interval:
    min_allocation_violators = np.argwhere(np.logical_and(
        result != 0,
        result < constraints.allowed_allocation_per_index.min
    )).ravel()
    if len(min_allocation_violators) > 0:
        money_needed_by_violators = len(min_allocation_violators) * constraints.allowed_allocation_per_index.min
        money_among_violators = sum(result[min_allocation_violators])
        money_to_raise = money_needed_by_violators - money_among_violators

        min_allocation_satisfiers = np.argwhere(np.logical_and(
            result != 0,
            result > constraints.allowed_allocation_per_index.min
        )).ravel()
        total_money_among_satisfiers = sum(result[min_allocation_satisfiers])
        money_needed_by_satisfiers = len(min_allocation_satisfiers) * constraints.allowed_allocation_per_index.min
        surplus_money_among_satisfiers = total_money_among_satisfiers - money_needed_by_satisfiers

        if money_to_raise > surplus_money_among_satisfiers:
            # We need to kick out at least one index:
            occupied_indexes = np.argwhere(result > 0).ravel()
            index_to_kick_out = random.choice(occupied_indexes)
            result[index_to_kick_out] = 0
            result = result / sum(result)
            return force_portfolio_into_constraints(
                portfolio=result,
                constraints=constraints,
                recursion_level=recursion_level + 1)

        # Take money from satisfiers, proportionally to the surplus that they have:
        for s in min_allocation_satisfiers:
            money_held = result[s]
            surplus = money_held - constraints.allowed_allocation_per_index.min
            penalty_ratio = surplus / surplus_money_among_satisfiers
            penalty = penalty_ratio * money_to_raise
            result[s] -= penalty
            assert result[s] + EPSILON > constraints.allowed_allocation_per_index.min

        # Distribute money raised, among the violators:
        for v in min_allocation_violators:
            result[v] = constraints.allowed_allocation_per_index.min

        assert 1 - EPSILON < sum(result) < 1 + EPSILON

    # Check if max allocation of each index is within allowed interval:
    max_allocation_violators = np.argwhere(
        result > constraints.allowed_allocation_per_index.max
    ).ravel()
    if len(max_allocation_violators) > 0:
        max_money_for_violators = len(max_allocation_violators) * constraints.allowed_allocation_per_index.max
        money_among_violators = sum(result[max_allocation_violators])
        money_to_distribute = money_among_violators - max_money_for_violators

        max_allocation_satisfiers = np.argwhere(np.logical_and(
            result != 0,
            result < constraints.allowed_allocation_per_index.max
        )).ravel()
        total_money_among_satisfiers = sum(result[max_allocation_satisfiers])
        max_money_for_satisfiers = len(max_allocation_satisfiers) * constraints.allowed_allocation_per_index.max
        money_that_satisfiers_can_take = max_money_for_satisfiers - total_money_among_satisfiers

        if money_to_distribute > money_that_satisfiers_can_take:
            # We do not have enough indexes to distribute the surplus among. We need to add another one:
            available_indexes = np.argwhere(result == 0).ravel()
            index_to_add = random.choice(available_indexes)
            result[index_to_add] = EPSILON
            result = result / sum(result)
            return force_portfolio_into_constraints(
                portfolio=result,
                constraints=constraints,
                recursion_level=recursion_level + 1)

        # Add money to satisfiers, proportionally to how much more they can still take:
        for s in max_allocation_satisfiers:
            money_held = result[s]
            can_take = constraints.allowed_allocation_per_index.max - money_held
            bonus_ratio = can_take / money_that_satisfiers_can_take
            bonus = bonus_ratio * money_to_distribute
            result[s] += bonus
            assert result[s] - EPSILON < constraints.allowed_allocation_per_index.max

        # Take money from violators:
        for v in max_allocation_violators:
            result[v] = constraints.allowed_allocation_per_index.max

        assert 1 - EPSILON < sum(result) < 1 + EPSILON

    return result


def create_crossover_portfolio(portfolio1, portfolio2, constraints: Constraints):
    portfolio = np.array([
        (portfolio1[i] if bool(random.getrandbits(1)) else portfolio2[i])
        for i in range(0, len(portfolio1))
    ])
    portfolio_sum = sum(portfolio)
    if portfolio_sum == 0:
        return portfolio1
    return force_portfolio_into_constraints(portfolio / portfolio_sum, constraints)


def create_crossover_portfolio2(portfolio1, portfolio2, constraints: Constraints):
    ratio = random.uniform(0, 1)
    result = portfolio1 * ratio + portfolio2 * (1 - ratio)
    return force_portfolio_into_constraints(result / sum(result), constraints)


def create_portfolio_with_swapped_positions(portfolio, constraints: Constraints):
    """Mutation, which swaps position size on two indexes."""
    result = np.array(portfolio, copy=True)
    # TODO: Figure out how not to create an index array here:
    idx = range(len(portfolio))
    i1, i2 = random.sample(idx, 2)
    result[i1], result[i2] = result[i2], result[i1]
    # If portfolio already satisfies the constraints,
    # swapping position on two indexes will not change that.
    # But for good measure, lets pass it to the force_* method anyway:
    return force_portfolio_into_constraints(result, constraints)


def create_portfolio_with_tweaked_position(portfolio, constraints: Constraints):
    """Mutation, which adjusts position on a single index up or down,
    and then scales all others so that sum remains 1."""
    result = np.array(portfolio, copy=True)
    idx = range(len(portfolio))
    result[idx] = random.uniform(0, 2) * result[idx]
    return force_portfolio_into_constraints(result / sum(result), constraints)


def generate_new_population(
        population,
        target_function: TargetFunction,
        min_or_max,
        expected_returns,
        cov_matrix,
        risk_free_rate,
        single_round_params: SingleRoundParams,
        constraints: Constraints):
    # TODO: After switching to Python 3.8, use Literal['min', 'max'] to express this type constraint:
    assert min_or_max == 'min' or min_or_max == 'max'

    # crossovers:
    # crossovers1 = np.empty([0, len(expected_returns)])
    crossovers1 = np.empty([single_round_params.n_crossovers1, len(expected_returns)])
    for i in range(single_round_params.n_crossovers1):
        portfolio1 = population[random.randint(0, len(population) - 1)]
        portfolio2 = population[random.randint(0, len(population) - 1)]
        crossovers1[i] = create_crossover_portfolio(portfolio1, portfolio2, constraints)

    crossovers2 = np.empty([single_round_params.n_crossovers2, len(expected_returns)])
    for i in range(single_round_params.n_crossovers2):
        portfolio1 = population[random.randint(0, len(population) - 1)]
        portfolio2 = population[random.randint(0, len(population) - 1)]
        crossovers2[i] = create_crossover_portfolio2(portfolio1, portfolio2, constraints)

    # mutations:
    mutations1 = np.empty([single_round_params.n_mutations1, len(expected_returns)])
    for i in range(single_round_params.n_mutations1):
        idx = random.randint(0, len(population) - 1)
        mutations1[i] = create_portfolio_with_swapped_positions(population[idx], constraints)

    mutations2 = np.empty([single_round_params.n_mutations2, len(expected_returns)])
    for i in range(single_round_params.n_mutations2):
        idx = random.randint(0, len(population) - 1)
        mutations2[i] = create_portfolio_with_tweaked_position(population[idx], constraints)

    # Construct new_population from helper variables:
    new_population = np.concatenate((population, crossovers1, crossovers2, mutations1, mutations2), axis=0)

    # Remove duplicates:
    unique_population = np.unique(new_population, axis=0)

    # sort and selection of the best solutions:
    pairs = list(
        map(lambda p: [target_function(p, expected_returns, cov_matrix, risk_free_rate), p], unique_population))
    pairs.sort(key=lambda p: p[0], reverse=(min_or_max == 'max'))
    sorted_unique_population = list(map(lambda p: p[1], pairs))

    # ranking selection:
    return sorted_unique_population[:single_round_params.max_population_size]
