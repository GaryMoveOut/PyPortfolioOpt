import random

import numpy as np


def compute_standard_deviation(portfolio, cov_matrix):
    variance = 0
    for i in range(0, len(portfolio)):
        for j in range(0, len(portfolio)):
            variance += portfolio[i] * portfolio[j] * cov_matrix[i][j]
    return variance ** 0.5


def compute_portfolio_returns(portfolio, expected_returns):
    result = 0
    for i in range(0, len(portfolio)):
        result += expected_returns[i] * portfolio[i]
    return result


def compute_sharpe_ratio(portfolio, expected_returns, cov_matrix, risk_free_rate):
    standard_deviation = compute_standard_deviation(portfolio, cov_matrix)
    returns = compute_portfolio_returns(portfolio, expected_returns)
    return (returns - risk_free_rate) / standard_deviation


def create_crossover_portfolio(portfolio1, portfolio2):
    portfolio = np.array([
        (portfolio1[i] if bool(random.getrandbits(1)) else portfolio2[i])
        for i in range(0, len(portfolio1))
    ])
    portfolio_sum = sum(portfolio)
    return portfolio / portfolio_sum if portfolio_sum != 0 else portfolio1


def create_crossover_portfolio2(portfolio1, portfolio2):
    ratio = random.uniform(0, 1)
    result = portfolio1 * ratio + portfolio2 * (1-ratio)
    return result / sum(result)


def create_portfolio_with_swapped_positions(portfolio):
    """Mutation, which swaps position size on two indexes."""
    result = np.array(portfolio, copy=True)
    # TODO: Figure out how not to create an index array here:
    idx = range(len(portfolio))
    i1, i2 = random.sample(idx, 2)
    result[i1], result[i2] = result[i2], result[i1]
    return result


def create_portfolio_with_tweaked_position(portfolio):
    """Mutation, which adjusts position on a single index up or down,
    and then scales all others so that sum remains 1."""
    result = np.array(portfolio, copy=True)
    idx = range(len(portfolio))
    result[idx] = random.uniform(0,2)*result[idx]
    return result / sum(result)


def generate_new_population(population, expected_returns, cov_matrix, risk_free_rate, n_crossovers, n_mutations,
                            max_population_size):
    # crossovers:
    # crossovers1 = np.empty([0, len(expected_returns)])
    crossovers1 = np.empty([n_crossovers, len(expected_returns)])
    for i in range(n_crossovers):
        portfolio1 = population[random.randint(0, len(population) - 1)]
        portfolio2 = population[random.randint(0, len(population) - 1)]
        crossovers1[i] = create_crossover_portfolio(portfolio1, portfolio2)

    crossovers2 = np.empty([n_crossovers, len(expected_returns)])
    for i in range(n_crossovers):
        portfolio1 = population[random.randint(0, len(population) - 1)]
        portfolio2 = population[random.randint(0, len(population) - 1)]
        crossovers2[i] = create_crossover_portfolio2(portfolio1, portfolio2)

    # mutations:
    mutations1 = np.empty([n_mutations, len(expected_returns)])
    for i in range(n_mutations):
        idx = random.randint(0, len(population) - 1)
        mutations1[i] = create_portfolio_with_swapped_positions(population[idx])

    mutations2 = np.empty([n_mutations, len(expected_returns)])
    for i in range(n_mutations):
        idx = random.randint(0, len(population) - 1)
        mutations2[i] = create_portfolio_with_tweaked_position(population[idx])

    # Construct new_population from helper variables:
    new_population = np.concatenate((population, crossovers1, crossovers2, mutations1, mutations2), axis=0)

    # Remove duplicates:
    unique_population = np.unique(new_population, axis=0)

    # sort and selection of the best solutions:
    pairs = list(map(lambda p: [compute_sharpe_ratio(p, expected_returns, cov_matrix, risk_free_rate), p], unique_population))
    pairs.sort(key=lambda p: p[0], reverse=True)
    sorted_unique_population = list(map(lambda p: p[1], pairs))

    # ranking selection:
    return sorted_unique_population[:max_population_size]




