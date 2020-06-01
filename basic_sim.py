from random import random

import numpy as np

def f(m, theta):
    """A model of likelihoods using a sigmoid distribution"""
    return 1 / (1 + np.exp(-theta*(m-0.5)))

def g(m, theta):
    """Zero-one rescaling of `f`"""
    return (f(m, theta) - f(0, theta)) / (f(1, theta) - f(0, theta))

def get_agent_choice(m, theta):
    """Simulates an agent deciding to say yes or no. Returns True iff
    the agent decides to say yes.

    `m` and `theta` are passed into `g` to determine probability of a
    response in the affirmative.
    """
    return random() < g(m, theta)

def simulate(theta, count, initial=[]):
    """Performs a simulation of `count` agents deciding to either
    affirm or reject the event. The value of theta is passed to the `g`
    function for each decision. `m` is determined by the history of
    responses, defined as the "pseudocount" where each side starts with
    one vote.

    `initial` is the (possibly empty) list of responses which have been
    given before the first simulated agent runs (useful for determining
    the impact of a specific start on everyone else). It should be a
    list of booleans, where each one represents one response.

    It returns a list of bools containing every response (including
    those in `initial`.
    """
    result = list(initial)
    num_yes = result.count(True) + 1
    total = len(result) + 2
    for i in range(count):
        response = get_agent_choice(num_yes / total, theta)
        if response:
            num_yes += 1
        total += 1
        result.append(response)
    return result

def many_simulate(theta, agent_count, initial, num_sims):
    """Performs `num_sims` simulations, where each one has the specified
    theta value, number of agents, and initial responses.

    Returns a list containing the results for each simulation.
    """
    return [simulate(theta, agent_count, initial) for _ in range(num_sims)]

def get_agent_choice_with_pressure(m, theta, alpha, beta):
    """Simulates an agent deciding to say yes or no. Returns True iff
    the agent decides to say yes.

    `m` and `theta` are passed into `g` to determine probability of a
    response in the affirmative. `alpha` and `beta` are the parameters
    to which `g` is rescaled (see "External Pressure Model" in the
    paper).
    """
    return random() < (beta-alpha)*g(m, theta) + alpha

def simulate_modified(theta, r, alpha, beta, agent_count, initial=0.5):
    """Performs the modified simulation which includes the Recent
    Majority Vote model and the External Pressure model.

    `theta`, `agent_count`, and `initial` are all defined the same as in
    the `simulate` function.

    `r`, `alpha`, and `beta` are defined in the paper.
    """
    if type(initial) is list:
        result = list(initial)
        m = 0.5
        for agent in initial:
            m = (1-r)*m + r*agent
    else:
        result = []
        m = initial
    for i in range(agent_count):
        response = get_agent_choice_with_pressure(m, theta, alpha, beta)
        m = m*(1-r) + response*r
        result.append(response)
    return result

def many_simulate_modified(theta, r, alpha, beta, agent_count, num_sims, initial=0.5):
    """Performs `num_sims` simulations, where each one has the specified
    parameters (see `simulate_modified` for documentation on those
    parameters.

    It returns a list containing the results for each simulation (where
    each simulation's results are themselves a list of True/False
    values).
    """
    return [simulate_modified(theta, r, alpha, beta, agent_count, initial) for _ in range(num_sims)]
