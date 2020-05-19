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
