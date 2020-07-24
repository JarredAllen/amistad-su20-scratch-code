import ctypes
from functools import reduce
import math
import numpy as np

from basic_sim import g_external_pressure

clib = ctypes.cdll.LoadLibrary('./simulate.so')

f = clib.f
f.restype = ctypes.c_double

g = clib.g
g.restype = ctypes.c_double

get_agent_choice = clib.get_agent_choice
get_agent_choice.restype = ctypes.c_char

def sim_basic(theta, agent_count, initial_yes=1, initial_total=2):
    """Run one iteration of the basic simulation (see the sim_basic
    function in simulate.c for documentation).
    """
    buf = np.array([0]*agent_count, dtype=np.byte)
    clib.sim_basic(ctypes.c_char_p(buf.ctypes.data), ctypes.c_double(theta), ctypes.c_int(agent_count), ctypes.c_int(initial_yes), ctypes.c_int(initial_total))
    return buf

def many_simulate_basic(theta, agent_count, num_sims, initial_yes=1, initial_total=2):
    """Run many iterations of the basic simulation and return them as a
    list.

    `theta` is the parameter in the paper, `agent_count` is the number
    of witnesses to simulate for, `num_sims` is the number of times to
    run the simulation, and `initial_yes` and `initial_total` handle
    the pseudocounts defined in the paper.
    """
    return [sim_basic(theta, agent_count, initial_yes, initial_total) for _ in range(num_sims)]

g_complex = clib.g_complex
g_complex.restype = ctypes.c_double

get_agent_choice_complex = clib.get_agent_choice_complex
get_agent_choice_complex.restype = ctypes.c_char

def sim_complex(theta, r, alpha, beta, agent_count, initial_w=0.5):
    """Run one iteration of the external pressure model simulation.

    `theta`, `r`, `alpha`, and `beta` are the model parameters.
    `agent_count` is the number of witnesses to simulate for.
    `initial_w` is the value of W_1 to use.
    """
    buf = np.zeros(agent_count, dtype=np.byte)
    clib.sim_complex(ctypes.c_char_p(buf.ctypes.data), ctypes.c_double(theta), ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_double(r), ctypes.c_int(agent_count), ctypes.c_double(initial_w))
    return buf

def many_simulate_complex(theta, r, alpha, beta, agent_count, num_sims, initial_m=0.5):
    """Run many iterations of the external pressure model simulation and
    return them as a list.

    `num_sims` is the number of times to run it. See the documentation
    of `sim_complex` for the other arguments.
    """
    return [sim_complex(theta, r, alpha, beta, agent_count, initial_m) for _ in range(num_sims)]

def sim_complex_get_ws(theta, r, alpha, beta, agent_count, initial_w=0.5):
    """Like `sim_complex`, but returns the value of w at each witness
    instead of their vote.

    See `sim_complex` for the parameter meanings.
    """
    buf = np.empty(agent_count+1, dtype=np.float64)
    clib.sim_complex_get_ws(ctypes.c_void_p(buf.ctypes.data), ctypes.c_double(theta), ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_double(r), ctypes.c_int(agent_count), ctypes.c_double(initial_w))
    return buf

def many_simulate_complex_get_ws(theta, r, alpha, beta, agent_count, num_sims, initial_w=0.5):
    """Run many iterations of the external pressure model simulation and
    return them as a 2D numpy array.

    `num_sims` is the number of times to run it. See the documentation
    of `sim_complex` for the other arguments.
    """
    return np.array([sim_complex_get_ws(theta, r, alpha, beta, agent_count, initial_w) for _ in range(num_sims)])

def compute_error_estimates(data, num_trials):
    """Computes estimates on the error bars by using the formula
    1.96*sqrt(p*(1-p)/n)
    """
    buf = np.zeros(data.shape, dtype=np.double)
    clib.compute_error_estimates(ctypes.c_void_p(buf.ctypes.data), ctypes.c_void_p(data.ctypes.data), ctypes.c_int(data.size), ctypes.c_int(num_trials))
    return buf


last_n_unanimous = clib.last_n_unanimous
last_n_unanimous.restype = ctypes.c_char

clib.prob_last_n_unanimous.restype = ctypes.c_double
def prob_last_n_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_reps):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    all of the next `tail_count` witnesses all affirm the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = clib.prob_last_n_unanimous(ctypes.c_double(theta), ctypes.c_double(r), ctypes.c_double(alpha), ctypes.c_double(beta), ctypes.c_int(agent_count), ctypes.c_double(initial_m), ctypes.c_int(tail_count), ctypes.c_int(num_reps))
    err = (prob * (1-prob) / num_reps) ** .5
    return prob, err

def prob_last_n_unanimous_with_fanout(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_full_reps, tail_fanout, min_successful_reps=9):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    all of the next `tail_count` witnesses all affirm the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = ctypes.c_double()
    err = ctypes.c_double()
    clib.prob_last_n_unanimous_with_fanout(
        ctypes.c_double(theta),
        ctypes.c_double(r),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
        ctypes.c_int(agent_count),
        ctypes.c_double(initial_m),
        ctypes.c_int(tail_count),
        ctypes.c_int(num_full_reps),
        ctypes.c_int(tail_fanout),
        ctypes.c_int(min_successful_reps),
        ctypes.byref(prob),
        ctypes.byref(err),
    )
    return prob.value, err.value


last_n_near_unanimous = clib.last_n_near_unanimous
last_n_near_unanimous.restype = ctypes.c_char

def prob_last_n_near_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_reps, frac_required, min_successful_reps=0):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    all of the next `tail_count` witnesses all affirm the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = ctypes.c_double()
    err = ctypes.c_double()
    clib.prob_last_n_near_unanimous(
        ctypes.c_double(theta),
        ctypes.c_double(r),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
        ctypes.c_int(agent_count),
        ctypes.c_double(initial_m),
        ctypes.c_int(tail_count),
        ctypes.c_int(num_reps),
        ctypes.c_int(int(math.ceil((1-frac_required)*tail_count))),
        ctypes.c_int(min_successful_reps),
        ctypes.byref(prob),
        ctypes.byref(err),
    )
    return prob.value, err.value

def prob_last_n_near_unanimous_with_fanout(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_full_reps, tail_fanout, frac_required, min_successful_reps=0):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    all of the next `tail_count` witnesses all affirm the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    It will run the simulation num_full_reps*tail_fanout times,
    recomputing the initial people (before a consensus is considered)
    after each tail_fanout iterations.

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = ctypes.c_double()
    err = ctypes.c_double()
    clib.prob_last_n_near_unanimous_with_fanout(
        ctypes.c_double(theta),
        ctypes.c_double(r),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
        ctypes.c_int(agent_count),
        ctypes.c_double(initial_m),
        ctypes.c_int(tail_count),
        ctypes.c_int(num_full_reps),
        ctypes.c_int(tail_fanout),
        ctypes.c_int(int(math.ceil((1-frac_required)*tail_count))),
        ctypes.c_int(min_successful_reps),
        ctypes.byref(prob),
        ctypes.byref(err),
    )
    return prob.value, err.value


last_n_near_unanimous_bidirectional = clib.last_n_near_unanimous_bidirectional
last_n_near_unanimous_bidirectional.restype = ctypes.c_char

def prob_last_n_near_unanimous_bidirectional(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_reps, frac_required, min_successful_reps=0):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    all of the next `tail_count` witnesses  either all affirm  or all
    reject the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = ctypes.c_double()
    err = ctypes.c_double()
    clib.prob_last_n_near_unanimous_bidirectional(
        ctypes.c_double(theta),
        ctypes.c_double(r),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
        ctypes.c_int(agent_count),
        ctypes.c_double(initial_m),
        ctypes.c_int(tail_count),
        ctypes.c_int(num_reps),
        ctypes.c_int(int(math.ceil((1-frac_required)*tail_count))),
        ctypes.c_int(min_successful_reps),
        ctypes.byref(prob),
        ctypes.byref(err),
    )
    return prob.value, err.value

def prob_last_n_near_unanimous_with_fanout_bidirectional(theta, r, alpha, beta, agent_count, initial_m, tail_count, num_full_reps, tail_fanout, frac_required, min_successful_reps=0):
    """Runs the simulation `num_reps` times, and finds the probability
    that, after `agent_count` witnesses have given their testimony, that
    of the next `tail_count` witnesses a sufficient majority will either
    affirm or reject the hypothesis.

    It returns a tuple consisting of the simulated probability and the
    error on that (1 sigma).

    It will run the simulation num_full_reps*tail_fanout times,
    recomputing the initial people (before a consensus is considered)
    after each tail_fanout iterations.

    For the other parameters, see the documentation of `sim_complex`.
    """
    prob = ctypes.c_double()
    err = ctypes.c_double()
    clib.prob_last_n_near_unanimous_with_fanout_bidirectional(
        ctypes.c_double(theta),
        ctypes.c_double(r),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
        ctypes.c_int(agent_count),
        ctypes.c_double(initial_m),
        ctypes.c_int(tail_count),
        ctypes.c_int(num_full_reps),
        ctypes.c_int(tail_fanout),
        ctypes.c_int(int(math.ceil((1-frac_required)*tail_count))),
        ctypes.c_int(min_successful_reps),
        ctypes.byref(prob),
        ctypes.byref(err),
    )
    return prob.value, err.value

clib.sim_complex_get_w.res_type = ctypes.c_double

def prob_last_n_unanimous_closed_form(theta, r, alpha, beta, agent_count, initial_w, tail_count, num_reps):
    """Runs the simulation for `num_reps` warm-up periods, going through
    agent_count iterations. For each iteration, it observes the final
    value of W_i, and computes the probability that the next
    `tail_count` witnesses will all affirm the hypothesis.

    It returns a tuple containing the estimated probability and the (1
    sigma) error on that estimate.

    For the other parameters, see the documentation on `sim_complex`.
    """
    def get_local_consensus_probability(theta, r, alpha, beta, tail_count, w1):
        """Returns the probability that the first `tail_count` witnesses
        will affirm, for the given parameters of the simulation.
        """
        return reduce(lambda x,y: x*y, [g_external_pressure(1+(w1-1)*(1-r)**i, theta, alpha, beta) for i in range(tail_count)], 1.0)
    def get_final_w_i_value(theta, r, alpha, beta, agent_count, initial_w):
        """Runs the external pressure model simulation and returns the
        last value of W_i after running for `agent_count` witnesses.
        """
        return clib.sim_complex_get_w(
                ctypes.c_double(theta),
                ctypes.c_double(alpha),
                ctypes.c_double(beta),
                ctypes.c_double(r),
                ctypes.c_int(agent_count),
                ctypes.c_double(initial_w),
            )
    probs = np.array([
        get_local_consensus_probability(theta, r, alpha, beta, tail_count, get_final_w_i_value(theta, r, alpha, beta, agent_count, initial_w))
        for _ in range(num_reps)
    ])
    return np.average(probs), np.std(probs) / np.sqrt(num_reps)
