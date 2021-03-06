#include <math.h>
#include <pthread.h>
#include <stdlib.h>

// Computes the value of $f(theta, M_i)$
double f(double m, double theta) {
    return 1.0 / (1.0 + exp(-1 * theta * (m - 0.5)));
}

// Computes the value of $g_\theta(M_i)$
double g(double m, double theta) {
    return (f(m, theta) - f(0, theta)) / (f(1, theta) - f(0, theta));
}

pthread_mutex_t rand_lock = PTHREAD_MUTEX_INITIALIZER;

// Returns 1 if the agent chooses to affirm, 0 if the agent chooses to
// reject
//
// It relies on rand(), so it might not work as well on other platforms.
// The man page for rand(3) on the computer we ran it on says:
//
// > on older rand() implementations, and on current implementations on
// > different systems, the lower-order bits are much less random than
// > the higher-order bits.
//
// This has no bearing on our execution of the code, but it may skew
// your results if you try to reproduce our results.
char get_agent_choice(double m, double theta) {
    int rng;
    pthread_mutex_lock(&rand_lock);
    rng = rand();
    pthread_mutex_unlock(&rand_lock);
    return ((double)rng) / RAND_MAX < g(m, theta);
}

// Runs the basic simulation and stores its result in the array passed
// to result
//
// result needs to already have enough space allocated, and it will
// contain a 1 at an index if that witness affirmed, and a 0 otherwise.
void sim_basic(char* result, double theta, int agent_count, int initial_yes, int initial_total) {
    int i,
        num_yes = initial_yes,
        num_total = initial_total;
    char response;
    double m;
    for (i=0; i < agent_count; i++) {
        m = (double)num_yes / num_total;
        response = get_agent_choice(m, theta);
        num_yes += response;
        num_total++;
        result[i] = response;
    }
}

// Returns the more complex g function which includes an alpha and a
// beta
double g_complex(double m, double theta, double alpha, double beta) {
    return (beta-alpha)*g(m, theta) + alpha;
}
// Returns 1 if the agent chooses to affirm, 0 if the agent chooses to
// reject. This function also includes alpha and beta
//
// It relies on rand(), so it might not work as well on other platforms.
// The man page for rand(3) on the computer we ran it on says:
//
// > on older rand() implementations, and on current implementations on
// > different systems, the lower-order bits are much less random than
// > the higher-order bits.
//
// This has no bearing on our execution of the code, but it may skew
// your results if you try to reproduce our results.
char get_agent_choice_complex(double m, double theta, double alpha, double beta) {
    int rng;
    pthread_mutex_lock(&rand_lock);
    rng = rand();
    pthread_mutex_unlock(&rand_lock);
    return ((double)rng) / RAND_MAX < g_complex(m, theta, alpha, beta);
}

// Runs the external pressure simulation, outputting responses the same
// way as `sim_basic`
void sim_complex(char* result, double theta, double alpha, double beta, double r, int agent_count, double initial_m) {
    int i;
    char response;
    double m = initial_m;
    for (i=0; i < agent_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        m = m*(1-r) + response*r;
        result[i] = response;
    }
}

// Runs the complex simulation, and outputs the values of W_i observed
// by each witness into `result`.
void sim_complex_get_ws(double* result, double theta, double alpha, double beta, double r, int agent_count, double initial_w) {
    int i;
    char response;
    double w = initial_w;
    for (i=0; i < agent_count; i++) {
        result[i] = w;
        response = get_agent_choice_complex(w, theta, alpha, beta);
        w = w*(1-r) + response*r;
    }
    result[agent_count] = w;
}

// Runs the complex simulation and returns the final value of W_i
double sim_complex_get_w(double theta, double alpha, double beta, double r, int agent_count, double initial_w) {
    int i;
    char response;
    double w = initial_w;
    for (i=0; i < agent_count; i++) {
        response = get_agent_choice_complex(w, theta, alpha, beta);
        w = w*(1-r) + response*r;
    }
    return w;
}

// Returns 1 if the last `tail_count` witnesses unanimously affirm,
// after running a "warm-up" simulation with agent_count witnesses.
char last_n_unanimous(double theta, double r, double alpha, double beta, int agent_count, double intial_m, int tail_count) {
    double m = intial_m;
    int i;
    char response;
    for (i=0; i < agent_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        m = m*(1-r) + response*r;
    }
    for (i=0; i < tail_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        if (!response) {
            return 0;
        }
        m = m*(1-r) + response*r;
    }
    return 1;
}

// Returns the probability of the last n witnesses unanimously
// affirming, by running `last_n_unanimous` for `num_reps` iterations
// and counting the number of successful runs.
double prob_last_n_unanimous(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_reps) {
    int i;
    int successes = 0;
    double ans;
    for (i = 0; i < num_reps; i++) {
        if (last_n_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count)) {
            successes++;
        }
    }
    ans = ((double) successes) / num_reps;
    return ans;
}

// The same as `prob_last_n_unanimous`, but each "warm-up" run gets
// associated with many different tails, which get averaged together
void prob_last_n_unanimous_with_fanout(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_full_reps, int tail_fanout, int min_successful_reps, double* prob_out, double* err_out) {
    int i, j;
    int successes = 0;
    double ans;
    double current_m;
    for (i = 0; i < num_full_reps; i++) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_unanimous(theta, r, alpha, beta, 0, current_m, tail_count)) {
                successes++;
            }
        }
    }
    while (successes < min_successful_reps) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_unanimous(theta, r, alpha, beta, 0, current_m, tail_count)) {
                successes++;
            }
        }
        num_full_reps++;
    }
    ans = ((double) successes) / (num_full_reps * tail_fanout);
    *prob_out = ans;
    *err_out = sqrt(ans*(1-ans) / (num_full_reps * tail_fanout));
}

// Returns 1 if the last `tail_count` witnesses nearly unanimously
// affirm (allowing up to `reject_allowance` witnesses to disagree),
// after running a "warm-up" simulation with agent_count witnesses.
char last_n_near_unanimous(double theta, double r, double alpha, double beta, int agent_count, double intial_m, int tail_count, int reject_allowance) {
    double m = intial_m;
    int i;
    char response;
    for (i=0; i < agent_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        m = m*(1-r) + response*r;
    }
    for (i=0; i < tail_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        if (!response) {
            reject_allowance--;
            if (reject_allowance <= 0) {
                return 0;
            }
        }
        m = m*(1-r) + response*r;
    }
    return 1;
}

// Returns the probability of the last n witnesses nearly unanimously
// affirming, by running `last_n_near_unanimous` for `num_reps`
// iterations and counting the number of successful runs.
void prob_last_n_near_unanimous(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_reps, int reject_allowance, int min_successful_reps, double* prob_out, double* err_out) {
    int i;
    int successes = 0;
    double ans;
    for (i = 0; i < num_reps; i++) {
        if (last_n_near_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count, reject_allowance)) {
            successes++;
        }
    }
    while (successes < min_successful_reps) {
        if (last_n_near_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count, reject_allowance)) {
            successes++;
        }
        num_reps++;
    }
    *prob_out = ((double) successes) / num_reps;
    *err_out = sqrt(*prob_out*(1.0 - *prob_out) / num_reps);
}

// The same as `prob_last_n_near_unanimous`, but each "warm-up" run gets
// associated with many different tails, which get averaged together
void prob_last_n_near_unanimous_with_fanout(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_full_reps, int tail_fanout, int reject_allowance, int min_successful_reps, double* prob_out, double* err_out) {
    int i, j;
    int successes = 0;
    double ans;
    double current_m;
    for (i = 0; i < num_full_reps; i++) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_near_unanimous(theta, r, alpha, beta, 0, current_m, tail_count, reject_allowance)) {
                successes++;
            }
        }
    }
    while (successes < min_successful_reps) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_near_unanimous(theta, r, alpha, beta, 0, current_m, tail_count, reject_allowance)) {
                successes++;
            }
        }
        num_full_reps++;
    }
    *prob_out = ((double) successes) / (num_full_reps * tail_fanout);
    *err_out = sqrt(*prob_out*(1.0 - *prob_out) / (num_full_reps * tail_fanout));
}

// The same as `prob_last_n_near_unanimous`, but a consensus affirming
// or rejecting is counted as a consensus.
char last_n_near_unanimous_bidirectional(double theta, double r, double alpha, double beta, int agent_count, double intial_m, int tail_count, int reject_allowance) {
    double m = intial_m;
    int i;
    char response;
    char required_response;
    for (i=0; i < agent_count; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        m = m*(1-r) + response*r;
    }
    required_response = get_agent_choice_complex(m, theta, alpha, beta);
        m = m*(1-r) + required_response*r;
    for (i=0; i < tail_count-1; i++) {
        response = get_agent_choice_complex(m, theta, alpha, beta);
        if (response != required_response) {
            reject_allowance--;
            if (reject_allowance <= 0) {
                return 0;
            }
        }
        m = m*(1-r) + response*r;
    }
    return 1;
}

// Returns the probability of the last n witnesses having a near
// consensus, by running `last_n_near_unanimous_bidirectional` for
// `num_reps` iterations and counting the number of successful runs.
void prob_last_n_near_unanimous_bidirectional(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_reps, int reject_allowance, int min_successful_reps, double* prob_out, double* err_out) {
    int i;
    int successes = 0;
    double ans;
    for (i = 0; i < num_reps; i++) {
        if (last_n_near_unanimous_bidirectional(theta, r, alpha, beta, agent_count, initial_m, tail_count, reject_allowance)) {
            successes++;
        }
    }
    while (successes < min_successful_reps) {
        if (last_n_near_unanimous_bidirectional(theta, r, alpha, beta, agent_count, initial_m, tail_count, reject_allowance)) {
            successes++;
        }
        num_reps++;
    }
    *prob_out = ((double) successes) / num_reps;
    *err_out = sqrt(*prob_out*(1.0 - *prob_out) / num_reps);
}

// The same as `prob_last_n_near_unanimous_bidirectional`, but each
// "warm-up" run gets associated with many different tails, which get
// averaged together
void prob_last_n_near_unanimous_with_fanout_bidirectional(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_full_reps, int tail_fanout, int reject_allowance, int min_successful_reps, double* prob_out, double* err_out) {
    int i, j;
    int successes = 0;
    double ans;
    double current_m;
    for (i = 0; i < num_full_reps; i++) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_near_unanimous_bidirectional(theta, r, alpha, beta, 0, current_m, tail_count, reject_allowance)) {
                successes++;
            }
        }
    }
    while (successes < min_successful_reps) {
        current_m = sim_complex_get_w(theta, alpha, beta, r, agent_count, initial_m);
        for (j = 0; j < tail_fanout; j++) {
            if (last_n_near_unanimous_bidirectional(theta, r, alpha, beta, 0, current_m, tail_count, reject_allowance)) {
                successes++;
            }
        }
        num_full_reps++;
    }
    *prob_out = ((double) successes) / (num_full_reps * tail_fanout);
    *err_out = sqrt(*prob_out*(1.0 - *prob_out) / (num_full_reps * tail_fanout));
}

// Given an array of values, and the number of trials those values were
// computed for, compute the standard error for each trial and put it
// into result.
void compute_error_estimates(double* result, double* values, int num_values, int num_trials) {
    int i;
    for (i=0; i < num_values; i++) {
        result[i] = sqrt(values[i]*(1-values[i])/num_trials);
    }
}
