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
char get_agent_choice(double m, double theta) {
    int rng;
    pthread_mutex_lock(&rand_lock);
    rng = rand();
    pthread_mutex_unlock(&rand_lock);
    return ((double)rng) / RAND_MAX < g(m, theta);
}

// Runs the basic simulation and stores its result in the array passed to result
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
char get_agent_choice_complex(double m, double theta, double alpha, double beta) {
    int rng;
    pthread_mutex_lock(&rand_lock);
    rng = rand();
    pthread_mutex_unlock(&rand_lock);
    return ((double)rng) / RAND_MAX < g_complex(m, theta, alpha, beta);
}

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

double prob_last_n_near_unanimous(double theta, double r, double alpha, double beta, int agent_count, double initial_m, int tail_count, int num_reps, int reject_allowance) {
    int i;
    int successes = 0;
    double ans;
    for (i = 0; i < num_reps; i++) {
        if (last_n_near_unanimous(theta, r, alpha, beta, agent_count, initial_m, tail_count, reject_allowance)) {
            successes++;
        }
    }
    ans = ((double) successes) / num_reps;
    return ans;
}

void compute_error_estimates(double* result, double* values, int num_values, int num_trials) {
    int i;
    for (i=0; i < num_values; i++) {
        result[i] = 1.96 * sqrt(values[i]*(1-values[i])/num_trials);
    }
}
