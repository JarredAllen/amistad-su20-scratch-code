from concurrent import futures
import numpy as np
import matplotlib.pyplot as plt

from simulate_c import many_simulate_complex, many_simulate_basic, compute_error_estimates

def shade_error_region(xs, ys, errors, *args, **kwargs):
    plt.fill_between(xs, ys-errors, ys+errors, *args, **kwargs)

def visualize_agreement(thetas, agent_count=500, num_reps=3500):
    xs = np.arange(1, agent_count, 1)
    for theta in thetas:
        sims = many_simulate_basic(theta, agent_count, num_reps)
        agree_odds = np.array([sum(sim[i] == sim[i-1] for sim in sims) / num_reps for i in range(1, agent_count)])
        # agree_odds = np.array([(agree_odds[0]*2+agree_odds[1])/3]
                              # + [(agree_odds[i-1] + agree_odds[i] + agree_odds[i+1])/3 for i in range(1, len(agree_odds)-1)]
                              # + [(agree_odds[-2] + agree_odds[-1]*2)/3])
        plt.plot(xs, agree_odds, label=f'\u03b8 = {theta}')
        shade_error_region(xs, agree_odds, compute_error_estimates(agree_odds, 5000), alpha=0.3)
    plt.xlim(1, agent_count)
    # plt.ylim(0, 1)
    plt.legend()
    plt.title('Evolution of Agreement among witnesses')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of witness agreeing with previous')
    plt.show()

def compute_counts_for_args(args):
    theta, r, alpha, beta, agent_count, num_reps, initial = args
    sim_res = many_simulate_complex(theta, r, alpha, beta, agent_count, num_reps, initial_m=initial)
    freqs = np.array([sum(sim[i] for sim in sim_res) / num_reps for i in range(agent_count)])
    return freqs

def plot_prob_affirm_vs_position(betas, theta=2.0, r=0.035, alpha=0.0, agent_count=400, num_reps=3500, initial=0.5):
    executor = futures.ProcessPoolExecutor()
    sims = executor.map(compute_counts_for_args, [(theta, r, alpha, beta, agent_count, num_reps, initial) for beta in betas])
    xs = np.arange(1, agent_count+1, 1)
    for sim, beta in zip(sims, betas):
        plt.plot(xs, sim, label=f'\u03b2 = {beta}')
        shade_error_region(xs, sim, compute_error_estimates(sim, num_reps), alpha=0.5)
    plt.xlim(0, agent_count)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('\u03b2 vs. likelihood of witness affirming')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of affirming')
    plt.show()

def plot_prob_affirm_vs_position_with_initial_g(betas, theta=2.0, r=0.035, agent_count=400, num_reps=5000, initial_g=0.5):
    def compute_initial_m(beta, theta, initial_g):
        # wolfram|alpha (I used this query to solve for the initial value of M
        # https://www.wolframalpha.com/input/?i=solve+1%2F%281%2Bexp%28-t*%28M-0.5%29%29%29+%3D+f%2Fb+*+%281%2F%281%2Bexp%28-t%2F2%29%29+-+1%2F%281%2Bexp%28t%2F2%29%29%29+%2B+1%2F%281%2Bexp%28t%2F2%29%29%2C+b%3Ef%3E0%2C+t%3E0+for+M
        if beta <= initial_g:
            return 1
        else:
            return (np.log(np.exp(theta/2)) + np.log(beta + initial_g*(np.exp(theta/2) - 1)) - np.log((beta-initial_g)*np.exp(theta/2)+initial_g))/theta
    executor = futures.ProcessPoolExecutor()
    sims = executor.map(
        compute_counts_for_args,
        [(theta, r, 0.0, beta, agent_count, num_reps, compute_initial_m(beta, theta, initial_g))
         for beta in betas
        ]
    )
    xs = np.arange(1, agent_count+1, 1)
    for sim, beta in zip(sims, betas):
        plt.plot(xs, sim, label=f'\u03b2 = {beta}')
        shade_error_region(xs, sim, compute_error_estimates(sim, num_reps), alpha=0.5)
    plt.xlim(0, agent_count)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('\u03b2 vs. likelihood of witness affirming')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of affirming')
    plt.show()

def plot_prob_h_given_e_for_lambdas(lambdas, pf, pt, ph, N, theta, r, agent_count, num_reps, initial=0.5):
    executor = futures.ProcessPoolExecutor()
    sim_res = executor.map(compute_counts_for_args,
        [(theta, r, (1-l_val)*pf, pf + (1-pf)*l_val, agent_count, num_reps, initial)
         for l_val in lambdas
        ]
    )
    def prob_from_results(result):
        """Take a single simulation run and return the probability of h
        given e and its error bars.
        """
        peh = pt*ph
        pelnh = result[-1] ** N
        pelnh_d = (pelnh * (1-pelnh) / num_reps) ** .5  # TODO something more precise for this
        pe = peh + pelnh*(1-ph)
        phle = peh / pe
        phle_d = pelnh_d * peh * (1-ph) / (peh + (1-ph)*pelnh)**2
        return phle, phle_d
    probs, error_bars = map(np.array, zip(*map(prob_from_results, sim_res)))
    plt.plot(lambdas, probs)
    shade_error_region(lambdas, probs, error_bars, alpha=0.5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('$P(E|H)$')
    plt.show()

def main():
    # plot_prob_affirm_vs_position([0.5, 0.75, 0.9, 0.95, 1.0])
    # plot_prob_affirm_vs_position_with_initial_g([0.5, 0.75, 0.9, 0.95, 1.0], theta=2.0, initial_g=0.5)
    # plot_prob_affirm_vs_position_with_initial_g([0.5, 0.75, 0.9, 0.95, 1.0], theta=2.0, initial_g=0.7)
    # plot_prob_affirm_vs_position_with_initial_g([0.5, 0.75, 0.9, 0.95, 1.0], theta=2.0, initial_g=0.7, agent_count=2000)
    # visualize_agreement([2.0, 5.0, 7.0, 10, 20], agent_count=100, num_reps=5000)
    plot_prob_h_given_e_for_lambdas(np.linspace(0, 1, 10), 0.95, 0.1, 1e-6, 500, 2.0, 0.035, 250, 500)

if __name__ == '__main__':
    main()
