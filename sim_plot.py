from concurrent import futures
import numpy as np
import matplotlib.pyplot as plt

from basic_sim import g
from simulate_c import many_simulate_complex, many_simulate_basic, compute_error_estimates, \
        prob_last_n_unanimous, prob_last_n_near_unanimous

process_executor = futures.ProcessPoolExecutor()

def plot_g(thetas, save_file=None):
    """For each value of theta in the given list, it produces a plot of
    g(x) for that value of theta. The values then get put on the same
    axes and plotted.
    """
    xs = np.linspace(0, 1, 1000)
    for theta in thetas:
        plt.plot(xs, g(xs, theta), label=f'theta = {theta}')
    plt.title('Plots of g for different values of theta')
    plt.xlabel('$x$')
    plt.ylabel('$g_\\theta(x)$')
    plt.legend()
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def r_vs_majority_time(xlog=False, ylog=False, save_file=None):
    xs = np.linspace(0, 0.5, 2000)[1:]
    ys = np.ceil(np.log(0.5)/np.log(1-xs))
    fig, ax = plt.subplots()
    if ylog:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.set_yticklabels(['1', '10', '100', '1000', '10000'])
    ax.set_ylim(top=400)
    ax.set_ylabel("Number of witnesses to overturn a majority")
    if xlog:
        ax.set_xscale("log")
        ax.set_xticks([0.001, 0.01, 0.1, 0.5])
        ax.set_xticklabels(['0.001', '0.01', '0.1', '0.5'])
        ax.grid(which='minor', linewidth=0.3)
        ax.grid(which='major', linewidth=0.8)
    ax.set_xlabel("Value of 1-r")
    ax.plot(1-xs, ys)
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def weights_vs_r(rvals, xvals, save_file=None):
    for rval in rvals:
        plt.plot(xvals, rval*np.power(1-rval, xvals-1), label=f'r={rval}')
    plt.title('Weighting decay for different values of r')
    plt.xlabel('$\\delta$')
    plt.ylabel('$w(r, \\delta)$')
    plt.legend()
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def shade_error_region(xs, ys, errors, *args, **kwargs):
    plt.fill_between(xs, ys-errors, ys+errors, in_layout=True, *args, **kwargs)

def compute_agree_odds(args):
    theta, agent_count, num_reps = args
    sims = many_simulate_basic(theta, agent_count, num_reps)
    return np.array([sum(sim[i] == sim[i-1] for sim in sims) / num_reps for i in range(1, agent_count)])
def visualize_agreement(thetas, agent_count=500, num_reps=3500, save_file=None):
    all_agree_odds = process_executor.map(compute_agree_odds, [(theta, agent_count, num_reps) for theta in thetas])
    xs = np.arange(1, agent_count, 1)
    for theta, agree_odds in zip(thetas, all_agree_odds):
        plt.plot(xs, agree_odds, label=f'\u03b8 = {theta}')
        errors = np.sqrt(agree_odds*(1-agree_odds)/num_reps)*1.96
        shade_error_region(xs, agree_odds, errors, alpha=0.5)
    plt.xlim(1, agent_count)
    plt.legend()
    plt.title('Evolution of Agreement among witnesses')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of witness agreeing with previous')
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def compute_counts_for_args(args):
    theta, r, alpha, beta, agent_count, num_reps, initial = args
    sim_res = many_simulate_complex(theta, r, alpha, beta, agent_count, num_reps, initial_m=initial)
    freqs = np.array([sum(sim[i] for sim in sim_res) / num_reps for i in range(agent_count)])
    return freqs
def plot_prob_affirm_vs_position(betas, theta=2.0, r=0.035, alpha=0.0, agent_count=400, num_reps=3500, initial=0.5, save_file=None):
    sims = process_executor.map(compute_counts_for_args, [(theta, r, alpha, beta, agent_count, num_reps, initial) for beta in betas])
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
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def plot_prob_affirm_vs_position_with_initial_g(betas, theta=2.0, r=0.035, agent_count=400, num_reps=5000, initial_g=0.5, save_file=None):
    def compute_initial_m(beta, theta, initial_g):
        # wolfram|alpha (I used this query to solve for the initial value of M
        # https://www.wolframalpha.com/input/?i=solve+1%2F%281%2Bexp%28-t*%28M-0.5%29%29%29+%3D+f%2Fb+*+%281%2F%281%2Bexp%28-t%2F2%29%29+-+1%2F%281%2Bexp%28t%2F2%29%29%29+%2B+1%2F%281%2Bexp%28t%2F2%29%29%2C+b%3Ef%3E0%2C+t%3E0+for+M
        if beta <= initial_g:
            return 1
        else:
            return (np.log(np.exp(theta/2)) + np.log(beta + initial_g*(np.exp(theta/2) - 1)) - np.log((beta-initial_g)*np.exp(theta/2)+initial_g))/theta
    sims = process_executor.map(
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
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def prob_from_args(args):
    return prob_last_n_unanimous(*args)
def plot_prob_h_given_e_for_lambdas(lambdas, phs, pf, pt, N, theta, r, agent_count, num_reps, initial=0.5, save_file=None):
    def phle_from_probs(prob, error, ph):
        peh = pt*ph
        pelnh = prob
        pelnh_d = error
        pe = peh + pelnh*(1-ph)
        phle = peh / (peh + pelnh*(1-ph))
        phle_d = phle * pelnh_d*(1-ph) / (peh + pelnh*(1-ph))
        return phle, phle_d
    sim_data = list(process_executor.map(prob_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps)
        for l in lambdas
    ]))
    for ph in phs:
        probs, error_bars = map(np.array, zip(*map(lambda t: phle_from_probs(*t, ph), sim_data)))
        plt.plot(lambdas, probs, label=f'$P(H) = {ph}$')
        shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('$P(H|E)$')
    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.legend()
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def plot_prob_of_consensus_for_lambdas(lambdas, pf, N, theta, r, agent_count, num_reps, initial=0.5, plot_log=True, save_file=None):
    probs, error_bars = map(np.array, zip(*process_executor.map(prob_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps)
        for l in lambdas
    ])))
    fig, ax = plt.subplots()
    if plot_log: ax.set_yscale("log")
    plt.plot(lambdas, probs)
    shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('Probability of Consensus')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def prob_near_unanimous_from_args(args):
    return prob_last_n_near_unanimous(*args)
def plot_prob_of_near_consensus_for_lambdas(lambdas, pf, N, theta, r, agent_count, num_reps, frac_required, initial=0.5, plot_log=True, save_file=None, min_successful_reps=16):
    probs, error_bars = map(np.array, zip(*process_executor.map(prob_near_unanimous_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps, frac_required, min_successful_reps)
        for l in lambdas
    ])))
    fig, ax = plt.subplots()
    if plot_log: ax.set_yscale("log")
    plt.plot(lambdas, probs)
    shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5)
    plt.xlabel('$\\lambda$')
    plt.ylabel('Probability of Near Consensus')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file)
    plt.clf()

def main():
    plt.style.use('ggplot')
    from cycler import cycler
    style_cycler = cycler(color=['#eb3d02', '#ff0dff', '#0004e8', '#ffba00', '#00b800'])
    style_cycler += cycler(marker=['o', '^', 's', '+', '*'])
    style_cycler *= cycler(markevery=[0.2])
    plt.rc('axes', prop_cycle=style_cycler)

    plot_g([1, 5, 10, 30, 100], save_file='../plots/g-vs-theta.pdf')
    visualize_agreement([2.0, 5.0, 7.0, 10, 20], agent_count=100, num_reps=10000, save_file='../plots/simple-dependence-agreement.pdf')
    visualize_agreement([2.0, 5.0, 7.0, 10, 20], agent_count=1000, num_reps=25000, save_file='../plots/simple-dependence-agreement-long.pdf')
    print('Simple Majority Vote Done!')

    r_vs_majority_time(save_file='../plots/r-vs-majority-time.pdf')
    weights_vs_r([0.4, 0.3, 0.2, 0.1, 0.001], np.arange(1, 50, 1), save_file='../plots/weight-function.pdf')
    print('Recent Majority Vote Done!')

    ep_betas = [0.5, 0.75, 0.9, 0.95, 1.0]
    plot_prob_affirm_vs_position(ep_betas, save_file='../plots/beta-vs-affirm-rate-over-time.pdf')
    plot_prob_affirm_vs_position_with_initial_g(ep_betas, theta=2.0, initial_g=0.5, save_file='../plots/beta-vs-affirm-rate-over-time-initial-g-half.pdf')
    plot_prob_affirm_vs_position_with_initial_g(ep_betas, theta=2.0, initial_g=0.7, save_file='../plots/beta-vs-affirm-rate-over-time-initial-g-majority.pdf')
    plot_prob_affirm_vs_position_with_initial_g(ep_betas, theta=2.0, initial_g=0.7, agent_count=2000, save_file='../plots/beta-vs-affirm-rate-over-time-initial-g-majority-long.pdf')
    print('External Pressure Done!')

    plot_prob_h_given_e_for_lambdas(np.linspace(0, 1, 100), [.5, .1, 1e-2, 1e-3, 1e-4], 0.95, 0.5, 150, 5.0, 0.035, 1600, 80000, save_file='../plots/sod-h-given-e.pdf')
    plot_prob_of_consensus_for_lambdas(np.linspace(0, 1, 100), 0.5, 13, 5.0, 0.035, 1600, 80000, plot_log=True, save_file='../plots/sod-pconsensus-log.pdf')
    plot_prob_of_consensus_for_lambdas(np.linspace(0, 1, 100), 0.5, 13, 5.0, 0.035, 1600, 80000, plot_log=False, save_file='../plots/sod-pconsensus-linear.pdf')
    plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 100), 0.9, 120, 5.0, 0.035, 1600, 80000, .99, save_file='../plots/sod-near-consensus-1-log.pdf')
    plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 40), 0.5, 30, 5.0, 0.035, 1600, 80000, .9, save_file='../plots/sod-near-consensus-2-log.pdf')
    plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 100), 0.95, 150, 5.0, 0.035, 1600, 80000, .99, save_file='../plots/sod-near-consensus-3-linear.pdf', plot_log=False)
    print('Spectrum of Dependence Done!')

if __name__ == '__main__':
    main()
