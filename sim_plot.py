from concurrent import futures
from sys import argv as args

import numpy as np
import matplotlib.pyplot as plt

from basic_sim import g, g_external_pressure
from simulate_c import many_simulate_complex, many_simulate_basic, compute_error_estimates, \
        prob_last_n_unanimous_with_fanout, prob_last_n_unanimous, prob_last_n_near_unanimous, prob_last_n_near_unanimous_with_fanout, \
        prob_last_n_near_unanimous_with_fanout_bidirectional, many_simulate_complex_get_ws

process_executor = futures.ProcessPoolExecutor(max_workers=20)

# Remove excess whitespace from the figures when saving
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0}

def plot_g(thetas, save_file=None):
    """For each value of theta in the given list, it produces a plot of
    g(M_i) for that value of theta. The values then get put on the same
    axes and plotted.
    """
    xs = np.linspace(0, 1, 1000)
    for theta in thetas:
        plt.plot(xs, g(xs, theta), label=f'$\\theta$ = {theta}')
    plt.title('Plots of $g$ for different values of $\\theta$')
    plt.xlabel('$M_i$')
    plt.ylabel('$g_\\theta(M_i)$')
    plt.legend()
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
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
    ax.plot(1-xs, ys, marker=',', color='#324dbe')
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def weights_vs_r(rvals, xvals, save_file=None):
    for rval in rvals:
        plt.plot(xvals, rval*np.power(1-rval, xvals-1), label=f'r={rval}')
    plt.title('Weighting decay for different values of r')
    plt.xlabel('$\\delta$')
    plt.ylabel('$w(r, \\delta)$')
    plt.legend()
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
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
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def compute_counts_for_args(args):
    theta, r, alpha, beta, agent_count, num_reps, initial = args
    sim_res = many_simulate_complex(theta, r, alpha, beta, agent_count, num_reps, initial_m=initial)
    freqs = np.array([sum(sim[i] for sim in sim_res) / num_reps for i in range(agent_count)])
    return freqs
def plot_prob_affirm_vs_position(alphas, theta=2.0, r=0.035, beta=1.0, agent_count=400, num_reps=3500, initial=0.5, save_file=None):
    sims = process_executor.map(compute_counts_for_args, [(theta, r, alpha, beta, agent_count, num_reps, initial) for alpha in alphas])
    xs = np.arange(1, agent_count+1, 1)
    for sim, alpha in zip(sims, alphas):
        plt.plot(xs, sim, label=f'\u03b1 = {alpha}')
        shade_error_region(xs, sim, compute_error_estimates(sim, num_reps), alpha=0.5)
    plt.xlim(0, agent_count)
    # plt.ylim(0, 1)
    plt.legend()
    plt.title('\u03b1 vs. likelihood of witness affirming')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of affirming')
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

# def compute_initial_m_from_beta(beta, theta, initial_g):
    # # wolfram|alpha (I used this query to solve for the initial value of M
    # # https://www.wolframalpha.com/input/?i=solve+1%2F%281%2Bexp%28-t*%28M-0.5%29%29%29+%3D+f%2Fb+*+%281%2F%281%2Bexp%28-t%2F2%29%29+-+1%2F%281%2Bexp%28t%2F2%29%29%29+%2B+1%2F%281%2Bexp%28t%2F2%29%29%2C+b%3Ef%3E0%2C+t%3E0+for+M
    # if beta <= initial_g:
        # return 1
    # else:
        # return (np.log(np.exp(theta/2)) + np.log(beta + initial_g*(np.exp(theta/2) - 1)) - np.log((beta-initial_g)*np.exp(theta/2)+initial_g))/theta
# def compute_initial_m_from_alpha(alpha, theta, initial_g):
    # # wolfram|alpha (I used this query to solve for the initial value of M
    # # https://www.wolframalpha.com/input/?i=solve+1%2F%281%2Bexp%28-t*%28M-0.5%29%29%29+%3D+%28c-a%29%28exp%28t%2F2%29-exp%28-t%2F2%29%29%2F%5B%281-a%29%281%2Bexp%28t%2F2%29%29%281%2Bexp%28-t%2F2%29%29%5D+%2B+1%2F%281%2Bexp%28t%2F2%29
    # if alpha >= initial_g:
        # return 0
    # else:
        # a = alpha
        # c = initial_g
        # e = np.exp(theta)
        # f = np.exp(theta/2)
        # g = np.exp(3*theta/2)
        # t = theta
        # return np.log((a**2*e+a*c*f-2*a*c*e-a*c*g-a*f+a*g-c**2*(f-g)+c*(2*e+f-g)-e) / (a**2-2*a*c-(c**2-2*c+1)*e+c**2)) / t

def plot_prob_affirm_vs_position_with_initial_g(alphas, theta=2.0, r=0.035, beta=1.0, agent_count=400, num_reps=5000, initial_g=0.5, save_file=None):
    def compute_initial_m_from_alpha(alpha, theta, initial_g):
        # wolfram|alpha (I used this query to solve for the initial value of M
        # https://www.wolframalpha.com/input/?i=solve+1%2F%281%2Bexp%28-t*%28M-0.5%29%29%29+%3D+%28c-a%29%28exp%28t%2F2%29-exp%28-t%2F2%29%29%2F%5B%281-a%29%281%2Bexp%28t%2F2%29%29%281%2Bexp%28-t%2F2%29%29%5D+%2B+1%2F%281%2Bexp%28t%2F2%29
        if alpha >= initial_g:
            return 0.0
        elif initial_g >= 1.0:
            return 1.0
        else:
            upper_bound = 1.0
            lower_bound = 0.0
            while upper_bound - lower_bound > 1e-11:
                mid = (upper_bound + lower_bound)/2
                resulting_g = g_external_pressure(mid, theta, alpha, 1.0)
                if resulting_g < initial_g:
                    lower_bound = mid
                elif resulting_g > initial_g:
                    upper_bound = mid
                else:
                    return mid
            ans = (upper_bound + lower_bound)/2
            # print(f'The value of m for alpha={alpha}, theta={theta}, and an initial g of {initial_g} is {ans}, which results in an initial g of {g_external_pressure(ans, theta, alpha, 1.0)}.')
            return ans
    sims = process_executor.map(
        compute_counts_for_args,
        [(theta, r, alpha, beta, agent_count, num_reps, compute_initial_m_from_alpha(alpha, theta, initial_g))
         for alpha in alphas
        ]
    )
    xs = np.arange(1, agent_count+1, 1)
    for sim, alpha in zip(sims, alphas):
        plt.plot(xs, sim, label=f'\u03b1 = {alpha}')
        shade_error_region(xs, sim, compute_error_estimates(sim, num_reps), alpha=0.5)
    plt.xlim(0, agent_count)
    # plt.ylim(0, 1)
    plt.legend()
    plt.title('\u03b1 vs. likelihood of witness affirming')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of affirming')
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def plot_affirm_rate_each_trial(alpha=0.0, beta=1.0, theta=10.0, r=0.035, agent_count=200, num_reps=250, initial_w=0.5, initial_g=None, save_file=None, plot_w=True):
    """Plot the results of each trial as a thin line, so tendencies can
    be noticed.

    If a value is providded for `initial_g`, then the initial value of W
    is determined for the specified value of g for the first witness.
    Otherwise, the value of `initial_w` is used.

    If plot_w is True, then the value of W will be plotted. Otherwise,
    the value of g(w) will be plotted.
    """
    def g(w):
        f = lambda theta, m: 1. / (1 + np.exp(theta * (m-0.5)))
        plain_g = (f(theta, w) - f(theta, 0))/(f(theta, 1) - f(theta, 0))
        return alpha + plain_g*(beta-alpha)
    if initial_g is not None:
        if initial_w != 0.5:
            print('Warning: initial_w was specified but ignored, because initial_g was also specified.')
            print('     in: `plot_affirm_rate_each_trial`')
        if beta <= initial_g:
            initial_w = 1
        else:
            initial_w = (np.log(np.exp(theta/2)) + np.log(beta + initial_g*(np.exp(theta/2)-1)) - np.log((beta-initial_g)*np.exp(theta/2)+initial_g))/theta
    sims = many_simulate_complex_get_ws(theta, r, alpha, beta, agent_count, num_reps, initial_w)
    xs = np.arange(0, agent_count+1, 1)
    for sim in sims:
        if plot_w:plt.plot(xs, sim, color='#324dbe', marker='', linewidth=0.1)
        else: plt.plot(xs, g(sim), color='#324dbe', marker='', linewidth=0.1)
    if plot_w: plt.ylabel('$W_i$')
    else: plt.ylabel(f'$g_{{{theta},{alpha},{beta}}}(W_i)$')
    plt.xlabel('Witness number')
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def plot_expected_overall_and_individuals(alpha=0.0, beta=1.0, theta=10.0, r=0.035, agent_count=200, num_reps=250, initial_w=0.5, save_file=None, lower_bias=1, upper_bias=1):
    def g(w):
        f = lambda theta, m: 1. / (1 + np.exp(theta * (m-0.5)))
        plain_g = (f(theta, w) - f(theta, 0))/(f(theta, 1) - f(theta, 0))
        return alpha + plain_g*(beta-alpha)
    sims = many_simulate_complex_get_ws(theta, r, alpha, beta, agent_count, num_reps, initial_w)
    xs = np.arange(0, agent_count+1, 1)
    from random import random
    counted_runs = []
    for sim in sims:
        if sim[-1] <= 0.5:
            if random() < lower_bias:
                counted_runs.append(g(sim))
            else:
                continue
        else:
            if random() < upper_bias:
                counted_runs.append(g(sim))
            else:
                continue
        plt.plot(xs, sim, color='#324dbe', marker='', linewidth=0.1)
    average = np.array([sum(run[i] for run in counted_runs)/len(counted_runs) for i in range(len(sims[0]))])
    plt.plot(xs, average, color='k', marker='')
    plt.ylabel('$W_i$')
    plt.xlabel('Witness number')
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def prob_from_args(args):
    return prob_last_n_unanimous_with_fanout(*args)
def plot_prob_h_given_e_for_lambdas(lambdas, phs, pf, pt, N, theta, r, agent_count, num_reps, initial=0.5, save_file=None, tail_fanout=100):
    def phle_from_probs(prob, error, ph):
        peh = pt*ph
        pelnh = prob
        pelnh_d = error
        pe = peh + pelnh*(1-ph)
        phle = peh / (peh + pelnh*(1-ph))
        phle_d = phle * pelnh_d*(1-ph) / (peh + pelnh*(1-ph))
        return phle, phle_d
    sim_data = list(process_executor.map(prob_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps // tail_fanout, tail_fanout)
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
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def plot_prob_of_consensus_for_lambdas(lambdas, pf, N, theta, r, agent_count, num_reps, initial=0.5, plot_log=True, save_file=None, tail_fanout=100):
    probs, error_bars = map(np.array, zip(*process_executor.map(prob_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps // tail_fanout, tail_fanout)
        for l in lambdas
    ])))
    fig, ax = plt.subplots()
    if plot_log: ax.set_yscale("log")
    plt.plot(lambdas, probs, marker=',', color='#324dbe')
    shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5, color='#324dbe')
    plt.xlabel('$\\lambda$')
    plt.ylabel('Probability of Consensus Affirming')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def prob_near_unanimous_from_args(args):
    return prob_last_n_near_unanimous_with_fanout(*args)
def plot_prob_of_near_consensus_for_lambdas(lambdas, pf, N, theta, r, agent_count, num_reps, frac_required, initial=0.5, plot_log=True, save_file=None, min_successful_reps=9, tail_fanout=100):
    probs, error_bars = map(np.array, zip(*process_executor.map(prob_near_unanimous_from_args, [
        (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps // tail_fanout, tail_fanout, frac_required, min_successful_reps)
        for l in lambdas
    ])))
    fig, ax = plt.subplots()
    if plot_log: ax.set_yscale("log")
    plt.plot(lambdas, probs, marker=',', color='#324dbe')
    shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5, color='#324dbe')
    plt.xlabel('$\\lambda$')
    plt.ylabel('Probability of Near Consensus Affirming')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def prob_near_unanimous_bidirectional_from_args(args):
    return prob_last_n_near_unanimous_with_fanout_bidirectional(*args)
def plot_prob_of_near_bidirectional_consensus_xaxis_lambdas(lambdas, pfs, N, theta, r, agent_count, num_reps, frac_required, initial=0.5, plot_log=True, save_file=None, min_successful_reps=16, tail_fanout=10):
    if plot_log:
        fig, ax = plt.subplots()
        ax.set_yscale("log")
    for pf in pfs:
        probs, error_bars = map(np.array, zip(*process_executor.map(prob_near_unanimous_bidirectional_from_args, [
            (theta, r, (1-l)*pf, pf + (1-pf)*l, agent_count, initial, N, num_reps // tail_fanout, tail_fanout, frac_required, min_successful_reps)
            for l in lambdas
        ])))
        plt.plot(lambdas, probs, label=f'$p_f = {pf}$')
        shade_error_region(lambdas, probs, error_bars * 1.96, alpha=0.5)
    plt.legend()
    plt.xlabel('$\\lambda$')
    plt.ylabel('Probability of Near Consensus')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()
def plot_prob_of_near_bidirectional_consensus_xaxis_pfs(lambdas, pfs, N, theta, r, agent_count, num_reps, frac_required, initial=0.5, plot_log=True, save_file=None, min_successful_reps=16, tail_fanout=10):
    if plot_log:
        fig, ax = plt.subplots()
        ax.set_yscale("log")
    for lbd in lambdas:
        probs, error_bars = map(np.array, zip(*process_executor.map(prob_near_unanimous_bidirectional_from_args, [
            (theta, r, (1-lbd)*pf, pf + (1-pf)*lbd, agent_count, initial, N, num_reps // tail_fanout, tail_fanout, frac_required, min_successful_reps)
            for pf in pfs
        ])))
        plt.plot(pfs, probs, label=f'$\lambda = {lbd}$')
        shade_error_region(pfs, probs, error_bars * 1.96, alpha=0.5)
    plt.legend()
    plt.xlabel('$p_f$')
    plt.ylabel('Probability of Near Consensus')
    plt.xlim(0, 1)
    if save_file is None: plt.show()
    else: plt.savefig(save_file, **savefig_args)
    plt.clf()

def setup_plot_style():
    plt.style.use('ggplot')
    from cycler import cycler
    # Setup colors
    style_cycler = cycler(color=['#f23030', '#f2762e', '#f2d338', '#25c7d9', '#248ea6', '#324dbe'])
    # Add markers to help colorblind people/people with greyscale printouts
    style_cycler += cycler(marker=['o', '^', 's', 'P', '*', 'v'])
    style_cycler *= cycler(markevery=[0.2])
    plt.rc('axes', prop_cycle=style_cycler)

class AllContainer:
    def __contains__(self, item): return True

def main(sections=AllContainer()):
    setup_plot_style()

    if 'polarized' in sections:
        plot_g([1, 5, 10, 30, 100], save_file='../plots/g-vs-theta.pdf')
        visualize_agreement([2.0, 5.0, 7.0, 10, 20], agent_count=100, num_reps=10000, save_file='../plots/simple-dependence-agreement.pdf')
        visualize_agreement([2.0, 5.0, 7.0, 10, 20], agent_count=1000, num_reps=25000, save_file='../plots/simple-dependence-agreement-long.pdf')
        print('Polarized Majority Vote Done!')

    if 'recent' in sections:
        r_vs_majority_time(save_file='../plots/r-vs-majority-time.pdf')
        weights_vs_r([0.4, 0.3, 0.2, 0.1, 0.001], np.arange(1, 50, 1), save_file='../plots/weight-function.pdf')
        print('Recent Majority Vote Done!')
 
    if 'external' in sections:
        ep_alphas = [0.5, 0.25, 0.1, 0.05, 0.0]
        plot_prob_affirm_vs_position(alphas=ep_alphas, theta=10.0, save_file='../plots/alpha-vs-affirm-rate-over-time.pdf', agent_count=200)
        plot_prob_affirm_vs_position_with_initial_g(alphas=ep_alphas, theta=10.0, initial_g=0.5, save_file='../plots/alpha-vs-affirm-rate-over-time-initial-g-half.pdf', agent_count=200)
        print('External Pressure Done!')
    if 'external-betas' in sections:
        ep_betas = [0.5, 0.75, 0.9, 0.95, 1.0]
        # plot_prob_affirm_vs_position(betas=ep_betas, theta=10.0, save_file='../plots/beta-vs-affirm-rate-over-time.pdf')
        # plot_prob_affirm_vs_position_with_initial_g(betas=ep_betas, theta=10.0, initial_g=0.5, save_file='../plots/beta-vs-affirm-rate-over-time-initial-g-half.pdf')
        # plot_prob_affirm_vs_position_with_initial_g(betas=ep_betas, theta=10.0, initial_g=0.7, save_file='../plots/beta-vs-affirm-rate-over-time-initial-g-majority.pdf')
        # plot_prob_affirm_vs_position_with_initial_g(betas=ep_betas, theta=10.0, initial_g=0.7, agent_count=2000, save_file='../plots/betas-vs-affirm-rate-over-time-initial-g-majority-long.pdf')
        # print('External Pressure with betas Done!')
        print('External Pressure with betas not implemented yet')

    if 'spectrum' in sections:
        plot_prob_h_given_e_for_lambdas(np.linspace(0, 1, 100), [.5, .1, 1e-2, 1e-3, 1e-4], 0.95, 0.5, 150, 10.0, 0.035, 1600, 80000, tail_fanout=20, save_file='../plots/sod-h-given-e.pdf')
        # plot_prob_h_given_e_for_lambdas(np.linspace(0, 1, 40), [.5, 1e-3, 1e-5, 1e-7, 1e-8, 1e-9], 0.25, 0.5, 13, 10.0, 0.035, 1600, 100000, tail_fanout=250, save_file='../plots/sod-h-given-e-small-pf.pdf')
        plot_prob_of_consensus_for_lambdas(np.linspace(0, 1, 50), 0.5, 13, 10.0, 0.035, 1600, 150000, plot_log=True, tail_fanout=20, save_file='../plots/sod-pconsensus-log.pdf')
        plot_prob_of_consensus_for_lambdas(np.linspace(0, 1, 50), 0.5, 13, 10.0, 0.035, 1600, 150000, plot_log=False, tail_fanout=20, save_file='../plots/sod-pconsensus-linear.pdf')
        plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 40), 0.35, 20, 10.0, 0.035, 1600, 80000, .9, save_file='../plots/sod-near-consensus-4-log.pdf')
        plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 40), 0.5, 30, 10.0, 0.035, 1600, 80000, .9, save_file='../plots/sod-near-consensus-2-log.pdf')
        plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 50), 0.9, 120, 10.0, 0.035, 1600, 80000, .99, save_file='../plots/sod-near-consensus-1-log.pdf')
        plot_prob_of_near_consensus_for_lambdas(np.linspace(0, 1, 100), 0.95, 150, 10.0, 0.035, 1600, 80000, .99, save_file='../plots/sod-near-consensus-3-linear.pdf', plot_log=False)
        print('Spectrum of Dependence Done!')
        plot_prob_of_near_bidirectional_consensus_xaxis_lambdas(np.linspace(0, 1, 100), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 20, 10.0, 0.035, 1600, 240000, .9, save_file='../plots/sod-bidirectional-xlambdas-linear.pdf', plot_log=False)
        plot_prob_of_near_bidirectional_consensus_xaxis_lambdas(np.linspace(0, 1, 100), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 20, 10.0, 0.035, 1600, 240000, .9, save_file='../plots/sod-bidirectional-xlambdas-log.pdf')
        plot_prob_of_near_bidirectional_consensus_xaxis_pfs([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], np.linspace(0, 1, 100), 20, 10.0, 0.035, 1600, 240000, .9, save_file='../plots/sod-bidirectional-xpfs-linear.pdf', plot_log=False)
        plot_prob_of_near_bidirectional_consensus_xaxis_pfs([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], np.linspace(0, 1, 100), 20, 10.0, 0.035, 1600, 240000, .9, save_file='../plots/sod-bidirectional-xpfs-log.pdf')
        print('Overall Consensus Probability Done!')

    if 'misc' in sections:
        plot_affirm_rate_each_trial(beta=0.5, initial_g=0.5, plot_w=False, save_file='../plots/each-run-affirm-beta-5-g.pdf')
        plot_affirm_rate_each_trial(beta=0.5, initial_g=0.5, save_file='../plots/each-run-affirm-beta-5.pdf')
        plot_affirm_rate_each_trial(beta=0.95, initial_g=0.5, agent_count=400, save_file='../plots/each-run-affirm-beta-95-g-half.pdf')
        plot_affirm_rate_each_trial(beta=0.95, agent_count=400, save_file='../plots/each-run-affirm-beta-95-w-half.pdf')
        plot_affirm_rate_each_trial(beta=0.9, initial_g=0.5, agent_count=400, save_file='../plots/each-run-affirm-beta-9-g-half.pdf')
        plot_affirm_rate_each_trial(beta=0.9, agent_count=400, save_file='../plots/each-run-affirm-beta-9-w-half.pdf')
        plot_expected_overall_and_individuals(beta=0.9, agent_count=400, save_file='../plots/each-run-affirm-beta-9-upper-all.pdf')
        plot_expected_overall_and_individuals(beta=0.9, agent_count=400, save_file='../plots/each-run-affirm-beta-9-upper-most.pdf', upper_bias=0.7)
        plot_expected_overall_and_individuals(beta=0.9, agent_count=400, save_file='../plots/each-run-affirm-beta-9-upper-half.pdf', upper_bias=0.5)
        plot_expected_overall_and_individuals(beta=0.9, agent_count=400, save_file='../plots/each-run-affirm-beta-9-upper-few.pdf', upper_bias=0.3)

if __name__ == '__main__':
    if len(args) > 1:
        main(args[1:])
    else:
        main()
