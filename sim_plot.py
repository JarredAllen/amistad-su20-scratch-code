from concurrent import futures
import numpy as np
import matplotlib.pyplot as plt

from basic_sim import many_simulate_modified

def compute_counts_for_one_beta(args):
    theta, r, alpha, beta, agent_count, num_reps = args
    sim_res = many_simulate_modified(theta, r, alpha, beta, agent_count, num_reps)
    freqs = np.array([sum(sim[i] for sim in sim_res) / num_reps for i in range(agent_count)])
    # Perform a pass to average each element with its neighbors, to decrease
    # the jaggedness of the plot
    freqs = np.array([(freqs[0]*2+freqs[1])/3]
                     + [(freqs[i-1] + freqs[i] + freqs[i+1])/3 for i in range(1, len(freqs)-1)]
                     + [(freqs[-2]*2+freqs[-1])/3])
    return freqs

def plot_prob_affirm_vs_position(betas, theta=2.0, r=0.035, alpha=0.0, agent_count=500, num_reps=3500, random_seed=0xdeadcafe):
    executor = futures.ProcessPoolExecutor()
    sims = executor.map(compute_counts_for_one_beta, [(theta, r, alpha, beta, agent_count, num_reps) for beta in betas])
    xs = np.arange(1, agent_count+1, 1)
    for sim, beta in zip(sims, betas):
        plt.plot(xs, sim, label=f'\u03b2 = {beta}')
    plt.xlim(0, agent_count)
    plt.legend()
    plt.title('\u03b2 vs. likelihood of witness affirming')
    plt.xlabel('Witness number')
    plt.ylabel('Probability of affirming')
    plt.show()

def main():
    plot_prob_affirm_vs_position([0.5, 0.75, 0.9, 0.95, 1.0])

if __name__ == '__main__':
    main()
