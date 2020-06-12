import numpy as np
import matplotlib.pyplot as plt

from basic_sim import g

def plot_g(thetas, show=True):
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
    if show:
        plt.show()

def g_vs_theta():
    plot_g([1, 5, 10, 30, 100, -100])

def r_vs_majority_time():
    xs = np.linspace(0, 0.5, 2000)[1:]
    ys = np.ceil(np.log(0.5)/np.log(1-xs))
    fig, ax = plt.subplots()
    # ax.set_yscale("log")
    # ax.set_ylim(bottom=1)
    ax.set_ylim(top=400)
    # ax.set_yticks([1, 10, 100, 1000, 10000])
    # ax.set_yticklabels(['1', '10', '100', '1000', '10000'])
    ax.set_ylabel("Number of witnesses to overturn a majority")
    # ax.set_xscale("log")
    # ax.set_xticks([0.001, 0.01, 0.1, 0.5])
    # ax.set_xticklabels(['0.001', '0.01', '0.1', '0.5'])
    ax.set_xlabel("Value of 1-r")
    # ax.grid(which='minor', linewidth=0.3)
    # ax.grid(which='major', linewidth=0.8)
    ax.plot(1-xs, ys)
    plt.show()

def weights_vs_r(rvals, xvals):
    for rval in rvals:
        plt.plot(xvals, rval*np.power(1-rval, xvals-1), label=f'r={rval}')
    plt.title('Weighting decay for different values of r')
    plt.xlabel('$\\delta$')
    plt.ylabel('$w(r, \\delta)$')
    plt.legend()
    plt.show()

def main():
    plot_g([1, 5, 10, 30, 100])
    r_vs_majority_time()
    weights_vs_r([0.4, 0.3, 0.2, 0.1, 0.001], np.arange(1, 50, 1))

if __name__ == '__main__':
    main()
