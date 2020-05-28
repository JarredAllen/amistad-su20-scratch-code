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
    plt.legend()
    if show:
        plt.show()

def main():
    plot_g([1, 5, 10, 30, 100, -100])

if __name__ == '__main__':
    main()
