import math

import numpy as np

def visualize_one(sim_results):
    """Visualize the results of one simulation"""
    num_yes = sim_results.count(True)
    total = len(sim_results)
    print(f'Number who affirmed: {num_yes}/{total} ({num_yes/total*100}%)')
    print('Affirmers over time:')
    print(''.join({True: 'Y', False: '.'}[a] for a in sim_results))

# Number of rows in frequency chart for each position
FREQ_ROWS = 4

# Size of a histogram step and width on screen for the histogram
HIST_STEP = 0.05
HIST_WIDTH = 80

def visualize_many(many_results):
    """Visualize the results of many simulations"""
    many_results = np.array(many_results, dtype=np.int)
    num_yes = np.count_nonzero(many_results, axis=1)
    totals = many_results.shape[1]
    frac_yes = num_yes / totals
    mean_yes = np.mean(frac_yes)
    std_yes = np.std(frac_yes)
    print('"Yes" response statistics:')
    print(f'Mean:  {mean_yes}')
    print(f'StDev: {std_yes}')
    freqs = np.array([sum(1/many_results.shape[0]
                          for j in range(many_results.shape[0]) if many_results[j][i] == 1
                         )
                      for i in range(many_results.shape[1])
                    ])
    # freqs /= many_results.shape[0]
    print('Frequency of "Yes" response for each position:')
    def bar_char(f):
        """Gets the appropriate character for a bar chart of f.
        
        Rounds f to the nearest 1/8, and then finds the closest box in
        unicode to print. Will print a space if less than 1/16, or a
        full box if greater than 15/16 (including handling vals <0/>1).
        """
        if 16*f < 1:
            return ' '
        elif 16*f < 3:
            return '\u2581'
        elif 16*f < 5:
            return '\u2582'
        elif 16*f < 7:
            return '\u2583'
        elif 16*f < 9:
            return '\u2584'
        elif 16*f < 11:
            return '\u2585'
        elif 16*f < 13:
            return '\u2586'
        elif 16*f < 15:
            return '\u2587'
        else:
            return '\u2588'
    for row in range(FREQ_ROWS):
        print(''.join(bar_char(f*FREQ_ROWS - (FREQ_ROWS - row - 1)) for f in freqs))
    print('\nHistogram:')
    bins = [(bin*HIST_STEP,
            (bin+1)*HIST_STEP,
            sum(1 for item in frac_yes
                if item >= bin*HIST_STEP
                   and (item < (bin+1)*HIST_STEP
                        or (bin+1)*HIST_STEP == 1 and item == 1)),
            )
            for bin in range(int(math.ceil(1/HIST_STEP)))]
    max_bin_count = max(count for (lower, upper, count) in bins)
    for lower, upper, count in bins:
        print("{0:<05.3}-{1:<05.3}:".format(lower, upper), "*" * (HIST_WIDTH * count // max_bin_count))
