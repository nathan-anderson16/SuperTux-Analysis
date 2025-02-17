import os

import matplotlib
import matplotlib.pyplot as plt

import misc


def main():
    if not os.path.exists("figures/"):
        os.makedirs("figures")
    if not os.path.exists("figures/qoe_logs_per_round"):
        os.makedirs("figures/qoe_logs_per_round")

    matplotlib.use("qt5agg")

    # Change the size of the default font in the plot to 10.
    # This doesn't apply to the title font size.
    plt.rcParams.update({"font.size": 10})

    misc.qoe_distribution()
    misc.compute_lag_differences()
    misc.success_distribution()
    misc.failure_distribution()
    misc.player_score_distribution()


if __name__ == "__main__":
    main()
