import os

import matplotlib
import matplotlib.pyplot as plt

import misc
import ivdv


def main():
    if not os.path.exists("figures/"):
        os.makedirs("figures")
    if not os.path.exists("figures/qoe_logs_per_round"):
        os.makedirs("figures/qoe_logs_per_round")

    # qt5 isn't a requirement on Windows
    if os.name != "nt":
        matplotlib.use("qt5agg")

    # Change the size of the default font in the plot to 10.
    # This doesn't apply to the title font size.
    plt.rcParams.update({"font.size": 10})

    misc.qoe_distribution()
    misc.success_rate_vs_spike_time()
    misc.compute_lag_differences()
    misc.success_distribution()
    misc.success_rate()
    misc.failure_distribution()
    misc.player_score_distribution()
    misc.qoe_score_vs_acceptability()

    ivdv.graph_success_distribution()
    ivdv.graph_failure_distribution()
    ivdv.graph_success_rate()
    ivdv.graph_acceptability()

    demo_info = misc.demographics_info()
    misc.platformer_experience_vs_qoe(demo_info)
    misc.platformer_experience_vs_score(demo_info)
    misc.reaction_time_vs_qoe(demo_info)
    misc.reaction_time_vs_score(demo_info)


if __name__ == "__main__":
    main()
