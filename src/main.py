import math

import matplotlib.pyplot as plt
from numpy import ceil
import pandas as pd

from util import LOG_MANAGER, QoELog


figure = 0
def next_figure() -> int:
    global figure
    figure += 1
    return figure


def main():
    # print("Loading rounds...")
    # round_data = LOG_MANAGER.rounds()
    # print("Finished loading rounds.\n")

    all_qoe_logs = LOG_MANAGER.qoe_logs()
    
    # Distribution of QoE scores per-user
    print("Generating QoE distribution per-player...")
    plt.rcParams.update({"font.size": 10})
    plt.figure(next_figure())
    size = int(ceil(math.sqrt(float(len(all_qoe_logs)))))
    fig, axs = plt.subplots(nrows=size, ncols=size)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for i, uid in enumerate(all_qoe_logs):
        x = int(i / size)
        y = i % size
        
        qoe_logs: list[QoELog] = all_qoe_logs[uid]
        qoe_scores = [log.score for log in qoe_logs]
        curr_ax = axs[x][y]
        pd.Series(qoe_scores).plot(kind="density", ax=curr_ax, xlim=(1, 5), ylim=(0, 1), xticks=[1, 5], yticks=[0, 1])
        curr_ax.set_title(uid, fontsize=10)
        curr_ax.set_ylabel("")
        
    plt.axis("off")
    fig.savefig("figures/qoe_distribution_per_player.png")
    print("Finished generating QoE distribution per-player.")
    print("Saved QoE distribution per-player to figures/qoe_distribution_per_player.png.\n")

    # Distribution of QoE scores per-round
    print("Generating QoE distribution per-round...")
    plt.figure(next_figure())
    fig, axs = plt.subplots(nrows=4, ncols=8)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(32):
        ...


if __name__ == "__main__":
    main()

