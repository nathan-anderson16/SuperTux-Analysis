import math
import os

import matplotlib
from matplotlib.figure import figaspect
import matplotlib.pyplot as plt
from numpy import ceil
import pandas as pd

from util import LOG_MANAGER, QoELog, Round


figure = 0


def next_figure() -> int:
    global figure
    figure += 1
    return figure


def main():
    if not os.path.exists("figures/"):
        os.makedirs("figures")
    if not os.path.exists("figures/qoe_logs_per_round"):
        os.makedirs("figures/qoe_logs_per_round")

    matplotlib.use("qt5agg")

    # print("Loading rounds...")
    # round_data = LOG_MANAGER.rounds()
    # print("Finished loading rounds.\n")

    all_qoe_logs = LOG_MANAGER.qoe_logs()

    # Distribution of QoE scores per-user
    print("Generating QoE distribution per-player...")
    plt.rcParams.update({"font.size": 10})
    plt.figure(next_figure())
    size = int(ceil(math.sqrt(float(len(all_qoe_logs)))))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(6, 5))
    fig.canvas.get_width_height
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    for i, uid in enumerate(all_qoe_logs):
        x = int(i / size)
        y = i % size

        qoe_logs: list[QoELog] = all_qoe_logs[uid]
        qoe_scores = [log.score for log in qoe_logs]
        curr_ax = axs[x][y]
        pd.Series(qoe_scores).plot(
            kind="density",
            ax=curr_ax,
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        curr_ax.set_title(uid, fontsize=10)
        curr_ax.set_ylabel("")

    plt.axis("off")
    fig.savefig("figures/qoe_distribution_per_player.png")
    print("Finished generating QoE distribution per-player.")
    print(
        "Saved QoE distribution per-player to figures/qoe_distribution_per_player.png.\n"
    )

    # Distribution of QoE scores per-round
    print("Generating QoE distribution per-round...")

    all_round_logs = LOG_MANAGER.logs_per_round()

    for i in range(1, 33, 4):
        ms0 = all_round_logs[i]
        ms75 = all_round_logs[i + 1]
        ms150 = all_round_logs[i + 2]
        ms225 = all_round_logs[i + 3]

        level_name, _ = Round.from_unique_id(i)

        plt.figure(next_figure())

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))
        fig.subplots_adjust(wspace=0.5)

        pd.Series([round.logs["qoe"].score for round in ms0]).plot(
            kind="density",
            ax=axs[0],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[0].set_title("0 ms", fontsize=10)
        axs[0].set_ylabel("")

        pd.Series([round.logs["qoe"].score for round in ms75]).plot(
            kind="density",
            ax=axs[1],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[1].set_title("75 ms", fontsize=10)
        axs[1].set_ylabel("")

        pd.Series([round.logs["qoe"].score for round in ms150]).plot(
            kind="density",
            ax=axs[2],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[2].set_title("150 ms", fontsize=10)
        axs[2].set_ylabel("")

        pd.Series([round.logs["qoe"].score for round in ms225]).plot(
            kind="density",
            ax=axs[3],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[3].set_title("225 ms", fontsize=10)
        axs[3].set_ylabel("")

        fig.suptitle(level_name)
        fig.savefig(f"figures/qoe_logs_per_round/{level_name}.png")


if __name__ == "__main__":
    main()
