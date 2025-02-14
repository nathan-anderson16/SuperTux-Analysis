import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import LOG_MANAGER, QoELog, Round, parse_timestamp


def qoe_distribution():
    """
    Distribution of QoE scores per-player and per-round.
    """
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    # Distribution of QoE scores per-user
    print("Generating QoE distribution per-player...")
    plt.figure()

    # Number of rows and columns the plot will have.
    ncols = 6
    nrows = math.ceil(len(all_qoe_logs) / float(ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 15))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    for i, uid in enumerate(sorted(all_qoe_logs, key=str.lower)):
        # x- and y-index of the current axis
        x = int(i / nrows)
        y = i % ncols

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
        # `pad` changes the size of the padding between the title and the plot
        curr_ax.set_title(uid, fontsize=11, pad=-0.9)

        # `labelpad` changes the size of the padding between the label and the plot
        curr_ax.set_xlabel("QoE Score", labelpad=-5.0)
        curr_ax.set_ylabel("Density", labelpad=-5.0)

        curr_ax.set_xticks([1, 2, 3, 4, 5], labels=["1", "", "", "", "5"])
        curr_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "", "", "", "1"])

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
        # The round logs for each frame spike duration
        ms0 = all_round_logs[i]
        ms75 = all_round_logs[i + 1]
        ms150 = all_round_logs[i + 2]
        ms225 = all_round_logs[i + 3]

        level_name, _ = Round.from_unique_id(i)

        plt.figure()

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 3))
        fig.subplots_adjust(wspace=0.5)

        # 0 ms stutter
        pd.Series([round.logs["qoe"].score for round in ms0]).plot(
            kind="density",
            ax=axs[0],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[0].set_title("0 ms", fontsize=10)

        # `labelpad` changes the size of the padding between the label and the plot
        axs[0].set_xlabel("QoE Score", labelpad=-7.0)
        axs[0].set_ylabel("Density", labelpad=-5.0)

        axs[0].set_xticks([1, 2, 3, 4, 5], labels=["1", "", "", "", "5"])
        axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "", "", "", "1"])

        # 75 ms stutter
        pd.Series([round.logs["qoe"].score for round in ms75]).plot(
            kind="density",
            ax=axs[1],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[1].set_title("75 ms", fontsize=10)

        # `labelpad` changes the size of the padding between the label and the plot
        axs[1].set_xlabel("QoE Score", labelpad=-7.0)
        axs[1].set_ylabel("Density", labelpad=-5.0)

        axs[1].set_xticks([1, 2, 3, 4, 5], labels=["1", "", "", "", "5"])
        axs[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "", "", "", "1"])

        # 150 ms stutter
        pd.Series([round.logs["qoe"].score for round in ms150]).plot(
            kind="density",
            ax=axs[2],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[2].set_title("150 ms", fontsize=10)

        # `labelpad` changes the size of the padding between the label and the plot
        axs[2].set_xlabel("QoE Score", labelpad=-7.0)
        axs[2].set_ylabel("Density", labelpad=-5.0)

        axs[2].set_xticks([1, 2, 3, 4, 5], labels=["1", "", "", "", "5"])
        axs[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "", "", "", "1"])

        # 225 ms stutter
        pd.Series([round.logs["qoe"].score for round in ms225]).plot(
            kind="density",
            ax=axs[3],
            xlim=(1, 5),
            ylim=(0, 1),
            xticks=[1, 5],
            yticks=[0, 1],
        )
        axs[3].set_title("225 ms", fontsize=11)

        # `labelpad` changes the size of the padding between the label and the plot
        axs[3].set_xlabel("QoE Score", labelpad=-7.0)
        axs[3].set_ylabel("Density", labelpad=-5.0)

        axs[3].set_xticks([1, 2, 3, 4, 5], labels=["1", "", "", "", "5"])
        axs[3].set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "", "", "", "1"])

        # Set the title of the entire figure
        fig.suptitle(level_name, fontsize=12)
        fig.savefig(f"figures/qoe_logs_per_round/{level_name}.png")


def compute_lag_differences():
    # def compute_lag_differences() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Computes the difference between the expected lag and the actual lag.
    """
    print("Computing expected vs actual spike times...")
    dfs = LOG_MANAGER.cleaned_event_logs()

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    all_diffs = []
    non_abs_diffs = []

    for uid in dfs:
        dfs1 = dfs[uid]
        diffs = []
        for df in dfs1:
            starts = df[df["Event"].str.contains(r"Lag Severity \(start\):")]
            ends = df[df["Event"].str.contains(r"Lag Severity \(end\):")]

            for (_, start), (_, end) in zip(starts.iterrows(), ends.iterrows()):
                expected_severity = start["ExpectedLag"]
                start_timestamp = parse_timestamp(str(start["Timestamp"]))
                end_timestamp = parse_timestamp(str(end["Timestamp"]))
                diff = (end_timestamp - start_timestamp) * 1000
                non_abs_diffs.append(diff - expected_severity)
                diffs.append(abs(expected_severity - diff))

        means[uid] = float(np.mean(diffs))
        stds[uid] = float(np.std(diffs))
        all_diffs.extend(diffs)

    plt.figure()
    plt.boxplot(non_abs_diffs)
    plt.title("Expected vs Actual Spike Times")
    plt.ylabel("Difference (ms)")
    plt.savefig(Path("figures") / "expected_vs_actual_lag_differences.png")
