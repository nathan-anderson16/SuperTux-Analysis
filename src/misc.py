import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import LOG_MANAGER, QoELog, Round, parse_timestamp


def qoe_distribution():
    """
    Distribution of QoE scores per-user and per-round.
    """
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    # Distribution of QoE scores per-user
    print("Generating QoE distribution per-user...")
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

    fig.savefig("figures/qoe_distribution_per_user.png")

    print("Saved QoE distribution per-user to figures/qoe_distribution_per_user.png.\n")

    plt.close()

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

        plt.close()

    print("Saved QoE distribution per-round graphs to figures/qoe_logs_per_round/")


def compute_lag_differences():
    """
    Plots the difference between the expected lag and the actual lag.
    """
    print("Computing expected vs actual spike times...")
    dfs = LOG_MANAGER.cleaned_event_logs()

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    all_diffs = []

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

                diffs.append(abs(expected_severity - diff))

        means[uid] = float(np.mean(diffs))
        stds[uid] = float(np.std(diffs))
        all_diffs.extend(diffs)

    plt.figure()
    plt.boxplot(all_diffs)
    plt.title("Expected vs Actual Spike Times")
    plt.ylabel("Difference (ms)")
    plt.savefig("figures/expected_vs_actual_lag_differences.png")

    print(
        "Saved expected vs actual spike times to figures/expected_vs_actual_lag_differences.png"
    )

    plt.close()


def success_distribution():
    """
    Distribution of number of successes per-user and per-round.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # ----------Success distribution per-user----------

    print("Generating success distribution per-user...")

    successes: dict[str, list[pd.DataFrame]] = dict()

    for uid in event_logs:
        successes[uid] = [  # type: ignore
            df[df["Event"].str.contains("Success")] for df in event_logs[uid]
        ]

    plt.figure(figsize=(18, 12))

    success_counts = [
        sum([len(success) for success in successes[k]]) for k in successes
    ]

    plt.scatter([i for i in range(len(successes))], success_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(success_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 25 units
    plt.yticks([i * 25 for i in range(0, math.ceil(max(success_counts) / 25.0) + 1)])

    uids = sorted(successes.keys(), key=str.lower)
    plt.xticks(ticks=[i for i in range(len(successes))], labels=uids)

    plt.savefig("figures/success_distribution_per_user.png")

    print(
        "Saved success distribution per-user to figures/success_distribution_per_user.png"
    )

    plt.close()

    # ----------Success distribution per-round----------

    print("Generating success distribution per-round...")

    rounds = LOG_MANAGER.logs_per_round()

    round_successes: dict[int, int] = dict()

    for round_id in rounds:
        logs = rounds[round_id]

        round_successes[round_id] = 0

        for round in logs:
            df: pd.DataFrame = round.logs["event"]
            n_successes = len(df[df["Event"].str.contains("Success")])

            # three_three_three_level doesn't log successes for some reason, so we calculate the number of successes from the score.
            if df["Level"].iloc[1] == "three_three_three_level":
                start_num = int(df["Coins"].iloc[0])
                end_num = int(df["Coins"].iloc[-1])
                n_successes = int((end_num - start_num) / 100)

            round_successes[round_id] += n_successes

    with open("figures/success_distribution_per_round.txt", "w") as f:
        for round_id in range(1, 33, 4):
            level_name = rounds[round_id][0].level_name
            ms0 = round_successes[round_id]
            ms75 = round_successes[round_id + 1]
            ms150 = round_successes[round_id + 2]
            ms225 = round_successes[round_id + 3]

            f.writelines(
                [
                    f"{level_name}:\n",
                    f"      0 ms: {ms0}\n",
                    f"     75 ms: {ms75}\n",
                    f"    150 ms: {ms150}\n",
                    f"    225 ms: {ms225}\n\n",
                ]
            )

    print(
        "Saved success distribution per-round to figures/success_distribution_per_round.txt"
    )


def failure_distribution():
    """
    Distribution of number of failures per-user and per-round.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # ----------Failure distribution per-user----------

    print("Generating failure distribution per-user...")

    failures: dict[str, list[pd.DataFrame]] = dict()

    for uid in event_logs:
        failures[uid] = [  # type: ignore
            df[df["Event"].str.contains("Failure|Death")] for df in event_logs[uid]
        ]

    plt.figure(figsize=(18, 12))

    failure_counts = [sum([len(failure) for failure in failures[k]]) for k in failures]

    plt.scatter([i for i in range(len(failures))], failure_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(failure_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 25 units
    plt.yticks([i * 25 for i in range(0, math.ceil(max(failure_counts) / 25.0) + 1)])

    uids = sorted(failures.keys(), key=str.lower)
    plt.xticks(ticks=[i for i in range(len(failures))], labels=uids)

    plt.savefig("figures/failure_distribution_per_user.png")

    print(
        "Saved failure distribution per-user to figures/failure_distribution_per_user.png"
    )

    plt.close()

    # ----------Failure distribution per-round----------

    print("Generating failure distribution per-round...")

    rounds = LOG_MANAGER.logs_per_round()

    round_failures: dict[int, int] = dict()

    for round_id in rounds:
        logs = rounds[round_id]

        round_failures[round_id] = 0

        for round in logs:
            df: pd.DataFrame = round.logs["event"]
            n_failures = len(df[df["Event"].str.contains("Failure|Death")])

            round_failures[round_id] += n_failures

    with open("figures/failure_distribution_per_round.txt", "w") as f:
        for round_id in range(1, 33, 4):
            level_name = rounds[round_id][0].level_name
            ms0 = round_failures[round_id]
            ms75 = round_failures[round_id + 1]
            ms150 = round_failures[round_id + 2]
            ms225 = round_failures[round_id + 3]

            f.writelines(
                [
                    f"{level_name}:\n",
                    f"      0 ms: {ms0}\n",
                    f"     75 ms: {ms75}\n",
                    f"    150 ms: {ms150}\n",
                    f"    225 ms: {ms225}\n\n",
                ]
            )

    print(
        "Saved failure distribution per-round to figures/failure_distribution_per_round.txt"
    )
