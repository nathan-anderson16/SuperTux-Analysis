import math

import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd

from util import (
    LOG_MANAGER,
    ROUND_MAX_SUCCESSES,
    ROUND_NAMES,
    ROUND_TYPES,
    QoELog,
    Round,
    parse_timestamp,
    compute_r2
)

def demographics_info() :
    """
    Collects demographics info from CSV
    """
    file = open("./Results/Demographics Survey (Responses) - Form Responses 1.csv", 'r')

    first_line = file.readline()[:-1]
    first_line = first_line.split(',')
    results = {}
    for line in file :

        first_quote = line.find('"')
        line = line[:first_quote] + line[line.find('"', first_quote + 1):-1]
        parsed = line.split(',')
        results[parsed[1]] = {}
        player_file = open("./Results/" + parsed[1] + "/" + parsed[1] + ".txt", 'r')

        reaction_time_data = player_file.readline().split(',')
        results[parsed[1]]['Reaction Time'] = int(np.mean([int(s) for s in reaction_time_data[:-1]]))

        for i in range(2, 7) :
            results[parsed[1]][first_line[i]] = parsed[i]

    return results

def reaction_time_vs_score(demo_info) :
    all_event_logs = LOG_MANAGER.cleaned_event_logs()

    player_scores = {}
    for uid in all_event_logs:
        last = int(all_event_logs[uid][-1].iloc[-1]["Coins"])
        first = int(all_event_logs[uid][0].iloc[0]["Coins"])
        player_scores[uid] = int((last - first) / 100.0)

    total_max = 0
    for max_value in ROUND_MAX_SUCCESSES.values():
        total_max += max_value

    for uid, total in player_scores.items():
        player_scores[uid] = total / (total_max * 4)

    # (experience, score)
    graphable_data = []
    for (uid, scores) in player_scores.items() :
        graphable_data.append((int(demo_info[uid]["Reaction Time"]), np.mean(scores)))

    plt.figure()

    plt.scatter([x[0] for x in graphable_data], [x[1] for x in graphable_data])

    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.xlabel("Reaction Time")
    plt.ylabel("Score")

    plt.tight_layout()

    plt.savefig("figures/reaction_time_vs_score.png")

def reaction_time_vs_qoe(demo_info) :
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    player_qoe_scores = {}
    for uid in all_qoe_logs:
        logs = all_qoe_logs[uid]
        player_qoe_scores[uid] = []
        for log in logs:
            player_qoe_scores[uid].append(log.score)
    # (experience, score)
    graphable_data = []
    for (uid, scores) in player_qoe_scores.items() :
        graphable_data.append((int(demo_info[uid]["Reaction Time"]), np.mean(scores)))

    plt.figure()

    plt.scatter([x[0] for x in graphable_data], [x[1] for x in graphable_data])

    plt.yticks([1, 2, 3, 4, 5])

    plt.xlabel("Reaction Time")
    plt.ylabel("QoE Score")

    plt.tight_layout()

    plt.savefig("figures/reaction_time_vs_qoe.png")

def platformer_experience_vs_score(demo_info) :
    all_event_logs = LOG_MANAGER.cleaned_event_logs()

    player_scores = {}
    for uid in all_event_logs:
        last = int(all_event_logs[uid][-1].iloc[-1]["Coins"])
        first = int(all_event_logs[uid][0].iloc[0]["Coins"])
        player_scores[uid] = int((last - first) / 100.0)

    total_max = 0
    for max_value in ROUND_MAX_SUCCESSES.values():
        total_max += max_value

    for uid, total in player_scores.items():
        player_scores[uid] = total / (total_max * 4)

    # (experience, score)
    graphable_data = []
    for (uid, scores) in player_scores.items() :
        graphable_data.append((int(demo_info[uid]["How much experience do you have playing 2D platformers?"]), np.mean(scores)))

    plt.figure()

    plt.scatter([x[0] for x in graphable_data], [x[1] for x in graphable_data])

    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.xticks([1, 2, 3, 4, 5])

    plt.xlabel("2D Platformer Experience")
    plt.ylabel("Score")

    plt.tight_layout()

    plt.savefig("figures/platformer_experience_vs_score.png")

def platformer_experience_vs_qoe(demo_info) :
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    player_qoe_scores = {}
    for uid in all_qoe_logs:
        logs = all_qoe_logs[uid]
        player_qoe_scores[uid] = []
        for log in logs:
            player_qoe_scores[uid].append(log.score)
    # (experience, score)
    graphable_data = []
    for (uid, scores) in player_qoe_scores.items() :
        graphable_data.append((int(demo_info[uid]["How much experience do you have playing 2D platformers?"]), np.mean(scores)))

    plt.figure()

    plt.scatter([x[0] for x in graphable_data], [x[1] for x in graphable_data])

    plt.yticks([1, 2, 3, 4, 5])
    plt.xticks([1, 2, 3, 4, 5])

    plt.xlabel("2D Platformer Experience")
    plt.ylabel("QoE Score")

    plt.tight_layout()

    plt.savefig("figures/platformer_experience_vs_qoe.png")

def qoe_distribution():
    """
    Distribution of QoE scores per-user and per-round.
    """
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    # -----Overall QoE Score Distribution-----
    print("Generating overall QoE score distribution...")
    plt.figure()

    ax = plt.subplot()

    all_scores: list[float] = []
    for uid in all_qoe_logs:
        logs = all_qoe_logs[uid]
        for log in logs:
            all_scores.append(log.score)

    pd.Series(all_scores).plot(
        kind="density",
        ax=ax,
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )

    # `pad` changes the size of the padding between the title and the plot
    ax.set_title("Distribution of QoE Scores", fontsize=11)

    # `labelpad` changes the size of the padding between the label and the plot
    ax.set_xlabel("QoE Score")
    ax.set_ylabel("Density")

    ax.set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.savefig("figures/overall_qoe_distribution.png")

    print(
        "Saved overall QoE score distribution to figures/overall_qoe_distribution.png"
    )

    # -----Mean QoE Score Distribution-----
    print("Generating mean QoE score distribution...")
    plt.figure()

    ax = plt.subplot()

    all_scores: list[float] = []
    for uid in all_qoe_logs:
        scores: list[float] = []
        logs = all_qoe_logs[uid]
        for log in logs:
            scores.append(log.score)
        all_scores.append(float(np.array(scores).mean()))

    pd.Series(all_scores).plot(
        kind="density",
        ax=ax,
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )

    # `pad` changes the size of the padding between the title and the plot
    ax.set_title("Distribution of Mean QoE Scores", fontsize=11)

    # `labelpad` changes the size of the padding between the label and the plot
    ax.set_xlabel("QoE Score")
    ax.set_ylabel("Density")

    ax.set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.savefig("figures/mean_qoe_distribution.png")

    print("Saved mean QoE score distribution to figures/mean_qoe_distribution.png")

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

    print("Saved QoE distribution per-user to figures/qoe_distribution_per_user.png.")

    plt.close()

    # -----Distribution of QoE scores per-round-----
    print("Generating QoE distribution per-round...")

    all_round_logs = LOG_MANAGER.logs_per_round()

    fig = plt.figure(figsize=(16, 25))
    subfigs = fig.subfigures(nrows=8, ncols=1)

    plt.subplots_adjust(hspace=0.5)

    for i in range(1, 33, 4):
        ms0 = all_round_logs[i]
        ms75 = all_round_logs[i + 1]
        ms150 = all_round_logs[i + 2]
        ms225 = all_round_logs[i + 3]

        level_name, _ = Round.from_unique_id(i)

        subfig = subfigs[int(i / 4)]

        axs = subfig.subplots(nrows=1, ncols=4)

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

        subfig.suptitle(ROUND_NAMES[level_name], fontsize=12)

    plt.savefig("figures/qoe_scores_per_round.png")
    plt.close()

    print("Saved QoE distribution per-round graphs to figures/qoe_logs_per_round/")

    # -----Mean QoE Distribution Per-Round-----
    all_round_logs = LOG_MANAGER.logs_per_round()

    print("Generating QoE distribution per spike size...")

    fig = plt.figure(figsize=(12, 3.3))

    axs = fig.subplots(nrows=1, ncols=4)

    qoe_ms0 = []
    qoe_ms75 = []
    qoe_ms150 = []
    qoe_ms225 = []

    for i in range(1, 33, 4):
        ms0 = all_round_logs[i]
        ms75 = all_round_logs[i + 1]
        ms150 = all_round_logs[i + 2]
        ms225 = all_round_logs[i + 3]

        m0 = [log.logs["qoe"].score for log in ms0]
        m75 = [log.logs["qoe"].score for log in ms75]
        m150 = [log.logs["qoe"].score for log in ms150]
        m225 = [log.logs["qoe"].score for log in ms225]

        qoe_ms0.extend(m0)
        qoe_ms75.extend(m75)
        qoe_ms150.extend(m150)
        qoe_ms225.extend(m225)

    pd.Series(qoe_ms0).plot(
        kind="density",
        ax=axs[0],
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )
    axs[0].set_title("0 ms", fontsize=11)
    axs[0].set_xlabel("QoE Score")
    axs[0].set_ylabel("Density")
    axs[0].set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    axs[0].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"]
    )

    pd.Series(qoe_ms75).plot(
        kind="density",
        ax=axs[1],
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )
    axs[1].set_title("75 ms", fontsize=11)
    axs[1].set_xlabel("QoE Score")
    axs[1].set_ylabel("Density")
    axs[1].set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    axs[1].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"]
    )

    pd.Series(qoe_ms150).plot(
        kind="density",
        ax=axs[2],
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )
    axs[2].set_title("150 ms", fontsize=11)
    axs[2].set_xlabel("QoE Score")
    axs[2].set_ylabel("Density")
    axs[2].set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    axs[2].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"]
    )

    pd.Series(qoe_ms225).plot(
        kind="density",
        ax=axs[3],
        xlim=(1, 5),
        ylim=(0, 1),
        xticks=[1, 5],
        yticks=[0, 1],
    )
    axs[3].set_title("225 ms", fontsize=11)
    axs[3].set_xlabel("QoE Score")
    axs[3].set_ylabel("Density")
    axs[3].set_xticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    axs[3].set_yticks(
        [0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0.25", "0.5", "0.75", "1"]
    )

    fig.suptitle("Distribution of QoE Scores Per-Spike Size")

    fig.tight_layout()
    fig.savefig("figures/qoe_distribution_per_spike_size.png")
    plt.close()

    print(
        "Saved QoE distribution per spike size to figures/qoe_distribution_per_spike_size.png"
    )

    # -----QoE Distribution vs Spike Size-----
    print("Generating QoE distribution vs spike size...")

    all_round_logs = LOG_MANAGER.logs_per_round()

    print("Generating mean QoE per spike size...")

    fig = plt.figure(figsize=(4, 3))

    qoe_ms0 = []
    qoe_ms75 = []
    qoe_ms150 = []
    qoe_ms225 = []

    for i in range(1, 33, 4):
        ms0 = all_round_logs[i]
        ms75 = all_round_logs[i + 1]
        ms150 = all_round_logs[i + 2]
        ms225 = all_round_logs[i + 3]

        m0 = [log.logs["qoe"].score for log in ms0]
        m75 = [log.logs["qoe"].score for log in ms75]
        m150 = [log.logs["qoe"].score for log in ms150]
        m225 = [log.logs["qoe"].score for log in ms225]

        qoe_ms0.extend(m0)
        qoe_ms75.extend(m75)
        qoe_ms150.extend(m150)
        qoe_ms225.extend(m225)

    z = 1.96

    # Standard deviations
    sigma_0 = float(np.std(qoe_ms0))
    sigma_75 = float(np.std(qoe_ms75))
    sigma_150 = float(np.std(qoe_ms150))
    sigma_225 = float(np.std(qoe_ms225))

    n_0 = len(qoe_ms0)
    n_75 = len(qoe_ms75)
    n_150 = len(qoe_ms150)
    n_225 = len(qoe_ms225)

    # Confidence intervals
    ci_0 = z * sigma_0 / math.sqrt(n_0)
    ci_75 = z * sigma_75 / math.sqrt(n_75)
    ci_150 = z * sigma_150 / math.sqrt(n_150)
    ci_225 = z * sigma_225 / math.sqrt(n_225)

    x = [0, 75, 150, 225]
    y = [
        float(np.mean(qoe_ms0)),
        float(np.mean(qoe_ms75)),
        float(np.mean(qoe_ms150)),
        float(np.mean(qoe_ms225)),
    ]

    plt.scatter(x, y, s=10)
    plt.plot(x, y)
    # , color="#1f77b4", color="#ff7f0e"
    plt.errorbar(
        x, y, yerr=[ci_0, ci_75, ci_150, ci_225], linewidth=1, capsize=4, fmt="none"
    )
    # plt.errorbar(x, y, yerr=[sigma_0, sigma_75, sigma_150, sigma_225], fmt='o', color="#1f77b4")

    plt.xlabel("Spike Duration")
    plt.ylabel("QoE Score")
    plt.title("Mean QoE Score vs Spike Duration")

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])

    plt.ylim((1, 5))

    plt.tight_layout()

    plt.savefig("figures/mean_qoe_vs_spike_size.png")
    plt.close()

    print("Saved mean QoE vs spike size to figures/mean_qoe_vs_spike_size.png")

    # -----QoE distribution vs spike size per-task-----
    all_round_logs = LOG_MANAGER.logs_per_round()

    print("Generating mean QoE per spike size per-task...")

    fig = plt.figure(figsize=(5, 3.75))

    qoes_per_task: dict[str, list[list[float]]] = {
        "one_two_two_level": [[], [], [], []],
        "two_three_two_level": [[], [], [], []],
        "three_three_three_level": [[], [], [], []],
        "three_three_five_level": [[], [], [], []],
        "three_four_five_level": [[], [], [], []],
        "two_five_five_level": [[], [], [], []],
        "four_four_five_level": [[], [], [], []],
        "five_five_five_level": [[], [], [], []],
    }

    qoes: dict[str, list[list[float]]] = {
        "Collect Power-Up": [[], [], [], []],
        "Squish Enemy": [[], [], [], []],
        "Jump Over Gap": [[], [], [], []],
        "Special Jump": [[], [], [], []],
    }

    for round_id in all_round_logs:
        level_name, spike_duration = Round.from_unique_id(round_id)

        for current_round in all_round_logs[round_id]:
            qoe: float = current_round.logs["qoe"].score
            match spike_duration:
                case 0:
                    qoes[ROUND_TYPES[level_name]][0].append(qoe)
                    qoes_per_task[level_name][0].append(qoe)
                case 75:
                    qoes[ROUND_TYPES[level_name]][1].append(qoe)
                    qoes_per_task[level_name][1].append(qoe)
                case 150:
                    qoes[ROUND_TYPES[level_name]][2].append(qoe)
                    qoes_per_task[level_name][2].append(qoe)
                case 225:
                    qoes[ROUND_TYPES[level_name]][3].append(qoe)
                    qoes_per_task[level_name][3].append(qoe)

    handles = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = [".", "^", "s", "D"]

    for a, type in enumerate(qoes.keys()):
        qoe_ms0 = qoes[type][0]
        qoe_ms75 = qoes[type][1]
        qoe_ms150 = qoes[type][2]
        qoe_ms225 = qoes[type][3]

        for i in range(1, 33, 4):
            ms0 = all_round_logs[i]
            ms75 = all_round_logs[i + 1]
            ms150 = all_round_logs[i + 2]
            ms225 = all_round_logs[i + 3]

            m0 = [log.logs["qoe"].score for log in ms0]
            m75 = [log.logs["qoe"].score for log in ms75]
            m150 = [log.logs["qoe"].score for log in ms150]
            m225 = [log.logs["qoe"].score for log in ms225]

            qoe_ms0.extend(m0)
            qoe_ms75.extend(m75)
            qoe_ms150.extend(m150)
            qoe_ms225.extend(m225)

        z = 1.96

        # Standard deviations
        sigma_0 = float(np.std(qoe_ms0))
        sigma_75 = float(np.std(qoe_ms75))
        sigma_150 = float(np.std(qoe_ms150))
        sigma_225 = float(np.std(qoe_ms225))

        n_0 = len(qoe_ms0)
        n_75 = len(qoe_ms75)
        n_150 = len(qoe_ms150)
        n_225 = len(qoe_ms225)

        # Confidence intervals
        ci_0 = z * sigma_0 / math.sqrt(n_0)
        ci_75 = z * sigma_75 / math.sqrt(n_75)
        ci_150 = z * sigma_150 / math.sqrt(n_150)
        ci_225 = z * sigma_225 / math.sqrt(n_225)

        x = [0, 75, 150, 225]
        y = [
            float(np.mean(qoe_ms0)),
            float(np.mean(qoe_ms75)),
            float(np.mean(qoe_ms150)),
            float(np.mean(qoe_ms225)),
        ]

        handle = plt.scatter(x, y, s=12, marker=markers[a])
        plt.plot(x, y)
        # , color="#1f77b4", color="#ff7f0e"
        plt.errorbar(
            x,
            y,
            yerr=[ci_0, ci_75, ci_150, ci_225],
            linewidth=1,
            capsize=4,
            fmt="none",
            color=colors[a],
        )
        # plt.errorbar(x, y, yerr=[sigma_0, sigma_75, sigma_150, sigma_225], fmt='o', color="#1f77b4")
        handles.append(handle)

    plt.legend(handles, qoes.keys())

    plt.xlabel("Spike Duration")
    plt.ylabel("QoE Score")
    plt.title("Mean QoE Score vs Spike Duration Per-Task")

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])

    plt.ylim((1, 5))

    plt.tight_layout()

    fig.savefig("figures/mean_qoe_vs_spike_size_per_task.png")
    plt.close("all")

    print("Test time :D")

    fig = plt.figure(figsize=(12,5.8))

    axs = fig.subplots(nrows=2, ncols=4)

    slope_equations = dict()
    original_data = dict()
    r2 = dict()
    row = 0
    col = 0
    for a, type in enumerate(qoes_per_task.keys()):
        ax = axs[row][col]

        ms0 = qoes_per_task[type][0]
        ms75 = qoes_per_task[type][1]
        ms150 = qoes_per_task[type][2]
        ms225 = qoes_per_task[type][3]

        print(type, np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225))

        z = 1.96

        # Standard deviations
        sigma_0 = float(np.std(ms0))
        sigma_75 = float(np.std(ms75))
        sigma_150 = float(np.std(ms150))
        sigma_225 = float(np.std(ms225))

        n_0 = len(ms0)
        n_75 = len(ms75)
        n_150 = len(ms150)
        n_225 = len(ms225)

        # Confidence intervals
        ci_0 = z * sigma_0 / math.sqrt(n_0)
        ci_75 = z * sigma_75 / math.sqrt(n_75)
        ci_150 = z * sigma_150 / math.sqrt(n_150)
        ci_225 = z * sigma_225 / math.sqrt(n_225)

        x = [0, 75, 150, 225]
        y = [np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225)]

        slope_equations[type] = np.poly1d(np.polyfit(x, y, 1))
        original_data[type] = y

        r2[type] = round(compute_r2(y, [slope_equations[type](val) for val in x]), 2)

        markers = {
            "Collect Power-Up": ".",
            "Squish Enemy": "^",
            "Jump Over Gap": "s",
            "Special Jump": "D"
        }

        handle = ax.scatter(x, y, marker=markers["Collect Power-Up"])

        ax.plot(x, [slope_equations[type](val) for val in x], label = ROUND_NAMES[type] + " [" + str(r2[type]) + "]")

        handles.append(handle)

        col += 1
        if col >= 4 :
            col = 0
            row += 1


        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #         fancybox=True, shadow=True, ncol=2)

        ax.set_xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
        ax.set_yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])

        ax.set_ylim(0, 5)

        ax.set_title(ROUND_NAMES[type])
        ax.set_xlabel("Spike time (ms)")
        ax.set_ylabel("Average QoE")

    plt.tight_layout()
    fig.savefig("figures/qoe_vs_spike_time_each_task.png")

    plt.close("all")

    print("Test time :D")

    fig = plt.figure()

    slope_equations = dict()
    original_data = dict()
    r2 = dict()
    row = 0
    col = 0
    for a, type in enumerate(qoes_per_task.keys()):
        ms0 = qoes_per_task[type][0]
        ms75 = qoes_per_task[type][1]
        ms150 = qoes_per_task[type][2]
        ms225 = qoes_per_task[type][3]

        print(type, np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225))

        z = 1.96

        # Standard deviations
        sigma_0 = float(np.std(ms0))
        sigma_75 = float(np.std(ms75))
        sigma_150 = float(np.std(ms150))
        sigma_225 = float(np.std(ms225))

        n_0 = len(ms0)
        n_75 = len(ms75)
        n_150 = len(ms150)
        n_225 = len(ms225)

        # Confidence intervals
        ci_0 = z * sigma_0 / math.sqrt(n_0)
        ci_75 = z * sigma_75 / math.sqrt(n_75)
        ci_150 = z * sigma_150 / math.sqrt(n_150)
        ci_225 = z * sigma_225 / math.sqrt(n_225)

        x = [0, 75, 150, 225]
        y = [np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225)]

        slope_equations[type] = np.poly1d(np.polyfit(x, y, 1))
        original_data[type] = y

        r2[type] = round(compute_r2(y, [slope_equations[type](val) for val in x]), 2)

        markers = {
            "Collect Power-Up": ".",
            "Squish Enemy": "^",
            "Jump Over Gap": "s",
            "Special Jump": "D"
        }

        handle = plt.scatter(x, y, marker=markers[ROUND_TYPES[type]])

        plt.plot(x, [slope_equations[type](val) for val in x], label = ROUND_NAMES[type] + " [" + str(r2[type]) + "]")

        handles.append(handle)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=2)

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])

    plt.ylim(0, 5)

    plt.title("QoE vs Spike Time Per-Task")
    plt.xlabel("Spike time (ms)")
    plt.ylabel("Average QoE")

    plt.tight_layout()
    fig.savefig("figures/qoe_vs_spike_time_all_task.png")

    plt.close("all")




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
    plt.xticks(ticks=[])
    plt.xlabel("")
    plt.ylabel("Difference (ms)")
    plt.savefig("figures/expected_vs_actual_lag_differences.png")

    print(
        "Saved expected vs actual spike times to figures/expected_vs_actual_lag_differences.png"
    )

    plt.close()


def success_rate():
    """
    Successes rate per-user and per-round.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # ----------Success distribution per-user----------

    print("Generating success rate per-user...")

    success_rate = dict()

    for uid in event_logs:
        last = int(event_logs[uid][-1].iloc[-1]["Coins"])
        first = int(event_logs[uid][0].iloc[0]["Coins"])
        success_rate[uid] = int((last - first) / 100.0)

    total_max = 0
    for max_value in ROUND_MAX_SUCCESSES.values():
        total_max += max_value

    for uid, total in success_rate.items():
        success_rate[uid] = total / (total_max * 4)

    plt.figure(figsize=(14, 12))

    count_uid = [(success_rate[k], k) for k in success_rate.keys()]
    sorted_uids = sorted(count_uid, key=lambda x: x[0], reverse=True)
    success_counts = [success_rate[k[1]] for k in sorted_uids]

    plt.scatter([i for i in range(len(success_rate))], success_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(success_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # It looks like it’s z * s / sqrt(n)
    plt.errorbar(
        [i for i in range(len(success_rate))],
        success_counts,
        yerr=[
            1.96 * np.std(success_counts) / len(success_counts) ** 0.5
            for i in range(len(success_counts))
        ],
    )

    # Set yticks to every 0.5 units
    plt.yticks([i * 0.1 for i in range(0, 10 + 1)])

    plt.xticks(
        ticks=[i for i in range(len(success_rate))], labels=[k[1] for k in sorted_uids]
    )

    plt.xlabel("User ID")
    plt.ylabel("Success Rate")

    plt.savefig("figures/success_rate_per_user.png")

    print("Saved success rate per-user to figures/success_rate_per_user.png")

    plt.close()

    # ----------Success distribution per-round----------

    print("Generating success rate per-task...")

    rounds = LOG_MANAGER.logs_per_round()

    round_success_rate = dict()

    for round_name in rounds:
        logs = rounds[round_name]

        round_success_rate[rounds[round_name][0].level_name] = []

        for round in logs:
            df: pd.DataFrame = round.logs["event"]

            start_num = int(df["Coins"].iloc[0])
            end_num = int(df["Coins"].iloc[-1])
            n_successes = int((end_num - start_num) / 100)

            round_success_rate[rounds[round_name][0].level_name].append(
                n_successes / ROUND_MAX_SUCCESSES[rounds[round_name][0].level_name]
            )

    for round_name, successes in round_success_rate.items():
        round_success_rate[round_name] = np.mean(successes)

    plt.figure()

    count_uid = [(round_success_rate[k], k) for k in round_success_rate.keys()]
    sorted_uids = sorted(count_uid, key=lambda x: x[0], reverse=True)
    success_counts = [round_success_rate[k[1]] for k in sorted_uids]

    plt.scatter([i for i in range(len(round_success_rate))], success_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # It looks like it’s z * s / sqrt(n)
    plt.errorbar(
        [i for i in range(len(round_success_rate))],
        success_counts,
        capsize=4,
        yerr=[
            1.96 * np.std(success_counts) / len(success_counts) ** 0.5
            for _ in range(len(success_counts))
        ],
    )

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    # for i, count in enumerate(success_counts):
    #     plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 25 units
    plt.yticks([i * 0.1 for i in range(0, 10 + 1)])

    plt.xticks(
        ticks=[i for i in range(len(round_success_rate))],
        labels=[ROUND_NAMES[k[1]] for k in sorted_uids],
        fontsize=9,
    )

    plt.setp(plt.xticks()[1], rotation=30, horizontalalignment="right")

    plt.xlabel("Task")
    plt.ylabel("Success Rate")

    plt.tight_layout()

    plt.savefig("figures/success_rate_per_task.png")

    print("Saved success rate per-task to figures/success_rate_per_round.png")

    plt.close()


def qoe_score_vs_acceptability():
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    # -----Overall QoE Score Distribution-----
    print("Generating overall QoE score distribution...")

    all_scores = {}
    for uid in all_qoe_logs:
        logs = all_qoe_logs[uid]
        for log in logs:
            if log.score in all_scores.keys():
                all_scores[log.score].append(log.acceptable)
            else:
                all_scores[log.score] = [log.acceptable]

    for score, acceptability in all_scores.items():
        all_scores[score] = acceptability.count(True) / len(acceptability)

    plt.figure()

    count_uid = [(all_scores[k], k) for k in all_scores.keys()]
    sorted_uids = sorted(count_uid, key=lambda x: x[0], reverse=True)
    keys = [k[1] for k in sorted_uids]
    success_counts = [k[0] for k in sorted_uids]

    plt.scatter(keys, success_counts)

    poly_fit = np.poly1d(np.polyfit(keys, success_counts, 3))

    plt.plot(sorted(keys), [poly_fit(key) for key in sorted(keys)])

    plt.yticks([i * 0.1 for i in range(0, 10 + 1)])
    plt.xticks([1, 2, 3, 4, 5])

    plt.xlabel("QoE Score")
    plt.ylabel("Acceptibility Rate")

    plt.tight_layout()

    plt.savefig("figures/qoe_score_vs_acceptability.png")

    print("Saved success rate per-task to figures/qoe_score_vs_acceptability.png")

    plt.close()


def success_rate_vs_spike_time():
    """
    Success rate vs spike time
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # -----Success rate vs spike size-----

    print("Generating success rate vs spike time...")

    ms0_actual = []
    ms75_actual = []
    ms150_actual = []
    ms225_actual = []

    max_possible_successes = sum([ROUND_MAX_SUCCESSES[i] for i in ROUND_MAX_SUCCESSES])

    for uid in event_logs:
        user_ms0 = 0
        user_ms75 = 0
        user_ms150 = 0
        user_ms225 = 0

        for current_round in event_logs[uid]:
            last = int(current_round.iloc[-1]["Coins"])
            first = int(current_round.iloc[0]["Coins"])
            n_successes = int((last - first) / 100.0)
            match int(current_round["ExpectedLag"].iloc[1]):
                case 0:
                    user_ms0 += n_successes
                case 75:
                    user_ms75 += n_successes
                case 150:
                    user_ms150 += n_successes
                case 225:
                    user_ms225 += n_successes

        ms0_actual.append(user_ms0 / max_possible_successes)
        ms75_actual.append(user_ms75 / max_possible_successes)
        ms150_actual.append(user_ms150 / max_possible_successes)
        ms225_actual.append(user_ms225 / max_possible_successes)

    z = 1.96

    # Standard deviations
    sigma_0_actual = float(np.std(ms0_actual))
    sigma_75_actual = float(np.std(ms75_actual))
    sigma_150_actual = float(np.std(ms150_actual))
    sigma_225_actual = float(np.std(ms225_actual))

    n_0_actual = len(ms0_actual)
    n_75_actual = len(ms75_actual)
    n_150_actual = len(ms150_actual)
    n_225_actual = len(ms225_actual)

    # Confidence intervals
    ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
    ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
    ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
    ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

    fig = plt.figure()

    x = [0, 75, 150, 225]
    y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

    plt.scatter(x, y_actual)
    plt.plot(x, y_actual)
    plt.errorbar(
        x, y_actual, yerr=[ci_0_actual, ci_75_actual, ci_150_actual, ci_225_actual], linewidth=1, capsize=4, fmt="none"
    )

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.ylim(0, 1)

    plt.title("Success rate vs spike time")
    plt.xlabel("Spike time (ms)")
    plt.ylabel("Success rate")

    fig.savefig("figures/success_rate_vs_spike_time.png")

    plt.close("all")

    print("Saved success rate vs spike time to figures/success_rate_vs_spike_time.png")

    # -----Success rate vs spike time per-task type-----

    print("Generating success rate vs spike time per-type...")

    ms0_actual = []
    ms75_actual = []
    ms150_actual = []
    ms225_actual = []

    # success_rates_by_task = {
    #     "Collect Power-Up": {
    #         "one_two_two_level": [[], [], [], []],
    #         "two_three_two_level": [[], [], [], []],
    #     },
    #     "Squish Enemy": {
    #         "three_three_three_level": [[], [], [], []],
    #     },
    #     "Jump Over Gap": {
    #         "three_three_five_level": [[], [], [], []],
    #         "three_four_five_level": [[], [], [], []],
    #         "two_five_five_level": [[], [], [], []],
    #     },
    #     "Special Jump": {
    #         "four_four_five_level": [[], [], [], []],
    #         "five_five_five_level": [[], [], [], []],
    #     },
    # }

    success_rates_by_task_actual = {
        "one_two_two_level": [[], [], [], []],
        "two_three_two_level": [[], [], [], []],
        "three_three_three_level": [[], [], [], []],
        "three_three_five_level": [[], [], [], []],
        "three_four_five_level": [[], [], [], []],
        "two_five_five_level": [[], [], [], []],
        "four_four_five_level": [[], [], [], []],
        "five_five_five_level": [[], [], [], []],
    }

    success_rates_actual: dict[str, list[list[float]]] = {
        "Collect Power-Up": [[], [], [], []],
        "Squish Enemy": [[], [], [], []],
        "Jump Over Gap": [[], [], [], []],
        "Special Jump": [[], [], [], []],
    }

    for uid in event_logs:
        for current_round in event_logs[uid]:
            last = int(current_round.iloc[-1]["Coins"])
            first = int(current_round.iloc[0]["Coins"])
            n_successes = int((last - first) / 100.0)

            level_type = str(current_round.iloc[0]["Level"])
            max_successes = ROUND_MAX_SUCCESSES[level_type]

            success_rate_actual = n_successes / max_successes

            match int(current_round["ExpectedLag"].iloc[1]):
                case 0:
                    success_rates_actual[ROUND_TYPES[level_type]][0].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][0].append(success_rate_actual)
                case 75:
                    success_rates_actual[ROUND_TYPES[level_type]][1].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][1].append(success_rate_actual)
                case 150:
                    success_rates_actual[ROUND_TYPES[level_type]][2].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][2].append(success_rate_actual)
                case 225:
                    success_rates_actual[ROUND_TYPES[level_type]][3].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][3].append(success_rate_actual)

    handles = []
    markers = [".", "^", "s", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure()

    slope_equations_actual = dict()
    original_data = dict()
    r2_actual = dict()
    for a, type in enumerate(success_rates_actual.keys()):
        ms0_actual = success_rates_actual[type][0]
        ms75_actual = success_rates_actual[type][1]
        ms150_actual = success_rates_actual[type][2]
        ms225_actual = success_rates_actual[type][3]

        print(type, np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual))

        z = 1.96

        # Standard deviations
        sigma_0_actual = float(np.std(ms0_actual))
        sigma_75_actual = float(np.std(ms75_actual))
        sigma_150_actual = float(np.std(ms150_actual))
        sigma_225_actual = float(np.std(ms225_actual))

        n_0_actual = len(ms0_actual)
        n_75_actual = len(ms75_actual)
        n_150_actual = len(ms150_actual)
        n_225_actual = len(ms225_actual)

        # Confidence intervals
        ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
        ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
        ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
        ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

        x = [0, 75, 150, 225]
        y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

        slope_equations_actual[type] = np.poly1d(np.polyfit(x, y_actual, 1))
        original_data[type] = y_actual

        r2_actual[type] = round(compute_r2(y_actual, [slope_equations_actual[type](val) for val in x]), 3)

        handle_actual = plt.scatter(x, y_actual, marker=markers[a])

        plt.plot(x, [slope_equations_actual[type](val) for val in x])

        plt.errorbar(
            x,
            y_actual,
            yerr=[ci_0_actual, ci_75_actual, ci_150_actual, ci_225_actual],
            linewidth=1,
            capsize=4,
            fmt="none",
            color=colors[a],
        )

        handles.append(handle_actual)

    plt.legend(handles, [key + " " + str(r2_actual[key]) for key in success_rates_actual.keys()])

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.ylim(0, 1)

    plt.title("Practical Success Rate vs Spike Time Per-Task")
    plt.xlabel("Spike time (ms)")
    plt.ylabel("Success rate")

    fig.savefig("figures/practical_success_rate_vs_spike_time_per_task.png")

    plt.close("all")

    print(
        "Saved success rate vs spike time to figures/success_rate_vs_spike_time_per_task.png"
    )

    success_rates_by_task_actual = {
        "one_two_two_level": [[], [], [], []],
        "two_three_two_level": [[], [], [], []],
        "three_three_three_level": [[], [], [], []],
        "three_three_five_level": [[], [], [], []],
        "three_four_five_level": [[], [], [], []],
        "two_five_five_level": [[], [], [], []],
        "four_four_five_level": [[], [], [], []],
        "five_five_five_level": [[], [], [], []],
    }

    success_rates_actual: dict[str, list[list[float]]] = {
        "Collect Power-Up": [[], [], [], []],
        "Squish Enemy": [[], [], [], []],
        "Jump Over Gap": [[], [], [], []],
        "Special Jump": [[], [], [], []],
    }

    for uid in event_logs:
        for current_round in event_logs[uid]:
            last = int(current_round.iloc[-1]["Coins"])
            first = int(current_round.iloc[0]["Coins"])

            n_failures = 0
            for row in current_round["Event"] :
                if "Death" in row or "Failure" in row :
                    n_failures += 1

            n_successes = int((last - first) / 100.0)

            level_type = str(current_round.iloc[0]["Level"])
            max_successes = n_successes + n_failures

            success_rate_actual = n_successes / max_successes

            match int(current_round["ExpectedLag"].iloc[1]):
                case 0:
                    success_rates_actual[ROUND_TYPES[level_type]][0].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][0].append(success_rate_actual)
                case 75:
                    success_rates_actual[ROUND_TYPES[level_type]][1].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][1].append(success_rate_actual)
                case 150:
                    success_rates_actual[ROUND_TYPES[level_type]][2].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][2].append(success_rate_actual)
                case 225:
                    success_rates_actual[ROUND_TYPES[level_type]][3].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][3].append(success_rate_actual)

    handles = []
    markers = [".", "^", "s", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure()

    slope_equations_actual = dict()
    original_data = dict()
    r2_actual = dict()
    for a, type in enumerate(success_rates_actual.keys()):
        ms0_actual = success_rates_actual[type][0]
        ms75_actual = success_rates_actual[type][1]
        ms150_actual = success_rates_actual[type][2]
        ms225_actual = success_rates_actual[type][3]

        print(type, np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual))

        z = 1.96

        # Standard deviations
        sigma_0_actual = float(np.std(ms0_actual))
        sigma_75_actual = float(np.std(ms75_actual))
        sigma_150_actual = float(np.std(ms150_actual))
        sigma_225_actual = float(np.std(ms225_actual))

        n_0_actual = len(ms0_actual)
        n_75_actual = len(ms75_actual)
        n_150_actual = len(ms150_actual)
        n_225_actual = len(ms225_actual)

        # Confidence intervals
        ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
        ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
        ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
        ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

        x = [0, 75, 150, 225]
        y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

        slope_equations_actual[type] = np.poly1d(np.polyfit(x, y_actual, 1))
        original_data[type] = y_actual

        r2_actual[type] = round(compute_r2(y_actual, [slope_equations_actual[type](val) for val in x]), 3)

        handle_actual = plt.scatter(x, y_actual, marker=markers[a])

        plt.plot(x, [slope_equations_actual[type](val) for val in x])

        plt.errorbar(
            x,
            y_actual,
            yerr=[ci_0_actual, ci_75_actual, ci_150_actual, ci_225_actual],
            linewidth=1,
            capsize=4,
            fmt="none",
            color=colors[a],
        )

        handles.append(handle_actual)

    plt.legend(handles, [key + " " + str(r2_actual[key]) for key in success_rates_actual.keys()])

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.ylim(0, 1)

    plt.title("Actual Success Rate vs Spike Time Per-Task")
    plt.xlabel("Spike time (ms)")
    plt.ylabel("Success rate")

    fig.savefig("figures/actual_success_rate_vs_spike_time_per_task.png")

    plt.close("all")

    print("Test 2")

    success_rates_by_task_actual = {
        "one_two_two_level": [[], [], [], []],
        "two_three_two_level": [[], [], [], []],
        "three_three_three_level": [[], [], [], []],
        "three_three_five_level": [[], [], [], []],
        "three_four_five_level": [[], [], [], []],
        "two_five_five_level": [[], [], [], []],
        "four_four_five_level": [[], [], [], []],
        "five_five_five_level": [[], [], [], []],
    }

    success_rates_actual: dict[str, list[list[float]]] = {
        "Collect Power-Up": [[], [], [], []],
        "Squish Enemy": [[], [], [], []],
        "Jump Over Gap": [[], [], [], []],
        "Special Jump": [[], [], [], []],
    }

    success_rates_by_task_practical = {
        "one_two_two_level": [[], [], [], []],
        "two_three_two_level": [[], [], [], []],
        "three_three_three_level": [[], [], [], []],
        "three_three_five_level": [[], [], [], []],
        "three_four_five_level": [[], [], [], []],
        "two_five_five_level": [[], [], [], []],
        "four_four_five_level": [[], [], [], []],
        "five_five_five_level": [[], [], [], []],
    }

    success_rates_practical: dict[str, list[list[float]]] = {
        "Collect Power-Up": [[], [], [], []],
        "Squish Enemy": [[], [], [], []],
        "Jump Over Gap": [[], [], [], []],
        "Special Jump": [[], [], [], []],
    }

    for uid in event_logs:
        for current_round in event_logs[uid]:
            last = int(current_round.iloc[-1]["Coins"])
            first = int(current_round.iloc[0]["Coins"])

            n_failures = 0
            for row in current_round["Event"] :
                if "Death" in row or "Failure" in row :
                    n_failures += 1

            n_successes = int((last - first) / 100.0)

            level_type = str(current_round.iloc[0]["Level"])
            max_successes_practical = ROUND_MAX_SUCCESSES[level_type]

            level_type = str(current_round.iloc[0]["Level"])
            max_successes_actual = n_successes + n_failures

            success_rate_practical = n_successes / max_successes_practical
            success_rate_actual = n_successes / max_successes_actual

            match int(current_round["ExpectedLag"].iloc[1]):
                case 0:
                    success_rates_actual[ROUND_TYPES[level_type]][0].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][0].append(success_rate_actual)
                    success_rates_practical[ROUND_TYPES[level_type]][0].append(success_rate_practical)
                    success_rates_by_task_practical[level_type][0].append(success_rate_practical)
                case 75:
                    success_rates_actual[ROUND_TYPES[level_type]][1].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][1].append(success_rate_actual)

                    success_rates_practical[ROUND_TYPES[level_type]][1].append(success_rate_practical)
                    success_rates_by_task_practical[level_type][1].append(success_rate_practical)
                case 150:
                    success_rates_actual[ROUND_TYPES[level_type]][2].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][2].append(success_rate_actual)

                    success_rates_practical[ROUND_TYPES[level_type]][2].append(success_rate_practical)
                    success_rates_by_task_practical[level_type][2].append(success_rate_practical)
                case 225:
                    success_rates_actual[ROUND_TYPES[level_type]][3].append(success_rate_actual)
                    success_rates_by_task_actual[level_type][3].append(success_rate_actual)

                    success_rates_practical[ROUND_TYPES[level_type]][3].append(success_rate_practical)
                    success_rates_by_task_practical[level_type][3].append(success_rate_practical)

    handles = []
    markers = [".", "^", "s", "D"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure()

    slope_equations_actual = dict()
    slope_equations_practical = dict()
    original_data = dict()
    r2_actual = dict()
    r2_practical = dict()
    for a, type in enumerate(success_rates_actual.keys()):
        fig = plt.figure()
        ms0_actual = success_rates_actual[type][0]
        ms75_actual = success_rates_actual[type][1]
        ms150_actual = success_rates_actual[type][2]
        ms225_actual = success_rates_actual[type][3]

        ms0_practical = success_rates_practical[type][0]
        ms75_practical = success_rates_practical[type][1]
        ms150_practical = success_rates_practical[type][2]
        ms225_practical = success_rates_practical[type][3]

        # print(type, np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual))

        z = 1.96

        # Standard deviations
        sigma_0_actual = float(np.std(ms0_actual))
        sigma_75_actual = float(np.std(ms75_actual))
        sigma_150_actual = float(np.std(ms150_actual))
        sigma_225_actual = float(np.std(ms225_actual))

        sigma_0_practical = float(np.std(ms0_practical))
        sigma_75_practical = float(np.std(ms75_practical))
        sigma_150_practical = float(np.std(ms150_practical))
        sigma_225_practical = float(np.std(ms225_practical))

        n_0_actual = len(ms0_actual)
        n_75_actual = len(ms75_actual)
        n_150_actual = len(ms150_actual)
        n_225_actual = len(ms225_actual)

        n_0_practical = len(ms0_practical)
        n_75_practical = len(ms75_practical)
        n_150_practical = len(ms150_practical)
        n_225_practical = len(ms225_practical)

        # Confidence intervals
        ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
        ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
        ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
        ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

        ci_0_practical = z * sigma_0_practical / math.sqrt(n_0_practical)
        ci_75_practical = z * sigma_75_practical / math.sqrt(n_75_practical)
        ci_150_practical = z * sigma_150_practical / math.sqrt(n_150_practical)
        ci_225_practical = z * sigma_225_practical / math.sqrt(n_225_practical)

        x = [0, 75, 150, 225]
        y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

        y_practical = [np.mean(ms0_practical), np.mean(ms75_practical), np.mean(ms150_practical), np.mean(ms225_practical)]

        slope_equations_actual[type] = np.poly1d(np.polyfit(x, y_actual, 1))

        r2_actual[type] = round(compute_r2(y_actual, [slope_equations_actual[type](val) for val in x]), 3)

        slope_equations_practical[type] = np.poly1d(np.polyfit(x, y_practical, 1))

        r2_practical[type] = round(compute_r2(y_practical, [slope_equations_practical[type](val) for val in x]), 3)

        handle_actual = plt.scatter(x, y_actual, marker=markers[0])

        handle_practical = plt.scatter(x, y_practical, marker=markers[1])

        plt.plot(x, [slope_equations_actual[type](val) for val in x])
        plt.plot(x, [slope_equations_practical[type](val) for val in x])

        handles.append(handle_actual)
        handles.append(handle_practical)

        plt.legend(handles, ["Actual", "Practical"])

        plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
        plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

        plt.ylim(0, 1)

        plt.title(f"Actual And Practical Success Rate vs Spike Time {type}")
        plt.xlabel("Spike time (ms)")
        plt.ylabel("Success rate")

        fig.savefig("figures/actual_and_practical_success_rate_vs_spike_time_per_task_" + type + ".png")

    for a, type in enumerate(success_rates_by_task_actual.keys()):
        fig = plt.figure()
        ms0_actual = success_rates_by_task_actual[type][0]
        ms75_actual = success_rates_by_task_actual[type][1]
        ms150_actual = success_rates_by_task_actual[type][2]
        ms225_actual = success_rates_by_task_actual[type][3]

        ms0_practical = success_rates_by_task_practical[type][0]
        ms75_practical = success_rates_by_task_practical[type][1]
        ms150_practical = success_rates_by_task_practical[type][2]
        ms225_practical = success_rates_by_task_practical[type][3]

        z = 1.96

        # Standard deviations
        sigma_0_actual = float(np.std(ms0_actual))
        sigma_75_actual = float(np.std(ms75_actual))
        sigma_150_actual = float(np.std(ms150_actual))
        sigma_225_actual = float(np.std(ms225_actual))

        sigma_0_practical = float(np.std(ms0_practical))
        sigma_75_practical = float(np.std(ms75_practical))
        sigma_150_practical = float(np.std(ms150_practical))
        sigma_225_practical = float(np.std(ms225_practical))

        n_0_actual = len(ms0_actual)
        n_75_actual = len(ms75_actual)
        n_150_actual = len(ms150_actual)
        n_225_actual = len(ms225_actual)

        n_0_practical = len(ms0_practical)
        n_75_practical = len(ms75_practical)
        n_150_practical = len(ms150_practical)
        n_225_practical = len(ms225_practical)

        # Confidence intervals
        ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
        ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
        ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
        ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

        ci_0_practical = z * sigma_0_practical / math.sqrt(n_0_practical)
        ci_75_practical = z * sigma_75_practical / math.sqrt(n_75_practical)
        ci_150_practical = z * sigma_150_practical / math.sqrt(n_150_practical)
        ci_225_practical = z * sigma_225_practical / math.sqrt(n_225_practical)

        x = [0, 75, 150, 225]
        y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

        y_practical = [np.mean(ms0_practical), np.mean(ms75_practical), np.mean(ms150_practical), np.mean(ms225_practical)]

        slope_equations_actual[type] = np.poly1d(np.polyfit(x, y_actual, 1))

        r2_actual[type] = round(compute_r2(y_actual, [slope_equations_actual[type](val) for val in x]), 3)

        slope_equations_practical[type] = np.poly1d(np.polyfit(x, y_practical, 1))

        r2_practical[type] = round(compute_r2(y_practical, [slope_equations_practical[type](val) for val in x]), 3)

        handle_actual = plt.scatter(x, y_actual, marker=markers[0])

        handle_practical = plt.scatter(x, y_practical, marker=markers[1])

        plt.plot(x, [slope_equations_actual[type](val) for val in x])
        plt.plot(x, [slope_equations_practical[type](val) for val in x])

        handles.append(handle_actual)
        handles.append(handle_practical)

        plt.legend(handles, ["Actual", "Practical"])

        plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
        plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

        plt.ylim(0, 1)

        plt.title(f"Actual And Practical Success Rate vs Spike Time {ROUND_NAMES[type]}")
        plt.xlabel("Spike time (ms)")
        plt.ylabel("Success rate")

        fig.savefig("figures/actual_and_practical_success_rate_vs_spike_time_per_task_" + type + ".png")

    plt.close("all")

    print("Test 3")

    # fig = plt.figure()

    # base = slope_equations_actual["Collect Power-Up"](0)

    # handles = []
    # x = [0, 75, 150, 225]
    # for (a, type) in enumerate(slope_equations_actual.keys()) :

    #     offset = base - slope_equations_actual[type](0)

    #     y_actual = [(slope_equations_actual[type](val) + offset) for val in x]

    #     handle_actual = plt.scatter(x, y_actual, marker=markers[a])

    #     plt.plot(x, y_actual)

    #     handles.append(handle_actual)

    # plt.legend(handles, [key + " " + str(r2_actual[key]) for key in success_rates_actual.keys()])

    # plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    # plt.yticks([])

    # plt.ylim(0, 1)

    # plt.title("Comparison of Trendlines")
    # plt.xlabel("Spike time (ms)")
    # # plt.ylabel("Success rate")

    # fig.savefig("figures/slopes_success_rate_vs_spike_time_per_task.png")

    # plt.close("all")

    print(
        "Testing time :D"
    )

    # for category in success_rates_by_task.keys() :
    #     fig = plt.figure()
    #     slope_equations = dict()
    #     original_data = dict()
    #     r2 = dict()
    #     for a, type in enumerate(success_rates_by_task[category].keys()):
    #         ms0 = success_rates_by_task[category][type][0]
    #         ms75 = success_rates_by_task[category][type][1]
    #         ms150 = success_rates_by_task[category][type][2]
    #         ms225 = success_rates_by_task[category][type][3]

    #         print(type, np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225))

    #         z = 1.96

    #         # Standard deviations
    #         sigma_0 = float(np.std(ms0))
    #         sigma_75 = float(np.std(ms75))
    #         sigma_150 = float(np.std(ms150))
    #         sigma_225 = float(np.std(ms225))

    #         n_0 = len(ms0)
    #         n_75 = len(ms75)
    #         n_150 = len(ms150)
    #         n_225 = len(ms225)

    #         # Confidence intervals
    #         ci_0 = z * sigma_0 / math.sqrt(n_0)
    #         ci_75 = z * sigma_75 / math.sqrt(n_75)
    #         ci_150 = z * sigma_150 / math.sqrt(n_150)
    #         ci_225 = z * sigma_225 / math.sqrt(n_225)

    #         x = [0, 75, 150, 225]
    #         y = [np.mean(ms0), np.mean(ms75), np.mean(ms150), np.mean(ms225)]

    #         slope_equations[type] = np.poly1d(np.polyfit(x, y, 1))
    #         original_data[type] = y

    #         r2[type] = round(compute_r2(y, [slope_equations[type](val) for val in x]), 3)

    #         handle = plt.scatter(x, y, marker=markers[a % 4])

    #         plt.plot(x, [slope_equations[type](val) for val in x])

    #         plt.errorbar(
    #             x,
    #             y,
    #             yerr=[ci_0, ci_75, ci_150, ci_225],
    #             linewidth=1,
    #             capsize=4,
    #             fmt="none",
    #             color=colors[a % 4],
    #         )

    #         handles.append(handle)

    #     plt.legend(handles, [key + " " + str(r2[key]) for key in success_rates_by_task[category].keys()])

    #     plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    #     plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

    #     plt.ylim(0, 1)

    #     plt.title("Success Rate vs Spike Time Per-Task")
    #     plt.xlabel("Spike time (ms)")
    #     plt.ylabel("Success rate")

    #     fig.savefig("figures/success_rate_vs_spike_time_" + category + ".png")

    #     plt.close("all")

    fig = plt.figure()

    slope_equations_actual = dict()
    original_data = dict()
    r2_actual = dict()
    for a, type in enumerate(success_rates_by_task_actual.keys()):
        ms0_actual = success_rates_by_task_actual[type][0]
        ms75_actual = success_rates_by_task_actual[type][1]
        ms150_actual = success_rates_by_task_actual[type][2]
        ms225_actual = success_rates_by_task_actual[type][3]

        print(type, np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual))

        z = 1.96

        # Standard deviations
        sigma_0_actual = float(np.std(ms0_actual))
        sigma_75_actual = float(np.std(ms75_actual))
        sigma_150_actual = float(np.std(ms150_actual))
        sigma_225_actual = float(np.std(ms225_actual))

        n_0_actual = len(ms0_actual)
        n_75_actual = len(ms75_actual)
        n_150_actual = len(ms150_actual)
        n_225_actual = len(ms225_actual)

        # Confidence intervals
        ci_0_actual = z * sigma_0_actual / math.sqrt(n_0_actual)
        ci_75_actual = z * sigma_75_actual / math.sqrt(n_75_actual)
        ci_150_actual = z * sigma_150_actual / math.sqrt(n_150_actual)
        ci_225_actual = z * sigma_225_actual / math.sqrt(n_225_actual)

        x = [0, 75, 150, 225]
        y_actual = [np.mean(ms0_actual), np.mean(ms75_actual), np.mean(ms150_actual), np.mean(ms225_actual)]

        slope_equations_actual[type] = np.poly1d(np.polyfit(x, y_actual, 1))
        original_data[type] = y_actual

        r2_actual[type] = round(compute_r2(y_actual, [slope_equations_actual[type](val) for val in x]), 2)

        markers = {
            "Collect Power-Up": ".",
            "Squish Enemy": "^",
            "Jump Over Gap": "s",
            "Special Jump": "D"
        }

        handle_actual = plt.scatter(x, y_actual, marker=markers[ROUND_TYPES[type]])

        plt.plot(x, [slope_equations_actual[type](val) for val in x], label = ROUND_NAMES[type] + " [" + str(r2_actual[type]) + "]")

        handles.append(handle_actual)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=2)

    plt.xticks([0, 75, 150, 225], labels=["0", "75", "150", "225"])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])

    plt.ylim(0, 1)

    plt.tight_layout()
    plt.title("Success Rate vs Spike Time Per-Task")
    plt.xlabel("Spike time (ms)")
    plt.ylabel("Success rate")

    fig.savefig("figures/success_rate_vs_spike_time_all_task.png")

    plt.close("all")


def success_distribution():
    """
    Distribution of number of successes per-user and per-round.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # ----------Success distribution per-user----------

    print("Generating success distribution per-user...")

    successes: dict[str, int] = dict()

    for uid in event_logs:
        last = int(event_logs[uid][-1].iloc[-1]["Coins"])
        first = int(event_logs[uid][0].iloc[0]["Coins"])
        successes[uid] = int((last - first) / 100.0)

    plt.figure(figsize=(7, 6))

    count_uid = [(successes[k], k) for k in successes.keys()]
    sorted_uids = sorted(count_uid, key=lambda x: x[0], reverse=True)
    success_counts = [successes[k[1]] for k in sorted_uids]

    plt.scatter([i for i in range(len(successes))], success_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(success_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 25 units
    plt.yticks([i * 25 for i in range(0, math.ceil(max(success_counts) / 25.0) + 1)])

    plt.xticks(
        ticks=[i for i in range(len(successes))],
        labels=[k[1] for k in sorted_uids],
        fontsize=8,
    )

    plt.xlabel("User ID")
    plt.ylabel("Number of Successes")

    plt.setp(plt.xticks()[1], rotation=40, horizontalalignment="right")

    plt.savefig("figures/success_distribution_per_user.png")

    print(
        "Saved success distribution per-user to figures/success_distribution_per_user.png"
    )

    plt.close()

    # ----------Success distribution per-round----------

    # print("Generating success distribution per-round...")

    # rounds = LOG_MANAGER.logs_per_round()

    # round_successes: dict[int, int] = dict()

    # for round_id in rounds:
    #     logs = rounds[round_id]

    #     round_successes[round_id] = 0

    #     for round in logs:
    #         df: pd.DataFrame = round.logs["event"]

    #         start_num = int(df["Coins"].iloc[0])
    #         end_num = int(df["Coins"].iloc[-1])
    #         n_successes = int((end_num - start_num) / 100)

    #         round_successes[round_id] += n_successes

    # plt.scatter([i for i in range(len(success_counts))], success_counts)

    # _, ymax = plt.ylim()
    # plt.ylim(0, ymax)

    # # Draw lines from each data point to the graph.
    # # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    # for i, count in enumerate(success_counts):
    #     plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # # Set yticks to every 25 units
    # plt.yticks([i * 25 for i in range(0, math.ceil(max(success_counts) / 25.0) + 1)])

    # plt.xticks(
    #     ticks=[i for i in range(len(successes))], labels=[k[1] for k in sorted_uids]
    # )

    # plt.xlabel("User ID")
    # plt.ylabel("Number of Successes")

    # plt.savefig("figures/success_distribution_per_user.png")

    # print(
    #     "Saved success distribution per-user to figures/success_distribution_per_user.png"
    # )

    # plt.close()


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

    uids = sorted(failures.keys(), key=str.lower)
    failure_counts = [sum([len(failure) for failure in failures[k]]) for k in uids]

    plt.scatter([i for i in range(len(failures))], failure_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(failure_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 25 units
    plt.yticks([i * 25 for i in range(0, math.ceil(max(failure_counts) / 25.0) + 1)])

    plt.xticks(ticks=[i for i in range(len(failures))], labels=uids)

    plt.xlabel("User ID")
    plt.ylabel("Number of Failures")

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


def player_score_distribution():
    """
    Distribution of player score per-user and per-round.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()

    # ----------Success distribution per-user----------

    print("Generating player score distribution per-user...")

    scores: dict[str, list[pd.DataFrame]] = dict()

    for uid in event_logs:
        scores[uid] = [  # type: ignore
            df[df["Event"].str.contains("Success")] for df in event_logs[uid]
        ]

    plt.figure(figsize=(18, 12))

    uids = sorted(scores.keys(), key=str.lower)
    score_counts = [sum([len(score) * 100 for score in scores[k]]) for k in uids]

    plt.scatter([i for i in range(len(scores))], score_counts)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    # Draw lines from each data point to the graph.
    # `zorder` (Z-order) makes the lines draw below the points created with scatter().
    for i, count in enumerate(score_counts):
        plt.vlines(i, 0, count, linewidth=0.5, colors=["black"], zorder=0)

    # Set yticks to every 2500 units
    plt.yticks([i * 2500 for i in range(0, math.ceil(max(score_counts) / 2500.0) + 1)])

    uids = sorted(scores.keys(), key=str.lower)
    plt.xticks(ticks=[i for i in range(len(scores))], labels=uids)

    plt.savefig("figures/score_distribution_per_user.png")

    print(
        "Saved score distribution per-user to figures/score_distribution_per_user.png"
    )

    plt.close()

    # ----------Score distribution per-round----------

    print("Generating score distribution per-round...")

    rounds = LOG_MANAGER.logs_per_round()

    round_scores: dict[int, int] = dict()

    for round_id in rounds:
        logs = rounds[round_id]

        round_scores[round_id] = 0

        for round in logs:
            df: pd.DataFrame = round.logs["event"]
            n_scores = len(df[df["Event"].str.contains("Success")]) * 100

            # three_three_three_level doesn't log successes for some reason, so we calculate the number of successes from the score.
            if df["Level"].iloc[1] == "three_three_three_level":
                start_num = int(df["Coins"].iloc[0])
                end_num = int(df["Coins"].iloc[-1])
                n_scores = end_num - start_num

            round_scores[round_id] += n_scores

    with open("figures/score_distribution_per_round.txt", "w") as f:
        for round_id in range(1, 33, 4):
            level_name = rounds[round_id][0].level_name
            ms0 = round_scores[round_id]
            ms75 = round_scores[round_id + 1]
            ms150 = round_scores[round_id + 2]
            ms225 = round_scores[round_id + 3]

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
        "Saved score distribution per-round to figures/score_distribution_per_round.txt"
    )
