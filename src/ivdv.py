import math

import numpy as np
#import seaborn as sb
import pandas as pd
import os

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

import matplotlib.pyplot as plt

success_data = {
        "five_five_five_level": {0: 35, 75: 25, 150: 26, 225: 15},
        "four_four_five_level": {0: 88, 75: 82, 150: 67, 225: 64},
        "one_two_two_level": {0: 308, 75: 314, 150: 318, 225: 319},
        "three_four_five_level": {0: 203, 75: 169, 150: 97, 225: 99},
        "three_three_five_level": {0: 285, 75: 264, 150: 250, 225: 217},
        "three_three_three_level": {0: 132, 75: 135, 150: 120, 225: 132},
        "two_five_five_level": {0: 170, 75: 195, 150: 71, 225: 33},
        "two_three_two_level": {0: 144, 75: 150, 150: 132, 225: 104},
}

failure_data = {
        "five_five_five_level": {0: 333, 75: 341, 150: 335, 225: 374},
        "four_four_five_level": {0: 263, 75: 270, 150: 317, 225: 314},
        "one_two_two_level": {0: 11, 75: 11, 150: 15, 225: 9},
        "three_four_five_level": {0: 104, 75: 123, 150: 164, 225: 173},
        "three_three_five_level": {0: 51, 75: 67, 150: 74, 225: 92},
        "three_three_three_level": {0: 54, 75: 51, 150: 53, 225: 48},
        "two_five_five_level": {0: 124, 75: 106, 150: 188, 225: 211},
        "two_three_two_level": {0: 56, 75: 49, 150: 66, 225: 87},
}

total_trials = {}
for level in success_data:
    total_trials[level] = {delay: success_data[level][delay] + failure_data[level][delay] for delay in success_data[level]}

def graph_success_distribution():

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))  # 4 rows, 2 columns
    axes = axes.flatten()

    for idx, (level, delays) in enumerate(success_data.items()):
        ax = axes[idx]
        x = list(delays.keys())
        y = list(delays.values())

        ax.plot(x, y, marker="o", linestyle="-", color="b", label=level)
        ax.set_xlabel("Task Delay (ms)")
        ax.set_ylabel("Number of Successes")
        ax.set_title(level)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks([0, 75, 150, 225])

    plt.tight_layout()
    plt.savefig("figures/graph_success_distribution.png")
    print("Graph Success Distribution Created")


def graph_failure_distribution():

    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (level, delays) in enumerate(failure_data.items()):
        ax = axes[idx]  # Select subplot
        x = list(delays.keys())
        y = list(delays.values())

        ax.plot(x, y, marker="o", linestyle="-", color="b", label=level)
        ax.set_xlabel("Task Delay (ms)")
        ax.set_ylabel("Number of Failures")
        ax.set_title(level)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks([0, 75, 150, 225])

    plt.tight_layout()
    plt.savefig("figures/graph_failure_distribution.png")
    print("Graph Failure Distribution Created")

def graph_success_rate():
    success_rates = {}
    for level in success_data:
        success_rates[level] = {
            delay: (success_data[level][delay] / total_trials[level][delay]) * 100
            for delay in success_data[level]
        }

    num_levels = len(success_rates)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))  # Adjust grid size

    axes = axes.flatten()

    for i, (level, rates) in enumerate(success_rates.items()):
        ax = axes[i]
        ax.plot(rates.keys(), rates.values(), marker='o', linestyle='-', color='b', label=level)
        ax.set_xlabel("Task Delay (ms)")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(level)
        ax.legend()
        ax.grid(True)
        ax.set_xticks([0, 75, 150, 225])
        ax.set_yticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    plt.savefig("figures/graph_success_rate.png")
    print("Graph Success Rate Created")

def graph_acceptability():
    all_round_logs = LOG_MANAGER.logs_per_round()
    acceptability_rates = {}

    for i in range(1, 33, 4):
        level_name, _ = Round.from_unique_id(i)

        delays = [0, 75, 150, 225]
        acceptability_data = {
            delay: [round.logs["qoe"].acceptable for round in all_round_logs[i + j]]
            for j, delay in enumerate(delays)
        }

        acceptability_rates[level_name] = {
            delay: sum(1 for acc in values if acc) / len(values) * 100
            for delay, values in acceptability_data.items()
        }

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
    axes = axes.flatten()

    for i, (level, rates) in enumerate(acceptability_rates.items()):
        ax = axes[i]
        ax.plot(rates.keys(), rates.values(), marker='o', linestyle='-', color='g', label="QoE Acceptability")
        ax.set_xlabel("Task Delay (ms)")
        ax.set_ylabel("Acceptability Rate (%)")
        ax.set_title(level)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True)
        ax.set_xticks([0, 75, 150, 225])
        ax.set_yticks([0, 25, 50, 75, 100])

    plt.tight_layout()
    plt.savefig("figures/graph_acceptability.png")
    print("Graph Acceptability Created")

def graph_pdi():
    """
    Graphs success rate vs precision, deadline, and impact scores.
    """
    event_logs = LOG_MANAGER.cleaned_event_logs()
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

    success_rates_by_precision_actual = {}
    success_rates_by_deadline_actual = {}
    success_rates_by_impact_actual = {}

    success_rates_by_precision_practical = {}
    success_rates_by_deadline_practical = {}
    success_rates_by_impact_practical = {}

    qoe_by_precision = {}
    qoe_by_deadline = {}
    qoe_by_impact = {}

    all_qoe_logs = LOG_MANAGER.qoe_logs()

    qoe_scores_by_uid = {uid: [log.score for log in logs] for uid, logs in all_qoe_logs.items()}

    avg_value_to_practical = {}
    avg_value_to_actual = {}
    avg_value_to_qoe = {}

    for uid in event_logs:
        for current_round in event_logs[uid]:
            last = int(current_round.iloc[-1]["Coins"])
            first = int(current_round.iloc[0]["Coins"])

            n_failures = sum(1 for row in current_round["Event"] if "Death" in row or "Failure" in row)
            n_successes = int((last - first) / 100.0)

            level_type = str(current_round.iloc[0]["Level"])
            max_successes_practical = ROUND_MAX_SUCCESSES[level_type]
            max_successes_actual = n_successes + n_failures

            success_rate_practical = n_successes / max_successes_practical
            success_rate_actual = n_successes / max_successes_actual

            level_parts = level_type.split("_")
            precision_score = word_to_num[level_parts[0]]
            deadline_score = word_to_num[level_parts[1]]
            impact_score = word_to_num[level_parts[2]]

            avg = (precision_score + deadline_score + impact_score) / 3

            if not avg in avg_value_to_practical :
                avg_value_to_practical[avg] = []
                avg_value_to_actual[avg] = []
                avg_value_to_qoe[avg] = []

            qoe_by_precision.setdefault(precision_score, [])
            qoe_by_deadline.setdefault(deadline_score, [])
            qoe_by_impact.setdefault(impact_score, [])

            for score_dict in [success_rates_by_precision_actual, success_rates_by_precision_practical]:
                score_dict.setdefault(precision_score, [])
            for score_dict in [success_rates_by_deadline_actual, success_rates_by_deadline_practical]:
                score_dict.setdefault(deadline_score, [])
            for score_dict in [success_rates_by_impact_actual, success_rates_by_impact_practical]:
                score_dict.setdefault(impact_score, [])

            qoe_by_precision.setdefault(precision_score, [])
            qoe_by_deadline.setdefault(deadline_score, [])
            qoe_by_impact.setdefault(impact_score, [])

            if uid in all_qoe_logs:
                round_index = next(
                    (i for i, round_df in enumerate(event_logs[uid]) if round_df.equals(current_round)), None)

                if round_index is not None and round_index < len(all_qoe_logs[uid]):
                    qoe_score = all_qoe_logs[uid][round_index].score
                    qoe_by_precision[precision_score].append(qoe_score)
                    qoe_by_deadline[deadline_score].append(qoe_score)
                    qoe_by_impact[impact_score].append(qoe_score)

                    avg_value_to_qoe[avg].append(qoe_score)

            success_rates_by_precision_actual[precision_score].append(success_rate_actual)
            success_rates_by_precision_practical[precision_score].append(success_rate_practical)

            success_rates_by_deadline_actual[deadline_score].append(success_rate_actual)
            success_rates_by_deadline_practical[deadline_score].append(success_rate_practical)

            success_rates_by_impact_actual[impact_score].append(success_rate_actual)
            success_rates_by_impact_practical[impact_score].append(success_rate_practical)

            avg_value_to_practical[avg].append(success_rate_practical)
            avg_value_to_actual[avg].append(success_rate_actual)


    def plot_graph(success_rates_actual, success_rates_practical, xlabel, filename):
        x = sorted(success_rates_actual.keys())
        y_actual = [np.mean(success_rates_actual[p] or [0]) for p in x]
        y_practical = [np.mean(success_rates_practical[p] or [0]) for p in x]
        z = 1.96

        ci_actual = [
            z * (np.std(success_rates_actual[p]) / math.sqrt(len(success_rates_actual[p])))
            if success_rates_actual[p] else 0
            for p in x
        ]

        ci_practical = [
            z * (np.std(success_rates_practical[p]) / math.sqrt(len(success_rates_practical[p])))
            if success_rates_practical[p] else 0
            for p in x
        ]

        fig, ax = plt.subplots()
        ax.plot(x, y_actual, marker="o", linestyle="-", label="Actual Success Rate", color="b")
        ax.plot(x, y_practical, marker="s", linestyle="--", label="Practical Success Rate", color="r")
        ax.errorbar(x, y_actual, yerr=ci_actual, fmt="o", capsize=4, color="b", alpha=0.7)
        ax.errorbar(x, y_practical, yerr=ci_practical, fmt="s", capsize=4, color="r", alpha=0.7)

        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, 1)
        ax.set_title(f"Success Rate vs {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Success Rate")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_merged_vs_performance_graph(unpackable_items, title, filename):
        """
        Plots precision, deadline, and impact success rates on the same graph
        """
        fig, ax = plt.subplots()
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]

        x = []
        y = []
        z = 1.96
        ci = []
        for (pdi_score, values) in unpackable_items.items():
            x.append(pdi_score)
            y.append(np.mean(values))
            ci.append( (z * np.std(values)) / math.sqrt(len(values)) )

        ax.scatter(x, y, marker=markers[0], linestyle="-", label=f"{title}", color=colors[0])
        ax.errorbar(x, y, yerr=ci, fmt=markers[0], capsize=4, color=colors[0], alpha=0.7)
        #ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} vs PDI Averaged Score")
        ax.set_xlabel("PDI Averaged Score")
        ax.set_ylabel(f"{title}")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_merged_vs_qoe_graph(unpackable_items, title, filename):
        """
        Plots precision, deadline, and impact success rates on the same graph
        """
        fig, ax = plt.subplots()
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]

        x = []
        y = []
        z = 1.96
        ci = []
        for (pdi_score, values) in unpackable_items.items():
            x.append(pdi_score)
            y.append(np.mean(values))
            ci.append( (z * np.std(values)) / math.sqrt(len(values)) )

        ax.scatter(x, y, marker=markers[0], linestyle="-", label=f"{title}", color=colors[0])
        ax.errorbar(x, y, yerr=ci, fmt=markers[0], capsize=4, color=colors[0], alpha=0.7)
        #ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_ylim(1, 5)
        ax.set_title(f"{title} vs PDI Averaged Score")
        ax.set_xlabel("PDI Averaged Score")
        ax.set_ylabel(f"{title}")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_combined_graph_practical(success_rates_practical, xlabel, filename):
        """
        Plots precision, deadline, and impact success rates on the same graph
        """
        fig, ax = plt.subplots()
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]
        labels = ["Precision", "Deadline", "Impact"]
        z = 1.96

        for i, (success_rates, label) in enumerate(zip(
                [success_rates_by_precision_practical, success_rates_by_deadline_practical, success_rates_by_impact_practical],
                labels)):
            x = sorted(success_rates.keys())
            y = [np.mean(success_rates[p] or [0]) for p in x]
            ci = [
                z * (np.std(success_rates[p]) / math.sqrt(len(success_rates[p])))
                if success_rates[p] else 0
                for p in x
            ]
            ax.plot(x, y, marker=markers[i], linestyle="-", label=f"{label} Success Rate", color=colors[i])
            ax.errorbar(x, y, yerr=ci, fmt=markers[i], capsize=4, color=colors[i], alpha=0.7)

        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, 1)
        ax.set_title(f"Practical Success Rate vs {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Success Rate")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_combined_graph_actual(success_rates_actual, xlabel, filename):
        """
        Plots precision, deadline, and impact actual success rates on the same graph
        """
        fig, ax = plt.subplots()
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]
        labels = ["Precision", "Deadline", "Impact"]

        for i, (success_rates, label) in enumerate(zip(
                [success_rates_by_precision_actual, success_rates_by_deadline_actual, success_rates_by_impact_actual],
                labels)):
            x = sorted(success_rates.keys())
            y = [np.mean(success_rates[p] or [0]) for p in x]
            z = 1.96
            ci = [
                z * (np.std(success_rates[p]) / math.sqrt(len(success_rates[p])))
                if success_rates[p] else 0
                for p in x
            ]
            ax.plot(x, y, marker=markers[i], linestyle="-", label=f"{label} Success Rate", color=colors[i])
            ax.errorbar(x, y, yerr=ci, fmt=markers[i], capsize=4, color=colors[i], alpha=0.7)

        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, 1)
        ax.set_title(f"Actual Success Rate vs {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average Success Rate")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_qoe_graph(qoe_scores, xlabel, filename):
        """
        Plots QoE score vs. the given metric
        """
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]
        labels = ["Precision", "Deadline", "Impact"]
        x = sorted(qoe_scores.keys())
        y_qoe = [np.mean(qoe_scores[p] or [0]) for p in x]
        z = 1.96

        ci_qoe = [
            z * (np.std(qoe_scores[p]) / math.sqrt(len(qoe_scores[p])))
            if qoe_scores[p] else 0
            for p in x
        ]

        fig, ax = plt.subplots()
        ax.plot(x, y_qoe, marker=markers[0], linestyle="-", label="QoE Score", color=colors[0])
        ax.errorbar(x, y_qoe, yerr=ci_qoe, fmt="o", capsize=4, color="b", alpha=0.7)

        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_ylim(1, 5)
        ax.set_title(f"QoE Score vs {xlabel}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average QoE Score")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    def plot_combined_qoe_graph(qoe_by_precision, qoe_by_deadline, qoe_by_impact, filename):
        """
        Plots QoE scores for Precision, Deadline, and Impact on the same graph.
        """
        fig, ax = plt.subplots()
        markers = ["o", "s", "d"]
        colors = ["b", "g", "r"]
        labels = ["Precision", "Deadline", "Impact"]

        for i, (qoe_scores, label) in enumerate(zip(
                [qoe_by_precision, qoe_by_deadline, qoe_by_impact], labels)):
            x = sorted(qoe_scores.keys())
            y_qoe = [np.mean(qoe_scores[p] or [0]) for p in x]
            z = 1.96
            ci_qoe = [
                z * (np.std(qoe_scores[p]) / math.sqrt(len(qoe_scores[p])))
                if qoe_scores[p] else 0
                for p in x
            ]
            ax.plot(x, y_qoe, marker=markers[i], linestyle="-", label=f"{label} QoE", color=colors[i])
            ax.errorbar(x, y_qoe, yerr=ci_qoe, fmt=markers[i], capsize=4, color=colors[i], alpha=0.7)

        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels([str(val) for val in [1, 2, 3, 4, 5]])
        ax.set_ylim(1, 5)
        ax.set_title("QoE Score Comparison by Metric")
        ax.set_xlabel("Score")
        ax.set_ylabel("Average QoE Score")

        plt.tight_layout()
        fig.savefig(f"figures/{filename}.png")
        plt.close()

    plot_graph(success_rates_by_precision_actual, success_rates_by_precision_practical, "Precision Score", "success_vs_precision")
    plot_graph(success_rates_by_deadline_actual, success_rates_by_deadline_practical, "Deadline Score", "success_vs_deadline")
    plot_graph(success_rates_by_impact_actual, success_rates_by_impact_practical, "Impact Score", "success_vs_impact")

    plot_combined_graph_practical(success_rates_by_precision_practical, "Score", "practical_success_comparison")
    plot_combined_graph_actual(success_rates_by_precision_actual, "Score", "actual_success_comparison")

    plot_merged_vs_performance_graph(avg_value_to_actual, "Actual Success Rate", "merged_actual_success_comparison")
    plot_merged_vs_performance_graph(avg_value_to_practical, "Practical Success Rate", "merged_practical_success_comparison")
    plot_merged_vs_qoe_graph(avg_value_to_qoe, "QoE", "merged_qoe_comparison")

    plot_qoe_graph(qoe_by_precision, "Precision Score", "qoe_vs_precision")
    plot_qoe_graph(qoe_by_deadline, "Deadline Score", "qoe_vs_deadline")
    plot_qoe_graph(qoe_by_impact, "Impact Score", "qoe_vs_impact")

    plot_combined_qoe_graph(qoe_by_precision, qoe_by_deadline, qoe_by_impact, "qoe_comparison")


def graph_qoe_violin():

    all_round_logs = LOG_MANAGER.logs_per_round()
    fig = plt.figure(figsize=(4, 3))

    qoe_data = []
    spike_labels = []

    spike_sizes = [0, 75, 150, 225]

    for i in range(1, 33, 4):
        for j, spike_size in enumerate(spike_sizes):
            if (i + j) in all_round_logs:
                ms_logs = all_round_logs[i + j]
                qoe_scores = [log.logs["qoe"].score for log in ms_logs]

                qoe_data.extend(qoe_scores)
                spike_labels.extend([spike_size] * len(qoe_scores))

    df = pd.DataFrame({"Spike Size": spike_labels, "QoE Score": qoe_data})

    #sb.violinplot(x="Spike Size", y="QoE Score", data=df, inner="quartile")

    plt.xlabel("Frametime Spike Size (ms)")
    plt.ylabel("QoE Score")
    plt.title("QoE Score Distribution vs Frametime Spike Size")
    plt.xticks([0, 1, 2, 3], labels=["0", "75", "150", "225"])
    plt.yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])
    plt.ylim((1, 5))
    plt.tight_layout()

    plt.savefig("figures/violin_qoe_vs_spike_size.png")
    plt.close()
