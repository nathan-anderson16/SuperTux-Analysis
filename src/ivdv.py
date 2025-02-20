import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import util

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
    all_round_logs = util.LOG_MANAGER.logs_per_round()
    acceptability_rates = {}

    for i in range(1, 33, 4):
        level_name, _ = util.Round.from_unique_id(i)

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





