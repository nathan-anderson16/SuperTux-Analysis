# Purpose
This repo contains the data analysis code for the study "Analyzing the Impact of Frametime Spikes on Navigation-Based Tasks in 2D Platformers."

The modified SuperTux Classic code used in this study can be found at https://github.com/nathan-anderson16/SuperTux-Classic.

# Installing
Using [uv](https://docs.astral.sh/uv/getting-started/installation/) is recommended.

To install, run

`uv venv`

# Running
`uv run src/main.py`

# Logs
By default, logs are saved to the following locations:
- [Windows] AppData\Roaming\SuperTuxClassic\logs
- [Mac] /Library/Application Support/SuperTuxClassic/logs
- [Linux] ~/.local/share/SuperTuxClassic/logs

There are four types of logs:
- Event Logs (~/SuperTuxClassic/logs/event_logs)
    - Contain all information on in-game events and triggers
- Frame Logs (~/SuperTuxClassic/logs/frame_logs)
    - Contain frame-by-frame information 
- QoE Logs (~/SuperTuxClassic/logs/qoe_logs)
    - Contain all post round QoE survey information
- Summary Logs (~/SuperTuxClassic/logs/summary_logs)
    - Contain summaries of each round in a human-readable format

# Results Format
Logs should go in a directory called Results, in the following format:
- Results/
  - 338/
  - 339/
  - ...

## Logs
Each directory in Results corresponds to a user ID. They should contain the following files:
- The frame log (e.g. frame_log_2025-01-27_16-42-00.csv)
- The event log (e.g. event_log_2025-01-27_16-42-00.csv)
- The QoE log (e.g. qoe_log_2025-01-27_16-42-00.csv)

These logs can be copy/pasted from the logs generated by SuperTux Classic.

## Reaction Time Results
Each directory should contain the results from the user's reaction time test, renamed to USER_ID.txt, where USER_ID is the user ID (e.g. 338.txt).

An example of the contents of the file is:

259,250,160,230,0,257,223,178,190,0

0 means the user failed that round (clicked too early)

# Figures
Graphs are outputted to figures/.

# Modifying / Adding Figures
`src/util.py` contains a LOG_MANAGER, which manages logs. You'll probably want `LOG_MANAGER.qoe_logs()`, `LOG_MANAGER.rounds()`, or `LOG_MANAGER.logs_per_round()`.

All results are cached, so there's no drawback to calling the same function multiple times.

However, if you want to force reload the logs from disk, you can pass the parameter `force_reload=True`.
