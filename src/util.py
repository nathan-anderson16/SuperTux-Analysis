import csv
import datetime
import os
from pathlib import Path
from typing import Any, Union

import pandas as pd
import tqdm

# Event log headers:
# PlayerID,Timestamp,Level,ExpectedLag,State,Timer,Coins,Lives,Deaths,X-Position,Y-Position,X-Velocity,Y-Velocity,Event
#
# Frame log headers:
# Time,PlayerID,DeltaFrames,Timestamp,Level,State,Timer,Coins,Lives,Deaths,X-Position,Y-Position,X-Velocity,Y-Velocity,TickRate


figure = 0


def next_figure() -> int:
    """
    Increments a global figure number and returns it.
    Use this when you call `plt.figure()`.
    """
    global figure
    figure += 1
    return figure


def _find_user_ids(path: Path):
    """
    Finds the user IDs (folder names) of each folder in the given path.
    """
    return [os.path.basename(x[0]) for x in os.walk(path)][1:]


def _find_log_paths(path: Path) -> dict[str, Union[Path, None]]:
    """
    Finds the paths of the logs in the given directory.
    """
    files = os.listdir(path)

    frame_logs = [path / f for f in files if f.startswith("frame_log")]
    frame = frame_logs[0] if frame_logs else None

    event_logs = [path / f for f in files if f.startswith("event_log")]
    event = event_logs[0] if event_logs else None

    qoe_logs = [path / f for f in files if f.startswith("qoe_log")]
    qoe = qoe_logs[0] if qoe_logs else None

    summary_logs = [path / f for f in files if f.startswith("summary_log")]
    summary = summary_logs[0] if summary_logs else None

    return {"frame": frame, "event": event, "qoe": qoe, "summary": summary}


def _get_log_paths(
    path: Path, user_ids: list[str]
) -> dict[str, dict[str, Union[Path, None]]]:
    """
    Given a path and a list of user IDs, returns all the logs for each user ID.
    """
    d: dict[str, dict[str, Union[Path, None]]] = dict()
    for uid in user_ids:
        logs = _find_log_paths(path / uid)
        d[uid] = logs

    return d


RESULTS_PATH = Path("./Results")
"""The path to the directory containing the results."""

USER_IDS = _find_user_ids(RESULTS_PATH)
"""The user IDs of the users in the study."""

LOG_PATHS = _get_log_paths(RESULTS_PATH, USER_IDS)
"""A dict of the form "user_id": {"frame": ..., "event": ..., "qoe": ..., "summary": ...}.
Each entry is one of the user IDs.
If a log does not exist, its entry is set to None."""


class QoELog:
    score: float = 0.0
    acceptable: bool = False

    def __init__(self, score: float, acceptable: bool):
        self.score = score
        self.acceptable = acceptable

    def __str__(self) -> str:
        return f"QoELog('score': {self.score}, 'acceptable': {self.acceptable})"

    def __repr__(self) -> str:
        return str(self)


class Round:
    """
    Represents a single round.
    """

    level_name: str = ""
    """
    The name of the level this round was played on.
    """

    spike_duration: int = 0
    """
    The duration, in ms, of the frame spike this round was played on.
    """

    uid: int = 0
    """
    The user ID of the user who played this round. WARNING: MAY NOT BE UNIQUE.
    """

    logs: dict[str, Any] = dict()
    """
    The logs for this round.

    This is a dict of the form
    {
        "frame": ...,
        "event": ...,
        "qoe": ...,
    }
    """

    _level_names = [
        "five_five_five_level",
        "four_four_five_level",
        "one_two_two_level",
        "three_four_five_level",
        "three_three_five_level",
        "three_three_three_level",
        "two_five_five_level",
        "two_three_two_level",
    ]
    """
    A list of all the different names a level can have.
    """

    def __init__(self, frame: pd.DataFrame, event: pd.DataFrame, qoe: QoELog) -> None:
        self.logs = {
            "frame": frame,
            "event": event,
            "qoe": qoe,
        }

        self.level_name = frame["Level"].iloc[1]
        self.spike_duration = int(event["ExpectedLag"].iloc[1])
        self.uid = int(frame["PlayerID"].iloc[1])

    def __str__(self) -> str:
        return f"Round('level_name': {self.level_name}, 'spike_duration': {self.spike_duration}, 'uid': {self.uid}, 'qoe_log': {self.logs['qoe']})"

    def __repr__(self) -> str:
        return str(self)

    def unique_id(self) -> int:
        """
        Returns the unique ID for the task and frame spike duration of this round.
        This ranges from 1 to 32 (inclusive).
        Useful when grouping rounds according to the task and frame spike duration.
        """
        level_id = self._level_names.index(self.level_name)
        variant = int(self.spike_duration / 75)

        return (level_id * 4) + variant + 1

    @staticmethod
    def from_unique_id(id: int) -> tuple[str, int]:
        """
        Given a unique round ID generated from unique_id(), returns the level name and the frame spike duration.
        """
        id -= 1  # return to zero-based indexing
        return (Round._level_names[int(id / 4)], (id % 4) * 75)


class LogManager:
    _cache: dict[str, Any] = dict()
    """A cache of the logs."""

    def __init__(self) -> None:
        self._cache = {
            "raw_frame_logs": None,
            "raw_event_logs": None,
            "clean_frame_logs": None,
            "clean_event_logs": None,
            "qoe_logs": None,
            "rounds": None,
        }

    def raw_frame_logs(self, force_reload: bool = False) -> dict[str, pd.DataFrame]:
        """
        Reads all of the frame logs and returns them in a dict of the form
        {"user_id": log}
        where user_id is the user's ID and log is the user's frame log.

        This returns the raw, unmodified log. If you want to access only the round data, use `cleaned_frame_logs()`.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """

        if not force_reload and self._cache["raw_frame_logs"] is not None:
            return self._cache["raw_frame_logs"]

        print("Loading frame logs...")
        dfs: dict[str, pd.DataFrame] = {}
        for uid in tqdm.tqdm(LOG_PATHS):
            path = LOG_PATHS[uid]["frame"]
            if path is not None:
                df = pd.read_csv(path)
                dfs[uid] = df

        print(f"Loaded {len(dfs.keys())} frame logs.")

        self._cache["raw_frame_logs"] = dfs
        return self._cache["raw_frame_logs"]

    def _read_event_log(self, path: Path) -> pd.DataFrame:
        """
        Reads an event log located at the given path.
        This function also removes any commas that cause issues with Pandas' CSV parser.
        For example, the line:
        `Random Delay Exit: Exited at time: 1162.977000, Duration: 0.134000 seconds`
        has a comma, which will (understandably) break the parser.
        This function removes those commas.
        """
        with open(path) as f:
            reader = csv.reader(f, delimiter=",")
            lines = [line for line in reader]

        headers = lines[0]
        lines = lines[1:]
        for i, line in enumerate(lines):
            line: list[Any] = line
            if len(line) == len(headers) + 1:
                line[13] = line[13] + line[14]

            # Player ID: int
            line[0] = int(line[0])
            # Expected Lag: int
            line[3] = int(line[3])
            # Timer: float
            line[5] = float(line[5])
            # Coins: int
            line[6] = int(line[6])
            # Lives: int
            line[7] = int(line[7])
            # Deaths: int
            line[8] = int(line[8])
            # X-Position: float
            line[9] = float(line[9])
            # Y-Position: float
            line[10] = float(line[10])
            # X-Velocity: float
            line[11] = float(line[11])
            # Y-Velocity: float
            line[12] = float(line[12])

            lines[i] = line[:14]

        df = pd.DataFrame(lines, columns=headers)  # type: ignore
        return df

    def raw_event_logs(self, force_reload: bool = False) -> dict[str, pd.DataFrame]:
        """
        Reads all of the event logs and returns them in a dict of the form
        {"user_id": log}
        where user_id is the user's ID and log is the user's event log.

        This returns the raw, unmodified log. If you want to access only the round data, use `cleaned_event_logs()`.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """

        if not force_reload and self._cache["raw_event_logs"] is not None:
            return self._cache["raw_event_logs"]

        print("Loading event logs...")
        dfs: dict[str, pd.DataFrame] = dict()
        for uid in tqdm.tqdm(LOG_PATHS):
            path = LOG_PATHS[uid]["event"]
            if path is not None:
                df = self._read_event_log(path)
                dfs[uid] = df

        print(f"Loaded {len(dfs.keys())} event logs.")

        self._cache["raw_event_logs"] = dfs
        return self._cache["raw_event_logs"]

    def _clean_frame_event_df(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Given a frame or event log DataFrame, removes the practice rounds and creates a separate DataFrame for each round.
        """
        # Get every index where the timer is less than 0
        less_than_0 = []
        for i, time in enumerate(df["Timer"].to_numpy()):
            if time <= 0 and i < len(df) - 1 and df["Timer"][i + 1] > 0:
                less_than_0.append(i)

        dfs = []
        for i, item in enumerate(less_than_0):
            start = item
            end = len(df) - 1 if i + 1 >= len(less_than_0) else less_than_0[i + 1]
            new_df = df.loc[start + 1 : end - 1]
            new_df = new_df[new_df["Timer"] > 0]
            new_df = new_df[new_df["Timer"] < max(new_df["Timer"])]
            dfs.append(new_df)

        dfs = [df for df in dfs if df["Level"].unique()[0] != "practice"]
        return dfs

    def cleaned_frame_logs(
        self, force_reload: bool = False
    ) -> dict[str, list[pd.DataFrame]]:
        """
        Reads all of the frame logs, cleans them, and returns them in a dict of the form
        {"user_id": logs}
        where user_id is the user's ID and logs is a list of the round logs where the first entry corresponds to round 1, the second entry corresponds to round 2, and so on.

        This returns ONLY the round data in each log. For the full unmodified log data, use `raw_frame_logs()`.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """

        if not force_reload and self._cache["clean_frame_logs"] is not None:
            return self._cache["clean_frame_logs"]

        logs = self.raw_frame_logs(force_reload=force_reload)

        print("Cleaning frame logs...")
        dfs: dict[str, list[pd.DataFrame]] = dict()
        for uid in tqdm.tqdm(logs):
            dfs[uid] = self._clean_frame_event_df(logs[uid])
            # print(f"Read {len(dfs[uid])} rounds in the frame log")

        self._cache["clean_frame_logs"] = dfs
        return self._cache["clean_frame_logs"]

    def cleaned_event_logs(
        self, force_reload: bool = False
    ) -> dict[str, list[pd.DataFrame]]:
        """
        Reads all of the event logs, cleans them, and returns them in a dict of the form
        {"user_id": logs}
        where user_id is the user's ID and logs is a list of the round logs where the first entry corresponds to round 1, the second entry corresponds to round 2, and so on.

        This returns ONLY the round data in each log. For the full unmodified log data, use `raw_event_logs()`.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """

        if not force_reload and self._cache["clean_event_logs"] is not None:
            return self._cache["clean_event_logs"]

        logs = self.raw_event_logs(force_reload=force_reload)

        print("Cleaning event logs...")
        dfs: dict[str, list[pd.DataFrame]] = dict()
        for uid in tqdm.tqdm(logs):
            dfs[uid] = self._clean_frame_event_df(logs[uid])
            # print(f"Read {len(dfs[uid])} rounds in the event log")

        self._cache["clean_event_logs"] = dfs
        return self._cache["clean_event_logs"]

    def qoe_logs(self, force_reload: bool = False) -> dict[str, list[QoELog]]:
        """
        Reads all of the QoE logs and returns them in a dict of the form
        {"user_id": log}
        where user_id is the user's ID and log is `list[QoELog]`, where the first entry corresponds to the QoE log for the first round, etc.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """

        if not force_reload and self._cache["qoe_logs"] is not None:
            return self._cache["qoe_logs"]

        print("Loading QoE logs...")
        logs: dict[str, list[QoELog]] = dict()
        for uid in tqdm.tqdm(LOG_PATHS):
            log = LOG_PATHS[uid]["qoe"]
            user_logs = []
            if log is not None:
                with open(log) as f:
                    lines = f.readlines()[1:]
                    lines = [line.split(",")[-1] for line in lines]
                    for i in range(0, len(lines), 2):
                        score = float(lines[i].strip("QoE Score: "))
                        acceptable = True if "Yes" in lines[i + 1] else False
                        user_logs.append(QoELog(score, acceptable))
            user_logs = user_logs[2:]  # Remove practice rounds
            logs[uid] = user_logs
            # print(f"Read {len(user_logs)} rounds in the QoE log")

        self._cache["qoe_logs"] = logs
        return self._cache["qoe_logs"]

    def rounds(self, force_reload: bool = False) -> dict[str, list[Round]]:
        """
        Returns all of the frame, event, and QoE logs, in the form

        {
            user_id: log,
            ...
        }

        where user_id is the user ID and log is a list[Round] representing all the rounds the user played.
        """
        if not force_reload and self._cache["rounds"] is not None:
            return self._cache["rounds"]

        frame_logs = self.cleaned_frame_logs(force_reload)
        event_logs = self.cleaned_event_logs(force_reload)
        qoe_logs = self.qoe_logs(force_reload)

        rounds: dict[str, list[Round]] = dict()
        for uid in frame_logs:
            frames = frame_logs[uid]
            events = event_logs[uid]
            qoes = qoe_logs[uid]

            user_rounds = []
            for frame, event, qoe in zip(frames, events, qoes):
                user_rounds.append(Round(frame, event, qoe))

            # print(user_rounds)
            rounds[uid] = user_rounds

        self._cache["rounds"] = rounds
        return self._cache["rounds"]

    def logs_per_round(self, force_reload: bool = False) -> dict[int, list[Round]]:
        """
        Reads all of the round logs and returns them in a dict of the form
        {round_id: log}
        where user_id is the round's ID and log is a list of all instances of that round that were played.

        :param force_reload: Whether to force a reload of the logs from disk, invalidating any cache. Warning: this can be slow. Default False.
        """
        rounds = self.rounds(force_reload=force_reload)

        logs: dict[int, list[Round]] = dict()
        for uid in rounds:
            for round in rounds[uid]:
                round_id = round.unique_id()
                if round_id not in logs:
                    logs[round_id] = list()

                logs[round_id].append(round)

        return logs


def parse_timestamp(string: str) -> float:
    """
    Given a timestamp of the form "HH:MM:SS.ms", returns the timestamp as a float.
    """
    return datetime.datetime.timestamp(
        datetime.datetime.strptime(string, "%H:%M:%S.%f")
    )


LOG_MANAGER = LogManager()
"""
The log manager for the program. This should be the only log manager used.
"""


# Testing
def main():
    # log_manager = LogManager()
    # print(log_manager.raw_frame_logs())
    # print(log_manager.raw_event_logs())
    # print(log_manager.cleaned_frame_logs()["338"])
    # print(log_manager.cleaned_event_logs()["338"])
    # print(log_manager.qoe_logs())
    # print(log_manager.rounds())
    # print(Round.from_unique_id(31))
    pass


if __name__ == "__main__":
    main()
