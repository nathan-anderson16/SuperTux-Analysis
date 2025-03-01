from util import LOG_MANAGER
import numpy as np

def player_qoe_outliers() :
    outliers = []
    all_qoe_logs = LOG_MANAGER.qoe_logs()

    averages = {}
    for (player_id, qoe_list) in all_qoe_logs.items() :
        scores = []
        for qoe in qoe_list :
            scores.append(qoe.score)
        
        averages[player_id] = np.average(scores)
    
    total_average = np.average(list(averages.values()))

    standard_deviation = np.std(scores)

    for (player_id, average) in averages.items() :
        if average < total_average - standard_deviation or average > total_average + standard_deviation :
            outliers.append(player_id)
    return outliers

def player_score_outliers() :
    outliers = []
    player_scores = {}
    for (user_id, data) in LOG_MANAGER.cleaned_frame_logs().items() :
        player_scores[user_id] = data[-1].iloc[-1].get('Coins')
    
    player_score_average = np.average(list(player_scores.values()))
    player_score_std = np.std(list(player_scores.values()))

    for (user_id, score) in player_scores.items() :
        if score < player_score_average - player_score_std or score > player_score_average + player_score_std :
            outliers.append(user_id)
    
    return outliers

def player_failure_outliers() :
    outliers = []
    player_failures = {}
    for (user_id, data) in LOG_MANAGER.cleaned_event_logs().items() :
        player_failures[user_id] = 0 
        for df in data :
            player_failures[user_id] += len(df[df["Event"].str.contains("Failure|Death")]) #data[-1].iloc[-1].get('Deaths')
    
    player_failure_average = np.average(list(player_failures.values()))
    player_failure_std = np.std(list(player_failures.values()))

    for (user_id, failures) in player_failures.items() :
        if failures < player_failure_average - player_failure_std or failures > player_failure_average + player_failure_std :
            outliers.append(user_id)
    
    return outliers

def player_activity_outliers() :
    outliers = []
    player_entries = {}
    for (user_id, data) in LOG_MANAGER.cleaned_event_logs().items() :
        player_entries[user_id] = 0 
        for df in data :
            player_entries[user_id] += len(df) #data[-1].iloc[-1].get('Deaths')
    
    player_entry_average = np.average(list(player_entries.values()))
    player_entry_std = np.std(list(player_entries.values()))

    for (user_id, entries) in player_entries.items() :
        if entries < player_entry_average - player_entry_std or entries > player_entry_average + player_entry_std :
            outliers.append(user_id)

    return outliers

def round_qoe_outliers() :
    outliers = []
    round_scores = {}

    for player_rounds in LOG_MANAGER.rounds().values() :
        for player_round in player_rounds :
            if not player_round.level_name in round_scores :
                round_scores[player_round.level_name] = [player_round.logs['qoe'].score]
            else :
                round_scores[player_round.level_name].append(player_round.logs['qoe'].score)
    
    round_averages = {}
    round_standard_deviations = {}
    for (round_type, scores) in round_scores.items() :
        round_averages[round_type] = np.average(list(scores))
        round_standard_deviations[round_type] = np.std(list(scores))
    
    round_total_average = np.average(list(round_averages.values()))
    for (round_type, round_average) in round_averages.items() :
        if round_average < round_total_average - round_standard_deviations[round_type] or round_average > round_total_average + round_standard_deviations[round_type] :
            outliers.append(round_type)
    return outliers

def round_activity_outliers() :
    outliers = []
    round_entries = {}
    for (user_id, data) in LOG_MANAGER.cleaned_event_logs().items() :
        for df in data :
            if df.iloc[0]["Level"] in round_entries :
                round_entries[df.iloc[0]["Level"]].append(len(df["Level"]))
            else :
                round_entries[df.iloc[0]["Level"]] = [len(df["Level"])]

    round_entry_averages = {}
    for (round_type, entries) in round_entries.items() :
        round_entry_averages[round_type] = np.average(entries)

    player_entry_average = np.average(list(round_entry_averages.values()))
    player_entry_std = np.std(list(round_entry_averages.values()))

    for (user_id, entries) in round_entry_averages.items() :
        if entries < player_entry_average - player_entry_std or entries > player_entry_average + player_entry_std :
            outliers.append(user_id)
   
    return outliers

def player_round_activity_outliers() :
    outliers = []
    player_round_entries = {}
    for (user_id, data) in LOG_MANAGER.cleaned_event_logs().items() :
        player_round_entries[user_id] = {}
        for df in data :
            if df.iloc[0]["Level"] in player_round_entries :
                player_round_entries[user_id][df.iloc[0]["Level"]].append(len(df["Level"]))
            else :
                player_round_entries[user_id][df.iloc[0]["Level"]] = [len(df["Level"])]

    player_round_entry_averages = {}
    list_of_all_rounds = []
    for (user_id, data) in player_round_entries.items() :
        player_round_entry_averages[user_id] = {}
        for (round_type, entries) in data.items() :
            player_round_entry_averages[user_id][round_type] = np.average(entries)
            list_of_all_rounds.append(entries)

    player_entry_average = np.average(list_of_all_rounds)
    player_entry_std = np.std(list_of_all_rounds)

    count = 0
    for (user_id, data) in player_round_entry_averages.items() :
        for (round_id, entries) in data.items() :
            if entries < player_entry_average - player_entry_std or entries > player_entry_average + player_entry_std :
                if not round_id in ["three_three_three_level", "three_three_five_level", "two_five_five_level"] :
                    print(user_id, round_id, entries)
                    count += 1
                outliers.append((user_id, round_id))
    print(player_entry_average, player_entry_std)
    print(count)
    return outliers

def print_outliers() :
    player_outliers = {}
    round_outliers = {}

    for player_id in player_qoe_outliers() :
        if player_id in player_outliers.keys() :
            player_outliers[player_id] += 1
        else :
            player_outliers[player_id] = 1
   
    for player_id in player_score_outliers() :
        if player_id in player_outliers.keys() :
            player_outliers[player_id] += 1
        else :
            player_outliers[player_id] = 1

    for player_id in player_failure_outliers() :
        if player_id in player_outliers.keys() :
            player_outliers[player_id] += 1
        else :
            player_outliers[player_id] = 1

    for player_id in player_activity_outliers() :
        if player_id in player_outliers.keys() :
            player_outliers[player_id] += 1
        else :
            player_outliers[player_id] = 1

    for round_type in round_qoe_outliers() :
        if round_type in round_outliers.keys() :
            round_outliers[round_type] += 1
        else :
            round_outliers[round_type] = 1
    
    for round_type in round_activity_outliers() :
        if round_type in round_outliers.keys() :
            round_outliers[round_type] += 1
        else :
            round_outliers[round_type] = 1
   
    print("Player Outliers:")
    for (player_id, count) in player_outliers.items() :
        print(player_id, count)

    print("Round Outliers")
    for (round_type, count) in round_outliers.items() :
        print(round_type, count)
    
    print("Misc")
    _ = player_round_activity_outliers()

if __name__ == "__main__" :
    print_outliers()
