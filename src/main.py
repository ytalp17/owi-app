import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta
from datetime import date
from itertools import combinations
import src.variables as variables
import networkx as nx
import time
import matplotlib.pyplot as plt
import math
import seaborn as sb
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

RESULTS_DIR = variables.RESULTS_DIRECTORY
SPLITS_DIR = variables.SPLITS_DIRECTORY
RANKINGS_DIR = variables.RANKINGS_DIRECTORY
ATHLETE_DATA_DIR = variables.ATHLETE_DATA_DIRECTORY
RANKING_FILE_NAME = variables.RANKING_FILE_NAME
LAMBDA = variables.LAMBDA
DEPRECIATION_MODEL = variables.DEPRECIATION_MODEL
GENDER = variables.GENDER
if GENDER == "men":
    DEPRECIATION_PERIOD = variables.M_DEPRECIATION_PERIOD
    K = variables.M_K
elif GENDER == "women":
    DEPRECIATION_PERIOD = variables.W_DEPRECIATION_PERIOD
    K = variables.W_K
RANK_DIST = variables.RANK_DIST
FROM_RANK = variables.FROM_RANK
TO_RANK = variables.TO_RANK
event_type_weights = variables.event_weights
athlete_countries = variables.athlete_countries

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def get_age_weight(race_date_text, ranking_date):
    race_date = dt.strptime(race_date_text, "%m/%d/%Y")
    rank_date = dt.strptime(ranking_date, "%m/%d/%Y")

    if DEPRECIATION_MODEL == "linear":
        days_old = (rank_date.date() - race_date.date()).days
        weight = (DEPRECIATION_PERIOD - days_old) / DEPRECIATION_PERIOD
        return weight
    elif DEPRECIATION_MODEL == "exponential":
        days_old = (rank_date.date() - race_date.date()).days
        years_old = days_old / 365
        weight = math.exp(LAMBDA * years_old)
        return weight
    elif DEPRECIATION_MODEL == "sigmoid":
        days_old = (rank_date.date() - race_date.date()).days
        if days_old == 0:
            weight = 1
        elif DEPRECIATION_PERIOD - days_old == 0:
            weight = 0
        else:
            f = math.log((DEPRECIATION_PERIOD - days_old) / days_old)
            weight = 1 / (1 + math.exp(-K * f))
        return weight


def get_comp_weight(event_type):
    """
    :param event_type: event type as text, ie: "FINA World Cup"
    :return: weight as a float
    """
    weight = float(event_type_weights.weight[event_type_weights.event == event_type])
    return weight


def get_distance_weight(race_dist):
    """
    :param race_dist: distance of the race in km
    :return: weight as a float
    """
    if RANK_DIST == 0:
        weight = 1
    else:
        weight = min(race_dist, RANK_DIST) / max(race_dist, RANK_DIST)
    return weight

    # # only use races that are the same distance as RANK_DIST
    # if RANK_DIST == race_dist:
    #     weight = 1
    # else:
    #     weight = 0
    # return weight


def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()


def label(results_file_path):
    race_data = pd.read_csv(results_file_path)
    race_label = f"{race_data.event[0]} {race_data.location[0]} {str(race_data.distance[0])}km " \
                 f"{race_data.location[0]} {race_data.date[0]}"
    return race_label


def update_graph(race_result_file, ranking_date):
    """
    :param ranking_date:
    :param race_result_file: csv file with results from a single race
    :return: adds nodes and edges from race_result_file to existing graph
    """
    # is global G needed? seems to work without it...
    global G

    race_data = pd.read_csv(race_result_file)
    name_list = [name.title() for name in race_data.athlete_name.tolist()]
    age_weight = get_age_weight(race_data.date[0], ranking_date)
    comp_weight = get_comp_weight(race_data.event[0])
    dist_weight = get_distance_weight(race_data.distance[0])
    total_weight = age_weight * comp_weight * dist_weight
    race_label = custom_label(race_result_file, "event", "location", "distance", "date")

    # Check for a tie:
    this_race_has_a_tie = False
    if len(race_data.place) != len(set(race_data.place)):
        this_race_has_a_tie = True
        # print(f'{race_result_file} has a tie')

    combos = list(combinations(name_list, 2))
    combos = [tuple(reversed(combo)) for combo in combos]

    for combo in combos:
        loser = combo[0]
        winner = combo[1]
        if this_race_has_a_tie:
            if int(race_data.place[race_data.athlete_name == loser]) == int(race_data.place[race_data.athlete_name == winner]):
                # print(f'{winner} and {loser} tied for {int(race_data.place[race_data.athlete_name == loser])} place.')
                pass
            if combo in G.edges:
                current_weight = G[loser][winner]["weight"]
                new_weight = current_weight + total_weight
                G[loser][winner]["weight"] = new_weight
                G[loser][winner]["race_weights"][race_label] = total_weight
            else:
                label_dict = {
                    race_label: total_weight
                }
                G.add_edge(*combo, weight=total_weight, race_weights=label_dict)
        else:
            if combo in G.edges:
                current_weight = G[loser][winner]["weight"]
                new_weight = current_weight + total_weight
                G[loser][winner]["weight"] = new_weight
                G[loser][winner]["race_weights"][race_label] = total_weight
            else:
                label_dict = {
                    race_label: total_weight
                }
                G.add_edge(*combo, weight=total_weight, race_weights=label_dict)


def test_predictability(race_result_file):
    """
    :param race_result_file: a new race result csv file to compare against the ranking at that point in time
    :return: adds to correct_predictions (if applicable) and total_matchups running count
    """

    global correct_predictions
    global total_tests
    global FROM_RANK
    global TO_RANK

    instance_correct_predictions = 0
    instance_total_tests = 0
    # race_label = label(race_result_file, "event", "location", "date", "distance")
    # race_label = race_result_file

    ranking_data = pd.read_csv(RANKING_FILE_NAME).iloc[(FROM_RANK - 1):TO_RANK]
    race_data = pd.read_csv(race_result_file)
    name_list = race_data.athlete_name.tolist()
    combos = list(combinations(name_list, 2))

    for matchup in combos:
        winner_name = matchup[0].title()
        loser_name = matchup[1].title()
        if winner_name in list(ranking_data.name) and loser_name in list(ranking_data.name):
            winner_rank = int(ranking_data["rank"][ranking_data.name == winner_name])
            loser_rank = int(ranking_data["rank"][ranking_data.name == loser_name])
            total_tests += 1
            instance_total_tests += 1
            if winner_rank < loser_rank:
                correct_predictions += 1
                instance_correct_predictions += 1

    try:
        instance_predictability = instance_correct_predictions / instance_total_tests
        # print(f"{race_label} instance_predictability {instance_predictability}")
    except ZeroDivisionError:
        # print(f"cannot calculate predictability for {race_result_file} -- cannot divide by 0")
        pass
    else:
        instance_predictability = "{:.0%}".format(instance_predictability)
        # print(f"Ranking predictability at {race_label}: {instance_predictability}")


def create_ranking(ranking_date, test=False, comment=False, summary=False, display_list=0, vis=0):
    start = time.time()
    global correct_predictions
    global total_tests
    global G
    global RANKING_FILE_NAME
    G = nx.DiGraph()
    race_count = 0

    # first remove the ranking file that may exist from past function calls
    if os.path.exists(RANKING_FILE_NAME):
        os.remove(RANKING_FILE_NAME)

    # Loop through each race result file. If it's in the date range, update global G with that race's results by
    # calling update_rankings()
    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            if comment:
                print(f"Excluding {file}, race is not in date range.")
            else:
                pass
        elif os.path.exists(RANKING_FILE_NAME):
            if test:
                test_predictability(results_file_path)
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)
        else:
            if test:
                pass
            if comment:
                print(f"Loading {file}")
            update_graph(results_file_path, ranking_date)
            race_count += 1
            pr_dict = nx.pagerank(G)
            ranking_dict = {
                "name": list(pr_dict.keys()),
                "pagerank": list(pr_dict.values())
            }
            ranking_df = pd.DataFrame(ranking_dict)
            ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
            ranking_df["rank"] = range(1, len(pr_dict) + 1)
            ranking_df.to_csv(RANKING_FILE_NAME, index=False)

    if display_list > 0:
        ranking_data = pd.read_csv(RANKING_FILE_NAME)
        print(ranking_data[ranking_data["rank"] < display_list + 1])

    if vis > 0:

        pr_dict = nx.pagerank(G)

        ranking_dict = {
            "name": list(pr_dict.keys()),
            # "country": [athlete_countries.country[athlete_countries.proper_name == name] for name in
            #                 list(pr_dict.keys())],
            "pagerank": list(pr_dict.values())
        }

        ranking_df = pd.DataFrame(ranking_dict)
        ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
        ranking_df["rank"] = range(1, len(pr_dict) + 1)

        num_of_athletes = vis
        top_athletes = list(ranking_df.name[ranking_df["rank"] < num_of_athletes + 1])
        G = G.subgraph(top_athletes)

        size_map = []
        thicknesses = []
        for name in G.nodes:
            size_map.append(float(ranking_df.pagerank[ranking_df.name == name] * 3000))
        for edge in G.edges:
            thicknesses.append(G[edge[0]][edge[1]]["weight"] * .4)

        nx.draw_networkx(G, node_size=size_map, font_size=8, font_color="red", width=thicknesses,
                         pos=nx.spring_layout(G))
        plt.show()

    end = time.time()

    if test and not summary:
        predictability = correct_predictions / total_tests
        print(f"Predictability ({FROM_RANK} - {TO_RANK}): {predictability}")
        return predictability

    if summary:
        print(f"New ranking file created: {RANKING_FILE_NAME}")
        print(f"Time to execute: {round((end - start), 2)}s")
        print(f"Races included in ranking: {race_count}")
        print(f"Gender: {GENDER}")
        print(f"Distance: {RANK_DIST}km")
        print(f"Depreciation period: {DEPRECIATION_PERIOD / 365} years")
        print(f"Depreciation model: {DEPRECIATION_MODEL}")
        if DEPRECIATION_MODEL == "sigmoid":
            print(f"K value: {K}")
        elif DEPRECIATION_MODEL == "exponential":
            print(f"Lambda: {LAMBDA}")
        if test:
            predictability = correct_predictions / total_tests
            # predictability = "{:.0%}".format(predictability)
            # print(correct_predictions)
            # print(total_tests)
            # print(f"Predictability: {predictability}")
            print(f"Predictability ({FROM_RANK} - {TO_RANK}): {predictability}")
            return predictability


def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


def unalpha_date(date):
    """
    :param date: YYYY_MM_DD
    :return: MM/DD/YYYY
    """
    uad = date[5:] + "_" + date[:4]
    uad = uad.replace("_", "/")
    return uad


def archive_ranking(ranking_date):
    global G
    G = nx.DiGraph()

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            pass
        else:
            update_graph(results_file_path, ranking_date)

    pr_dict = nx.pagerank(G)

    ranking_dict = {
        "name": list(pr_dict.keys()),
        "pagerank": list(pr_dict.values())
    }

    ranking_df = pd.DataFrame(ranking_dict)
    ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(pr_dict) + 1)
    file_name = f"{alpha_date(ranking_date)}_{GENDER}_{RANK_DIST}km.csv"
    ranking_df.to_csv(f"{RANKINGS_DIR}/{file_name}", index=False)
    print(f"{RANKINGS_DIR}/{file_name} archived")


def archive_rankings_range(start_date, end_date, increment=1):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    rank_dates = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    files_created = 0
    total_files = len(rank_dates)

    for date in rank_dates:
        archive_ranking(date)
        files_created += 1
        progress = files_created / total_files
        print("{:.0%}".format(progress))


def ranking_progression_multi(start_date, end_date, athlete_names):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    increment = 1
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    traces = []
    scatter_df = pd.DataFrame()

    for athlete_name in athlete_names:
        athlete_name = athlete_name.title()
        rank_dates = []
        ranks = []

        for date in date_range:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIR}/{file_name}")
            ranked_athletes = list(ranking_data.name)
            if athlete_name in ranked_athletes:
                rank_dates.append(dt.strptime(date, "%m/%d/%Y"))
                rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
                ranks.append(rank_on_date)
        line_trace = go.Scatter(x=rank_dates,
                                y=ranks,
                                mode='lines',
                                opacity=0.8,
                                name=athlete_name)
        traces.append(line_trace)

        results_df = get_results(athlete_name)
        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_ranks = []
        for date in results_df.date:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIR}/{file_name}")
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            race_date_ranks.append(rank_on_date)
        results_df['rank'] = race_date_ranks
        print(results_df)
        scatter_df = pd.concat([scatter_df, results_df])
        print(scatter_df)

    for event_type in scatter_df['event'].unique():
        df = scatter_df[scatter_df['event'] == event_type]
        scatter_trace = go.Scatter(x=df['dt_date'],
                                   y=df['rank'],
                                   mode='markers',
                                   marker={'size': 10, 'line': {'width': 0.5, 'color': 'black'}},
                                   name=event_type)
        traces.append(scatter_trace)

    fig_layout = go.Layout(
        title="World Ranking Progression",
        xaxis={'title': 'Date'},
        yaxis={'title': 'World Ranking'},
        hovermode='closest')

    fig = go.Figure(data=traces, layout=fig_layout)
    pyo.plot(fig)

    print({'data': traces, 'layout': fig_layout})


def rating_progression_multi(start_date, end_date, *athlete_names):
    """
    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking on every day between (inclusive) start_date and end_date
    :param increment:
    """

    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    # Get a list of dates called date_range within the start and end range
    increment = 1
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]
    fig = go.Figure()

    # Empty lists for chart lines
    ln_athlete_names = []
    ln_dates = []
    ln_pageranks = []
    ln_ranks = []

    # Empty lists for chart scatter points:
    sp_athlete_names = []
    sp_dates = []
    sp_pageranks = []
    sp_ranks = []
    sp_events = []
    sp_locations = []
    sp_places = []
    sp_field_sizes = []
    sp_distances = []

    for athlete_name in athlete_names:
        # Loop through each of the dates in date_range and look up the ranking for that date in the archive. Add the date
        # to one list and add the athlete's rank to a separate list. Count loops to track progress.
        athlete_name = athlete_name.title()
        loop_count = 0

        # get the data for the progression line:
        for date in date_range:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIR}/{file_name}")
            ranked_athletes = list(ranking_data.name)
            if athlete_name in ranked_athletes:
                ln_athlete_names.append(athlete_name)
                ln_dates.append(dt.strptime(date, "%m/%d/%Y"))
                pagerank_on_date = float(ranking_data["pagerank"][ranking_data.name == athlete_name])
                rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
                ln_pageranks.append(pagerank_on_date)
                ln_ranks.append(rank_on_date)
            loop_count += 1
            progress = loop_count / len(date_range)
            print(athlete_name + ": " + "{:.0%}".format(progress))

        # get the data for the scatter points representing performances:
        results_df = get_results(athlete_name)
        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_pageranks = []
        race_date_ranks = []

        for date in results_df.date:
            file_name = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
            ranking_data = pd.read_csv(f"{RANKINGS_DIR}/{file_name}")
            pagerank_on_date = float(ranking_data["pagerank"][ranking_data.name == athlete_name])
            rank_on_date = float(ranking_data["rank"][ranking_data.name == athlete_name])
            race_date_pageranks.append(pagerank_on_date)
            race_date_ranks.append(rank_on_date)

        results_df["pagerank_after_race"] = race_date_pageranks
        results_df["rank_after_race"] = race_date_ranks

        sp_athlete_names.extend(results_df["athlete_name"])
        sp_dates.extend(results_df["dt_date"])
        sp_pageranks.extend(race_date_pageranks)
        sp_ranks.extend(race_date_ranks)
        sp_events.extend(results_df["event"])
        sp_locations.extend(results_df["location"])
        sp_places.extend(results_df["place"])
        sp_field_sizes.extend(results_df["field_size"])
        dists = [str(dist) + "km" for dist in results_df["distance"]]
        sp_distances.extend(dists)

    print(len(ln_athlete_names))
    print(len(ln_dates))
    print(len(ln_pageranks))

    # Create a dataframe for the line traces:
    progress_dict = {
        "athlete_name": ln_athlete_names,
        "date": ln_dates,
        "pagerank": ln_pageranks,
        "rank": ln_ranks
    }
    progress_df = pd.DataFrame(progress_dict)
    xticks = [ln_date for ln_date in ln_dates if ln_date.day == 1]

    # Create a dataframe for the scatter traces:
    all_results_dict = {
        "athlete_name": sp_athlete_names,
        "date": sp_dates,
        "pagerank": sp_pageranks,
        "rank": sp_ranks,
        "event": sp_events,
        "location": sp_locations,
        "place": sp_places,
        "field_size": sp_field_sizes,
        "distance": sp_distances,
    }
    all_results_df = pd.DataFrame(all_results_dict)
    print(all_results_df)

    sp_fig = px.scatter(all_results_df, x="date", y="pagerank", color="event", hover_name="athlete_name",
                        hover_data=["date", "event", "location", "place", "field_size", "distance", "rank"])
    ln_fig = px.line(progress_df, x="date", y="pagerank", color="athlete_name", hover_data=["rank"])
    # ln_fig['data'][0]["line"]["shape"] = 'hv'
    fig = go.Figure(data=sp_fig.data + ln_fig.data)
    fig.update_layout(xaxis_title="Date", yaxis_title="Pagerank Value",
                      title=f"Rating Value Progression",
                      yaxis=dict(dtick=.01),
                      xaxis=dict(tickmode="array", tickvals=xticks))
    pyo.plot(fig, filename='plots/rating_progression_multi.html')


def show_results(athlete_name, as_of=dt.strftime(date.today(), "%m/%d/%Y")):
    rows = []

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            # calculate the weight of this race and add it to the row:
            age_weight = max(0, get_age_weight(race_data.date[0], as_of))
            comp_weight = get_comp_weight(race_data.event[0])
            dist_weight = get_distance_weight(race_data.distance[0])
            total_weight = age_weight * comp_weight * dist_weight
            row = race_data[race_data.athlete_name == athlete_name.title()]
            row["weight"] = total_weight
            # calculate the WR of the top 10 finishers, average it, and add it to the row:
            top_ten = names_list[0:min(10, len(names_list))]
            rank_file = f"{RANKINGS_DIR}/{alpha_date(as_of)}_{GENDER}_{RANK_DIST}km.csv"
            rank_df = pd.read_csv(rank_file)
            # top_ten_ranks = [int(rank_df["rank"][rank_df["name"] == name]) for name in top_ten
            #                  if name in list(rank_file["name"])]
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[0]]))
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[1]]))
            # print(int(rank_df["rank"][rank_df["name"] == top_ten[2]]))
            top_ten_ranks = []
            for name in top_ten:
                try:
                    rank = int(rank_df["rank"][rank_df["name"] == name])
                except TypeError:
                    pass
                else:
                    top_ten_ranks.append(rank)
            top_ten_avg = sum(top_ten_ranks) / len(top_ten_ranks)
            ranks_list_as_str = ", ".join([str(num) for num in top_ten_ranks])
            row["top 10 avg rank"] = ranks_list_as_str
            rows.append(row)

    df = pd.concat(rows, ignore_index=True)
    # df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    print(df)
    df.to_csv(f"{GENDER}/show_results/{athlete_name}.csv")


def get_results(athlete_name):
    rows = []

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            row = race_data[race_data.athlete_name == athlete_name.title()]
            rows.append(row)
            # print(file)
            # print(row)

    return pd.concat(rows, ignore_index=True)


def show_edges(graph, athlete1, athlete2):
    """
    :param graph: a previously created ranking graph
    :param athlete1: string
    :param athlete2: string
    :return: shows all edges (wins/losses) between to two athletes in the graph that is passed in
    """

    results_dict = {
        "winner": [],
        "event": [],
        "weight": []
    }

    athlete1 = athlete1.title()
    athlete2 = athlete2.title()

    try:
        athlete_one_wins = graph[athlete2][athlete1]["race_weights"]
    except KeyError:
        pass
    else:
        for (key, value) in athlete_one_wins.items():
            results_dict["winner"].append(athlete1)
            results_dict["event"].append(key)
            results_dict["weight"].append(value)

    try:
        athlete_two_wins = graph[athlete1][athlete2]["race_weights"]
    except KeyError:
        pass
    else:
        for (key, value) in athlete_two_wins.items():
            results_dict["winner"].append(athlete2)
            results_dict["event"].append(key)
            results_dict["weight"].append(value)

    df = pd.DataFrame(results_dict)
    print(df)


def print_race_labels():
    race_list = []

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_list.append(custom_label(results_file_path, "event", "location", "date", "distance") + "km")

    race_list.reverse()
    for race_label in race_list:
        print(race_label)


def compare_place_wr(results_file_path):
    race_data = pd.read_csv(results_file_path)
    race_date = race_data.date[0]
    rank_date = (dt.strptime(race_date, "%m/%d/%Y") - timedelta(days=1)).strftime("%m/%d/%Y")
    create_ranking(rank_date, comment=False)
    places = list(race_data.place)
    athletes = list(race_data.athlete_name)
    athletes = [athlete.title() for athlete in athletes]
    ranking = pd.read_csv(RANKING_FILE_NAME)

    graph_places = []
    graph_athletes = []
    graph_ranks = []

    for i in range(len(places)):
        try:
            rank = int(ranking["rank"][ranking["name"] == athletes[i]])
        except:
            pass
        else:
            graph_places.append(places[i])
            graph_athletes.append(athletes[i])
            graph_ranks.append(rank)

    place_wr_dict = {
        "name": graph_athletes,
        "place": graph_places,
        "rank": graph_ranks
    }

    df = pd.DataFrame(place_wr_dict)
    print(df)

    print(place_wr_dict)

    plt.plot(graph_places, graph_ranks, "o")
    plt.xlabel("Place")
    plt.ylabel("World Ranking")
    title = custom_label(results_file_path, "event", "location", "date", "distance") + "km"
    plt.title(f"{title}")
    plt.show()


def sum_of_edges(graph, athlete):
    weight_dict = {
    }

    for node in graph.nodes:
        # print(node)
        try:
            dict = graph[node][athlete]["race_weights"]
        except KeyError:
            pass
        else:
            # print(dict)
            for key, value in dict.items():
                if key in weight_dict.keys():
                    # pass
                    weight_dict[key] += value
                else:
                    weight_dict[key] = value

    new_dict = {
        "race": list(weight_dict.keys()),
        "sum_of_weights": list(weight_dict.values()),
    }

    df = pd.DataFrame(new_dict)
    print(df.sort_values(by="sum_of_weights", ascending=False).reset_index(drop=True))
    print(f"Sum of edge weights directed at {athlete}: {sum(weight_dict.values())}")


def horse_race_rank(start_date, end_date, num_athletes, increment, type="rank"):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    athlete_list = []

    for date in date_range:
        file_path = f"{RANKINGS_DIR}/{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
        df = pd.read_csv(file_path)
        all_athletes = list(df.name)
        selected_athletes = all_athletes[0:num_athletes + 1]
        for athlete in selected_athletes:
            if athlete not in athlete_list:
                athlete_list.append(athlete)

    horse_race_dict = {
        "Name": athlete_list
    }

    for date in date_range:
        chart_values = []
        file_path = f"{RANKINGS_DIR}/{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
        df = pd.read_csv(file_path)
        print(file_path)
        for athlete in athlete_list:
            try:
                if type == "ranking":
                    chart_value = int(df["rank"][df["name"] == athlete])
                elif type == "rating":
                    chart_value = float(df["pagerank"][df["name"] == athlete])
            except TypeError:
                if type == "ranking":
                    chart_value = 1000
                elif type == "rating":
                    chart_value = 0
            chart_values.append(chart_value)
        horse_race_dict[date] = chart_values

    df = pd.DataFrame(horse_race_dict)
    df.to_csv("horserace.csv")


def time_diffs(dist, athlete, comp_to_athlete):
    diffs = []

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if athlete in list(race_data.athlete_name) and comp_to_athlete in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                main_time = float(race_data["time"][race_data["athlete_name"] == athlete])
                comp_to_time = float(race_data["time"][race_data["athlete_name"] == comp_to_athlete])
                diff = round(main_time - comp_to_time, 2)
                if not math.isnan(diff):
                    diffs.append(diff)
    return diffs


def time_diffs2(dist, athlete, comp_to_athlete, date_for_weights=""):
    time_diffs = []
    outcomes = []
    races = []
    events = []
    field_sizes = []
    weights = []

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if athlete in list(race_data.athlete_name) and comp_to_athlete in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                main_time = float(race_data["time"][race_data["athlete_name"] == athlete])
                comp_to_time = float(race_data["time"][race_data["athlete_name"] == comp_to_athlete])
                diff = round(main_time - comp_to_time, 2)
                race = custom_label(f"{RESULTS_DIR}/{file}", "location", "event", "date", "distance") + "km"
                event = race_data.event[0]
                field_size = race_data.field_size[0]
                athlete_place = int(race_data.place[race_data.athlete_name == athlete])
                comp_athlete_place = int(race_data.place[race_data.athlete_name == comp_to_athlete])
                if athlete_place < comp_athlete_place:
                    outcome = "win"
                elif athlete_place > comp_athlete_place:
                    outcome = "lose"
                else:
                    outcome = "tie"
                if date_for_weights != "":
                    age_weight = max(0, get_age_weight(race_data.date[0], date_for_weights))
                    comp_weight = get_comp_weight(race_data.event[0])
                    dist_weight = get_distance_weight(race_data.distance[0])
                    total_weight = age_weight * comp_weight * dist_weight
                    weights.append(total_weight)
                if not (math.isnan(diff)):
                    time_diffs.append(diff)
                    races.append(race)
                    events.append(event)
                    field_sizes.append(field_size)
                    outcomes.append(outcome)

    diff_dict = {
        "competitor": [comp_to_athlete for i in range(len(time_diffs))],
        "time_diff": time_diffs,
        "outcome": outcomes,
        "race": races,
        "event": events,
        "field_size": field_sizes,
    }

    if date_for_weights != "":
        diff_dict["weight"] = weights

    # df = pd.DataFrame(diff_dict)
    # print(df)
    return diff_dict


def plot_time_diffs(dist, max_diff, athlete_name, *comp_athletes):
    """
    :param dist: number or "all"
    :param max_diff: in seconds, max/min shown on chart
    :param athlete_name: athlete you are comparing to all others in comp_athletes
    :param comp_athletes: athletes that athlete_name is being compared to
    :return: chart
    """

    all_names = []
    all_diffs = []
    all_hues = []
    sb.set_style("darkgrid")

    for comp_athlete in comp_athletes:
        diffs = time_diffs(dist, athlete_name, comp_athlete)
        if len(diffs) > 0:
            for diff in diffs:
                all_names.append(comp_athlete)
                all_diffs.append(diff)
                if diff > 0:
                    win_lose = "lose"
                else:
                    win_lose = "win"
                all_hues.append(win_lose)

    diff_dict = {
        "Competitor": all_names,
        f"Time Difference: {athlete_name} compared to competitors": all_diffs,
        f"Outcome for {athlete_name}": all_hues
    }

    df = pd.DataFrame(diff_dict)
    # print(df)
    chart = sb.stripplot(y="Competitor", x=f"Time Difference: {athlete_name} compared to competitors",
                         hue=f"Outcome for {athlete_name}", linewidth=1, size=7, data=df)
    if dist == "all":
        dist_subtitle = "all race distances"
    else:
        dist_subtitle = f"{dist}km races"
    chart.set(title=f"{athlete_name}'s time differential to various competitors\n{dist_subtitle}, +/- {max_diff}s")
    chart.set_xlim(-max_diff, max_diff)
    chart.invert_xaxis()
    plt.show()


def plot_time_diffs2(dist, athlete_name, *comp_athletes, date_for_weights=""):
    """
    :param date_for_weights:
    :param dist: number or "all"
    :param max_diff: in seconds, max/min shown on chart
    :param athlete_name: athlete you are comparing to all others in comp_athletes
    :param comp_athletes: athletes that athlete_name is being compared to
    :return: chart
    """

    competitors = []
    time_diffs = []
    outcomes = []
    races = []
    events = []
    field_sizes = []
    weights = []

    for comp_athlete in comp_athletes:
        diff_dict = time_diffs2(dist, athlete_name, comp_athlete, date_for_weights=date_for_weights)
        competitors.extend(diff_dict["competitor"])
        time_diffs.extend(diff_dict["time_diff"])
        outcomes.extend(diff_dict["outcome"])
        races.extend(diff_dict["race"])
        events.extend(diff_dict["event"])
        field_sizes.extend(diff_dict["field_size"])

    diff_dict = {
        "competitor": competitors,
        "time_diff": time_diffs,
        "outcome": outcomes,
        "race": races,
        "event": events,
        "field_size": field_sizes,
    }

    df = pd.DataFrame(diff_dict)
    print(df)

    fig = px.strip(df, x="time_diff", y="competitor", color="outcome", stripmode="overlay",
                   hover_data=["competitor", "time_diff", "race"])
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig.update_xaxes(title_text=f"{athlete_name}'s time compared to competitors (seconds)")
    fig.update_yaxes(title_text="Competitor")
    pyo.plot(fig, filename='plots/plot_time_diffs2.html')


def plot_wr_num_races(date, max_rank):
    athlete_list = []

    ranking_file = f"{alpha_date(date)}_{GENDER}_{RANK_DIST}km.csv"
    df = pd.read_csv(f"{RANKINGS_DIR}/{ranking_file}")

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            print(f"{file} not in range")
        else:
            print(f"{file} names added")
            athlete_list.extend(list(race_data.athlete_name))

    race_counts = [athlete_list.count(athlete) for athlete in df.name]
    race_counts_per_yr = [num / DEPRECIATION_PERIOD * 365 for num in race_counts]
    df['race_count_per_year'] = race_counts_per_yr

    df = df.iloc[:max_rank]

    fig = px.scatter(df, x="race_count_per_year", y="rank", hover_name="name")
    fig['layout']['yaxis']['autorange'] = "reversed"
    pyo.plot(fig)


def optimization_test(year_start_value, year_end_value, increment, dates_to_test):
    global DEPRECIATION_PERIOD
    global correct_predictions
    global total_tests
    dates_to_test = ["04/30/2022"]

    year_values = [year_start_value]
    keep_going = True
    while keep_going:
        new_num = year_values[-1] + increment
        if new_num <= year_end_value:
            year_values.append(new_num)
        else:
            year_values.append(year_end_value)
            keep_going = False

    dates = []
    opt_year_values = []
    opt_predict_values = []

    for date in dates_to_test:
        year_value_list = []
        predict_value_list = []
        for year_value in year_values:
            DEPRECIATION_PERIOD = 365 * year_value  # reset the depreciation period with new value to test
            year_value_list.append(year_value)
            print(f"date: {date}, years: {year_value}")
            predict_value_list.append(create_ranking(date, test=True))
            # reset the counters after every ranking created
            total_tests = 0
            correct_predictions = 0
        dates.append(date)
        opt_predict_value = max(predict_value_list)
        opt_year_value = year_value_list[predict_value_list.index(opt_predict_value)]
        opt_year_values.append(opt_year_value)
        opt_predict_values.append(opt_predict_value)

    opt_dict = {
        "date": dates,
        "years": opt_year_values,
        "predictability": opt_predict_values
    }

    df = pd.DataFrame(opt_dict)
    print(df)
    df.to_csv(
        f"{GENDER}/depreciation optimization {alpha_date(dates_to_test[0])} to {alpha_date(dates_to_test[-1])}.csv")


def num_one_consec_days():
    names = []
    num_days = []
    start_dates = []
    end_dates = []
    day_count = 0
    prev_date = ""

    for file in os.listdir(RANKINGS_DIR):
        print(file)
        results_file_path = os.path.join(RANKINGS_DIR, file)
        ranking_data = pd.read_csv(results_file_path)
        num_one = ranking_data.name[0]
        ranking_date = unalpha_date(file[:10])
        if len(names) == 0:
            # handle the first file in the directory
            names.append(num_one)
            day_count += 1
            start_dates.append(ranking_date)
            prev_date = ranking_date
        elif num_one == names[-1]:
            # no change in number one
            day_count += 1
        else:
            # if there's a change in number one
            num_days.append(day_count)
            end_dates.append(prev_date)
            names.append(num_one)
            day_count = 1
            start_dates.append(ranking_date)
        if file == os.listdir(RANKINGS_DIR)[-1]:
            # check if the last file in ranking archive
            num_days.append(day_count)
            end_dates.append(ranking_date)
        prev_date = ranking_date

        # print(names)
        # print(num_days)
        # print(start_dates)
        # print(end_dates)

    cd_dict = {
        "name": names,
        "days": num_days,
        "from": start_dates,
        "to": end_dates
    }

    df = pd.DataFrame(cd_dict)
    df = df.sort_values(by="days", ascending=False).reset_index(drop=True)
    print(df)
    df.to_csv(f"num_one_consecutive_days.csv")


def country_ranks(lowest_rank, as_of):
    create_ranking(as_of)
    df = pd.read_csv(RANKING_FILE_NAME).iloc[0:lowest_rank]
    ranking_names = list(df.name)
    ranking_ranks = list(df["rank"])
    ac_names = list(df.athlete_name)
    ac_countries = list(df.country)
    ranking_countries = []

    for name in ranking_names:
        i = ac_names.index(name)
        country = ac_countries[i]
        ranking_countries.append(country)

    dct = {
        "name": ranking_names,
        "country": ranking_countries,
        "rank": ranking_ranks
    }

    df = pd.DataFrame(dct)
    print(df)

    fig = px.strip(df, x="country", y="rank", hover_name="name")
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_layout(
        title={
            'text': f"Federations with {GENDER} athletes in the world top {lowest_rank}\n(as of {as_of})",
            'y': 0.95,
            'x': 0.5},
        xaxis_title="Federation",
        yaxis_title=f"World Ranking",
    )
    pyo.plot(fig, filename="plots/country_ranks.html")


def predicttest():
    start = time.time()
    global total_tests
    global correct_predictions
    global last_test_time
    global DEPRECIATION_PERIOD
    global K
    total_tests = 0
    correct_predictions = 0

    oldest_race_file = os.listdir(RESULTS_DIR)[0]
    oldest_race_date = unalpha_date(oldest_race_file[:10])
    oldest_race_date = dt.strptime(oldest_race_date, "%m/%d/%Y")

    for file in os.listdir(RESULTS_DIR):
        instance_total_tests = 0
        instance_correct_predictions = 0
        race_name = label(os.path.join(RESULTS_DIR, file))
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_date = race_data.date[0]
        race_dist = race_data.distance[0]
        race_dt_date = dt.strptime(race_date, "%m/%d/%Y")
        if (race_dt_date.date() - oldest_race_date.date()).days < (DEPRECIATION_PERIOD + 1) \
                or race_dist != RANK_DIST:
            pass
        else:
            day_before = race_dt_date - timedelta(days=1)
            create_ranking(dt.strftime(day_before, "%m/%d/%Y"))
            ranking_data = pd.read_csv(RANKING_FILE_NAME).iloc[(FROM_RANK - 1):TO_RANK]
            race_data = pd.read_csv(results_file_path)
            name_list = race_data.athlete_name.tolist()
            combos = list(combinations(name_list, 2))

            for matchup in combos:
                winner_name = matchup[0].title()
                loser_name = matchup[1].title()
                if winner_name in list(ranking_data.name) and loser_name in list(ranking_data.name):
                    winner_rank = int(ranking_data["rank"][ranking_data.name == winner_name])
                    loser_rank = int(ranking_data["rank"][ranking_data.name == loser_name])
                    total_tests += 1
                    instance_total_tests += 1
                    if winner_rank < loser_rank:
                        correct_predictions += 1
                        instance_correct_predictions += 1

            try:
                instance_predictability = instance_correct_predictions / instance_total_tests
            except ZeroDivisionError:
                # print(f"cannot calculate predictability for {race_name} -- cannot divide by 0")
                pass
            else:
                instance_predictability = "{:.0%}".format(instance_predictability)
                # print(f"Ranking predictability at {race_name}: {instance_predictability} ({instance_correct_predictions}/{instance_total_tests})")

    print(f"Correct predictions: {correct_predictions}")
    print(f"Total tests: {total_tests}")
    predictability = correct_predictions / total_tests
    predictability_txt = "{:.0%}".format(predictability)
    # print(f"Predictability: {predictability_txt}")
    print(f"predictability: {predictability}")
    print(f"Gender: {GENDER}")
    print(f"Distance: {RANK_DIST}km")
    print(f"Depreciation period: {DEPRECIATION_PERIOD / 365} years")
    print(f"Depreciation model: {DEPRECIATION_MODEL}")
    if DEPRECIATION_MODEL == "sigmoid":
        print(f"K value: {K}")
    elif DEPRECIATION_MODEL == "exponential":
        print(f"Lambda: {LAMBDA}")
    end = time.time()
    secs = round(end - start, 0)
    last_test_time = timedelta(seconds=secs)
    print(f"Single-test run time: {timedelta(seconds=secs)}")
    return predictability


def opttest(dp_start, dp_stop, k_start, k_stop, num):
    start = time.time()
    dps = []
    ks = []
    pvs = []
    global last_test_time
    global DEPRECIATION_PERIOD
    global K

    dps_to_test = list(np.linspace(dp_start, dp_stop, num))
    print(f"depreciation_period values (in years): {[round(num, 2) for num in dps_to_test]}")
    ks_to_test = list(np.linspace(k_start, k_stop, num))
    print(f"k values: {[round(num, 2) for num in ks_to_test]}")

    max_reps = len(dps_to_test) * len(ks_to_test)
    reps_done = 0

    for dp in dps_to_test:
        DEPRECIATION_PERIOD = dp * 365
        for k in ks_to_test:
            K = k
            ks.append(k)
            dps.append(dp)
            pvs.append(predicttest())
            reps_done += 1
            time_remaining = last_test_time * (max_reps - reps_done)
            print("{:.0%}".format(reps_done / max_reps) + " complete")
            print(f"estimated time remaining: {time_remaining}\n---------")

    dct = {
        "depreciation_period": dps,
        "k_value": ks,
        "predictability": pvs
    }

    df = pd.DataFrame(dct)
    df.to_csv(f"{GENDER} opttest {dt.now().strftime('%m-%d-%Y at %H-%M-%S')}.csv")

    end = time.time()
    secs = round(end - start, 0)
    print(f"Total run time: {timedelta(seconds=secs)}")

    pivoted = df.pivot("k_value", "depreciation_period", "predictability").sort_values(by="k_value", ascending=False)
    fig = px.imshow(pivoted, origin="lower")
    fig.show()


def opttest_vis(file_path):
    df = pd.read_csv(file_path)
    pivotted = df.pivot("k_value", "depreciation_period", "predictability").sort_values(by="k_value", ascending=False)
    fig = px.imshow(pivotted, origin="lower")
    fig.show()


def outcome_table(athlete_name, dist='all'):
    wins = 0
    podiums = 0
    top25pct = 0
    total_races = 0

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        if athlete_name in list(race_data.athlete_name):
            race_dist = race_data.distance[0]
            finishers = len(race_data.athlete_name)
            if race_dist == dist or dist == "all":
                total_races += 1
                athlete_place = int(race_data.place[race_data.athlete_name == athlete_name])
                if athlete_place == 1:
                    wins += 1
                if athlete_place <= 3:
                    podiums += 1
                if athlete_place / finishers <= 0.25:
                    top25pct += 1

    tiers = [wins, podiums, top25pct]

    dct = {
        'Outcome': ['Win', 'Podium', 'Top 25%'],
        'Count': [str(tier) + ' / ' + str(total_races) for tier in tiers],
        'Percentage': ["{:.0%}".format(num / total_races) for num in tiers]
    }

    df = pd.DataFrame(dct)
    print(athlete_name)
    print(df)


def archive_athlete_data(athlete_name, start_date, end_date, mode='overwrite', increment=1):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]
    csv_path = f"{ATHLETE_DATA_DIR}/{athlete_name}.csv"

    dates = []
    ranks = []
    ratings = []

    # loop through the ranking files and if the athlete is ranked, add the date, rating, and rank to separate lists,
    # then create a df to be written to csv.
    for d in date_range:
        file_name = f"{alpha_date(d)}_{GENDER}_{RANK_DIST}km.csv"
        ranking_data = pd.read_csv(f"{RANKINGS_DIR}/{file_name}")
        ranked_athletes = list(ranking_data.name)
        if athlete_name in ranked_athletes:
            dates.append(d)
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            rating_on_date = float(ranking_data["pagerank"][ranking_data.name == athlete_name])
            ranks.append(rank_on_date)
            ratings.append(rating_on_date)

    df = pd.DataFrame(dict(date=dates, rank=ranks, rating=ratings))

    if os.path.exists(csv_path) and mode == 'new':
        df.to_csv(csv_path, index=False)
    elif os.path.exists(csv_path) and mode == 'overwrite':
        existing_df = pd.read_csv(csv_path)
        existing_df['dt_date'] = [dt.strptime(d, "%m/%d/%Y") for d in existing_df.date]
        filtered_df = existing_df[(existing_df.dt_date < start_date) | (existing_df.dt_date > end_date)]
        df['dt_date'] = [dt.strptime(d, "%m/%d/%Y") for d in df.date]
        df2 = pd.concat([filtered_df, df])
        df2 = df2.sort_values(by="dt_date", ascending=True).reset_index(drop=True)
        df2 = df2.drop('dt_date', axis='columns')
        df2.to_csv(csv_path, index=False)
    elif os.path.exists(csv_path) and mode == 'append':
        df.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(csv_path, index=False)


def top(n):
    most_recent_ranking = os.listdir(RANKINGS_DIR)[-1]
    df = pd.read_csv(RANKINGS_DIR + "/" + most_recent_ranking).iloc[0:n]
    return df['name']


def lastracedate(athlete_name):
    """
    :param athlete_name:
    :return: datetime format of the date of the athlete's last (most recent) race
    """

    last_race_date = dt.strptime('01/01/2015', "%m/%d/%Y")

    for file in os.listdir(RESULTS_DIR):
        results_file_path = os.path.join(RESULTS_DIR, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        if athlete_name in list(race_data.athlete_name):
            if race_date > last_race_date:
                last_race_date = race_date

    return last_race_date


def archive_athlete_data_range(athlete_names, start_date, end_date, mode='overwrite', increment=1):
    count = 0
    if athlete_names == 'all':
        athlete_list = athlete_countries.athlete_name.unique()
    else:
        athlete_list = [athlete_names]

    for athlete in athlete_list:
        count += 1
        ttl_count = len(athlete_countries.athlete_name.unique())
        archive_athlete_data(athlete, start_date, end_date, mode=mode, increment=increment)
        print(f'{athlete} file saved ({count} / {ttl_count})')
        print(count / ttl_count)


def system_update(start_date, end_date=''):
    if end_date == '':
        end_date = start_date
    start = time.time()
    archive_rankings_range(start_date, end_date)
    archive_athlete_data_range('all', start_date, end_date, mode='overwrite', increment=1)
    end = time.time()
    ttl_time = str(end - start)
    print(f"Time to execute: {ttl_time}")


def race_accuracy(race_result_file):
    """
    :param race_result_file: a new race result csv file to compare against the ranking at that point in time
    :return: prints predictability of the ranking as of the day before the race
    """

    global FROM_RANK
    global TO_RANK

    instance_correct_predictions = 0
    instance_total_tests = 0

    race_data = pd.read_csv(race_result_file)
    race_date = dt.strptime(race_data['date'][0], '%m/%d/%Y')
    rank_date = race_date - timedelta(days=1)
    rank_file_name = f"{RANKINGS_DIR}/{dt.strftime(rank_date, '%Y_%m_%d')}_{GENDER}_{RANK_DIST}km.csv"
    ranking_data = pd.read_csv(rank_file_name).iloc[(FROM_RANK - 1):TO_RANK]

    name_list = race_data.athlete_name.tolist()
    combos = list(combinations(name_list, 2))

    for matchup in combos:
        winner_name = matchup[0].title()
        loser_name = matchup[1].title()
        if winner_name in list(ranking_data.name) and loser_name in list(ranking_data.name):
            winner_rank = int(ranking_data["rank"][ranking_data.name == winner_name])
            loser_rank = int(ranking_data["rank"][ranking_data.name == loser_name])
            instance_total_tests += 1
            if winner_rank < loser_rank:
                instance_correct_predictions += 1

    try:
        instance_predictability = instance_correct_predictions / instance_total_tests
        print(f"{rank_file_name} predictability at {race_result_file} = {instance_predictability}")
        print(f'total matchups = {instance_total_tests}')
        print(f'correct predictions = {instance_correct_predictions}')
    except ZeroDivisionError:
        print(f"cannot calculate predictability for {race_result_file} -- cannot divide by 0")
        pass


def name_correction(current_name, correct_name):
    # 1. Change name in all existing ranking files

    for file in os.listdir(RANKINGS_DIR):
        results_file_path = os.path.join(RANKINGS_DIR, file)
        ranking_data = pd.read_csv(results_file_path)
        names = list(ranking_data['name'])
        if current_name in names:
            names[names.index(current_name)] = correct_name
            ranking_data['name'] = names
            ranking_data.to_csv(results_file_path, index=False)
            print(f'{results_file_path} updated')
        else:
            pass

    # 2. Change athlete archive file with wrong name to correct name
    # 3. Correct row in athlete_countries with wrong name

def name_merge(name_to_merge, merge_into_name):
    """

    :param name_to_merge: The name to be deprecated and merged into merge_into_name.
    :param merge_into_name: The name that is already existing and is preferred. Rankings for name_to_merge will be
    merged into this name, and results with name_to_merge will be updated with this name.
    :return:
    """
    pass
# 1. get earliest date where name_to_merge appears in results and store in variable starting_rewrite_date
# 2. get latest date where name_to_merge appears in results and store in variable ending_rewrite_date
# 2. Change name_to_merge to merge_into_name in all results files
# 3. Perform system_update() from starting_rewrite_date to ending_rewrite_date


def create_splits_file(splits_dict):
    '''
    :param splits_dict: dictionary with keys as filename and values as list of split distances,
    including 0 and total distance
    :return:
    '''
    for filename in splits_dict.keys():
        filepath = os.path.join(RESULTS_DIR, filename)
        df = pd.read_csv(filepath)
        df = df[['Split1', 'Split2', 'Split3', 'Split4', 'Split5', 'Split6', 'Split7', 'Split8', 'Split9', 'Split10',
                 'Split11', 'Split12', 'Split13', 'Split14', 'Split15', 'Split16', 'Split17', 'Split18', 'Split19',
                 'Split20',
                 'Split21', 'Split22', 'Split23', 'Split24', 'Split25', 'Split26', 'Split27', 'Split28', 'Split29',
                 'Split30', ]]
        cols = ['Split0']
        for col in df.columns:
            if not math.isnan(df[col][0]):
                cols.append(col)
            else:
                pass

        if len(cols) - len(splits_dict[filename]) != 0:
            print(f'!!! {filename} missing {len(splits_dict[filename]) - len(cols)} splits')
        else:
            d = {
                'split': cols,
                'distance': splits_dict[filename]
            }

            df2 = pd.DataFrame(d)
            df2.to_csv(f'{SPLITS_DIR}/{filename}', index=False)
            print(f'{filename} created')


G = nx.DiGraph()
total_tests = 0
correct_predictions = 0
last_test_time = timedelta(seconds=3117)

system_update("02/05/2024", "02/28/2024")
# compare_place_wr('app_data/men/results/2023_05_20_GolfoAranci_10km_M.csv')
# race_accuracy('app_data/women/results/2023_05_13_Piombino_10km_W.csv')
# archive_athlete_data('Saleh Mohammad', '02/04/2018', '05/19/2023')

# files_with_splits = {
#     # '2022_03_24_Eilat_10km_M.csv': [0, 1250, 2500, 3750, 5000, 6250, 7500, 8250, 10000],
#     # '2021_05_13_Budapest_10km_M.csv': [],
#     # '2020_02_15_Doha_10km_M.csv': [],
#     # '2019_06_15_Balatonfured_10km_M.csv': [],
#     # '2019_05_12_Seychelles_10km_M.csv': [],
#     # '2018_11_09_AbuDhabi_10km_M.csv': [],
#     # '2018_06_16_Balatonfured_10km_M.csv': [],
#     # '2018_05_20_Seychelles_10km_M.csv': [],
#     # '2018_03_17_Doha_10km_M.csv': [],
#     # '2017_03_11_AbuDhabi_10km_M.csv': [],
#     # '2016_07_10_Hoorn_10km_M.csv': [],
#     # '2023_05_20_GolfoAranci_10km_M.csv': [0, 1666, 3332, 5000, 6666, 8332, 10000],
#     # '2022_11_12_Eilat_10km_M.csv': [0, 1666, 3332, 5000, 6666, 8332, 10000],
# }




