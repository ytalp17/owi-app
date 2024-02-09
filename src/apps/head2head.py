import pandas as pd
import os
import math
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app

def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()

def age_opacity(race_date, oldest_date):
    min_opacity = .25
    max_opacity = .75
    today = date.today()
    period = (today - oldest_date.date()).days
    days_old = (race_date.date() - oldest_date.date()).days
    depreciation = days_old / period
    opacity = max_opacity - depreciation*(max_opacity - min_opacity)
    return opacity

h2h_default_gender = 'men'
h2h_default_athlete1 = 'Gregorio Paltrinieri'
h2h_default_athlete2 = 'Domenico Acerenza'

score_style = {'fontFamily': 'helvetica', 'fontSize': 96, 'textAlign': 'center'}
dropdown_div_style = {'width': '50%', 'float': 'left', 'display': 'block'}
head2head_app_description = "Select the names of two athletes to compare. The score at the top shows total wins " \
                            "(relative to each other) for each athlete, the graph shows the difference in finish " \
                            "time between the athletes for the races where data is available, and the table shows " \
                            "all the races where the athletes competed against each other along with some info " \
                            "about the race."

layout = html.Div([
    html.Div(children=head2head_app_description),
    dcc.RadioItems(id='gender-picker', value=h2h_default_gender,
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   persistence=True, persistence_type='session'),
    html.Div([
        html.Div(dcc.Dropdown(id='name-dropdown1', persistence=True, persistence_type='session', value=h2h_default_athlete1),
                 style=dropdown_div_style),
        html.Div(dcc.Dropdown(id='name-dropdown2', persistence=True, persistence_type='session', value=h2h_default_athlete2),
                 style=dropdown_div_style)]),
        html.H1(id='score', style=score_style),
        dcc.Graph(id='diff-graph'),
        html.H2('Competitive History:'),
        html.Div(id='table')
    ])


# Update names in both dropdown lists when a different gender is selected:
@app.callback([Output('name-dropdown1', 'options'),
               Output('name-dropdown2', 'options')],
              [Input('gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    names_list = [{'label': i, 'value': i} for i in names]
    return names_list, names_list


# Update the figure:
@app.callback([Output('table', 'children'),
               Output('score', 'children'),
               Output('diff-graph', 'figure')],
              [Input('name-dropdown1', 'value'),
               Input('name-dropdown2', 'value'),
               Input('gender-picker', 'value')])
def update(name1, name2, gender_choice):
    dist = 'all'
    results_directory = 'app_data/' + gender_choice + "/results"
    winners = []
    winner_places = []
    loser_places = []
    diffs = []
    races = []
    dates = []
    distances = []

    for file in os.listdir(results_directory):
        results_file_path = os.path.join(results_directory, file)
        race_data = pd.read_csv(results_file_path)
        race_dist = race_data.distance[0]
        if name1 in list(race_data.athlete_name) and name2 in list(race_data.athlete_name):
            if race_dist == dist or dist == "all":
                race_name = custom_label(os.path.join(results_directory, file), "event", "location")
                race_date = race_data["date"][0]
                distance = race_data['distance'][0]
                races.append(race_name)
                dates.append(race_date)
                distances.append(distance)
                name1time = float(race_data["time"][race_data["athlete_name"] == name1])
                name2time = float(race_data["time"][race_data["athlete_name"] == name2])
                diff = round(name1time - name2time, 2)
                name1place = int(race_data.place[race_data.athlete_name == name1])
                name2place = int(race_data.place[race_data.athlete_name == name2])
                if name1place < name2place:
                    winners.append(name1)
                    winner_places.append(name1place)
                    loser_places.append(name2place)
                elif name1place > name2place:
                    winners.append(name2)
                    winner_places.append(name2place)
                    loser_places.append(name1place)
                else:
                    winners.append('Tie')
                    winner_places.append(name1place)
                    loser_places.append(name2place)
                if not math.isnan(diff):
                    diffs.append(diff)
                else:
                    diffs.append('N/A')

    diff_dict = {
        'winner': winners,
        'winner_place': winner_places,
        'loser_place': loser_places,
        'time_diff': diffs,
        'race': races,
        'date': dates,
        'distance (km)': distances,
    }

    print(diffs)
    print(races)
    print([type(i) for i in diffs])

    table_df = pd.DataFrame(diff_dict)
    table_df['time_diff'] = [str(timedelta(seconds=abs(i))) if i != 'N/A' else 'N/A' for i in table_df['time_diff']]
    table_df = table_df.sort_values('date')
    print(table_df)
    data = table_df.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in table_df.columns]
    score = f"{str(winners.count(name1))} - {str(winners.count(name2))}"
    if winners.count("Tie") > 0:
        score = score + f" - {str(winners.count('Tie'))}"

    # create the data and layout for output to figure parameter in graph:

    fig_df = pd.DataFrame(diff_dict)
    fig_df = fig_df[fig_df.time_diff != 'N/A'].reset_index(drop=True)
    fig_df['dt_date'] = [dt.strptime(i, "%m/%d/%Y") for i in fig_df['date']]
    # oldest_date = min(fig_df['dt_date'])
    # fig_df['opacity'] = [age_opacity(i, oldest_date) for i in fig_df['dt_date']]
    # print(type(date.today()))
    # print(type(fig_df['dt_date'][0]))
    # fig_df['days_old'] = [(date.today() - d).days for d in [dt.strptime(i, "%m/%d/%Y").date() for i in fig_df['date']]]
    print(fig_df)

    chart_data = []
    colors = {
        name1: 'blue',
        name2: 'green',
        "Tie": 'orange'
    }
    unique_winners = list(fig_df['winner'].unique())

    # think i should try plotly.express for this because for loop below is based on traces (unique winners / tie),
    # not rows of fig_df

    for i in unique_winners:
        df = fig_df[fig_df['winner'] == i]
        trace = go.Scatter(
                    x=df['time_diff'],
                    y=['str' for i in df['time_diff']],
                    mode='markers',
                    marker={'size': 20, 'line': {'width': 2, 'color': 'black'}, 'opacity': 0.4, 'color': colors[i]},
                    name=i
    )
        chart_data.append(trace)

    rangemax = max([abs(i) for i in fig_df['time_diff'] if i != 'N/A']) * 1.1

    layout = go.Layout(
        xaxis={'title': 'Finish Time Difference (s)', 'range': [-rangemax, rangemax]},
        yaxis={'showticklabels': False},
        legend_title_text='Winner:',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=.5
        )
    )


    fig = {
        'data': chart_data,
        'layout': layout
    }

    return [dash_table.DataTable(data=data, columns=columns)], score, fig