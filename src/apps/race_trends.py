import pandas as pd
import os
from datetime import datetime as dt

import plotly.graph_objs as go

from dash import html, dcc
from dash.dependencies import Input, Output
from app import app
import math

def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0]) + " "
    return race_label.strip()


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 12, 'display': 'block'}
dropdown_div_style = {'width': '100%', 'float': 'left', 'display': 'block'}
graph_style = {'width': '58%', 'display': 'block', 'float': 'left'}
summary_style = {'width': '100%', 'float': 'left', 'display': 'block'}
outcome_stats_style = {'width': '38%', 'display': 'block', 'float': 'left'}

trends_default_comp_to = 'leader'
trends_default_measure = 'time'
trends_default_gender = 'men'
trends_default_athlete = 'Ferry Weertman'
trends_default_names_list = pd.read_csv('app_data/' + trends_default_gender + "/athlete_countries.csv").sort_values('athlete_name')
trends_app_description = "Select an athlete of interest to see data from intermediate time checkpoints and " \
                         "understand how they compared to either the leader or median (middle of the pack) " \
                         "athlete at those checkpoints. In the 'Measure by' field, select whether you would like " \
                         "to see the comparison in terms of time (seconds) or positions. In the 'Compare to' " \
                         "field, select whether you want to compare to the leader, the median, or (if you have " \
                         "selected 'time' in the 'Measure by field') the average swimmer in the race. Data is only " \
                         "available from the 2017 (coming soon), 2019, and 2022 FINA World Championships. If an " \
                         "athlete is selected who did not compete in any of those events, the chart will not update."

name_style = {'fontFamily': 'helvetica', 'fontSize': 72, 'textAlign': 'left'}
summary_stats_style = {'fontFamily': 'helvetica', 'fontSize': 36, 'textAlign': 'left'}

layout = html.Div([
    html.Div(children=trends_app_description),
    dcc.RadioItems(id='trends-gender-picker',
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}], value=trends_default_gender,
                   persistence=True, persistence_type='session'),
    html.Div(dcc.Dropdown(id='trends-name-dropdown', persistence=True, persistence_type='session',
                          options=[{'label': i, 'value': i} for i in trends_default_names_list], value=trends_default_athlete),
             style=dropdown_div_style),
    html.Div([
            html.Label('Measure by:'),
            dcc.Dropdown(id='measure-dropdown', value=trends_default_measure, options=[{'label': i.title(), 'value': i} for i in ['time', 'position']],
                    persistence=True, persistence_type='session'),
            html.Label('Compare to:'),
            dcc.Dropdown(id='comp-to-dropdown', value=trends_default_comp_to, options=[{'label': i.title(), 'value': i} for i in ['leader', 'average', 'median']],
                    persistence=True, persistence_type='session'),
            ],
        style=input_dates_style),
    dcc.Loading(children=[dcc.Graph(id='trends-graph')], color="#119DFF", type="dot", fullscreen=True),
])


# Update names in dropdown list when a different gender is selected:
@app.callback(Output('trends-name-dropdown', 'options'),
              [Input('trends-gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    return [{'label': i, 'value': i} for i in names]

# Update comp-to dropdown options based on measure selected:
@app.callback(Output('comp-to-dropdown', 'options'),
              [Input('measure-dropdown', 'value')])
def adjust_comparison_options(measure):

    options_dict = {
        'time': [{'label': i.title(), 'value': i} for i in ['leader', 'average', 'median']],
        'position': [{'label': i.title(), 'value': i} for i in ['leader', 'median']],
    }

    return options_dict[measure]


@app.callback(
    Output('trends-graph', 'figure'),
    [Input('trends-gender-picker', 'value'),
     Input('trends-name-dropdown', 'value'),
     Input('comp-to-dropdown', 'value'),
     Input('measure-dropdown', 'value')])
def trend_graph(gender_choice, athlete_name, comp_to, measure):

    results_directory = 'app_data/' + gender_choice + "/results"
    splits_directory = 'app_data/' + gender_choice + "/splits"

    split_cols = ['Split' + str(i + 1) for i in range(-1, 30)]
    traces = []

    for file in os.listdir(results_directory):
        # First, check to see if this race even has splits data.
        if os.path.exists(os.path.join(splits_directory, file)):
            results_data = pd.read_csv(os.path.join(results_directory, file))
            if athlete_name in list(results_data['athlete_name']):
                # If so, create lists to be used in a dataframe to be created for each race for this athlete (where
                # there are splits available.
                split_labels = []
                split_dists = []
                athlete_times = []
                athlete_split_types = []
                leader_names = []
                leader_times = []
                leader_split_types = []
                avg_times = []
                median_times = []

                athlete_positions = []
                median_positions = []

                # Read in the split distances file for the race. Fill out the split_dists column and add Split0 to
                # results df.
                split_dist_data = pd.read_csv(os.path.join(splits_directory, file))
                split_dists.extend(split_dist_data['distance'])
                results_data['Split0'] = [0 for i in results_data['athlete_name']]
                results_data = results_data.set_index('athlete_name')

                race_dist = results_data['distance'][0]
                race_label = custom_label(os.path.join(results_directory, file), 'event', 'location', 'date', 'distance') + 'km'
                median_position = results_data['place'].median()

                # Make a copy of the results df and change the split values to True if there's an actual split, and
                # False if empty, meaning their timing chip didn't register. Fill out the split_labels list while
                # you're at it.
                split_bools = results_data.copy()
                for col in split_cols:
                    bools = [False if math.isnan(cell) else True for cell in results_data[col]]
                    split_bools[col] = bools
                    if bools.count(True) > 0:
                        split_labels.append(col)

                # Remove unnecessary columns in the results and bools dataframes.
                results_data = results_data[split_labels]
                split_bools = split_bools[split_labels]

                # Fill in missing splits in the results dataframe with estimates.
                for athlete in results_data.index:
                    df = pd.DataFrame(results_data.loc[athlete]).reset_index()
                    df.rename(columns={'index': 'split', athlete: 'time'}, inplace=True)
                    df['distance'] = split_dists
                    mod_splits = []
                    prev_split = 0
                    prev_dist = 0
                    for i in range(len(df)):
                        split_time = df['time'][i]
                        if math.isnan(split_time):
                            this_dist = df['distance'][i]
                            next_split = df['time'][i + 1]
                            next_dist = df['distance'][i + 1]
                            if math.isnan(next_split):
                                next_split_index = list(df['time']).index(df[i:]['time'].min())
                                next_split = df['time'][next_split_index]
                                next_dist = df['distance'][next_split_index]
                            distance_swam = this_dist - prev_dist
                            rate = (next_dist - prev_dist) / (next_split - prev_split)
                            time_swam = distance_swam / rate
                            estimated_split = prev_split + time_swam
                            mod_splits.append(estimated_split)
                            prev_split = estimated_split
                            prev_dist = df['distance'][i]
                        else:
                            mod_splits.append(split_time)
                            prev_split = df['time'][i]
                            prev_dist = df['distance'][i]

                    results_data.loc[athlete] = mod_splits

                # loop through each split and, referencing the results df and bools df, get all the data needed for the
                # figure.
                for split in split_labels:
                    if measure == 'time':
                        if split == 'Split0':
                            athlete_times.append(0)
                            athlete_split_types.append(True)
                            leader_names.append('N/A')
                            leader_times.append(0)
                            leader_split_types.append(True)
                            avg_times.append(0)
                            median_times.append(0)
                        else:
                            athlete_times.append(results_data[split][athlete_name])
                            athlete_split_types.append(split_bools[split][athlete_name])
                            leader_times.append(results_data[split].min())
                            leaders = results_data.index[results_data[split] == results_data[split].min()].tolist()
                            leader_split_type = False
                            for leader in leaders:
                                if split_bools[split][leader]:
                                    leader_split_type = True
                            leader_split_types.append(leader_split_type)
                            leader_name = ' / '.join(leaders)
                            leader_names.append(leader_name)
                            avg_times.append(results_data[split].mean())
                            median_times.append(results_data[split].median())
                    elif measure == 'position':
                        if split == 'Split0':
                            athlete_positions.append(1)
                            athlete_split_types.append(True)
                            leader_names.append('N/A')
                            leader_split_types.append(True)
                            median_positions.append(1)
                        else:
                            athlete_time = results_data[split][athlete_name]
                            sorted_split_times = list(results_data[split].sort_values(ascending=True))
                            athlete_position = sorted_split_times.index(athlete_time) + 1
                            athlete_positions.append(athlete_position)
                            athlete_split_types.append(split_bools[split][athlete_name])
                            leaders = results_data.index[results_data[split] == results_data[split].min()].tolist()
                            leader_split_type = False
                            for leader in leaders:
                                if split_bools[split][leader]:
                                    leader_split_type = True
                            leader_split_types.append(leader_split_type)
                            leader_name = ' / '.join(leaders)
                            leader_names.append(leader_name)
                            median_positions.append(median_position)

                if measure == 'time':
                    fig_df = pd.DataFrame({
                        'split': split_labels,
                        'distance': split_dists,
                        'distance_pct': [i / (race_dist * 1000) for i in split_dists],
                        'athlete_time': athlete_times,
                        'athlete_split_type': ['actual' if i else 'estimate' for i in athlete_split_types],
                        'leader': leader_names,
                        'leader_time': leader_times,
                        'leader_split_type': ['actual' if i else 'estimate' for i in leader_split_types],
                        'average_time': avg_times,
                        'median_time': median_times,
                    })
                    comp_time_col = comp_to + '_time'
                    fig_df['var'] = [fig_df['athlete_time'][i] - fig_df[comp_time_col][i] for i in range(len(fig_df))]
                elif measure == 'position':
                    fig_df = pd.DataFrame({
                        'split': split_labels,
                        'distance': split_dists,
                        'distance_pct': [i / (race_dist * 1000) for i in split_dists],
                        'athlete_position': athlete_positions,
                        'athlete_split_type': ['actual' if i else 'estimate' for i in athlete_split_types],
                        'leader': leader_names,
                        'leader_split_type': ['actual' if i else 'estimate' for i in leader_split_types],
                        'median_position': median_positions,
                    })
                    fig_df['leader_position'] = [1 for i in range(len(fig_df))]
                    comp_pos_col = comp_to + '_position'
                    fig_df['var'] = [fig_df['athlete_position'][i] - fig_df[comp_pos_col][i] for i in range(len(fig_df))]
                var_types = []
                for i in range(len(fig_df)):
                    if comp_to == 'leader':
                        if fig_df['athlete_split_type'][i] == 'actual' and fig_df['leader_split_type'][i] == 'actual':
                            type = 'actual'
                        else:
                            type = 'estimate'
                    else:
                        type = fig_df['athlete_split_type'][i]
                    var_types.append(type)
                fig_df['var_type'] = var_types

                print(fig_df)

                trace = go.Scatter(
                    x=fig_df['distance_pct'],
                    y=fig_df['var'],
                    mode='lines+markers',
                    name=race_label,
                    showlegend=True
                )
                traces.append(trace)

                # trace = px.line(fig_df,
                #                 x='distance_pct',
                #                 y='var')
                # traces.append(trace['data'])
                # print(trace)

    yaxis_titles = {
        'time': {
            'leader': "Time (seconds) Behind Race Leader",
            'average': "Time (seconds) Compared to Average Time",
            'median': "Time (seconds) Compared to Median Time"
        },
        'position': {
            'leader': "Positions Behind Race Leader",
            'median': "Position Compared to Median Swimmer"
        }
    }

    layout = go.Layout(
            title=f"Race Trends by {measure.title()}: {athlete_name}",
            xaxis={'title': 'Percent of Race Completed'},
            yaxis={'title': yaxis_titles[measure][comp_to], 'autorange': 'reversed'},
            hovermode='closest')

    fig = go.Figure(data=traces, layout=layout)

    return fig


