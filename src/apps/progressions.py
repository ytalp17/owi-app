import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta, date
import plotly.graph_objs as go
from dash import html, dcc
from dash.dependencies import Input, Output
from app import app
import dash_bootstrap_components as dbc

# MM/DD/YYYY string format of today's date
today = dt.strftime(date.today(), "%Y-%m-%d")


def alpha_date(date):
    """
    :param date: MM/DD/YYYY
    :return: YYYY_MM_DD
    """
    date = date.replace("/", "_")
    alphadate = date[6:] + "_" + date[:5]
    return alphadate


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 12, 'display': 'block'}
dropdown_div_style = {'width': '100%', 'float': 'left', 'display': 'block'}
graph_style = {'width': '58%', 'display': 'block', 'float': 'left'}
stats_style = {'width': '38%', 'display': 'block', 'float': 'left'}

# Defaults:
prog_default_start_date = '2022-01-01'
prog_default_end_date = dt.strftime(date.today(), "%Y-%m-%d")
prog_default_gender_choice = 'women'
prog_default_names = ['Leonie Beck', 'Sharon Van Rouwendaal', 'Ana Marcela Cunha']
prog_default_mode = 'rating'
prog_date_display_format = 'Y-M-D'
prog_app_description = "Select multiple athletes to see either their ranking or rating progressions " \
                       "over time compared to each other. The rating value is an arbitrary number produced " \
                       "by the ranking algorithm that is used to create the ranking, and cannot be compared " \
                       "across men and women. This tool, especially when displaying rating values, allows a " \
                       "user to see how close or how far an athlete is from surpassing or being surpassed by " \
                       "other athletes in the ranking pool."

layout = html.Div([
    html.Div(children=prog_app_description),
    dcc.RadioItems(id='progressions-gender-picker',
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   persistence=True, persistence_type='session', value=prog_default_gender_choice,
                   labelStyle={'margin-right': '20px'}),
    html.Div(dcc.Dropdown(id='progressions-name-dropdown',
                          multi=True, persistence=True, persistence_type='session', value=prog_default_names),
             style=dropdown_div_style),
    dbc.Row(
        [dbc.Col(html.Label('Select mode: '), width={'size': 1, 'offset': 2}, style={'font-weight': 'bold'}),
         dbc.Col(dcc.RadioItems(id='mode-selector',
                                options=[{'label': 'Rating', 'value': 'rating'}, {'label': 'Ranking', 'value': 'rank'}],
                                persistence=True, persistence_type='session', value=prog_default_mode,
                                labelStyle={'margin-right': '20px'}), width={'size': 2, 'offset': 0}),
         dbc.Col(html.Label('Select time range: '), width={'size': 1, 'offset': 1}, style={'font-weight': 'bold'}),
         dbc.Col(dcc.RadioItems(id='time-range-picker',
                                options=[{'label': i, 'value': i} for i in
                                         ['All time', '1 year', '6 months', '1 month']],
                                value='1 year', labelStyle={'margin-left': '20px'}
                                ), width={'size': 4, 'offset': 0})], style={'margin-top': '20px'}),
    dcc.Loading(children=[dcc.Graph(id='multi-progression-graph')], color="#119DFF", type="dot", fullscreen=False),
])


# Create / update ranking progression graph:
@app.callback(
    Output('multi-progression-graph', 'figure'),
    [Input('time-range-picker', 'value'),
     Input('progressions-name-dropdown', 'value'),
     Input('progressions-gender-picker', 'value'),
     Input('mode-selector', 'value')])
def ranking_progression(time_range, athlete_names, gender_choice, mode):
    rating_multiplier = 10000
    athlete_names = list(athlete_names)
    increment = 1
    rank_dist = 10
    rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
    results_directory = 'app_data/' + gender_choice + "/results"
    athlete_data_directory = 'app_data/' + gender_choice + "/athlete_data"
    global today
    if time_range == 'All time':
        start_date = dt.strptime('2017-1-1', "%Y-%m-%d")
        end_date = dt.strptime(today, '%Y-%m-%d')
    elif time_range == '1 year':
        start_date = dt.strptime(today, "%Y-%m-%d") - timedelta(days=365)
        end_date = dt.strptime(today, '%Y-%m-%d')
    elif time_range == '6 months':
        start_date = dt.strptime(today, "%Y-%m-%d") - timedelta(days=180)
        end_date = dt.strptime(today, '%Y-%m-%d')
    elif time_range == '1 month':
        start_date = dt.strptime(today, "%Y-%m-%d") - timedelta(days=30)
        end_date = dt.strptime(today, '%Y-%m-%d')
    # start_date = dt.strptime(start_date, "%Y-%m-%d")
    # end_date = dt.strptime(end_date, "%Y-%m-%d")
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    traces = []
    scatter_df = pd.DataFrame()
    if mode == 'rating':
        col = 'pagerank'
    elif mode == 'rank':
        col = 'rank'
    else:
        col = 'balls'

    # Loop through each athlete and grab their progression csv. Create a dataframe, add a datetime format date column,
    # and filter it to include only rows in the selected date range. Then make a ranking and rating trace for them.
    # If no csv available, calculate the old way by looping through dated ranking files.
    for athlete_name in athlete_names:
        athlete_name = athlete_name.title()
        if os.path.exists(f"{athlete_data_directory}/{athlete_name}.csv"):
            df = pd.read_csv(f"{athlete_data_directory}/{athlete_name}.csv")
            df['dt_date'] = [dt.strptime(d, "%m/%d/%Y") for d in df['date']]
            df = df[df['dt_date'] >= start_date]
            df = df[df['dt_date'] <= end_date]
            if mode == 'rating':
                df['rating'] = [i * rating_multiplier for i in df['rating']]

            # Use the columns of the modified dataframe to construct a trace for the rank line and the rating line.
            # Add the traces to the list that gets passed into the figure data parameter and returned to the graph.

            trace = go.Scatter(x=df['dt_date'],
                               y=df[mode],
                               mode='lines',
                               opacity=0.8,
                               name=athlete_name)
            traces.append(trace)


        else:
            pass
            # rank_dates = []
            # ranks = []
            # ratings = []
            #
            # # For each date in the date range, look up the athlete's rank and rating on that date and create a trace for each.
            # # Then add those traces to the respective lists to be used in the data parameter of each of the figures
            # for date in date_range:
            #     file_name = f"{alpha_date(date)}_{gender_choice}_{rank_dist}km.csv"
            #     ranking_data = pd.read_csv(f"{rankings_directory}/{file_name}")
            #     ranked_athletes = list(ranking_data.name)
            #     if athlete_name in ranked_athletes:
            #         rank_dates.append(dt.strptime(date, "%m/%d/%Y"))
            #         rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            #         rating_on_date = float(ranking_data["pagerank"][ranking_data.name == athlete_name])
            #         ranks.append(rank_on_date)
            #         ratings.append(rating_on_date)
            # ratings = [i * rating_multiplier for i in ratings]
            # trace = go.Scatter(x=rank_dates,
            #                              y=ranks,
            #                              mode='lines',
            #                              opacity=0.8,
            #                              name=athlete_name)
            # traces.append(trace)
            # rating_line_trace = go.Scatter(x=rank_dates,
            #                                y=ratings,
            #                                mode='lines',
            #                                opacity=0.8,
            #                                name=athlete_name)
            # rating_traces.append(rating_line_trace)

        # Create the scatter traces.
        # this block is the get_results function from main.py basically. Get a df of all the times the athlete raced.
        rows = []
        for file in os.listdir(results_directory):
            results_file_path = os.path.join(results_directory, file)
            race_data = pd.read_csv(results_file_path)
            names_list = list(race_data.athlete_name)
            names_list = [name.title() for name in names_list]
            race_data.athlete_name = names_list
            if athlete_name.title() in names_list:
                row = race_data[race_data.athlete_name == athlete_name.title()]
                rows.append(row)
        results_df = pd.concat(rows, ignore_index=True)

        # add dt_date column and filter down to dates within selection
        results_df["dt_date"] = [dt.strptime(date, "%m/%d/%Y") for date in results_df.date]
        results_df = results_df[results_df.dt_date >= start_date]
        results_df = results_df[results_df.dt_date <= end_date]
        race_date_vals = []

        # for dates where athlete raced, look up their rank/rating on that date and create another column in the
        # results_df for that value. Then, concatenate the athlete's results_df with the running overall scatter_df.
        for d in results_df.date:
            file_name = f"{alpha_date(d)}_{gender_choice}_{rank_dist}km.csv"
            df = pd.read_csv(f"{rankings_directory}/{file_name}")
            if mode == 'rank':
                val_on_date = int(df[col][df.name == athlete_name])
            else:
                val_on_date = float(df[col][df.name == athlete_name]) * rating_multiplier
            race_date_vals.append(val_on_date)
        results_df[col] = race_date_vals
        scatter_df = pd.concat([scatter_df, results_df])

    for event_type in scatter_df['event'].unique():
        df = scatter_df[scatter_df['event'] == event_type]
        scatter_trace = go.Scatter(x=df['dt_date'],
                                   y=df[col],
                                   mode='markers',
                                   marker={'size': 10, 'line': {'width': 0.5, 'color': 'black'}},
                                   name=event_type)
        traces.append(scatter_trace)

    if mode == 'rank':
        fig = {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Date'},
                yaxis={'title': 'World Ranking', 'autorange': 'reversed'},
                hovermode='closest')
        }
    else:
        fig = {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'Date'},
                yaxis={'title': 'Rating Value'},
                hovermode='closest')
        }

    return fig


# Update names in dropdown list when a different gender is selected:
@app.callback(Output('progressions-name-dropdown', 'options'),
              [Input('progressions-gender-picker', 'value')])
def list_names(gender_choice):
    df = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    names = df['athlete_name'].unique()
    return [{'label': i, 'value': i} for i in names]
