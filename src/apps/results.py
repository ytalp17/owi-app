import pandas as pd
import os
from datetime import timedelta
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app


def custom_label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for arg in args:
        race_label = race_label + str(race_data[arg][0])
        if arg == 'distance':
            race_label = race_label + "km "
        else:
            race_label = race_label + " "
    return race_label.strip()


layout = html.Div([
    dcc.RadioItems(id='results-gender-picker',
                   options=[{'label': 'Men', 'value': 'men'}, {'label': 'Women', 'value': 'women'}],
                   labelStyle={'margin-left': '20px'},
                   # value='men',
                   persistence=True, persistence_type='session'),
    dcc.Dropdown(id='results-race-dropdown', persistence=True, persistence_type='session'),
    html.Div(id='results-output-table', children='this is where the results table will go')
])


@app.callback(Output('results-race-dropdown', 'options'),
              [Input('results-gender-picker', 'value')])
def list_races(gender_choice):
    results_path = 'app_data/' + gender_choice + '/results'
    race_choices = os.listdir(results_path)

    return [{'label': custom_label('app_data/' + gender_choice + '/results/' + i, 'event', 'location', 'distance',
                                   'date'), 'value': i} for i in race_choices]


@app.callback(Output('results-output-table', 'children'),
              [Input('results-gender-picker', 'value'),
               Input('results-race-dropdown', 'value')])
def insert_results(gender_choice, race_choice):
    results_data = pd.read_csv('app_data/' + gender_choice + "/results/" + race_choice)
    results_data = results_data[['place', 'athlete_name', 'country', 'time']]
    results_data['time'] = [str(timedelta(seconds=time_in_secs)) for time_in_secs in results_data['time']]
    data = results_data.to_dict('rows')
    columns = [{"name": i, "id": i, } for i in results_data.columns]
    # columns = [
    #     {"name": 'Date', "id": 'dt_date'},
    #     {"name": 'Event', "id": 'event'},
    #     {"name": 'Location', "id": 'location'},
    #     {"name": 'Distance (km)', "id": 'distance'},
    #     {"name": 'Place', "id": 'place'},
    #     {"name": 'Field Size', "id": 'field_size'},
    # ]
    table = [dash_table.DataTable(data=data, columns=columns, sort_action="native", sort_mode="multi",
                                  style_as_list_view=True)]
    return table


