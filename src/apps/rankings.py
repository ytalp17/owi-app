import pandas as pd
from datetime import datetime as dt
from datetime import date
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from app import app
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import variables as variables



# MM/DD/YYYY string format of today's date
# today = dt.strftime(date.today(), "%m/%d/%Y")
today = '09/30/2022'


# Style dictionaries for dashboard elements:
input_dates_style = {'fontFamily': 'helvetica', 'fontSize': 12, 'display': 'block'}
title_style = {'width': '35%', 'float': 'left', 'display': 'block', 'fontFamily': 'helvetica', 'textAlign': 'center'}
table_div_style = {'width': '35%', 'float': 'center', 'display': 'block'}
date_display_format = 'Y-M-D'

table_formatting = (
            [
                {'if': {
                        'filter_query': '{change} contains "▲"',
                        'column_id': 'change'
                    },
                    'color': 'green'
                },
                {'if': {
                        'filter_query': '{change} contains "▼"',
                        'column_id': 'change'
                    },
                    'color': 'red'
                }
            ]
)

default_ranking_date = '09/30/2022'
default_comparison_date = ''
default_gender = 'women'
rankings_app_description = "See men's and women's world rankings as of any date after 01/01/2017. Select a previous " \
                           "date as the Comparison Date to see every athlete's change in ranking between the two " \
                           "dates selected."

# if you change these, change the table_formatting above to match!
up_arrow = '▲'
down_arrow = '▼'

#default_color
#xs, sm, md, lg, xl and xxl



layout = dmc.MantineProvider(dmc.MantineProvider(
    theme= variables.THEME,
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        
    html.Div([
        dbc.Row([
            dbc.Col([
                dmc.Text(rankings_app_description, color="dimmed",align="left", size='xl')
                ]),
        ]),
        dbc.Row([
            dbc.Col([
                dmc.SegmentedControl(id='rankings-gender-picker', 
                                      value=default_gender,
                                      orientation='horizontal',
                                      size='sm',
                                      radius='xl',
                                      #color= variables.COLOR_CYAN,
                                      data = [{'label': 'Men', 'value': 'men'}, 
                                              {'label': 'Women', 'value': 'women'}])
                ],width=2, xs=11, sm=9, md=7, lg=5, xl=4, xxl= 3),
                ], justify='center'),
            

    
    dcc.RadioItems(id='rankings-gender-picker', value=default_gender,
                   options=[],
                   persistence=True, persistence_type='session', labelStyle={'margin-right': '20px'}),
    html.Div([
            html.Label('Ranking Date'),
            dcc.DatePickerSingle(id='ranking-date', date=date(2024, 1, 1), display_format=date_display_format,
                                 clearable=False, persistence=True, persistence_type='session',
                                 max_date_allowed=date.today(),
                                 min_date_allowed=date(2017, 1, 1)),
            html.Label('Comparison Date'),
            dcc.DatePickerSingle(id='comparison-date', display_format=date_display_format, clearable=True,
                                 persistence=True, persistence_type='session',
                                 min_date_allowed=date(2017, 1, 1)),
            ], style=input_dates_style),
    html.Div([
            html.H2(id='title-main'),
            html.H3(id='title-date'),
            html.P(id='title-change')
            ], style=title_style),
    html.Div(id='rankings-table', style=table_div_style)
])
        
        ],
))



@app.callback(
    [Output('comparison-date', 'max_date_allowed'),
     Output('comparison-date', 'initial_visible_month')],
    Input('ranking-date', 'date')
)
def set_max_comp_date(max_comp_date):
    return max_comp_date, max_comp_date

@app.callback(
    Output('rankings-table', 'children'),
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'date'),
     Input('comparison-date', 'date')])
def update_ranking(gender_choice, rank_date, comp_date):
    print(f'rank date is {rank_date}')
    athlete_countries = pd.read_csv('app_data/' + gender_choice + "/athlete_countries.csv")
    if comp_date is None:
        rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
        rank_file = f"{rankings_directory}/{rank_date.replace('-', '_')}_{gender_choice}_10km.csv"
        rank_df = pd.read_csv(rank_file)
        rank_df["country"] = [athlete_countries['country'][athlete_countries['athlete_name'] == athlete_name]
                              for athlete_name in rank_df['name']]
        table_df = rank_df[['rank', 'name', 'country']]
        data = table_df.to_dict('rows')
        columns = [{"name": i.title(), "id": i, } for i in table_df.columns]
        table = [dash_table.DataTable(data=data, columns=columns, page_size=100)]
    elif dt.strptime(comp_date, "%Y-%m-%d") >= dt.strptime(rank_date, "%Y-%m-%d"):
        table = "Please choose a comparison date that is prior to the ranking date chosen!"
    else:
        rankings_directory = 'app_data/' + gender_choice + "/rankings_archive"
        rank_file = f"{rankings_directory}/{rank_date.replace('-', '_')}_{gender_choice}_10km.csv"
        rank_df = pd.read_csv(rank_file)
        rank_df["country"] = [athlete_countries['country'][athlete_countries['athlete_name'] == athlete_name]
                              for athlete_name in rank_df['name']]
        comp_file = f"{rankings_directory}/{comp_date.replace('-', '_')}_{gender_choice}_10km.csv"
        comp_df = pd.read_csv(comp_file)

        changes = []

        for athlete_name in rank_df['name']:
            if athlete_name in list(comp_df['name']):
                prev_rank = int(comp_df['rank'][comp_df['name'] == athlete_name])
                rank = int(rank_df['rank'][rank_df['name'] == athlete_name])
                rank_change = prev_rank - rank
                if rank_change == 0:
                    rank_change = '-'
                elif rank_change > 0:
                    rank_change = f"{up_arrow}{rank_change}"
                else:
                    rank_change = f"{down_arrow}{abs(rank_change)}"
            else:
                rank_change = '-'
            changes.append(rank_change)

        rank_df['change'] = changes
        table_df = rank_df[['rank', 'name', 'country', 'change']]
        data = table_df.to_dict('rows')
        columns = [{"name": i.title(), "id": i, } for i in table_df.columns]
        table = [dash_table.DataTable(data=data, columns=columns, page_size=100,
                                      style_data_conditional=table_formatting,)]

    return table


@app.callback(
    [Output('title-main', 'children'),
     Output('title-date', 'children'),
     Output('title-change', 'children')],
    [Input('rankings-gender-picker', 'value'),
     Input('ranking-date', 'date'),
     Input('comparison-date', 'date')])
def update_title(gender_choice, rank_date, comp_date):

    if comp_date is None:
        title_main = f"{gender_choice}'s 10km Open Water World Rankings"
        title_main = title_main.upper()
        title_date = dt.strftime(dt.strptime(rank_date, "%Y-%m-%d"), "%d %B, %Y")
        title_change = ''
    elif dt.strptime(comp_date, "%Y-%m-%d") > dt.strptime(rank_date, "%Y-%m-%d"):
        title_main = ''
        title_date = ''
        title_change = ''
    else:
        title_main = f"{gender_choice}'s Open Water World Rankings"
        title_main = title_main.upper()
        title_date = dt.strftime(dt.strptime(rank_date, "%Y-%m-%d"), "%d %B, %Y")
        formatted_comp_date = dt.strftime(dt.strptime(comp_date, "%Y-%m-%d"), "%d %B, %Y")
        title_change = f"(change since {formatted_comp_date})"

    return title_main, title_date, title_change
