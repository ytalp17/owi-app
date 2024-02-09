import dash
import dash_bootstrap_components as dbc

# meta_tags are required for the app layout to be mobile responsive
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Open Water Insights",
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                     "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;900&display=swap"
                                    ],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                )
server = app.server
