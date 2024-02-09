from dash import html, dcc
from dash.dependencies import Input, Output
# Connect to main app.py file
from app import app
# Connect to your app pages
from apps import profile, head2head, progressions, rankings, race_trends, results


link_style = {'display': 'inline', 'padding-left': '40px', 'padding-right': '40px'}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('RANKINGS', href='/apps/rankings'),
        dcc.Link('ATHLETE PROFILES', href='/apps/profile'),
        dcc.Link('RANKING PROGRESSIONS', href='/apps/progressions'),
        dcc.Link('RACE TRENDS', href='/apps/race_trends'),
        dcc.Link('HEAD-TO-HEAD', href='/apps/head2head'),
        dcc.Link('RESULTS', href='/apps/results'),
    ], className="row", style=link_style),
    html.Div(id='page-content', children=[], style={'padding-left': '40px', 'padding-right': '40px'})
])

default = html.Div(
    [html.Div('Please click on a tool above to get started!'),
    html.Ul([
        html.Li("RANKINGS: See current and historical world rankings, and how they've changed over time."),
        html.Li('ATHLETE PROFILES: Look up summary stats, world ranking, and results for a single athlete.'),
        html.Li('RANKING PROGRESSIONS: Compare ranking progressions of multiple athletes in one figure.'),
        html.Li('RACE TRENDS: See where in the pack an athlete was positioned at various points throughout a race.'),
        html.Li('HEAD-TO-HEAD: Match up two athletes to see a win/loss record and finish time differences.'),
        html.Li('RESULTS: See race results details.'),
    ])]
)

layouts = {
    '/apps/head2head': head2head.layout,
    '/apps/progressions': progressions.layout,
    '/apps/profile': profile.layout,
    '/apps/rankings': rankings.layout,
    '/apps/race_trends': race_trends.layout,
    '/apps/results': results.layout,
    '/': default
}


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    return layouts[pathname]


if __name__ == '__main__':
    app.run_server(debug=False)