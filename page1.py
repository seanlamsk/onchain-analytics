import dash
from dash import dcc
from dash import html

class page1:

    def __init__(self,app):
        self.app = app

        self.page_1_layout = html.Div([
            html.H1('Page 1'),
            html.Div(id='page-1-content'),
            html.Br(),
            dcc.Link('Go to Page 2', href='/page-2'),
            html.Br(),
            dcc.Link('Go back to home', href='/'),
        ])

        @self.app.callback(dash.dependencies.Output('page-1-content', 'children'),
                    [dash.dependencies.Input('page-1-dropdown', 'value')])
        def page_1_dropdown(value):
            active_model = ...
            return 'You have selected "{}"'.format(value)