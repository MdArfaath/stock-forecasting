import pandas as pd
from pandas.io.formats import style
from pandas_datareader import data as pdr
from datetime import datetime as date
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from dash.exceptions import PreventUpdate
from model import predict
from plotly.graph_objects import Layout
from plotly.validator_cache import ValidatorCache
app = dash.Dash()

def get_stock_price_fig(df):
    fig = px.line(df,x= "Date" ,y= ["Close","Open"], title="Closing and Opening Price vs Date",markers=True)
    fig.update_layout(title_x=0.5)
    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,x= "Date",y= "EWA_20",title="Exponential Moving Average vs Date")
    fig.update_traces(mode= "lines+markers")
    return fig

app.layout =html.Div([
        html.Div([
            html.Div([
            html.H1(children="Welcome to the Stock Dash App!")
            ],
            className='start',
            style = {'padding-top' : '1%'}
            ),
            html.Div([
              # stock code input
              dcc.Input(id='input', type='text',style={'align':'center'}),
              html.Button('Submit', id='submit-name', n_clicks=0),
            ]),
            html.Div(
              # Date range picker input
              ['Select a date range: ',
                            dcc.DatePickerRange(
                              id='my-date-picker-range',
                              min_date_allowed=date(1995, 8, 5),
                              max_date_allowed=date.now(),
                              initial_visible_month=date.now(),
                              end_date=date.now().date(),
                              style = {'font-size': '18px','display': 'inline-block','align':'center', 'border-radius' : '2px', 'border' : '1px solid #ccc', 'color': '#333', 'border-spacing' : '0', 'border-collapse' :'separate'}
                            ),
              html.Div(id='output-container-date-picker-range',children='You have selected a date')

              ]),
            html.Div([
              # Stock price button
              html.Button('Stock Price', id='submit-val', n_clicks=0,style={'float':'left','padding':'15px 32px','background-color':'red','display': 'inline'}),
              html.Div(id='container-button-basic'),
              # Indicators button
              html.Button('Indicator', id='submit-ind', n_clicks=0),
             
      
              # Number of days of forecast input
              html.Div([dcc.Input(id='Forcast_Input', type='text',)]),
              html.Button('No of days to forcast', id='submit-forc', n_clicks=0),
              html.Div(id='forcast')
              # Forecast button
            ])
      ],className='nav'),
      html.Div(
          [
            html.Div(
                  [  html.Img(id='logo'),
                    html.H1(id='name')
                    # Company Name
                  ],
                className="header"),
            html.Div( #Description
              id="description", className="decription_ticker"),
            html.Div([],
                # Stock price plot
             id="graphs-content"),
            html.Div([
                # Indicator plot
            ], id="main-content"),
            html.Div([
                # Forecast plot
            ], id="forecast-content")
          ],
        className="content")],
        className="container")


@app.callback(
    [
        Output('description', 'children'),
        Output('logo', 'src'),
        Output('name', 'children'),
        Output('submit-val', 'n_clicks'),
        Output('submit-ind', 'n_clicks'),
        Output('submit-forc', 'n_clicks'),
    ],
    Input('submit-name', 'n_clicks'),
    State('input', 'value'),
)
def update_data(n, val):
    if not n:
        return (
            "Hey there! Please enter a legitimate stock code to get details.",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXiMwetRCK45yWzynINQg4fJn3iUXetiQoSw&s",
            "Stocks",
            None,
            None,
            None,
        )
    if not val:
        raise PreventUpdate

    try:
        ticker = yf.Ticker(val)
        inf = ticker.info

        # Check if keys exist; use fallback values if not
        description = inf.get("longBusinessSummary", "No description available.")
        logo_url = inf.get("logo_url")  # Placeholder image
        short_name = inf.get("shortName", "N/A")

        return description, logo_url, short_name, None, None, None

    except Exception as e:
        print(f"Error: {e}")
        return (
            "Failed to retrieve stock data. Please check the stock code and try again.",
            "https://via.placeholder.com/150",
            "Error",
            None,
            None,
            None,
        )




@app.callback([
    Output('graphs-content','children'),
    Input('submit-val', 'n_clicks'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    State('input', 'value')])

def update_graph(n,start_date,end_date,val):
  if n == None:
        return [""]
        #raise PreventUpdate
  if val == None:
    raise PreventUpdate
  else:
    if start_date != None:
      df = yf.download(val,str( start_date) ,str( end_date ))
    else:
      df = yf.download(val)  
  df.reset_index(inplace=True)
  fig = get_stock_price_fig(df)
  return [dcc.Graph(figure=fig)]

@app.callback([Output("main-content", "children")], [
    Input("submit-ind", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("input", "value")])
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]

@app.callback([
  Output("forecast-content","children"),
  Input("submit-forc","n_clicks"),
  State("Forcast_Input","value"),
  State("input","value")

])
def forecast(n, n_days, val):
    if n == None:
        return [""]

    if val == None:
        raise PreventUpdate

    # Get stock data
    df = yf.download(val, period='max')  # Download stock data for the max available period
    if df.empty:
        return ["No data found for this stock code."]

    # Ensure we have data
    if df.shape[0] < 10:  # If there are fewer than 10 data points, it's likely not enough for forecasting
        return ["Not enough data for forecasting."]

    # If you reach here, proceed with prediction
    try:
        x = int(n_days)
        fig = predict(val, x + 1)
        return [dcc.Graph(figure=fig)]
    except Exception as e:
        return [f"An error occurred: {e}"]


if __name__ == '__main__':
    app.run_server(debug=True)