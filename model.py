def predict(stock, days_n):
    import pandas as pd
    from datetime import datetime as date
    from datetime import timedelta
    import yfinance as yf
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    import plotly.graph_objs as go

    print(f"Fetching data for stock: {stock}")
    
    # Fetch the stock data
    try:
        df = yf.download(stock, period='3mo')
        print(df.head())  # Check the first few rows of data
        if df.empty:
            raise ValueError(f"No data found for the stock symbol provided: {stock}")
    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        raise
    
    df.reset_index(inplace=True)
    df['Day'] = df.index
    days = [[i] for i in range(len(df.Day))]

    X = days
    Y = df[['Close']]
    
    # If there's insufficient data, raise an error
    if len(df) < 10:
        raise ValueError("Not enough data for training.")
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    y_train = y_train.values.ravel()
    
    # Grid Search to find best parameters
    gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    
    # Fit the best SVR model
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"])
    best_svr.fit(x_train, y_train)

    # Prepare for forecasting
    output_days = [[i + x_test[-1][0]] for i in range(1, days_n)]

    dates = []
    current = date.today()
    for i in range(days_n):
        current += timedelta(days=1)
        dates.append(current)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=best_svr.predict(output_days), mode='lines+markers', name='Forecasted Data'))
    fig.update_layout(title=f"Predicted Close Price for the next {days_n} days", xaxis_title="Date", yaxis_title="Close Price")
    
    return fig
