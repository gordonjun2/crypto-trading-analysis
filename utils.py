import time
import pandas as pd
import matplotlib.pyplot as plt


def get_interval_seconds(cex, interval):

    if cex == 'binance':
        if interval not in [
                '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
                '12h', '1d', '3d', '1w', '1M'
        ]:
            print(
                '\nThe interval is invalid. Available options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.\n'
            )
            interval_seconds = 0
        elif interval == '1m':
            interval_seconds = 60
        elif interval == '3m':
            interval_seconds = 180
        elif interval == '5m':
            interval_seconds = 300
        elif interval == '15m':
            interval_seconds = 900
        elif interval == '30m':
            interval_seconds = 1800
        elif interval == '1h':
            interval_seconds = 3600
        elif interval == '2h':
            interval_seconds = 7200
        elif interval == '4h':
            interval_seconds = 14400
        elif interval == '6h':
            interval_seconds = 21600
        elif interval == '8h':
            interval_seconds = 28800
        elif interval == '12h':
            interval_seconds = 43200
        elif interval == '1d':
            interval_seconds = 86400
        elif interval == '3d':
            interval_seconds = 259200
        elif interval == '1w':
            interval_seconds = 604800
        else:
            interval_seconds = 2592000

    elif cex == 'okx':
        if interval not in [
                '1s', '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H',
                '12H', '1D', '2D', '3D', '1W', '1M', '3M'
        ]:
            print(
                '\nThe interval is invalid. Availble options: 1s, 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 2D, 3D, 1W, 1M, 3M.\n'
            )
            interval_seconds = 0
        elif interval == '1s':
            interval_seconds = 1
        elif interval == '1m':
            interval_seconds = 60
        elif interval == '3m':
            interval_seconds = 180
        elif interval == '5m':
            interval_seconds = 300
        elif interval == '15m':
            interval_seconds = 900
        elif interval == '30m':
            interval_seconds = 1800
        elif interval == '1H':
            interval_seconds = 3600
        elif interval == '2H':
            interval_seconds = 7200
        elif interval == '4H':
            interval_seconds = 14400
        elif interval == '6H':
            interval_seconds = 21600
        elif interval == '12H':
            interval_seconds = 43200
        elif interval == '1D':
            interval_seconds = 86400
        elif interval == '2D':
            interval_seconds = 172800
        elif interval == '3D':
            interval_seconds = 259200
        elif interval == '1W':
            interval_seconds = 604800
        elif interval == '1M':
            interval_seconds = 2592000
        else:
            interval_seconds = 7776000

    elif cex == 'bybit':
        if interval not in [
                '1', '3', '5', '15', '30', '60', '120', '240', '360', '720',
                'D', 'M', 'W'
        ]:
            print(
                '\nThe interval is invalid. Availble options: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W.\n'
            )
            interval_seconds = 0
        elif interval == '1':
            interval_seconds = 60
        elif interval == '3':
            interval_seconds = 180
        elif interval == '5':
            interval_seconds = 300
        elif interval == '15':
            interval_seconds = 900
        elif interval == '30':
            interval_seconds = 1800
        elif interval == '60':
            interval_seconds = 3600
        elif interval == '120':
            interval_seconds = 7200
        elif interval == '240':
            interval_seconds = 14400
        elif interval == '360':
            interval_seconds = 21600
        elif interval == '720':
            interval_seconds = 43200
        elif interval == 'D':
            interval_seconds = 86400
        elif interval == 'M':
            interval_seconds = 2592000
        else:
            interval_seconds = 604800

    else:
        print('\nInvalid CEX.\n')
        interval_seconds = 0

    return interval_seconds


def calculate_start_ts(end_time_ms, interval_seconds, limit):
    interval_ms = interval_seconds * 1000
    total_time_span_ms = interval_ms * limit
    start_time_ms = end_time_ms - total_time_span_ms

    return start_time_ms


def convert_timestamp_to_date(timestamp_ms):
    datetime_obj = pd.to_datetime(timestamp_ms, unit='ms')
    formatted_datetime = datetime_obj.strftime('%Y-%m-%d')

    return formatted_datetime


def get_current_timestamp_ms():
    current_time_seconds = time.time()
    current_time_ms = int(current_time_seconds * 1000)

    return current_time_ms


def calculate_profit(signals, prices):
    """
    Calculate cumulative profit based on trading signals and stock prices.
    Parameters:
    - signals (pandas.DataFrame): A DataFrame containing trading signals (1 for buy, -1 for sell).
    - prices (pandas.Series): A Series containing stock prices corresponding to the signal dates.
    Returns:
    - cum_profit (pandas.Series): A Series containing cumulative profit over time.
    """
    profit = pd.Series(index=prices.index)
    profit.fillna(0, inplace=True)

    buys = signals[signals['orders'] == 1].index
    sells = signals[signals['orders'] == -1].index
    skip = 0
    for bi in buys:
        if skip > 0:
            skip -= 1
            continue
        sis = sells[sells > bi]
        if len(sis) > 0:
            si = sis[0]
            profit[si] = prices[si] - prices[bi]
            skip = len(buys[(buys > bi) & (buys < si)])
        else:
            profit[-1] = prices[-1] - prices[bi]
    cum_profit = profit.cumsum()

    return cum_profit


def plot_strategy(prices_df, signal_df, profit):
    """
    Plot a trading strategy with buy and sell signals and cumulative profit.
    Parameters:
    - prices (pandas.Series): A Series containing stock prices.
    - signals (pandas.DataFrame): A DataFrame with buy (1) and sell (-1) signals.
    - profit (pandas.Series): A Series containing cumulative profit over time.
    Returns:
    - ax1 (matplotlib.axes.Axes): The top subplot displaying stock prices and signals.
    - ax2 (matplotlib.axes.Axes): The bottom subplot displaying cumulative profit.
    """
    fig, (ax1, ax2) = plt.subplots(2,
                                   1,
                                   gridspec_kw={'height_ratios': (3, 1)},
                                   figsize=(24, 12))

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price in $')
    ax1.plot(prices_df.index, prices_df, color='g', lw=0.25)

    # Plot the Buy and Sell signals
    ax1.plot(signal_df.loc[signal_df.orders == 1.0].index,
             prices_df[signal_df.orders == 1.0],
             '^',
             markersize=12,
             color='blue',
             label='Buy')
    ax1.plot(signal_df.loc[signal_df.orders == -1.0].index,
             prices_df[signal_df.orders == -1.0],
             'v',
             markersize=12,
             color='red',
             label='Sell')

    ax2.plot(profit.index, profit, color='b')
    ax2.set_ylabel('Cumulative Profit (%)', fontsize=18)
    ax2.set_xlabel('Date', fontsize=18)

    return ax1, ax2
