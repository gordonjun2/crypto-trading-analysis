import time
import pandas as pd
import matplotlib.pyplot as plt


def get_start_end_date(end_timestamp, interval_seconds, limit):
    if not end_timestamp:
        end_timestamp = get_current_timestamp_ms()
    else:
        end_timestamp = int(end_timestamp)

    start_timestamp = calculate_start_ts(end_timestamp, interval_seconds,
                                         limit)

    end_date = convert_timestamp_to_date(end_timestamp)
    start_date = convert_timestamp_to_date(start_timestamp)

    return start_date, end_date


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
    ax2.set_ylabel('Cumulative Profit (%)')
    ax2.set_xlabel('Date')

    return ax1, ax2


# def prep_ohlcv_data(self, symbols: List[str],
#                     interval: str) -> Dict[str, pd.DataFrame]:
#     """
#     prepare data for multiple symbols in a dictionary
#     """
#     data = {symbol: self.load_data(symbol, interval) for symbol in symbols}
#     return data

# TODO need a function to slow down the API calls to avoid rate limiting
