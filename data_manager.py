import os
import argparse
import shutil
import pickle
from utils import *
from cex_api.query_binance_data import *
from cex_api.query_okx_data import *
from cex_api.query_bybit_data import *


def save_ts_df(candlestick_data, dir_path, pair, start_date, end_date):
    """
    Save time series financial data and associated metadata.
    """

    cached_file_path = '{}/{}_{}_{}.pkl'.format(dir_path, pair, start_date,
                                                end_date)

    columns = ["Open Time", "Close", "Volume in USDT"]
    df = pd.DataFrame(candlestick_data, columns=columns)

    df["Close"] = pd.to_numeric(df["Close"])
    df["Volume in USDT"] = pd.to_numeric(df["Volume in USDT"])
    df["Open Time"] = pd.to_numeric(df["Open Time"])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')

    mean_volume = df["Volume in USDT"].mean()
    metadata = {
        'pair': pair,
        'start_date': start_date,
        'end_date': end_date,
        'mean_volume': mean_volume
    }

    data_to_save = {'dataframe': df, 'metadata': metadata}

    os.makedirs(dir_path, exist_ok=True)
    with open(cached_file_path, 'wb') as file:
        pickle.dump(data_to_save, file)


def load_ts_df(file_path):
    """
    Load time series financial data and associated metadata.
    """

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    df = data['dataframe']
    metadata = data['metadata']

    return df, metadata


if __name__ == "__main__":

    # Get arguments from terminal
    parser = argparse.ArgumentParser(
        description="Get parameters for the script.")
    parser.add_argument('-c',
                        '--cex',
                        type=str,
                        default='binance',
                        help="CEX. Available values: binance, okx, bybit.")
    parser.add_argument(
        '-i',
        '--interval',
        type=str,
        default='',
        help=
        "Interval. Refer to the relevant query script for the available intervals."
    )
    parser.add_argument(
        '-e',
        '--end',
        type=str,
        default='',
        help=
        "End Time. Start time will be interval x limit before end time. Timestamp in ms."
    )
    parser.add_argument('-l',
                        '--limit',
                        type=int,
                        default=365,
                        help="No. of candlesticks to return.")
    args = parser.parse_args()

    cex = args.cex.lower()
    interval = args.interval
    end_timestamp = args.end
    limit = args.limit

    if cex == 'binance':
        if interval not in [
                '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
                '12h', '1d', '3d', '1w', '1M'
        ]:
            print(
                '\nThe interval is invalid. Available options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.\n'
            )
            sys.exit(1)
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

        start_date, end_date = get_start_end_date(end_timestamp,
                                                  interval_seconds, limit)

        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))
        print("Start Date: {}".format(start_date))
        print("End Date: {}".format(end_date))

        perpetual_futures_pairs = get_binance_perpetual_futures_pairs()

        dir_path = './saved_data/{}/{}'.format(cex, interval)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('\nDeleted existing directory: {}'.format(dir_path))

        for pair in perpetual_futures_pairs:
            print(
                '\nRetrieving candlestick data for pair {} from {}...'.format(
                    pair, cex.capitalize()))

            candlestick_data = get_binance_perpetual_futures_candlestick_data(
                pair, interval, end_timestamp, limit)

            if candlestick_data:
                print('Saving pair {} candlestick data...'.format(pair))

                save_ts_df(candlestick_data, dir_path, pair, start_date,
                           end_date)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    elif cex == 'okx':
        if interval not in [
                '1s', '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H',
                '12H', '1D', '2D', '3D', '1W', '1M', '3M'
        ]:
            print(
                '\nThe interval is invalid. Availble options: 1s, 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 2D, 3D, 1W, 1M, 3M.\n'
            )
            sys.exit(1)
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

        start_date, end_date = get_start_end_date(end_timestamp,
                                                  interval_seconds, limit)

        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))
        print("Start Date: {}".format(start_date))
        print("End Date: {}".format(end_date))

        perpetual_futures_pairs = get_okx_perpetual_futures_pairs()

        dir_path = './saved_data/{}/{}'.format(cex, interval)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('\nDeleted existing directory: {}'.format(dir_path))

        for pair in perpetual_futures_pairs:
            print(
                '\nRetrieving candlestick data for pair {} from {}...'.format(
                    pair, cex.capitalize()))

            candlestick_data = get_okx_perpetual_futures_candlestick_data(
                pair, interval, end_timestamp, limit)

            if candlestick_data:
                print('Saving pair {} candlestick data...'.format(pair))

                save_ts_df(candlestick_data, dir_path, pair, start_date,
                           end_date)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    elif cex == 'bybit':
        if interval not in [
                '1', '3', '5', '15', '30', '60', '120', '240', '360', '720',
                'D', 'M', 'W'
        ]:
            print(
                '\nThe interval is invalid. Availble options: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W.\n'
            )
            sys.exit(1)
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

        start_date, end_date = get_start_end_date(end_timestamp,
                                                  interval_seconds, limit)

        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))
        print("Start Date: {}".format(start_date))
        print("End Date: {}".format(end_date))

        perpetual_futures_pairs = get_bybit_perpetual_futures_pairs()

        dir_path = './saved_data/{}/{}'.format(cex, interval)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print('\nDeleted existing directory: {}'.format(dir_path))

        for pair in perpetual_futures_pairs:
            print(
                '\nRetrieving candlestick data for pair {} from {}...'.format(
                    pair, cex.capitalize()))

            candlestick_data = get_bybit_perpetual_futures_candlestick_data(
                pair, interval, end_timestamp, limit)

            if candlestick_data:
                print('Saving pair {} candlestick data...'.format(pair))

                save_ts_df(candlestick_data, dir_path, pair, start_date,
                           end_date)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    else:
        print('\nInvalid CEX.\n')
        sys.exit(1)

    print("\nData downloaded successfully. Please run 'main.py' next.\n")
