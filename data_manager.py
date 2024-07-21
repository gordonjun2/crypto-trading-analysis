import os
import argparse
import shutil
import pickle
from datetime import datetime
import numpy as np
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


def process_data(cex,
                 interval,
                 nan_remove_threshold,
                 selected_pairs,
                 top_n_volume_pairs,
                 volume_filter_mode='rolling',
                 day_limit=7):
    """
    Process and data.
    """

    cex = str(cex).lower()
    interval = str(interval)
    volume_filter_mode = str(volume_filter_mode).lower()

    if volume_filter_mode not in ['rolling', 'mean']:
        print("\nInvalid volume filter mode. Using 'rolling' mode instead.")
        volume_filter_mode = 'rolling'

    dir_path = './saved_data/{}/{}'.format(cex, interval)
    volume_dict = {}
    df_concat_list = []
    column_to_drop_list = []
    earliest_date_obj = datetime.max.date()
    latest_date_obj = datetime.min.date()

    if volume_filter_mode != 'mean':
        interval_seconds = get_interval_seconds(cex, interval)

        if interval_seconds == 0:
            return None

        rolling_window_value = int(day_limit * 86400 / interval_seconds)

    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        if files:
            for file in files:
                file_path = dir_path + '/' + file
                try:
                    df, metadata = load_ts_df(file_path)
                except:
                    print(
                        '\nUnable to load the file at {}. Skipping...'.format(
                            file_path))
                    continue

                if df is None:
                    print("\nNo data found for pair {}. Skipping...".format(
                        pair))
                    continue

                pair = metadata['pair']

                if not selected_pairs or pair in selected_pairs:

                    if volume_filter_mode == 'mean':
                        volume_dict[pair] = metadata["mean_volume"]
                    else:
                        volume_dict[pair] = df['Volume in USDT'].rolling(
                            window=rolling_window_value).mean().iloc[-1]

                    df = df.drop(columns=["Volume in USDT"]).rename(
                        columns={"Close": pair})
                    df.set_index("Open Time", inplace=True)
                    df_concat_list.append(df)

                    start_date = metadata['start_date']
                    end_date = metadata['end_date']

                    start_date_obj = datetime.strptime(start_date,
                                                       "%Y-%m-%d").date()
                    end_date_obj = datetime.strptime(end_date,
                                                     "%Y-%m-%d").date()

                    if start_date_obj < earliest_date_obj:
                        earliest_date_obj = start_date_obj
                    if end_date_obj > latest_date_obj:
                        latest_date_obj = end_date_obj

            if df_concat_list:
                merged_df = pd.concat(df_concat_list, axis=1,
                                      join="outer").reset_index()
                merged_df = merged_df.sort_values("Open Time")

                nan_columns = merged_df.columns[
                    merged_df.isna().any()].tolist()
                nan_counts = merged_df[nan_columns].isna().sum()
                threshold = nan_remove_threshold * len(merged_df)

                nan_columns_df = pd.DataFrame({
                    'Pair':
                    nan_counts.index,
                    'NaN Count':
                    nan_counts.values,
                    'Remark':
                    np.where(nan_counts > threshold, 'To Remove',
                             'To Interpolate')
                })

                if not nan_columns_df.empty:
                    nan_columns_df_sorted = nan_columns_df.sort_values(
                        by='NaN Count', ascending=False)
                    print("\nColumns that contains NaN values:")
                    print(nan_columns_df_sorted)

                    prev_df_column_len = merged_df.shape[1]

                    column_to_drop_list = merged_df.columns[
                        merged_df.isna().sum() > threshold]
                    merged_df = merged_df.drop(columns=column_to_drop_list)

                    curr_df_column_len = merged_df.shape[1]

                    print(
                        "\nRemoved {} pairs as they contain too many NaN values."
                        .format(prev_df_column_len - curr_df_column_len))

                filtered_volume_dict = {
                    k: v
                    for k, v in volume_dict.items()
                    if k not in column_to_drop_list
                }
                sorted_pairs = sorted(filtered_volume_dict.keys(),
                                      key=lambda x: filtered_volume_dict[x],
                                      reverse=True)
                filtered_sorted_pairs = sorted_pairs[:top_n_volume_pairs]

                merged_df = merged_df[['Open Time'] + filtered_sorted_pairs]

                print("\nFiltered top {} mean volume pairs.".format(
                    top_n_volume_pairs))
                print(
                    "Successfully loaded candlestick dataframe for all available pairs."
                )

                print("\nEarliest time series start date: {}".format(
                    earliest_date_obj))
                print(
                    "Latest time series end date: {}".format(latest_date_obj))

                return merged_df

            else:
                print(
                    "\nNo pair data found. Please check if the selected pairs are keyed in correctly."
                )

                return None

        else:
            print(
                "\nNo files found in the selected directory {}. Please run 'data_manager.py' to generate the data."
                .format(dir_path))

            return None

    else:
        print(
            "\nNo files found in the selected directory {}. Please run 'data_manager.py' to generate the data."
            .format(dir_path))

        return None


def sanitize_data(merged_df, start_date, end_date):

    try:
        start_date = pd.to_datetime(start_date)
    except:
        print(
            "Invalid start date entered. Please enter the start date in YYYY-MM-DD format."
        )
        return {}, []

    try:
        end_date = pd.to_datetime(end_date)
    except:
        print(
            "Invalid end date entered. Please enter the end date in YYYY-MM-DD format."
        )
        return {}, []

    data_sanitized = {}

    filtered_df = merged_df[(merged_df["Open Time"] >= start_date)
                            & (merged_df["Open Time"] <= end_date)]
    filtered_df.set_index("Open Time", inplace=True)

    pairs = [col for col in filtered_df.columns]

    for pair in pairs:
        filtered_df.loc[:, pair] = filtered_df[pair].replace([np.inf, -np.inf],
                                                             np.nan)
        filtered_df.loc[:,
                        pair] = filtered_df[pair].interpolate(method='linear')
        filtered_df.loc[:, pair] = filtered_df[pair].ffill()
        filtered_df.loc[:, pair] = filtered_df[pair].bfill()

        assert not np.any(np.isnan(filtered_df[pair])) and not np.any(
            np.isinf(filtered_df[pair]))

        data_sanitized[pair] = pd.DataFrame({"Close": filtered_df[pair]},
                                            index=filtered_df.index)

    sorted_available_pairs = sorted(pairs)

    return data_sanitized, sorted_available_pairs


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

    interval_seconds = get_interval_seconds(cex, interval)

    if interval_seconds == 0:
        sys.exit(1)

    if cex == 'binance':
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
