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


def save_ts_df(candlestick_data, dir_path, pair):
    """
    Save time series financial data and associated metadata.
    """

    columns = ["Open Time", "Open", "High", "Low", "Close", "Volume in USDT"]
    df = pd.DataFrame(candlestick_data, columns=columns)

    df["Open"] = pd.to_numeric(df["Open"])
    df["High"] = pd.to_numeric(df["High"])
    df["Low"] = pd.to_numeric(df["Low"])
    df["Close"] = pd.to_numeric(df["Close"])
    df["Volume in USDT"] = pd.to_numeric(df["Volume in USDT"])
    df["Open Time"] = pd.to_numeric(df["Open Time"])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df = df.sort_values(by='Open Time', ascending=True).reset_index(drop=True)
    start_datetime = df.iloc[0]['Open Time']
    end_datetime = df.iloc[-1]['Open Time']

    metadata = {
        'pair': pair,
        'start_datetime': start_datetime,
        'end_datetime': end_datetime,
    }

    data_to_save = {'dataframe': df, 'metadata': metadata}

    cached_file_path = '{}/{}_{}_{}.pkl'.format(dir_path, pair, start_datetime,
                                                end_datetime)

    os.makedirs(dir_path, exist_ok=True)
    with open(cached_file_path, 'wb') as file:
        pickle.dump(data_to_save, file)


def save_df(dataframe, dir_path, start_datetime, end_datetime):
    """
    Save dataframe.
    """

    cached_file_path = '{}/{}_{}.pkl'.format(dir_path, start_datetime,
                                             end_datetime)

    os.makedirs(dir_path, exist_ok=True)
    with open(cached_file_path, 'wb') as file:
        pickle.dump(dataframe, file)


def save_sentiment_score_df(sentiment_score_df, dir_path, start_date,
                            end_date):
    """
    Save sentiment score data.
    """

    merged_df = pd.DataFrame(columns=['Open Time', 'Sentiment Score'])
    date_format = '%Y-%m-%d'
    parsed_input_start_datetime = datetime.strptime(start_date, date_format)
    parsed_input_end_datetime = datetime.strptime(
        end_date, date_format).replace(hour=23, minute=59, second=59)

    merged_df, files_to_delete = load_presaved_df(merged_df, dir_path)

    merged_df = pd.concat([merged_df, sentiment_score_df],
                          axis=0,
                          ignore_index=True)
    merged_df = merged_df.drop_duplicates(
        subset='Open Time')  # Drop rows with duplicated 'Date Time'
    merged_df = merged_df.sort_values(by='Open Time', ascending=True)
    start_datetime = merged_df.iloc[0]['Open Time']
    end_datetime = merged_df.iloc[-1]['Open Time']

    delete_files(files_to_delete)
    save_df(merged_df, dir_path, start_datetime, end_datetime)
    print("\n")

    merged_df = merged_df[
        (merged_df['Open Time'] >= parsed_input_start_datetime)
        & (merged_df['Open Time'] <= parsed_input_end_datetime)]
    merged_df = merged_df.reset_index(drop=True)

    return merged_df


def load_ts_df(file_path):
    """
    Load time series financial data and associated metadata.
    """

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    df = data['dataframe']
    metadata = data['metadata']

    return df, metadata


def load_df(file_path):
    """
    Load dataframe.
    """

    with open(file_path, 'rb') as file:
        dataframe = pickle.load(file)

    return dataframe


def load_presaved_df(merged_df, dir_path):

    files_to_delete = []

    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        if files:
            expected_columns = set(merged_df.columns)

            for file in files:
                if not file.endswith('.pkl'):
                    continue

                file_path = dir_path + '/' + file
                try:
                    df = load_df(file_path)
                except:
                    continue

                if df is None:
                    continue

                if set(df.columns) == expected_columns:
                    merged_df = pd.concat([merged_df, df],
                                          axis=0,
                                          ignore_index=True)

                files_to_delete.append(file_path)

    return merged_df, files_to_delete


def process_data(strategy,
                 cex,
                 interval,
                 nan_remove_threshold,
                 selected_pairs,
                 top_n_volume_pairs,
                 volume_filter_mode='rolling',
                 day_limit=7):
    """
    Process and data.
    """

    strategy = str(strategy).lower()
    cex = str(cex).lower()
    interval = str(interval)
    volume_filter_mode = str(volume_filter_mode).lower()

    if strategy not in [
            'mean_reversion', 'low_correlation', 'beta_neutral',
            'sentiment_on_chart', 'volatility'
    ]:
        print(
            "\nInvalid strategy. Available options: mean_reversion, low_correlation, beta_neutral."
        )
        return None
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

                if (not selected_pairs and strategy
                        != 'beta_neutral') or pair in selected_pairs:
                    if volume_filter_mode == 'mean':
                        volume_dict[pair] = df["Volume in USDT"].mean()
                    else:
                        volume_dict[pair] = df['Volume in USDT'].rolling(
                            window=rolling_window_value).mean().iloc[-1]

                    if strategy == 'beta_neutral':
                        df['OHLC Average'] = (df['Open'] + df['Close']) / 2
                        df = df.rename(columns={"OHLC Average": pair})
                        df = df.drop(columns=[
                            "Open", "High", "Low", "Close", "Volume in USDT"
                        ])
                    elif strategy == 'volatility':
                        df = df.rename(columns={"Close": pair + '_Close'})
                        df = df.rename(columns={"High": pair + '_High'})
                        df = df.rename(columns={"Low": pair + '_Low'})
                        df = df.drop(columns=["Open", "Volume in USDT"])
                    else:
                        df = df.rename(columns={"Close": pair})
                        df = df.drop(
                            columns=["Open", "High", "Low", "Volume in USDT"])

                    df.set_index("Open Time", inplace=True)
                    df_concat_list.append(df)

                    start_date = metadata['start_datetime']
                    end_date = metadata['end_datetime']

                    start_date_obj = start_date.to_pydatetime().date()
                    end_date_obj = end_date.to_pydatetime().date()

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

                if strategy == 'volatility':
                    filtered_sorted_pairs = [
                        f"{pair}_{suffix}" for pair in filtered_sorted_pairs
                        for suffix in ["Close", "High", "Low"]
                    ]

                merged_df = merged_df[['Open Time'] + filtered_sorted_pairs]

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


def sanitize_data(merged_df,
                  start_date,
                  end_date,
                  is_volatility_strategy=False):

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
    filtered_df = filtered_df.copy()

    if is_volatility_strategy:
        sorted_available_pairs = sorted(
            set(col.split("_")[0] for col in filtered_df.columns))

        for pair in sorted_available_pairs:
            close_col = f"{pair}_Close"
            high_col = f"{pair}_High"
            low_col = f"{pair}_Low"

            if not all(col in filtered_df.columns
                       for col in [close_col, high_col, low_col]):
                print(f"Skipping {pair} due to missing HLC columns.")
                continue

            for col in [close_col, high_col, low_col]:
                filtered_df[col] = filtered_df[col].replace([np.inf, -np.inf],
                                                            np.nan)
                filtered_df[col] = filtered_df[col].interpolate(
                    method='linear')
                filtered_df[col] = filtered_df[col].ffill().bfill()

                assert not np.any(np.isnan(filtered_df[col])) and not np.any(
                    np.isinf(filtered_df[col]))

            data_sanitized[pair] = filtered_df[[close_col, high_col,
                                                low_col]].rename(columns={
                                                    close_col: "Close",
                                                    high_col: "High",
                                                    low_col: "Low"
                                                })

    else:
        pairs = [col for col in filtered_df.columns]

        for pair in pairs:
            filtered_df.loc[:, pair] = filtered_df[pair].replace(
                [np.inf, -np.inf], np.nan)
            filtered_df.loc[:, pair] = filtered_df[pair].interpolate(
                method='linear')
            filtered_df.loc[:, pair] = filtered_df[pair].ffill()
            filtered_df.loc[:, pair] = filtered_df[pair].bfill()

            assert not np.any(np.isnan(filtered_df[pair])) and not np.any(
                np.isinf(filtered_df[pair]))

            data_sanitized[pair] = pd.DataFrame({"Close": filtered_df[pair]},
                                                index=filtered_df.index)

        sorted_available_pairs = sorted(pairs)

    return data_sanitized, sorted_available_pairs


def delete_files(file_list):
    for file_path in file_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print("Deleted: {}".format(file_path))
            else:
                print("File to delete not found: {}".format(file_path))
        except Exception as e:
            print("Error deleting {}: {}".format(file_path, e))


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
        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))

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

                save_ts_df(candlestick_data, dir_path, pair)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    elif cex == 'okx':
        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))

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

                save_ts_df(candlestick_data, dir_path, pair)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    elif cex == 'bybit':
        print("\nCEX: {}".format(cex.capitalize()))
        print("Interval: {}".format(interval))
        print("No. of candlesticks to save: {}".format(limit))

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

                save_ts_df(candlestick_data, dir_path, pair)

            else:
                print('No candlestick data found for pair {}. Skipping...'.
                      format(pair))

    else:
        print('\nInvalid CEX.\n')
        sys.exit(1)

    print(
        "\nData downloaded successfully. Please use any of the Jupyter Notebook next.\n"
    )
