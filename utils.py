import os
import time
import pandas as pd


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


# def prep_ohlcv_data(self, symbols: List[str],
#                     interval: str) -> Dict[str, pd.DataFrame]:
#     """
#     prepare data for multiple symbols in a dictionary
#     """
#     data = {symbol: self.load_data(symbol, interval) for symbol in symbols}
#     return data

# TODO need a function to slow down the API calls to avoid rate limiting
