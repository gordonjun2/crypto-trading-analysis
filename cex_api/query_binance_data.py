import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys

urllib3.disable_warnings(InsecureRequestWarning)
'''
Binance API Information:
    Frequently Asked Questions on API: https://www.binance.com/en/support/faq/frequently-asked-questions-on-api-360004492232
    USD-M Futures API Documentation: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
'''


def get_binance_perpetual_futures_pairs():
    '''
    Get all perpetual futures pairs from Binance.

    Request Weight:
        1

    Response Example:
        ['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'TRXUSDT', 'ETCUSDT', 'LINKUSDT', 'XLMUSDT', 
        'ADAUSDT', 'XMRUSDT', 'DASHUSDT', 'ZECUSDT', 'XTZUSDT', 'BNBUSDT', 'ATOMUSDT', 'ONTUSDT', 'IOTAUSDT', ...]

    More information:
        https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information
    '''

    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        data = response.json()
        perpetual_futures_pairs = [
            symbol['symbol'] for symbol in data['symbols']
            if symbol['contractType'] == 'PERPETUAL'
            and symbol['quoteAsset'] == 'USDT'
        ]

    elif response.status_code == 429:
        print('\nUnable to query Binance USD-M pairs. Rate limit exceeded.\n')

        sys.exit(1)

    else:
        print('\nUnable to query Binance USD-M pairs. Response code is {}.\n'.
              format(response.status_code))

        sys.exit(1)

    # print(sorted(perpetual_futures_pairs))

    print('\nSuccessfully queried Binance Perpetual Futures pairs.')

    return perpetual_futures_pairs


def get_binance_perpetual_futures_candlestick_data(symbol,
                                                   interval='1d',
                                                   endTime='',
                                                   limit=365):
    '''
    Get perpetual futures candlestick data from Binance.

    Request Weight:
        based on parameter LIMIT
        If LIMIT [1,100), 1; [100,500), 2; [500,1000], 5; > 1000, 10

    Request Parameters:
        symbol: str
            The symbol of the USD-M pair.
        interval: str
            The interval of the candlestick data. 
            Possible values: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.
        endTime: int
            Timestamp in ms to get candlestick data before this time.
        limit: int
            The number of candlesticks to return. 
            Default 500; max 1500.

    Response Example:
        [
            [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "17928899.62484339" // Ignore.
            ],
            ...
        ]

    More information:
        https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
        https://dev.binance.vision/t/did-kline-api-ignore-starttime/4364/2
    '''

    if symbol in ['', None]:
        print('\nThe symbol cannot be empty.\n')

        sys.exit(1)

    if interval not in [
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h',
            '12h', '1d', '3d', '1w', '1M'
    ]:
        print(
            '\nThe interval is invalid. Available options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.\n'
        )

        sys.exit(1)

    url = 'https://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&endTime={}&limit={}'.format(
        symbol, interval, endTime, limit)
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        data = response.json()
        data = [[
            candlestick[0],
            candlestick[1],
            candlestick[2],
            candlestick[3],
            candlestick[4],
            candlestick[7],
        ] for candlestick in data]

    elif response.status_code == 429:
        print(
            '\nUnable to query Binance USD-M pair {} candlestick data. Rate limit exceeded.\n'
            .format(symbol))

        sys.exit(1)

    else:
        print(
            '\nUnable to query Binance USD-M pair {} candlestick data. Response code is {}.\n'
            .format(symbol, response.status_code))

        sys.exit(1)

    # print(data[0])
    # print(data[-1])
    # print(len(data))

    return data


def get_binance_perpetual_futures_24hr_price_change_statistics_data():
    '''
    Get perpetual futures 24hr price change statistics data from Binance.

    Request Weight:
        1 for a single symbol;
        40 when the symbol parameter is omitted

    Response Example:
        [
            {
                "symbol": "BTCUSDT",
                "priceChange": "-94.99999800",
                "priceChangePercent": "-95.960",
                "weightedAvgPrice": "0.29628482",
                "lastPrice": "4.00000200",
                "lastQty": "200.00000000",
                "openPrice": "99.00000000",
                "highPrice": "100.00000000",
                "lowPrice": "0.10000000",
                "volume": "8913.30000000",
                "quoteVolume": "15.30000000",
                "openTime": 1499783499040,
                "closeTime": 1499869899040,
                "firstId": 28385,   // First tradeId
                "lastId": 28460,    // Last tradeId
                "count": 76         // Trade count
            },
            ...
        ]

    More information:
        https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/24hr-Ticker-Price-Change-Statistics
    '''

    url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        data = response.json()
        data = [{
            'symbol': pair['symbol'],
            'priceChange': pair['priceChange'],
            'priceChangePercent': pair['priceChangePercent'],
            'lastPrice': pair['lastPrice'],
            'openPrice': pair['openPrice'],
            'highPrice': pair['highPrice'],
            'lowPrice': pair['lowPrice'],
            'volume': pair['volume'],
            'quoteVolume': pair['quoteVolume'],
        } for pair in data]

    elif response.status_code == 429:
        print(
            '\nUnable to query Binance USD-M tickers 24hr price change statistics data. Rate limit exceeded.\n'
        )

        sys.exit(1)

    else:
        print(
            '\nUnable to query Binance USD-M tickers 24hr price change statistics data. Response code is {}.\n'
            .format(response.status_code))

        sys.exit(1)

    # print(data)

    return data


# get_binance_perpetual_futures_pairs()
# get_binance_perpetual_futures_candlestick_data('BTCUSDT', '1d', None, 365)
# get_binance_perpetual_futures_24hr_price_change_statistics_data()
