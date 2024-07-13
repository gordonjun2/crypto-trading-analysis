import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys

urllib3.disable_warnings(InsecureRequestWarning)
'''
OKX API FAQ:
    API Documentation: https://www.okx.com/docs-v5/en/?python#overview
'''


def get_okx_perpetual_futures_pairs():
    '''
    Get all perpetual futures pairs from OKX.

    Rate Limit: 
        20 requests per 2 seconds

    Rate limit rule:
        IP + instrumentType

    Response Example:
        ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'LTC-USDT-SWAP', 'ETC-USDT-SWAP', 'XRP-USDT-SWAP', 'EOS-USDT-SWAP', 'BCH-USDT-SWAP', 
        'BSV-USDT-SWAP', 'TRX-USDT-SWAP', 'LINK-USDT-SWAP', 'ADA-USDT-SWAP', 'DOT-USDT-SWAP', 'UNI-USDT-SWAP', 'FIL-USDT-SWAP', 
        'YFI-USDT-SWAP', 'SUSHI-USDT-SWAP', 'AAVE-USDT-SWAP', 'MKR-USDT-SWAP', 'COMP-USDT-SWAP', 'SNX-USDT-SWAP', ...]

    More information:
        https://www.okx.com/docs-v5/en/?python#public-data-rest-api-get-instruments
    '''

    url = 'https://www.okx.com/api/v5/public/instruments'
    params = {
        'instType':
        'SWAP',  # Type of instrument, 'SWAP' for perpetual contracts
    }
    response = requests.get(url, params=params, verify=False)

    if response.status_code == 200:
        json_data = response.json()
        code = json_data.get('code')

        if code == '0':
            data = json_data.get('data', [])

            perpetual_futures_pairs = [
                instrument['instId'] for instrument in data
                if instrument['settleCcy'] == 'USDT'
            ]

        else:
            msg = json_data.get('msg', '')
            print(
                "\nUnable to query OKX Perpetual Futures pairs. Return code is {} and error message is '{}'.\n"
                .format(code, msg))

            sys.exit(1)

    else:
        print(
            '\nUnable to query OKX Perpetual Futures pairs. Response code is {}.\n'
            .format(response.status_code))

        sys.exit(1)

    # print(len(sorted(perpetual_futures_pairs)))

    print('\nSuccessfully queried OKX Perpetual Futures pairs.')

    return perpetual_futures_pairs


def get_okx_perpetual_futures_candlestick_data(symbol,
                                               interval='1D',
                                               endTime='',
                                               limit='300'):
    '''
    Get perpetual futures candlestick data from OKX. This endpoint can retrieve the latest 1,440 data entries. 

    Request Parameters:
        instId: str
            Instrument ID, e.g. BTC-USDT-SWAP.
        bar: str
            Bar size. 
            Possible values: 1s, 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 2D, 3D, 1W, 1M, 3M.
        after: str
            Pagination of data to return records earlier than the requested ts.
        limit: str
            Number of results per request.
            Default 100; max 300.

    Response Parameters:
        ts: str
            Opening time of the candlestick, Unix timestamp format in milliseconds, e.g. 1597026383085.
            Data sorted in ascending order by ts.
        o: str
            Open price
        h: str
            Highest price
        l: str
            Lowest price
        c: str
            Close price
        vol: str
            Trading volume, with a unit of contract.
        volCcy: str
            Trading volume, with a unit of currency.
        volCcyQuote: str
            Trading volume, the value is the quantity in quote currency
            e.g. The unit is USDT for BTC-USDT and BTC-USDT-SWAP;
            The unit is USD for BTC-USD-SWAP.
        confirm: str
            The state of candlesticks.
            0: K line is uncompleted
            1: K line is completed

    Response Example:
        [
            ['1719936000000', '61958.5', '62375', '59531.3', '60187.1', '12398107', '123981.07', '7546291286.4393', '1'], 
            ['1720022400000', '60187.2', '60664.8', '56758', '58140.1', '19073867.4', '190738.674', '11146414317.0544', '1'], 
            ['1720108800000', '58140.2', '58821.8', '53222.3', '56678.3', '26484417.1', '264844.171', '14773857129.44', '1'], 
            ['1720195200000', '56678.3', '57466.4', '55820', '57310.2', '8919585.2', '89195.852', '5050067792.233', '1'], 
            ['1720281600000', '57310.2', '58499.8', '57066', '57634.4', '6227498.2', '62274.982', '3602758626.9132', '0']
        ]

    More information:
        https://www.okx.com/docs-v5/en/?shell#order-book-trading-market-data-get-candlesticks
    '''

    if symbol in ['', None]:
        print('\nThe symbol cannot be empty.\n')

        sys.exit(1)

    if interval not in [
            '1s', '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H',
            '12H', '1D', '2D', '3D', '1W', '1M', '3M'
    ]:
        print(
            '\nThe interval is invalid. Availble options: 1s, 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 2D, 3D, 1W, 1M, 3M.\n'
        )

        sys.exit(1)

    url = 'https://www.okx.com/api/v5/market/candles'
    params = {
        'instId': symbol,
        'bar': interval,
        'after': endTime,
        'limit': limit,
    }
    response = requests.get(url, params=params, verify=False)

    if response.status_code == 200:
        json_data = response.json()
        code = json_data.get('code')

        if code == '0':
            data = json_data.get('data', [])
            data = data[::-1]
            data = [[
                candlestick[0],
                candlestick[4],
                candlestick[7],
            ] for candlestick in data if candlestick[-1] == '1']

        else:
            msg = json_data.get('msg', '')
            print(
                "\nUnable to query OKX Perpetual Futures pair {} candlestick data. Return code is {} and error message is '{}'.\n"
                .format(symbol, code, msg))

            sys.exit(1)

    else:
        print(
            '\nUnable to query OKX Perpetual Futures pair {} candlestick data. Response code is {}.\n'
            .format(symbol, response.status_code))

        sys.exit(1)

    # print(data[0])
    # print(data[-1])
    # print(len(data))

    return data


def get_okx_perpetual_futures_24hr_price_change_statistics_data():
    '''
    Get perpetual futures 24hr price change statistics data from OKX.

    Response Parameters:
        instType: str
            Instrument type
        instId: str
            Instrument ID
        last: str
            Last traded price
        lastSz: str
            Last traded size. 0 represents there is no trading volume
        askPx: str
            Best ask price
        askSz: str
            Best ask size
        bidPx: str
            Best bid price
        bidSz: str
            Best bid size
        open24h: str
            Open price in the past 24 hours
        high24h: str
            Highest price in the past 24 hours
        low24h: str
            Lowest price in the past 24 hours
        volCcy24h: str
            24h trading volume, with a unit of currency.
        vol24h: str
            24h trading volume, with a unit of contract.
        sodUtc0: str
            Open price in the UTC 0
        sodUtc8: str
            Open price in the UTC 8
        ts: str
            Ticker data generation time, Unix timestamp format in milliseconds, e.g. 1597026383085

    Response Example:
        [
            {
                "instType":"SWAP",
                "instId":"LTC-USD-SWAP",
                "last":"9999.99",
                "lastSz":"1",
                "askPx":"9999.99",
                "askSz":"11",
                "bidPx":"8888.88",
                "bidSz":"5",
                "open24h":"9000",
                "high24h":"10000",
                "low24h":"8888.88",
                "volCcy24h":"2222",
                "vol24h":"2222",
                "sodUtc0":"0.1",
                "sodUtc8":"0.1",
                "ts":"1597026383085"
            },
            ...
        ]

    More information:
        https://www.okx.com/docs-v5/en/?shell#order-book-trading-market-data-get-tickers
    '''

    url = 'https://www.okx.com/api/v5/market/tickers'
    params = {
        'instType': 'SWAP',
    }
    response = requests.get(url, params=params, verify=False)

    if response.status_code == 200:
        json_data = response.json()
        code = json_data.get('code')

        if code == '0':
            data = json_data.get('data', [])
            data = [{
                'symbol':
                pair['instId'],
                'priceChange':
                str(float(pair['last']) - float(pair['open24h'])),
                'priceChangePercent':
                str(((float(pair['last']) - float(pair['open24h'])) /
                     float(pair['open24h'])) * 100),
                'lastPrice':
                pair['last'],
                'openPrice':
                pair['open24h'],
                'highPrice':
                pair['high24h'],
                'lowPrice':
                pair['low24h'],
                'volume':
                pair['vol24h'],
                'quoteVolume':
                pair['volCcy24h'],
            } for pair in data if pair['instId'].endswith('-USDT-SWAP')]

        else:
            msg = json_data.get('msg', '')
            print(
                "\nUnable to query OKX Perpetual Futures tickers 24hr price change statistics data. Return code is {} and error message is '{}'.\n"
                .format(code, msg))

            sys.exit(1)

    else:
        print(
            '\nUnable to query OKX Perpetual Futures tickers 24hr price change statistics data. Response code is {}.\n'
            .format(response.status_code))

        sys.exit(1)

    # print(data[0])

    return data


# get_okx_perpetual_futures_pairs()
# get_okx_perpetual_futures_candlestick_data(
#     'BTC-USDT-SWAP',
#     bar='1D',
#     #    after='1704065053000',
#     limit='365')
# get_okx_perpetual_futures_24hr_price_change_statistics_data()
