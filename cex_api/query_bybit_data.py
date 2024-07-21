import requests
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import sys

urllib3.disable_warnings(InsecureRequestWarning)
'''
Bybit API Information:
    API Documentation: https://bybit-exchange.github.io/docs/v5/intro
'''


def get_bybit_perpetual_futures_pairs():
    '''
    Get all perpetual futures pairs from Bybit.

    Response Example:
        ['10000000AIDOGEUSDT', '1000000BABYDOGEUSDT', '1000000MOGUSDT', '1000000PEIPEIUSDT', '10000COQUSDT', '10000LADYSUSDT', 
        '10000NFTUSDT', '10000SATSUSDT', '10000STARLUSDT', '10000WENUSDT', '1000APUUSDT', '1000BEERUSDT', '1000BONKUSDT', 
        '1000BTTUSDT', ...]

    More information:
        https://bybit-exchange.github.io/docs/v5/market/instrument
    '''

    all_perpetual_futures_pairs = []
    next_page_cursor = ''

    while True:

        url = 'https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000&cursor={}'.format(
            next_page_cursor)
        response = requests.get(url, verify=False)

        if response.status_code == 200:
            json_data = response.json()
            code = json_data.get('retCode')

            if code == 0:
                result = json_data.get('result', {})
                data = result.get('list', [])

                perpetual_futures_pairs = [
                    instrument['symbol'] for instrument in data
                ]

                all_perpetual_futures_pairs.extend(perpetual_futures_pairs)
                next_page_cursor = result.get('nextPageCursor', '')

                if not next_page_cursor:
                    break

            else:
                msg = json_data.get('retMsg', '')
                print(
                    "\nUnable to query Bybit Perpetual Futures pairs. Return code is {} and error message is '{}'.\n"
                    .format(code, msg))

                sys.exit(1)
        else:
            print(
                '\nUnable to query Bybit Perpetual Futures pairs. Response code is {}.\n'
                .format(response.status_code))

            sys.exit(1)

    # print(len(all_perpetual_futures_pairs))

    print('\nSuccessfully queried Bybit Perpetual Futures pairs.')

    return all_perpetual_futures_pairs


def get_bybit_perpetual_futures_candlestick_data(symbol,
                                                 interval='D',
                                                 endTime='',
                                                 limit=365):
    '''
    Get perpetual futures candlestick data from Bybit. Charts are returned in groups based on the requested interval.

    Request Parameters:
        symbol: str
            Symbol name, like BTCUSDT, uppercase only.
        interval: str
            The interval of the candlestick data. 
            Possible values: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W.
        end: int
            The end timestamp (ms)
        limit: int
            Limit for data size per page. [1, 1000]. 
            Default: 200

    Response Parameters:
        startTime: str
            Start time of the candle (ms)
        openPrice: str
            Open price
        highPrice: str
            Highest price
        lowPrice: str
            Lowest price
        closePrice: str
            Close price. Is the last traded price when the candle is not closed.
        volume: str
            Trade volume. Unit of contract: pieces of contract. Unit of spot: quantity of coins.
        turnover: str
            Turnover. Unit of figure: quantity of quota coin

    Response Example:
        [
            [
                "1670608800000",
                "17071",
                "17073",
                "17027",
                "17055.5",
                "268611",
                "15.74462667"
            ],
            ...
        ]

    More information:
        https://bybit-exchange.github.io/docs/v5/market/kline
    '''

    if symbol in ['', None]:
        print('\nThe symbol cannot be empty.\n')

        sys.exit(1)

    if interval not in [
            '1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D',
            'M', 'W'
    ]:
        print(
            '\nThe interval is invalid. Availble options: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W.\n'
        )

        sys.exit(1)

    url = 'https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&end={}&limit={}'.format(
        symbol, interval, endTime, limit)
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        json_data = response.json()
        code = json_data.get('retCode')

        if code == 0:
            result = json_data.get('result', {})
            data = result.get('list', [])
            data = data[::-1]
            data = [[
                candlestick[0],
                candlestick[1],
                candlestick[2],
                candlestick[3],
                candlestick[4],
                candlestick[6],
            ] for candlestick in data]

        else:
            msg = json_data.get('retMsg', '')
            print(
                "\nUnable to query Bybit Perpetual Futures pair {} candlestick data. Return code is {} and error message is '{}'.\n"
                .format(code, msg))

            sys.exit(1)
    else:
        print(
            '\nUnable to query Bybit Perpetual Futures pair {} candlestick data. Response code is {}.\n'
            .format(response.status_code))

        sys.exit(1)

    # print(data[0])
    # print(data[-1])
    # print(len(data))

    return data


def get_bybit_perpetual_futures_24hr_price_change_statistics_data():
    '''
    Get perpetual futures 24hr price change statistics data from Bybit.

    Response Parameters:
        symbol: str
            Symbol name
        lastPrice: str
            Last price
        indexPrice: str
            Index price
        markPrice: str
            Mark price
        prevPrice24h: str
            Market price 24 hours ago
        price24hPcnt: str
            Percentage change of market price relative to 24h
        highPrice24h: str
            The highest price in last 24 hours
        lowPrice24h: str
            The lowest price in last 24 hours
        prevPrice1h: str
            Market price an hour ago
        openInterest: str
            Open interest size
        openInterestValue: str
            Open interest value
        turnover24h: str
            Turnover for 24h
        volume24h: str
            Volume for 24h
        fundingRate: str
            Funding rate
        nextFundingTime: str
            Next funding time (ms)
        predictedDeliveryPrice: str
            Predicted delivery price. It has value when 30 min before delivery
        basisRate: str
            Basis rate
        basis: str
            Basis
        deliveryFeeRate: str
            Delivery fee rate
        deliveryTime: str
            Delivery timestamp (ms)
        ask1Size: str
            Best ask size
        bid1Price: str
            Best bid price
        ask1Price: str
            Best ask price
        bid1Size: str
            Best bid size
        preOpenPrice: str
            Estimated pre-market contract open price
                - The value is meaningless when entering continuous trading phase
        preQty: str
            Estimated pre-market contract open qty
                - The value is meaningless when entering continuous trading phase
        curPreListingPhase: str
            The current pre-market contract phase

    Response Example:
        [
            {
                "symbol": "BTCUSD",
                "lastPrice": "16597.00",
                "indexPrice": "16598.54",
                "markPrice": "16596.00",
                "prevPrice24h": "16464.50",
                "price24hPcnt": "0.008047",
                "highPrice24h": "30912.50",
                "lowPrice24h": "15700.00",
                "prevPrice1h": "16595.50",
                "openInterest": "373504107",
                "openInterestValue": "22505.67",
                "turnover24h": "2352.94950046",
                "volume24h": "49337318",
                "fundingRate": "-0.001034",
                "nextFundingTime": "1672387200000",
                "predictedDeliveryPrice": "",
                "basisRate": "",
                "deliveryFeeRate": "",
                "deliveryTime": "0",
                "ask1Size": "1",
                "bid1Price": "16596.00",
                "ask1Price": "16597.50",
                "bid1Size": "1",
                "basis": ""
            },
            ...
        ]

    More information:
        https://bybit-exchange.github.io/docs/v5/market/tickers
    '''

    url = 'https://api.bybit.com/v5/market/tickers?category=linear'
    response = requests.get(url, verify=False)

    if response.status_code == 200:
        json_data = response.json()
        code = json_data.get('retCode')

        if code == 0:
            result = json_data.get('result', {})
            data = result.get('list', [])
            data = [{
                'symbol':
                pair['symbol'],
                'priceChange':
                float(pair['lastPrice']) - float(pair['prevPrice24h']),
                'priceChangePercent':
                pair['price24hPcnt'],
                'lastPrice':
                pair['lastPrice'],
                'openPrice':
                pair['prevPrice24h'],
                'highPrice':
                pair['highPrice24h'],
                'lowPrice':
                pair['lowPrice24h'],
                'volume':
                pair['volume24h'],
                'quoteVolume':
                pair['turnover24h'],
            } for pair in data]

        else:
            msg = json_data.get('retMsg', '')
            print(
                "\nUnable to query Bybit Perpetual Futures tickers 24hr price change statistics data. Return code is {} and error message is '{}'.\n"
                .format(code, msg))

            sys.exit(1)
    else:
        print(
            '\nUnable to query Bybit Perpetual Futures tickers 24hr price change statistics data. Response code is {}.\n'
            .format(response.status_code))

        sys.exit(1)

    # print(data)

    return data


# get_bybit_perpetual_futures_pairs()

# get_bybit_perpetual_futures_candlestick_data(
#     'BTCUSDT',
#     interval='D',
#     #    end=1688860800000,
#     limit=365)

# get_bybit_perpetual_futures_24hr_price_change_statistics_data()
