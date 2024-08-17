# crypto-pair-trading

### **Overview**
This repository contains Jupyter notebooks implementing several pair trading strategies designed to minimize risk while maximizing returns. These strategies focus on profiting from relative price movements between cryptocurrencies, leveraging correlations and market dynamics.
<br>
<br>

### Included Strategies
Please look into the respective Jupyter notebook for more explanations and details.

1. **Pairs-Trading (Mean Reversion) Strategy Analysis** (*cointegration-pair-trading.ipynb*)

Pair Trading Mean Reversion strategy is a market-neutral trading strategy that involves simultaneously buying and selling two correlated financial instruments, such as stocks, ETFs, or currencies, to profit from the relative price movement between them.

2. **Pairs-Trading (Negative / Low Correlation) Strategy Analysis** (*correlation-pair-trading.ipynb*)

The negative/low correlation pairs trading strategy involves taking advantage of the performance divergence between two loosely correlated or negatively correlated assets by longing the stronger asset and shorting the weaker one.

3. **Pairs-Trading (Beta Neutral) Strategy Analysis (Auto)** (*beta-neutral-pair-trading-auto.ipynb*)

The beta neutral trading strategy is a market-neutral approach designed to eliminate systematic market risk by constructing a portfolio that has a net beta of zero. This strategy aims to generate alpha (excess returns above the market) by taking both long and short positions in securities such that the overall portfolio is insulated from broad market movements. This notebook uses CVXPY package to optimise portfolio weights given a set of data.

4. **Pairs-Trading (Beta Neutral) Strategy Analysis (Manual)** (*beta-neutral-pair-trading-manual.ipynb*)

The strategy analysis is the same as 3., but this notebook allow users to manually set the portfolio weights and plot the returns of portfolio compared to market.

5. **Crypto Sentiment on Chart Analysis** (*crypto-sentiment-on-chart.ipynb*)

This notebook aims to explore the potential relationship between sentiment on 4chan's Business and Finance board and the price action of selected cryptocurrencies. 4chan is a valuable source for sentiment analysis as its posts are freely accessible via its API, making it a cost-effective alternative to platforms like Twitter. The primary objective of this analysis is to investigate whether sentiment derived from 4chan posts can be correlated with the price movements of specific cryptocurrencies. By binning 4chan posts into specific time intervals and aligning them with cryptocurrency price charts, we can calculate a net sentiment score for each bin and observe any patterns or trends that may emerge.
<br>
<br>

### Notes
- This is a **personal project** of mine and was not created under any entity.
- Please be aware that the results provided by this project might not be 100% accurate due to potential bugs.
- Please do not rely on this software to make financial decisions. **NFA**.
<br>

### Future Updates
- Collect open-source news data and overlay it on the price charts of selected cryptocurrencies to analyze how news events influence price movements.
<br>

### Data Sources and APIs
- Binance API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - GET /api/v3/exchangeInfo (get tickers)
        - GET /api/v3/klines (get OHLC data)
        - GET /api/v3/ticker/24hr (get 24hr price change percent)
- Bybit API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - /v5/market/instruments-info (get tickers)
        - /v5/market/kline (get OHLC data)
        - /v5/market/tickers (get 24hr price change percent)
- OKX API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - GET /api/v5/public/instruments (get tickers)
        - GET /api/v5/market/candles (get recent OHLC data)
        - GET /api/v5/market/history-candles (get historical OHLC data)
        - GET /api/v5/market/tickers (get 24hr price change percent)
<br>

### **Installation**
1. Clone the repository using
    ```
    git clone https://github.com/gordonjun2/crypto-pair-trading.git
    ```
2. Navigate to the root directory of the repository.
    ```
    cd crypto-pair-trading
    ```
3. Create a Python virtual environment for this project.
    ```
    python3 -m venv venv
    ```
4. Activate the Python virtual environment.
    ```
    source venv/bin/activate
    ```
5. Install required packages.
    ```
    pip install -r requirements.txt
    ```
6. Install jupyter kernel for the virtual environment.
    ```
    python -m ipykernel install --user --name venv --display-name "crypto-trading-analysis"
    ```
<br>

### How to Use

#### Price Data Download
- Data Manager Arguments
    ```
    -c      : Select CEX to get the perpetual futures' price data. 
              Available values: binance, okx, bybit.

    -i      : Select the price data interval. 
              Available values for each CEX:
                - binance: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 
                           12h, 1d, 3d, 1w, 1M.
                - okx:     1s, 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 
                           12H, 1D, 2D, 3D, 1W, 1M, 3M.
                - bybit:   1, 3, 5, 15, 30, 60, 120, 240, 360, 720, 
                           D, M, W.
                

    -e      : Enter the price data end time in timestamp ms. Start 
              time will be interval x limit before end time.

    -l      : Select the no. of candlesticks to return. For example,
              if 'binance' is chosen as the CEX and '1d' is chosen
              as the interval, then selecting 1000 in this argument
              will mean that 1000 days of price data for all 
              available perpetual futures' assets will be downloaded.
              Maximum available values for each CEX:
                - binance: max 1500
                - okx:     max 300
                - bybit:   max 1000
    ```
- Run the command below to download the price data:
    ```
    python data_manager.py <set options and arguments here>

    Eg. 
    python data_manager.py -c binance -i 1d -l 365
    ```
- The data will be downloaded as *.pkl* file in the ***saved_data*** directory.
    - The ***saved_data*** directory is organised in this manner:
        ```
        saved_data/
            binance/
                1h/
                    <Asset 1 Price Data .pkl>
                    <Asset 2 Price Data .pkl>
                    ...
                4h/
                    ...
                1d/
                    ...
                ...
            okx/
                1H/
                    <Asset 1 Price Data .pkl>
                    <Asset 2 Price Data .pkl>
                    ...
                4H/
                    ...
                1D/
                    ...
                ...
            bybit/
                60/
                    <Asset 1 Price Data .pkl>
                    <Asset 2 Price Data .pkl>
                    ...
                360/
                    ...
                D/
                    ...
                ...
        ```
    - Price data in each path will be overwritten *data_manager.py* is ran with the same argument value for *CEX* and *Interval*. Eg.
        - The command below will download price data into the ./saved_data/binance/1d/ path.
            ```
            python data_manager.py -c binance -i 1d -l 365
            ```
        - The command below will delete and replace the price data in the ./saved_data/binance/1d/ path.
            ```
            python data_manager.py -c binance -i 1d -l 730
            ```

#### Social Media Post Data Download
- To use the *crypto-sentiment-on-chart.ipynb* notebook, you need to download social media post data. 
- Currently, only post data from 4chan and selected dataset from Hugging Face can be download.
- To download the latest few post data from 4chan:
    - Change directory to *social_media_analysis*
        ```
        cd social_media_analysis
        ```
    - Run the command below:
        ```
        python data_manager.py
        ```
    - The data will be downloaded as *.pkl* file in the ***social_media_analysis/saved_data/4chan/*** directory.
    - The data will be updated whenever the download command is ran. The dataframe in the *.pkl* will increase in size.
- To download the selected dataset from Hugging Face:
    - Change directory to *social_media_analysis*
        ```
        cd social_media_analysis
        ```
    - Run the command below:
        ```
        python download_hugging_face_data.py
        ```
    - The data will be downloaded as *.pkl* file in the ***social_media_analysis/saved_data/hugging_face/*** directory.

#### Analysis
- After the price data is downloaded, you can start to use the Jupyter notebooks.
- To open the Jupyter notebook, run
    ```
    jupyter notebook <selected .ipynb>

    Eg.
    jupyter notebook beta-neutral-pair-trading-auto.ipynb
    ```
- Continue to follow the instructions and explanations in the respective notebook to perform the trading analysis.
- To execute the cell in the notebook, press 'SHIFT' + 'ENTER'.

<br>

### Others
- [Add, Update, and Remove Git Submodule](https://phoenixnap.com/kb/git-add-remove-update-submodule)

<br>
