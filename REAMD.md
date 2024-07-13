# crypto-pair-trading

### **Overview**

<br>

### Notes
- This is a **personal project** of mine and was not created under any entity.
- Please be aware that the results provided by this project might not be 100% accurate due to potential bugs.
- Please do not rely on this software to make financial decisions. **NFA**.

<br>

### Future Updates

<br>

### How It Works

<br>

### Data Sources
- Yahoo Finance API
- Binance API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - https://api.binance.com/api/v3/exchangeInfo (get tickers)
        - GET /api/v3/klines (get OHLC data)
        - GET /api/v3/ticker/24hr (get 24hr price change percent)
- Bybit API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - /v5/market/kline (get OHLC data)
        - /v5/market/instruments-info (get tickers)
        - /v5/market/tickers (get 24hr price change percent)
- OKX API
    - DO NOT NEED KEYS
    - Endpoints to use:
        - GET /api/v5/market/candles (get recent OHLC data)
        - GET /api/v5/market/history-candles (get historical OHLC data)
        - GET /api/v5/public/instruments (get tickers)
        - GET /api/v5/market/tickers (get 24hr price change percent)


### **APIs**
1. [Bitquery](https://docs.bitquery.io/docs/intro/)
    - Used to query on-chain DEX trades data
    - ***REQUIRED*** and ***API keys REQUIRED***

<br>

### **Installation**
1. Clone the repository using
    ```
    git clone https://github.com/gordonjun2/underground-dex-trades.git
    ```
2. Navigate to the root directory of the repository.
    ```
    cd underground-dex-trades
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
    python -m ipykernel install --user --name myenv --display-name "beta_neutral"
    ```
7. Rename private keys file.
    ```
    mv private_template.ini private.ini
    ```
8. Sign up for *Bitquery API* [here](https://bitquery.io/) and get the respective keys in the link below.
    - BITQUERY_CLIENT_ID: https://account.bitquery.io/user/api_v2/applications
    - BITQUERY_CLIENT_SECRET: https://account.bitquery.io/user/api_v2/applications
    - BITQUERY_V1_API_KEY: https://account.bitquery.io/user/api_v1/api_keys
9. Sign up for *Vybe Network API* [here](https://www.vybenetwork.com/) and get the respective keys in the link below.
    - VYBE_NETWORK_X_API_KEY: https://alpha.vybenetwork.com/dashboard/api-management
10. Copy and save the keys into the private keys file.
<br>

### **Configuration**
- ```BITQUERY_CLIENT_ID```: Bitquery Client ID from ```private.ini``` file

<br>
