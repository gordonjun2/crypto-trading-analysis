import data_manager

# Retrieve the data
prices = data_manager.get_prices(['bitcoin', 'ethereum'])

# Calculate the daily returns
returns = prices.pct_change().dropna()
