import pandas as pd

trades = pd.read_csv("data/historical_data.csv")
sentiment = pd.read_csv("data/fear_greed_index.csv")

print("Trades Columns:")
print(trades.columns)

print("\nSentiment Columns:")
print(sentiment.columns)