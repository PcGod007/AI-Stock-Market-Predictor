import yfinance as yf
try:
    data = yf.download('GOOG', start='2023-01-01', end='2023-12-31', progress=False)
    print("Columns:", data.columns)
    print("Empty:", data.empty)
    print("Head:\n", data.head())
except Exception as e:
    print("Error:", e)
