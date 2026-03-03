from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Latest_stock_price_model.keras')

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

@app.route('/api/stock-data', methods=['GET'])
def get_stock_data():
    ticker = request.args.get('ticker', 'GOOG')
    start  = request.args.get('start', '2004-01-01')
    end    = request.args.get('end',   datetime.now().strftime('%Y-%m-%d'))

    try:
        data = fetch_stock_data(ticker, start, end)
        if data.empty:
            return jsonify({'error': 'No data found for the given ticker / date range.'}), 404

        data = data.reset_index()
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        # Round all numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = data[col].round(4)

        records = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient='records')
        return jsonify({'ticker': ticker, 'data': records})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/moving-average', methods=['GET'])
def get_moving_average():
    ticker  = request.args.get('ticker', 'GOOG')
    start   = request.args.get('start', '2004-01-01')
    end     = request.args.get('end',   datetime.now().strftime('%Y-%m-%d'))
    windows_param = request.args.get('windows', '100,200,250')
    windows = [int(w) for w in windows_param.split(',') if w.strip().isdigit()]

    try:
        data = fetch_stock_data(ticker, start, end)
        if data.empty:
            return jsonify({'error': 'No data found.'}), 404

        result = {
            'dates':  [d.strftime('%Y-%m-%d') for d in data.index],
            'close':  [round(float(v), 4) for v in data['Close']],
            'mas':    {}
        }

        for w in windows:
            ma = data['Close'].rolling(w).mean()
            result['mas'][str(w)] = [None if pd.isna(v) else round(float(v), 4) for v in ma]

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['GET'])
def get_prediction():
    ticker = request.args.get('ticker', 'GOOG')
    start  = request.args.get('start', '2004-01-01')
    end    = request.args.get('end',   datetime.now().strftime('%Y-%m-%d'))

    try:
        data = fetch_stock_data(ticker, start, end)
        if data.empty:
            return jsonify({'error': 'No data found.'}), 404

        splitting_len = int(len(data) * 0.7)
        x_test_df    = pd.DataFrame(data['Close'][splitting_len:])

        scaler      = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test_df[['Close']])

        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        model       = load_model(MODEL_PATH)
        predictions = model.predict(x_data)

        inv_pre    = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        pred_index = data.index[splitting_len + 100:]

        result = {
            'dates':      [d.strftime('%Y-%m-%d') for d in pred_index],
            'original':   [round(float(v), 4) for v in inv_y_test.reshape(-1)],
            'predicted':  [round(float(v), 4) for v in inv_pre.reshape(-1)],
            'close_full': {
                'dates': [d.strftime('%Y-%m-%d') for d in data.index[splitting_len + 100:]],
                'values': [round(float(v), 4) for v in data['Close'][splitting_len + 100:]]
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Stock Predictor API running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
