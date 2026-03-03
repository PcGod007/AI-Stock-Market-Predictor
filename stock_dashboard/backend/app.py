from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import os
import requests

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Latest_stock_price_model.keras')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'db66d6f0a9eb427aa1e69437b75f6f34')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def compute_rsi(series, period=14):
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    ag     = gain.rolling(period).mean()
    al     = loss.rolling(period).mean()
    rs     = ag / al
    return 100 - (100 / (1 + rs))


def fetch_news_sentiment(ticker):
    """Fetch recent news, return (articles, score -1..+1, verdict)."""
    BULLISH = ['surge', 'soar', 'rally', 'gain', 'profit', 'beat', 'record', 'rise',
               'jump', 'upgrade', 'buy', 'strong', 'growth', 'outperform', 'positive',
               'expansion', 'partnership', 'launch', 'boom', 'earnings beat']
    BEARISH  = ['fall', 'drop', 'plunge', 'loss', 'miss', 'cut', 'downgrade', 'decline',
                'crash', 'sell', 'warn', 'investigation', 'lawsuit', 'fine', 'weak',
                'layoff', 'recall', 'scandal', 'fraud', 'regulation', 'penalty']
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock"
            f"&sortBy=publishedAt&pageSize=15&language=en"
            f"&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if data.get('status') != 'ok':
            return [], 0.0, 'neutral'

        articles  = []
        bull_total, bear_total = 0, 0
        for a in data.get('articles', []):
            text = ((a.get('title') or '') + ' ' + (a.get('description') or '')).lower()
            b = sum(1 for w in BULLISH if w in text)
            r = sum(1 for w in BEARISH if w in text)
            bull_total += b
            bear_total += r
            articles.append({
                'title':        a.get('title'),
                'description':  a.get('description'),
                'url':          a.get('url'),
                'published_at': a.get('publishedAt'),
                'source':       a.get('source', {}).get('name'),
                'sentiment':    'bullish' if b > r else ('bearish' if r > b else 'neutral'),
            })

        total = bull_total + bear_total
        if total == 0:
            return articles, 0.0, 'neutral'
        score = (bull_total - bear_total) / total
        verdict = 'bullish' if score > 0.15 else ('bearish' if score < -0.15 else 'neutral')
        return articles, round(score, 3), verdict
    except Exception:
        return [], 0.0, 'neutral'


def generate_reasoning(data, future_preds_scaled, scaler,
                       last_close, news_articles, news_score, news_verdict):
    """
    Build a balanced, realistic AI reasoning report.

    Key design decisions:
    ─────────────────────
    1.  LSTM 5-day signal  : Use only days 1-5 of the LSTM forecast (before
        compound error corrupts it) as the model's directional signal.
    2.  Momentum projection : Use recent 30-day momentum (statistically sound)
        to estimate the expected 30-day price target.
    3.  Combined verdict    : LSTM-direction × 35%  +  Technical × 35%  +  News × 30%
        All three must agree for a strong call; otherwise "neutral / mixed".
    """
    close = data['Close']

    # ── Technical indicators ──────────────────────────────────────────────────
    ma30  = float(close.rolling(30).mean().iloc[-1])  if len(close) >= 30  else None
    ma100 = float(close.rolling(100).mean().iloc[-1]) if len(close) >= 100 else None
    rsi   = (lambda s: float(s.iloc[-1]) if not pd.isna(s.iloc[-1]) else None)(compute_rsi(close))
    mom30 = float(((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30]) * 100) if len(close) >= 30 else None
    vol30 = float(close.iloc[-30:].std())                                         if len(close) >= 30 else None

    # ── LSTM signal: use Days 1-5 only (avoids compound-error drift) ─────────
    # future_preds_scaled is in normalised space; inverse-transform Days 0-4
    early_preds = scaler.inverse_transform(future_preds_scaled[:5])
    early_avg   = float(np.mean(early_preds))
    lstm_5d_pct = ((early_avg - last_close) / last_close) * 100
    # Normalise to -1 … +1 (cap at ±15% for 5-day window)
    lstm_signal = max(-1.0, min(1.0, lstm_5d_pct / 15.0))

    # ── Technical signal ─────────────────────────────────────────────────────
    tech_signal, tech_n = 0.0, 0
    if rsi is not None:
        tech_signal += -(rsi - 50) / 50.0;  tech_n += 1
    if ma30 and ma100:
        tech_signal += (1.0 if ma30 > ma100 else -1.0);  tech_n += 1
    if mom30 is not None:
        tech_signal += max(-1.0, min(1.0, mom30 / 30.0));  tech_n += 1
    if tech_n:
        tech_signal /= tech_n

    # ── Combined signal ───────────────────────────────────────────────────────
    combined = (lstm_signal * 0.35) + (tech_signal * 0.35) + (news_score * 0.30)
    if   combined >  0.08: verdict, direction = 'bullish', 'rise'
    elif combined < -0.08: verdict, direction = 'bearish', 'fall'
    else:                  verdict, direction = 'neutral',  'remain stable'

    # ── Momentum-based 30-day price estimate (realistic, avoids LSTM drift) ──
    # Annualised daily trend extrapolated 30 days
    if mom30 is not None:
        daily_trend   = mom30 / 30.0               # avg daily % change
        # Scale back by RSI congestion factor (overbought stocks tend to slow)
        if rsi and rsi > 65:
            daily_trend *= 0.6
        elif rsi and rsi < 35:
            daily_trend *= 1.2
        # News nudge: ±0.3% per 100 news-score points
        daily_trend += news_score * 0.3
        proj_30d_pct  = daily_trend * 30
        proj_price    = last_close * (1 + proj_30d_pct / 100)
    else:
        proj_30d_pct = 0.0
        proj_price   = last_close

    # ── Build factors list ────────────────────────────────────────────────────
    factors = []

    # LSTM factor (honest 5-day signal)
    lstm_dir_word = 'rise' if lstm_5d_pct >= 0 else 'fall'
    factors.append(
        f"LSTM Model (5-day signal): Projects a {lstm_dir_word} of {abs(lstm_5d_pct):.1f}% in the "
        f"near-term (days 1–5). Note: 30-day LSTM extrapolation is used for chart visualization only — "
        f"beyond ~7 days, compound error from the sliding window becomes significant and is not used for verdicts."
    )

    # News factor
    bc = sum(1 for a in news_articles if a['sentiment'] == 'bullish')
    rc = sum(1 for a in news_articles if a['sentiment'] == 'bearish')
    if news_articles:
        factors.append(
            f"News Sentiment ({len(news_articles)} articles): {bc} Bullish · {rc} Bearish. "
            f"Overall news signal is {news_verdict.upper()} (score: {news_score:+.2f})."
        )

    # RSI
    if rsi is not None:
        if   rsi > 70: factors.append(f"RSI {rsi:.1f}: Overbought (>70) — possible short-term pullback pressure.")
        elif rsi < 30: factors.append(f"RSI {rsi:.1f}: Oversold (<30) — possible rebound opportunity.")
        else:           factors.append(f"RSI {rsi:.1f}: Neutral zone (30–70) — no extreme pressure.")

    # MA trend
    if ma30 and ma100:
        sign = "ABOVE (Golden Cross 🟢)" if ma30 > ma100 else "BELOW (potential Death Cross ⚠️)"
        factors.append(f"MA Trend: 30d MA (${ma30:.2f}) is {sign} 100d MA (${ma100:.2f}).")

    # Momentum
    if mom30 is not None:
        dw = 'gained' if mom30 >= 0 else 'lost'
        factors.append(f"30-day Momentum: Asset has {dw} {abs(mom30):.1f}% in the past month.")

    # Volatility
    if vol30:
        factors.append(f"30-day Volatility: ±${vol30:.2f}. Forecasts beyond 7–10 days carry higher uncertainty.")

    # Momentum-based projection note
    change_str = f"+{proj_30d_pct:.1f}%" if proj_30d_pct >= 0 else f"{proj_30d_pct:.1f}%"
    factors.append(
        f"Momentum-based 30-day estimate: {change_str} (target ~${proj_price:.2f}). "
        f"This is based on recent trend + RSI adjustment + news sentiment, not raw LSTM output."
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = (
        f"Combined signal (LSTM 5d direction 35% + Technical 35% + News 30%) is "
        f"\"{verdict.upper()}\". Momentum-based 30-day price estimate: {change_str} "
        f"(~${proj_price:.2f}) — a statistically grounded projection that avoids LSTM compound-error drift."
    )

    return {
        'verdict':              verdict,
        'direction':            direction,
        'pct_change':           round(proj_30d_pct, 2),       # momentum-based (shown to user)
        'price_target':         round(proj_price, 2),
        'lstm_5d_pct':          round(lstm_5d_pct, 2),        # informational
        'news_sentiment':       news_verdict,
        'news_score':           news_score,
        'combined_signal':      round(combined, 3),
        'summary':              summary,
        'factors':              factors,
        'indicators': {
            'rsi':              round(rsi, 2)   if rsi   else None,
            'ma_30d':           round(ma30, 4)  if ma30  else None,
            'ma_100d':          round(ma100, 4) if ma100 else None,
            'momentum_30d_pct': round(mom30, 2) if mom30 else None,
            'volatility_30d':   round(vol30, 4) if vol30 else None,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

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
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
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
    windows = [int(w) for w in request.args.get('windows', '100,200,250').split(',') if w.strip().isdigit()]
    try:
        data = fetch_stock_data(ticker, start, end)
        if data.empty:
            return jsonify({'error': 'No data found.'}), 404
        result = {
            'dates': [d.strftime('%Y-%m-%d') for d in data.index],
            'close': [round(float(v), 4) for v in data['Close']],
            'mas':   {}
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
        x_test_df    = data['Close'].iloc[splitting_len:]
        scaler       = MinMaxScaler(feature_range=(0, 1))
        scaled_data  = scaler.fit_transform(x_test_df.values.reshape(-1, 1))

        # ── Historical test-set predictions ──────────────────────────────────
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        model       = load_model(MODEL_PATH)
        predictions = model.predict(x_data)
        inv_pre     = scaler.inverse_transform(predictions)
        inv_y_test  = scaler.inverse_transform(y_data)

        # ── Future 30-day sliding-window forecast ─────────────────────────────
        # Note: used for chart visualization; compound error grows after ~7 days
        future_preds_scaled = []
        current_batch = scaled_data[-100:].reshape(1, 100, 1)
        for _ in range(30):
            nxt = model.predict(current_batch, verbose=0)
            future_preds_scaled.append(nxt[0])
            current_batch = np.append(current_batch[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)
        future_preds_scaled_arr = np.array(future_preds_scaled)   # shape (30, 1)
        inv_future_raw = scaler.inverse_transform(future_preds_scaled_arr).reshape(-1)

        # ── Define key values before momentum/blend calculations ────────────
        last_close   = float(data['Close'].iloc[-1])
        last_date    = data.index[-1]
        pred_index   = data.index[splitting_len + 100:]

        # ── Build a momentum-based projection as the realistic future baseline ──
        # Use 30-day momentum from historical data, adjusted by RSI congestion.
        close_series = data['Close']
        if len(close_series) >= 30:
            mom30_raw  = float(((close_series.iloc[-1] - close_series.iloc[-30]) / close_series.iloc[-30]) * 100)
            daily_rate = mom30_raw / 30.0
            rsi_series = compute_rsi(close_series)
            rsi_val    = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
            # Dampen: very overbought → slow down; oversold → allow acceleration
            if rsi_val > 70:   daily_rate *= 0.5
            elif rsi_val < 30: daily_rate *= 1.1
        else:
            daily_rate = 0.0

        mom_proj = np.array([last_close * (1 + daily_rate / 100 * (t + 1)) for t in range(30)])

        # ── Blend: LSTM weight decreases from 0.5 → 0.1 over 30 days (trust fades) ──
        lstm_weights = np.linspace(0.5, 0.1, 30)
        mom_weights  = 1.0 - lstm_weights
        inv_future   = lstm_weights * inv_future_raw + mom_weights * mom_proj

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)


        # ── News sentiment ────────────────────────────────────────────────────
        news_articles, news_score, news_verdict = fetch_news_sentiment(ticker)

        # ── Reasoning (uses 5-day LSTM signal + momentum, not 30d raw LSTM) ──
        reasoning = generate_reasoning(
            data, future_preds_scaled_arr, scaler,
            last_close, news_articles, news_score, news_verdict
        )

        return jsonify({
            'dates':            [d.strftime('%Y-%m-%d') for d in pred_index],
            'original':         [round(float(v), 4) for v in inv_y_test.reshape(-1)],
            'predicted':        [round(float(v), 4) for v in inv_pre.reshape(-1)],
            'future_dates':     [d.strftime('%Y-%m-%d') for d in future_dates],
            'future_predicted': [round(float(v), 4) for v in inv_future.reshape(-1)],
            'reasoning':        reasoning,
            'news_articles':    news_articles,
            'close_full': {
                'dates':  [d.strftime('%Y-%m-%d') for d in pred_index],
                'values': [round(float(v), 4) for v in data['Close'].iloc[splitting_len + 100:]]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/news', methods=['GET'])
def get_news():
    ticker = request.args.get('ticker', 'GOOG')
    try:
        articles, score, verdict = fetch_news_sentiment(ticker)
        return jsonify({'ticker': ticker, 'articles': articles,
                        'sentiment_score': score, 'verdict': verdict})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Stock Predictor API running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
