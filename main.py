import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import warnings
warnings.filterwarnings('ignore')

# ── Setup ──────────────────────────────────────────────────────────


app       = FastAPI(title="Stock Direction Predictor")
templates = Jinja2Templates(directory="templates")

# ── Load model, scaler, features once at startup ──────────────────
with open('models/best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print(f"Model loaded with {len(feature_cols)} features")

# ── Feature engineering (same as training) ────────────────────────
def engineer_features(df):
    # Returns
    df['return_1d']  = df['Close'].pct_change(1)
    df['return_5d']  = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)

    # Moving averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['price_to_sma20'] = df['Close'] / df['sma_20']
    df['price_to_sma50'] = df['Close'] / df['sma_50']
    df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)

    # RSI
    delta    = df['Close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs       = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger Bands
    sma = df['Close'].rolling(20).mean()
    std = df['Close'].rolling(20).std()
    df['bb_upper']    = sma + (2 * std)
    df['bb_lower']    = sma - (2 * std)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / sma

    # Volume
    df['volume_sma20'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma20']

    # Lag features
    for col in ['rsi_14', 'macd_hist', 'bb_position', 'return_1d', 'volume_ratio']:
        for lag in [1, 2, 3]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Sentiment (neutral baseline for live prediction)
    df['sentiment_mean']  = 0.0
    df['sentiment_count'] = 0.0
    df['sentiment_std']   = 0.0

    return df

def get_prediction(ticker: str):
    try:
        # Fetch 6 months of data — need enough history for 50-day SMA
        raw = yf.download(ticker, period='6mo', auto_adjust=True, progress=False)

        if raw.empty:
            return None, "Ticker not found or no data available"

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Engineer features
        df = engineer_features(raw.copy())
        df.dropna(inplace=True)

        if df.empty:
            return None, "Not enough data to compute features"

        # Get the most recent row — this is today's prediction
        latest = df[feature_cols].iloc[-1:].values
        latest_scaled = scaler.transform(latest)

        # Predict
        prob      = model.predict_proba(latest_scaled)[0]
        direction = "UP ↑" if prob[1] > 0.5 else "DOWN ↓"
        confidence = max(prob[0], prob[1]) * 100

        # Get current price info
        current_price = float(raw['Close'].iloc[-1])
        prev_price    = float(raw['Close'].iloc[-2])
        day_change    = ((current_price - prev_price) / prev_price) * 100

        return {
            "ticker":        ticker.upper(),
            "direction":     direction,
            "confidence":    float(round(confidence, 1)),
            "prob_up":       float(round(float(prob[1]) * 100, 1)),
            "prob_down":     float(round(float(prob[0]) * 100, 1)),
            "current_price": float(round(current_price, 2)),
            "day_change":    float(round(day_change, 2)),
            "is_up":         bool(prob[1] > 0.5)
        }, None

    except Exception as e:
        return None, str(e)

# ── Routes ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/predict")
async def predict(ticker: str):
    result, error = get_prediction(ticker.upper())
    if error:
        return {"error": error}
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "features": len(feature_cols)}