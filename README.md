# Stock Direction Predictor

An end-to-end machine learning system that predicts 5-day stock price direction using technical indicators and NLP sentiment analysis.

## Live Demo
🔗 **Live demo:** https://stock-ml-predictor.onrender.com

## What it does
- Fetches live stock data via yfinance
- Engineers 31 features (RSI, MACD, Bollinger Bands, lag features, sentiment)
- Predicts whether a stock will be higher or lower in 5 trading days
- Serves predictions via a FastAPI REST endpoint with a web UI

## Tech stack
- **Data**: yfinance, pandas, numpy
- **ML**: scikit-learn, XGBoost
- **NLP**: HuggingFace Transformers, FinBERT
- **API**: FastAPI, uvicorn
- **Frontend**: HTML, CSS, JavaScript

## Key ML decisions
- Walk-forward validation to prevent lookahead bias in time-series data
- Class balancing via scale_pos_weight to handle market imbalance
- Lag features to capture sequential price patterns
- FinBERT sentiment scoring on financial news headlines

## Results
| Model | AUC |
|-------|-----|
| Logistic Regression | 0.492 |
| Random Forest | 0.459 |
| XGBoost (tuned) | 0.454 |
| XGBoost + Sentiment | 0.454 |

AUC scores near 0.5 are consistent with the Efficient Market Hypothesis for short-term technical prediction — the project demonstrates honest ML evaluation methodology.

## Project structure
