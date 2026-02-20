# AGENTS.md — sqlserver_copilot (NASDAQ ML Pipeline)

## Overview
This repo does NOT contain CrewAI agents. It is a **scikit-learn ML training pipeline** for NASDAQ 100 stock predictions.

## ML Pipeline Architecture

```
[nasdaq_100_hist_data] (SQL Server)
        │
        ▼
  feature_engineering.py  (50+ features)
        │
        ▼
  feature_selection.py  (SelectKBest → top 20)
        │
        ▼
  train_model.py  (Gradient Boosting Classifier)
        │
        ▼
  models/nasdaq_gb_model.pkl  (serialized model)
        │
        ▼
  predict_daily.py  (daily predictions → SQL Server)
        │
        ▼
  [ml_trading_predictions]  +  [ml_prediction_summary]
```

## Downstream Consumers
- **stockdata_agenticai** — 7 agents read these predictions for daily briefing
- **streamlit-trading-dashboard** — Displays predictions and tracks accuracy
- Cross-strategy analysis joins `ml_trading_predictions` with `vw_PowerBI_AI_Technical_Combos`

## Model Details
- **Algorithm**: Gradient Boosting Classifier (sklearn)
- **Input**: 50+ engineered features from OHLCV data
- **Selection**: Top 20 features via SelectKBest (f_classif)
- **Output**: Buy/Sell signal + confidence % + signal strength
- **Schedule**: Daily 6:00 AM predictions, weekly full retrain

## Ecosystem Role
This is one of 3 ML pipeline repos (NASDAQ/NSE/Forex) that feed the shared `stockdata_db`. The agentic AI layer and dashboard consume predictions downstream.
