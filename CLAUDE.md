# CLAUDE.md — sqlserver_copilot (NASDAQ ML Training Pipeline)

> **Project context file for AI assistants (Claude, Copilot, Cursor).**

---

## 1. SYSTEM OVERVIEW

This is the **NASDAQ ML training pipeline** — one of **7 interconnected repositories** that form an end-to-end AI-powered stock trading analytics platform. All repos share a single SQL Server database (`stockdata_db`).

### Repository Map

| Layer | Repo | Purpose |
|-------|------|---------|
| Data Ingestion | `stockanalysis` | ETL: yfinance/Alpha Vantage → SQL Server |
| SQL Infrastructure | `sqlserver_mcp` | .NET 8 MCP Server (Microsoft MssqlMcp) — 7 tools (ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData) via stdio transport for AI IDE ↔ SQL Server |
| Dashboard | `streamlit-trading-dashboard` | 40+ views, signal tracking, Streamlit UI |
| **ML: NASDAQ** ⭐ | **`sqlserver_copilot`** | **THIS REPO** — Gradient Boosting → `ml_trading_predictions` |
| ML: NSE | `sqlserver_copilot_nse` | 5-model ensemble → `ml_nse_trading_predictions` |
| ML: Forex | `sqlserver_copilot_forex` | XGBoost/LightGBM → `forex_ml_predictions` |
| Agentic AI | `stockdata_agenticai` | 7 CrewAI agents, daily briefing email |

---

## 2. THIS REPO: sqlserver_copilot

### Purpose
Trains a **Gradient Boosting classifier** on NASDAQ 100 stocks to predict Buy/Sell signals, then writes predictions to `ml_trading_predictions`. Also manages data freshness checks and conditional retraining.

### Daily Schedule (Windows Task Scheduler)
```
06:00 AM  Daily prediction run     → ml_trading_predictions
06:30 AM  Data freshness check     → Conditional retrain if stale
Weekly     Full retrain             → Updated model files
```

### Key Files

```
sqlserver_copilot/
├── src/
│   ├── predict_daily.py         # Daily prediction script (entry point for scheduler)
│   ├── train_model.py           # Full training pipeline
│   ├── check_data_freshness.py  # Freshness check + conditional retrain
│   ├── feature_engineering.py   # 50+ feature calculations
│   ├── feature_selection.py     # SelectKBest top-20 feature selection
│   ├── sql_queries.py           # All SQL queries for data retrieval
│   └── model_utils.py          # Model save/load utilities
├── models/
│   ├── nasdaq_gb_model.pkl      # Trained Gradient Boosting model
│   └── feature_columns.pkl     # Selected feature names
├── config/
│   └── settings.py              # .env configuration loader
├── logs/
│   └── *.log                    # Execution logs
└── tests/
    └── test_predictions.py      # Validation tests
```

---

## 3. ML MODEL DETAILS

### Model Architecture
- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
- **Target**: Buy/Sell signal (binary classification)
- **Features**: 50+ engineered features → top 20 selected via SelectKBest
- **Training Data**: `nasdaq_100_hist_data` (127,889 rows) + technical indicators

### Feature Categories (50+)
| Category | Examples |
|----------|---------|
| Price-based | Returns (1d/5d/10d/20d), price ratios, gap analysis |
| Moving Averages | SMA (5/10/20/50/200), EMA (12/26), crossover signals |
| Momentum | RSI (14), MACD, Stochastic %K/%D, ROC, Williams %R |
| Volatility | Bollinger Bands (width/position), ATR, historical volatility |
| Volume | Volume ratios, OBV, volume-price trend |
| Market Context | Sector relative strength, market breadth |

### Feature Selection
- Method: `SelectKBest` with `f_classif` scoring
- Selects top 20 features from 50+ candidates
- Selected features saved to `models/feature_columns.pkl`

### Market Context Features (Phase 4)
Merged from shared `market_context_daily` table on `trading_date` via `_merge_market_context()`:
- **Market Regime**: `vix_close`, `vix_change_pct`, `sp500_return_1d`, `nasdaq_comp_return_1d`
- **Rates/Currency**: `dxy_return_1d`, `us_10y_yield_close`, `us_10y_yield_change`
- **Sector Rotation**: `sector_etf_return_1d` (mapped from stock's sector → XLK/XLF/XLE/etc.)
- All go through feature selection — only survivors appear in `data/selected_features.json`

### Output Table: `ml_trading_predictions`
| Column | Type | Description |
|--------|------|-------------|
| ticker | VARCHAR | Stock symbol |
| trading_date | DATE | Prediction date |
| predicted_signal | VARCHAR | 'Buy' or 'Sell' |
| confidence_percentage | FLOAT | Model confidence (0-100) |
| signal_strength | VARCHAR | 'Strong'/'Moderate'/'Weak' based on confidence |
| RSI | FLOAT | Current RSI value |
| buy_probability | FLOAT | P(Buy) from model |
| sell_probability | FLOAT | P(Sell) from model |

### Also Writes
- `ml_prediction_summary` — Daily aggregate stats (buy/sell counts, avg confidence, trend counts)
- `ml_technical_indicators` — Technical indicator snapshots used for predictions

---

## 4. DATABASE CONTEXT

### Shared SQL Server
- **Server**: `192.168.87.27\MSSQLSERVER01` (Machine A LAN IP)
- **Database**: `stockdata_db`
- **Auth**: SQL Auth (`remote_user`, `SQL_TRUSTED_CONNECTION=no`)

### Tables This Repo READS
| Table | Purpose |
|-------|---------|
| `nasdaq_100_hist_data` | Historical OHLCV data (VARCHAR prices — CAST to FLOAT!) |
| `nasdaq_top100` | Ticker master list with sector/industry |
| `nasdaq_100_fundamentals` | 37 fundamental metrics per ticker |

### Tables This Repo WRITES
| Table | Purpose |
|-------|---------|
| `ml_trading_predictions` | Daily Buy/Sell predictions with confidence |
| `ml_prediction_summary` | Aggregate daily prediction stats |
| `ml_technical_indicators` | Indicator snapshots |

### Tables Written by Other Repos (consumed downstream)
- `ml_nse_trading_predictions` ← `sqlserver_copilot_nse`
- `forex_ml_predictions` ← `sqlserver_copilot_forex`
- `ai_prediction_history` ← `streamlit-dashboard`
- `signal_tracking_history` ← `streamlit-dashboard`

---

## 5. CODING CONVENTIONS

### Critical Data Issues
- **VARCHAR Price Columns**: `open_price`, `high_price`, `low_price`, `close_price` in `nasdaq_100_hist_data` are VARCHAR, not numeric. Always: `CAST(close_price AS FLOAT)`
- **Null Handling**: Feature engineering must handle NaN values from technical indicator calculations (initial periods)
- **Date Alignment**: Predictions dated to trading days only — check for weekends/holidays

### Environment Variables
All loaded from `.env`:
- `SQL_SERVER`, `SQL_DATABASE`, `SQL_DRIVER`, `SQL_TRUSTED_CONNECTION`
- `ANTHROPIC_API_KEY` (if using LLM features)
- Model hyperparameters may be in .env or hardcoded

### Testing
```bash
python -m pytest tests/
python src/predict_daily.py --test  # Test prediction without writing
```

---

## 6. KNOWN ISSUES & IMPROVEMENTS

### Current Limitations
- Single model (Gradient Boosting) — no ensemble
- Binary classification only (Buy/Sell, no Hold class)
- No confidence calibration (raw probabilities used as confidence)
- No model versioning or drift detection
- Feature selection is static (recomputed only on full retrain)

### Downstream Consumers
- `stockdata_agenticai` reads `ml_trading_predictions` in:
  - ML Analyst queries (accuracy tracking)
  - Strategy Trade queries (NASDAQ ML signals)
  - Cross-Strategy queries (NASDAQ alignment analysis)
- `streamlit-trading-dashboard` reads for display

---

## 7. MCP SERVER FOR DEVELOPMENT

The `sqlserver_mcp` repo provides an MCP server for AI IDEs to query `stockdata_db` directly during development.

### VS Code Configuration
```json
"MSSQL MCP": {
    "type": "stdio",
    "command": "C:\\Users\\sreea\\OneDrive\\Desktop\\sqlserver_mcp\\SQL-AI-samples\\MssqlMcp\\dotnet\\MssqlMcp\\bin\\Debug\\net8.0\\MssqlMcp.exe",
    "env": {
        "CONNECTION_STRING": "Server=192.168.87.27\\MSSQLSERVER01;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```

### 7 MCP Tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData

Useful for: checking `ml_trading_predictions` output format, verifying `nasdaq_100_hist_data` schema, exploring prediction accuracy data.
