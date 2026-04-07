# Copilot Instructions — sqlserver_copilot

## Project Context
This is the **NASDAQ ML training pipeline** — part of a 7-repo stock trading analytics platform. Trains a Gradient Boosting classifier to predict Buy/Sell signals for NASDAQ 100 stocks.

## Key Architecture Rules
- Reads from `nasdaq_100_hist_data` (VARCHAR prices — always CAST to FLOAT)
- Writes to `ml_trading_predictions`, `ml_prediction_summary`, `ml_technical_indicators`
- Uses scikit-learn Gradient Boosting with 50+ engineered features → top 20 selected
- Connected to shared database `stockdata_db` on `192.168.86.28\MSSQLSERVER01` (SQL Auth)

## Key Technologies
- **Database**: SQL Server (`stockdata_db` on `192.168.86.28\MSSQLSERVER01`, SQL Auth)
- **Language**: Python 3.11+
- **ML Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Database Connectivity**: pyodbc (Trusted_Connection=yes)

## Pipeline Flow
1. `feature_engineering.py` — Calculates 50+ technical features from OHLCV
2. `feature_selection.py` — SelectKBest picks top 20 features
3. `train_model.py` — Trains and saves Gradient Boosting model
4. `predict_daily.py` — Daily predictions written to SQL Server

## Schedule
- Daily 6:00 AM: Prediction run
- Daily 6:30 AM: Data freshness check (conditional retrain)
- Weekly: Full retrain with updated data

## Database Notes
- Price columns in `nasdaq_100_hist_data` are **VARCHAR** — always use `CAST(close_price AS FLOAT)`
- Server: `192.168.86.28\MSSQLSERVER01`, DB: `stockdata_db`, Auth: SQL Auth

## Code Guidelines
- Use pyodbc for database connections (Windows Integrated Auth)
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Include proper error handling for database operations
- Use environment variables for database credentials
- Structure SQL queries for readability and maintainability

## Security Considerations
- Never hardcode database credentials
- Use environment variables or secure configuration files
- Use parameterized queries to prevent SQL injection

## Sibling Repositories (same database)
- `sqlserver_copilot_nse` — NSE ML pipeline (5-model ensemble)
- `sqlserver_copilot_forex` — Forex ML pipeline (XGBoost/LightGBM)
- `stockdata_agenticai` — CrewAI agents that consume predictions
- `streamlit-trading-dashboard` — Displays predictions and tracks accuracy
- `sqlserver_mcp` — .NET 8 MCP Server (Microsoft MssqlMcp) with 7 tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData. Stdio transport. Use to explore DB schemas and verify query results during development.
- `stockanalysis` — Data ingestion ETL

## MCP Server for Development
Configure in `.vscode/mcp.json` to query stockdata_db directly from your AI IDE:
```json
"MSSQL MCP": {
    "type": "stdio",
    "command": "C:\\Users\\sreea\\OneDrive\\Desktop\\sqlserver_mcp\\SQL-AI-samples\\MssqlMcp\\dotnet\\MssqlMcp\\bin\\Debug\\net8.0\\MssqlMcp.exe",
    "env": {
        "CONNECTION_STRING": "Server=192.168.86.28\\MSSQLSERVER01;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```
Useful for: verifying `ml_trading_predictions` output, checking `nasdaq_100_hist_data` schema, exploring prediction accuracy.
