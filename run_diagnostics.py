"""
Phase 1 Diagnostic Script — NASDAQ ML Pipeline
Investigates root cause of 0 Buy signals on May 26-27 2026.
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection
import pandas as pd

db = SQLServerConnection()

# ============================================================
# DIAGNOSTIC 1: Buy probability distribution for last 5 trading days
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 1: Buy probability distribution (last 5 days)")
print("="*60)
q1 = """
SELECT
    trading_date,
    COUNT(*) AS total_stocks,
    SUM(CASE WHEN predicted_signal IN ('Buy','Up') THEN 1 ELSE 0 END) AS buy_signals,
    SUM(CASE WHEN predicted_signal IN ('Sell','Down') THEN 1 ELSE 0 END) AS sell_signals,
    ROUND(AVG(buy_probability*100),2) AS avg_buy_prob_pct,
    ROUND(MIN(buy_probability*100),2) AS min_buy_prob_pct,
    ROUND(MAX(buy_probability*100),2) AS max_buy_prob_pct,
    SUM(CASE WHEN buy_probability >= 0.50 THEN 1 ELSE 0 END) AS above_50pct,
    SUM(CASE WHEN buy_probability >= 0.60 THEN 1 ELSE 0 END) AS above_60pct,
    SUM(CASE WHEN buy_probability >= 0.67 THEN 1 ELSE 0 END) AS above_67pct,
    SUM(CASE WHEN buy_probability >= 0.72 THEN 1 ELSE 0 END) AS above_72pct
FROM dbo.ml_trading_predictions
WHERE trading_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
GROUP BY trading_date
ORDER BY trading_date
"""
df1 = db.execute_query(q1)
print(df1.to_string())

# ============================================================
# DIAGNOSTIC 2: Sector breakdown for May 26-27
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 2: Signal breakdown by sector (last 5 days)")
print("="*60)
q2 = """
SELECT
    p.trading_date,
    t.sector,
    COUNT(*) AS total,
    SUM(CASE WHEN p.predicted_signal IN ('Buy','Up') THEN 1 ELSE 0 END) AS buys,
    ROUND(AVG(p.buy_probability*100),2) AS avg_buy_prob,
    ROUND(MAX(p.buy_probability*100),2) AS max_buy_prob,
    ROUND(AVG(CAST(p.RSI AS FLOAT)),1) AS avg_rsi,
    SUM(CASE WHEN CAST(p.RSI AS FLOAT) > 70 THEN 1 ELSE 0 END) AS rsi_above_70
FROM dbo.ml_trading_predictions p
LEFT JOIN dbo.nasdaq_top100 t ON p.ticker = t.ticker
WHERE p.trading_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
  AND t.sector IS NOT NULL
GROUP BY p.trading_date, t.sector
ORDER BY p.trading_date, t.sector
"""
df2 = db.execute_query(q2)
print(df2.to_string())

# ============================================================
# DIAGNOSTIC 3: RSI distribution across NASDAQ 100 for May 26-27
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 3: RSI distribution in ml_trading_predictions (last 5 days)")
print("="*60)
q3 = """
SELECT
    trading_date,
    SUM(CASE WHEN CAST(RSI AS FLOAT) < 30 THEN 1 ELSE 0 END) AS rsi_oversold,
    SUM(CASE WHEN CAST(RSI AS FLOAT) BETWEEN 30 AND 50 THEN 1 ELSE 0 END) AS rsi_30_50,
    SUM(CASE WHEN CAST(RSI AS FLOAT) BETWEEN 50 AND 60 THEN 1 ELSE 0 END) AS rsi_50_60,
    SUM(CASE WHEN CAST(RSI AS FLOAT) BETWEEN 60 AND 70 THEN 1 ELSE 0 END) AS rsi_60_70,
    SUM(CASE WHEN CAST(RSI AS FLOAT) BETWEEN 70 AND 75 THEN 1 ELSE 0 END) AS rsi_70_75,
    SUM(CASE WHEN CAST(RSI AS FLOAT) > 75 THEN 1 ELSE 0 END) AS rsi_above_75,
    ROUND(AVG(CAST(RSI AS FLOAT)),1) AS avg_rsi,
    ROUND(MAX(CAST(RSI AS FLOAT)),1) AS max_rsi,
    ROUND(MIN(CAST(RSI AS FLOAT)),1) AS min_rsi
FROM dbo.ml_trading_predictions
WHERE trading_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
GROUP BY trading_date
ORDER BY trading_date
"""
df3 = db.execute_query(q3)
print(df3.to_string())

# ============================================================
# DIAGNOSTIC 4: Market context for May 26-27
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 4: Market context (VIX, NASDAQ, S&P500) - last 7 days")
print("="*60)
q4 = """
SELECT
    trading_date,
    ROUND(vix_close,2) AS vix_close,
    ROUND(vix_change_pct,2) AS vix_chg_pct,
    ROUND(sp500_return_1d*100,2) AS sp500_ret_pct,
    ROUND(nasdaq_comp_return_1d*100,2) AS nasdaq_ret_pct,
    ROUND(dxy_return_1d*100,2) AS dxy_ret_pct,
    ROUND(us_10y_yield_close,3) AS us_10y_yield,
    ROUND(xlk_return_1d*100,2) AS xlk_tech_pct,
    ROUND(xlf_return_1d*100,2) AS xlf_fin_pct,
    ROUND(xlv_return_1d*100,2) AS xlv_health_pct
FROM dbo.market_context_daily
WHERE trading_date >= DATEADD(day, -7, CAST(GETDATE() AS DATE))
ORDER BY trading_date
"""
df4 = db.execute_query(q4)
print(df4.to_string())

# ============================================================
# DIAGNOSTIC 5: Sector sentiment data quality
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 5: Sector sentiment data quality (last 30 days)")
print("="*60)
q5 = """
SELECT
    sector,
    COUNT(*) AS days_with_data,
    ROUND(AVG(sentiment_score),4) AS avg_score,
    ROUND(MIN(sentiment_score),4) AS min_score,
    ROUND(MAX(sentiment_score),4) AS max_score,
    ROUND(AVG(confidence),4) AS avg_confidence,
    SUM(CASE WHEN sentiment_score = 0 THEN 1 ELSE 0 END) AS zero_score_days,
    SUM(CASE WHEN sentiment_score IS NULL THEN 1 ELSE 0 END) AS null_days,
    ROUND(AVG(CAST(news_count AS FLOAT)),1) AS avg_news_count,
    MIN(trading_date) AS earliest_date,
    MAX(trading_date) AS latest_date
FROM dbo.nasdaq_sector_sentiment
WHERE trading_date >= DATEADD(day, -30, CAST(GETDATE() AS DATE))
GROUP BY sector
ORDER BY sector
"""
try:
    df5 = db.execute_query(q5)
    print(df5.to_string())
except Exception as e:
    print(f"  [ERROR] Could not query sentiment table: {e}")

# ============================================================
# DIAGNOSTIC 6: Sentiment coverage (days present per sector per month)
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 6: Sentiment monthly coverage")
print("="*60)
q6 = """
SELECT
    FORMAT(trading_date, 'yyyy-MM') AS month,
    COUNT(DISTINCT sector) AS sectors_covered,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN sentiment_score != 0 THEN 1 ELSE 0 END) AS non_zero_rows
FROM dbo.nasdaq_sector_sentiment
WHERE trading_date >= DATEADD(month, -6, CAST(GETDATE() AS DATE))
GROUP BY FORMAT(trading_date, 'yyyy-MM')
ORDER BY month
"""
try:
    df6 = db.execute_query(q6)
    print(df6.to_string())
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================================
# DIAGNOSTIC 7: Stocks with highest buy_probability on May 26-27
# (even if blocked by filters — to confirm model CAN generate Buy)
# ============================================================
print("\n" + "="*60)
print("DIAGNOSTIC 7: Top 20 stocks by buy_probability on last prediction day")
print("="*60)
q7 = """
SELECT TOP 20
    ticker, trading_date, sector, predicted_signal,
    ROUND(buy_probability*100,1) AS buy_prob_pct,
    ROUND(confidence_percentage,1) AS confidence_pct,
    ROUND(CAST(RSI AS FLOAT),1) AS rsi,
    signal_strength
FROM dbo.ml_trading_predictions
WHERE trading_date = (SELECT MAX(trading_date) FROM dbo.ml_trading_predictions)
ORDER BY buy_probability DESC
"""
df7 = db.execute_query(q7)
print(df7.to_string())

print("\n" + "="*60)
print("DIAGNOSTICS COMPLETE")
print("="*60)
