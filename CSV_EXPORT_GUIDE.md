# 📊 CSV Export Guide for Trading Signals

This guide explains how to export trading signal predictions to CSV files with various formatting and filtering options.

## 📁 Results Location
All CSV exports are saved in the `results/` folder automatically created in your project directory.

## 🚀 Quick Export Options

### Simple CSV Export with Batch Predictions
```bash
# Export all predictions with default settings
python predict_trading_signals.py --batch --export-csv

# Export with custom confidence threshold
python predict_trading_signals.py --batch --export-csv --confidence 0.8

# Export specific ticker
python predict_trading_signals.py --ticker AAPL --export-csv
```

### Advanced Export Options
```bash
# Enhanced format with additional analysis columns
python export_results.py --batch --export-csv --format enhanced

# Export only high-confidence signals
python export_results.py --batch --export-csv --filter high-confidence

# Export only buy signals in trading format
python export_results.py --batch --export-csv --filter buy-signals --format trading

# Export segmented by confidence levels (creates multiple files)
python export_results.py --segmented --confidence 0.7
```

## 📋 Export Formats Available

### 1. **Standard Format** (Default)
Basic prediction data with essential columns:
- `export_timestamp`, `trading_date`, `ticker`, `company`
- `predicted_signal`, `confidence`, `confidence_level`
- `close_price`, `RSI`, `high_confidence`

```bash
python predict_trading_signals.py --batch --export-csv
```

### 2. **Enhanced Format** 
Comprehensive analysis with additional calculated fields:
- All standard columns plus:
- `signal_strength`, `confidence_percentage`, `rsi_category`
- `price_risk`, `sell_probability`, `buy_probability`

```bash
python export_results.py --batch --export-csv --format enhanced
```

### 3. **Summary Format**
Compact view with key information only:
- `ticker`, `company`, `predicted_signal`, `confidence_pct`, `close_price`, `RSI`

```bash
python export_results.py --batch --export-csv --format summary
```

### 4. **Trading Format**
Optimized for trading decisions:
- `ticker`, `action` (BUY/SELL/HOLD), `priority` (1-4), `risk_level`
- `confidence`, `close_price`, `RSI`

```bash
python export_results.py --batch --export-csv --format trading
```

## 🔍 Filter Options

### Filter Types
- `all` - Export all predictions (default)
- `high-confidence` - Only predictions > 70% confidence
- `buy-signals` - Only buy recommendations
- `sell-signals` - Only sell recommendations  
- `medium-high` - Predictions > 60% confidence

### Filter Examples
```bash
# Only high-confidence buy signals
python export_results.py --batch --export-csv --filter buy-signals --format enhanced

# Medium and high confidence signals
python export_results.py --batch --export-csv --filter medium-high

# Only sell signals in trading format
python export_results.py --batch --export-csv --filter sell-signals --format trading
```

## 📊 Segmented Export (Multiple Files)

Creates separate CSV files for different confidence levels:

```bash
python export_results.py --segmented --confidence 0.7
```

**Creates 3 files:**
1. `high_confidence_signals_TIMESTAMP.csv` - Signals > 70% confidence
2. `medium_confidence_signals_TIMESTAMP.csv` - Signals 60-70% confidence  
3. `trading_signals_summary_TIMESTAMP.csv` - Trading-focused summary

## 📝 File Naming Convention

Files are automatically named with descriptive information:
- `batch_predictions_TIMESTAMP.csv` - Standard batch export
- `predictions_TICKER_DATE_TIMESTAMP.csv` - Specific ticker/date
- `predictions_FILTER_FORMAT_TIMESTAMP.csv` - Filtered/formatted export

**Timestamp format:** `YYYYMMDD_HHMMSS`

## 💡 Usage Examples

### Daily Trading Workflow
```bash
# Morning: Get all high-confidence signals for the day
python export_results.py --batch --export-csv --filter high-confidence --format trading

# Research: Get detailed analysis of specific stocks
python export_results.py --ticker AAPL --export-csv --format enhanced

# Portfolio review: Get summary of all signals
python export_results.py --batch --export-csv --format summary
```

### Weekly Analysis
```bash
# Full segmented export for comprehensive review
python export_results.py --segmented --confidence 0.6

# Buy signals only for accumulation strategy
python export_results.py --batch --export-csv --filter buy-signals --format enhanced

# Sell signals for profit-taking review
python export_results.py --batch --export-csv --filter sell-signals --format trading
```

### Research & Backtesting
```bash
# Enhanced format with all analysis metrics
python export_results.py --batch --export-csv --format enhanced

# Historical analysis for specific date
python predict_trading_signals.py --date 2024-11-20 --batch --export-csv --format enhanced
```

## 📈 CSV Columns Explained

### Standard Columns
- **export_timestamp** - When the prediction was generated
- **trading_date** - Stock market date for the data
- **ticker** - Stock symbol (e.g., AAPL)
- **company** - Company name
- **predicted_signal** - Buy/Sell recommendation
- **confidence** - Model confidence (0.0 to 1.0)
- **confidence_level** - High/Medium/Low classification
- **close_price** - Stock closing price
- **RSI** - Relative Strength Index value
- **high_confidence** - Boolean flag for high confidence

### Enhanced Columns (Additional)
- **signal_strength** - Strong/Moderate/Weak classification
- **confidence_percentage** - Confidence as percentage
- **rsi_category** - Oversold/Overbought/Neutral
- **price_risk** - High/Medium/Low based on stock price
- **sell_probability** - Probability of sell signal
- **buy_probability** - Probability of buy signal

### Trading Format Columns
- **action** - BUY/SELL/HOLD recommendation
- **priority** - 1 (highest) to 4 (lowest) priority
- **risk_level** - Investment risk assessment

## 🔧 Tips & Best Practices

### 1. **File Management**
```bash
# Create dated subfolder for organization
mkdir results/2024-11-25
python export_results.py --batch --export-csv --results-dir results/2024-11-25
```

### 2. **Automation Scripts**
Create batch files for regular exports:

**daily_export.bat:**
```batch
@echo off
python export_results.py --segmented --confidence 0.7
python export_results.py --batch --export-csv --filter high-confidence --format trading
echo Daily exports completed!
```

### 3. **Data Analysis**
Import CSV files into Excel, Python pandas, or your preferred analysis tool:

```python
import pandas as pd

# Load and analyze exported data
df = pd.read_csv('results/high_confidence_signals_20241125_174418.csv')
print(f"High confidence signals: {len(df)}")
print(f"Average confidence: {df['confidence'].mean():.1%}")
```

## 🚨 Troubleshooting

### Issue: No CSV files created
```bash
# Check if results directory exists
ls results/

# Verify export command includes --export-csv flag
python export_results.py --batch --export-csv
```

### Issue: Empty CSV files
```bash
# Check if predictions are available
python predict_trading_signals.py --batch

# Verify database connection
python check_data_status_fixed.py
```

### Issue: Wrong format/filter
```bash
# List available options
python export_results.py --help

# Use correct format names: standard, enhanced, summary, trading
# Use correct filters: all, high-confidence, buy-signals, sell-signals, medium-high
```

---

## 📞 Quick Reference Commands

```bash
# Standard daily export
python predict_trading_signals.py --batch --export-csv

# Advanced analysis export  
python export_results.py --batch --export-csv --format enhanced --filter high-confidence

# Trading-focused export
python export_results.py --batch --export-csv --format trading --filter medium-high

# Segmented export (multiple files)
python export_results.py --segmented

# Specific stock analysis
python export_results.py --ticker AAPL --export-csv --format enhanced
```

All CSV files are automatically saved in the `results/` folder with descriptive filenames and timestamps for easy organization and tracking.
