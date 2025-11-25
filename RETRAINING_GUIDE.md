# 🔄 Model Retraining Guide

This guide explains how to retrain your ML model with the latest data from SQL Server.

## 📋 Quick Steps Summary

1. **🤖 Daily Automation** (Recommended for Production)
2. **🚀 Automated Retraining** (Manual Execution)
3. **👨‍💻 Manual Step-by-Step Process**  
4. **✅ Validation and Deployment**

---

## 🤖 Option 1: Daily Automation (Recommended for Production)

### ⭐ NEW: Complete Daily Workflow
The daily automation system handles everything automatically:
- 🔍 **Data status checking** (connectivity, freshness)
- 🤔 **Smart retraining decisions** (only when data is fresh)
- 📊 **CSV export generation** (multiple formats)
- 📋 **Comprehensive logging** and reporting

```bash
# Complete daily workflow - handles everything automatically
python daily_automation.py

# Force retraining regardless of data age
python daily_automation.py --force-retrain

# Only generate CSV exports (skip retraining)
python daily_automation.py --csv-only

# Only check data status
python daily_automation.py --check-only
```

**🎯 Set up once, run automatically:**
1. **Windows Task Scheduler**: See `DAILY_AUTOMATION_GUIDE.md` for setup
2. **Batch file**: Double-click `run_daily_automation.bat`
3. **Monitor**: Use `python monitor_automation.py` to check status

---

## 🚀 Option 2: Manual Automated Retraining

### Simple Full Retrain
```bash
# Full retrain with latest data
python retrain_model.py

# Quick retrain (skip detailed EDA)
python retrain_model.py --quick

# Backup old model before retrain
python retrain_model.py --backup-old
```

### What the automated script does:
1. ✅ **Backs up existing model** (if requested)
2. ✅ **Loads latest data** from SQL Server
3. ✅ **Compares data** with previous version
4. ✅ **Performs EDA** (or skips if quick mode)
5. ✅ **Engineers features** (18 technical indicators)
6. ✅ **Trains 4 models** (Random Forest, Gradient Boosting, etc.)
7. ✅ **Selects best model** based on F1-score
8. ✅ **Saves all artifacts** (model, scaler, encoder)
9. ✅ **Compares performance** with previous model

---

## 🛠️ Option 2: Manual Step-by-Step Process

### Step 1: Check Data Availability
```bash
# Test database connection
python -c "from src.database.connection import SQLServerConnection; SQLServerConnection().test_connection()"

# Check latest data in database
python -c "
from src.database.connection import SQLServerConnection
db = SQLServerConnection()
result = db.execute_query('SELECT COUNT(*) as count, MAX(trading_date) as latest_date FROM dbo.nasdaq_100_hist_data')
print(f'Records: {result.iloc[0][\"count\"]:,}')
print(f'Latest date: {result.iloc[0][\"latest_date\"]}')
"
```

### Step 2: Run Data Exploration
```bash
# Option A: Use Jupyter notebook (interactive)
jupyter lab --notebook-dir=notebooks

# Then open and run: 02_data_exploration.ipynb
```

### Step 3: Run Model Training
```bash
# Option A: Use Jupyter notebook
# Open and run: 03_model_development.ipynb

# Option B: Use the complete analysis script
python continue_analysis.py
```

### Step 4: Validate New Model
```bash
# Test predictions with new model
python predict_trading_signals.py --ticker AAPL
python predict_trading_signals.py --batch --confidence 0.7
```

---

## 📊 Monitoring Data Updates

### Check for New Data
```bash
# Quick data check script
python -c "
import pandas as pd
from src.database.connection import SQLServerConnection
from datetime import datetime, timedelta

db = SQLServerConnection()

# Get data summary
query = '''
SELECT 
    COUNT(*) as total_records,
    MIN(trading_date) as earliest_date,
    MAX(trading_date) as latest_date,
    COUNT(DISTINCT ticker) as unique_tickers,
    COUNT(DISTINCT trading_date) as trading_days
FROM dbo.nasdaq_100_hist_data
'''

result = db.execute_query(query).iloc[0]
print('📊 DATA SUMMARY')
print(f'Total Records: {result[\"total_records\"]:,}')
print(f'Date Range: {result[\"earliest_date\"]} to {result[\"latest_date\"]}')
print(f'Unique Tickers: {result[\"unique_tickers\"]}')
print(f'Trading Days: {result[\"trading_days\"]}')

# Check recent data
recent_query = '''
SELECT COUNT(*) as recent_count
FROM dbo.nasdaq_100_hist_data 
WHERE trading_date >= DATEADD(day, -7, GETDATE())
'''
recent = db.execute_query(recent_query).iloc[0]['recent_count']
print(f'Recent Data (7 days): {recent:,} records')
"
```

---

## ⚡ Quick Retraining Commands

### For Regular Updates (Weekly/Monthly)
```bash
# Quick retrain with backup
python retrain_model.py --quick --backup-old
```

### For Full Analysis (Quarterly/Semi-annual)
```bash
# Full retrain with detailed analysis
python retrain_model.py --backup-old
```

### Emergency Model Update
```bash
# Quick retrain without backup (fastest)
python retrain_model.py --quick
```

---

## 🔍 Validation Steps After Retraining

### 1. Test Model Performance
```bash
# Get batch predictions with CSV export
python predict_trading_signals.py --batch --confidence 0.7 --export-csv

# Advanced export with enhanced analysis
python export_results.py --batch --export-csv --format enhanced --filter high-confidence

# Export segmented by confidence levels
python export_results.py --segmented --confidence 0.7

# Check specific stocks
python predict_trading_signals.py --ticker AAPL --ticker MSFT --ticker GOOGL --export-csv
```

### 2. Compare Model Metrics
```bash
# Load and compare results
python -c "
import pickle
with open('data/model_results.pkl', 'rb') as f:
    results = pickle.load(f)

print('📊 MODEL PERFORMANCE')
print(f'Best Model: {results[\"best_model_name\"]}')
for model_name, metrics in results['model_results'].items():
    print(f'{model_name}:')
    print(f'  Accuracy: {metrics[\"accuracy\"]:.3f}')
    print(f'  F1-Score: {metrics[\"f1_score\"]:.3f}')
    print(f'  CV Score: {metrics[\"cv_mean\"]:.3f} (±{metrics[\"cv_std\"]:.3f})')
    print()
"
```

### 3. Visual Validation
```bash
# Check generated reports
ls -la reports/
# Look for: confusion_matrix.png, feature_importance.png

# Open reports in browser
start reports/confusion_matrix.png    # Windows
start reports/feature_importance.png  # Windows
```

---

## 📅 Recommended Retraining Schedule

### **Weekly** (if market data updates frequently)
```bash
# Quick check for new data and retrain if needed
python retrain_model.py --quick
```

### **Monthly** (standard recommendation)
```bash
# Full retrain with backup
python retrain_model.py --backup-old
```

### **Quarterly** (comprehensive analysis)
```bash
# Full analysis with detailed EDA
python retrain_model.py --backup-old
# Then run: jupyter lab --notebook-dir=notebooks
# Review all notebooks for insights
```

---

## 🚨 Troubleshooting

### Issue: No new data found
```bash
# Check database connection
python -c "from src.database.connection import SQLServerConnection; SQLServerConnection().test_connection()"

# Verify data update process
# Check with your data provider/ETL process
```

### Issue: Model performance decreased
```bash
# Backup current model
python retrain_model.py --backup-old

# Try different parameters or investigate data quality issues
# Review EDA results in notebooks/02_data_exploration.ipynb
```

### Issue: Memory/performance problems
```bash
# Use quick mode to skip heavy computations
python retrain_model.py --quick

# Or limit data range for testing
# Modify retrain_model.py to use fewer days of data
```

---

## 💡 Pro Tips

### 1. **Automate with Windows Task Scheduler**
Create a batch file for automated retraining:

```batch
@echo off
cd /d "c:\Users\sreea\OneDrive\Desktop\sqlserver_copilot"
python retrain_model.py --quick --backup-old > logs\retrain_%date:~-4,4%_%date:~-10,2%_%date:~-7,2%.log 2>&1
```

### 2. **Monitor Performance Over Time**
Keep a log of model performance:
```bash
python -c "
import pickle
from datetime import datetime
try:
    with open('data/model_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    best_model = results['best_model_name']
    f1_score = results['model_results'][best_model]['f1_score']
    timestamp = results.get('training_timestamp', 'unknown')
    
    with open('model_performance_log.txt', 'a') as log:
        log.write(f'{datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}, {best_model}, {f1_score:.4f}, {timestamp}\n')
    
    print(f'Performance logged: {best_model} - {f1_score:.4f}')
except Exception as e:
    print(f'Error logging performance: {e}')
"
```

### 3. **A/B Testing New Models**
Keep multiple model versions:
```bash
# Backup with specific name
cp data/best_model_gradient_boosting.joblib data/models/model_v1.joblib

# After retraining
cp data/best_model_gradient_boosting.joblib data/models/model_v2.joblib

# Compare in production
```

---

## 📞 Quick Reference Commands

```bash
# 🚀 DAILY AUTOMATION (NEW - RECOMMENDED)
python daily_automation.py                    # Complete daily workflow
python daily_automation.py --force-retrain    # Force retrain regardless of data age
python daily_automation.py --csv-only         # Skip retraining, only CSV export
python daily_automation.py --check-only       # Only check data status

# Monitor automation status
python monitor_automation.py                  # Current status
python monitor_automation.py --history        # Last 7 days performance
python monitor_automation.py --alerts         # Issues requiring attention

# Manual retraining (if needed)
python retrain_model.py --backup-old          # Full automated retrain
python retrain_model.py --quick              # Quick retrain

# CSV exports only
python predict_trading_signals.py --batch --export-csv
python export_results.py

# Data checks
python check_data_status_fixed.py

# Interactive development
jupyter lab --notebook-dir=notebooks
```

**🎯 Recommended Approach:**
- **🔄 Daily Automation**: Set up `daily_automation.py` with Windows Task Scheduler for hands-free operation
- **📊 Monitoring**: Use `monitor_automation.py` to check system health weekly
- **🚀 Manual Override**: Use individual scripts only when automation fails or for development

**Choose your workflow:**
- **🤖 Fully Automated** → Set up daily automation + weekly monitoring
- **🔧 Semi-Automated** → Run `daily_automation.py` manually when needed
- **👨‍💻 Development Mode** → Use individual scripts and notebooks for fine-tuning
