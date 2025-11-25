# 📅 Daily Automation & Scheduling Guide

This guide explains how to set up and use the daily automation system for your SQL Server ML Trading Signals project.

## 🚀 Quick Start

### Option 1: Manual Execution
```bash
# Run complete daily workflow
python daily_automation.py

# Force retraining regardless of data age
python daily_automation.py --force-retrain

# Only generate CSV exports (skip retraining)
python daily_automation.py --csv-only

# Only check data status
python daily_automation.py --check-only
```

### Option 2: Windows Batch File
```cmd
# Double-click or run from command prompt
run_daily_automation.bat
```

---

## 🔄 What the Daily Automation Does

### Complete Workflow
1. **🔍 Data Status Check**
   - Tests SQL Server connectivity
   - Checks data freshness (age in days)
   - Validates record counts

2. **🤔 Retraining Decision**
   - Automatically retrains if data is ≤ 2 days old
   - Skips retraining for older data
   - Can be forced with `--force-retrain`

3. **🚀 Model Retraining** (if needed)
   - Backs up existing model
   - Trains with latest data
   - Saves new model artifacts
   - Quick mode for faster execution

4. **📊 CSV Export Generation**
   - Creates trading signals for all NASDAQ 100 stocks
   - Generates multiple export formats
   - Saves timestamped files to `results/` folder

5. **📋 Reporting**
   - Creates detailed logs in `logs/` folder
   - Generates daily summary reports in `daily_reports/`
   - Provides success/failure status

---

## ⏰ Setting Up Automated Scheduling

### Option 1: Windows Task Scheduler (Recommended)

#### Step 1: Open Task Scheduler
1. Press `Win + R`, type `taskschd.msc`, press Enter
2. Click "Create Basic Task" in the right panel

#### Step 2: Configure Task
1. **Name**: `Trading Signals Daily Automation`
2. **Description**: `Daily ML model retraining and CSV generation for trading signals`
3. **Trigger**: Daily at your preferred time (e.g., 6:00 AM)
4. **Action**: Start a program
   - **Program**: `C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot\run_daily_automation.bat`
   - **Start in**: `C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot`

#### Step 3: Advanced Settings
1. Right-click the created task → Properties
2. **General Tab**:
   - ☑️ Run whether user is logged on or not
   - ☑️ Run with highest privileges
3. **Conditions Tab**:
   - ☑️ Start the task only if the computer is on AC power (optional)
   - ☑️ Wake the computer to run this task
4. **Settings Tab**:
   - ☑️ Allow task to be run on demand
   - ☑️ Stop the task if it runs longer than: 1 hour

### Option 2: Command Line Task Creation
```cmd
# Create daily task at 6:00 AM
schtasks /create /sc daily /mo 1 /tn "Trading Signals Automation" /tr "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot\run_daily_automation.bat" /st 06:00 /ru SYSTEM

# View the task
schtasks /query /tn "Trading Signals Automation"

# Delete the task (if needed)
schtasks /delete /tn "Trading Signals Automation" /f
```

### Option 3: PowerShell Script for Advanced Scheduling
```powershell
# Create a more advanced scheduled task
$action = New-ScheduledTaskAction -Execute "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot\run_daily_automation.bat" -WorkingDirectory "C:\Users\sreea\OneDrive\Desktop\sqlserver_copilot"
$trigger = New-ScheduledTaskTrigger -Daily -At 6:00AM
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
$principal = New-ScheduledTaskPrincipal -RunLevel Highest

Register-ScheduledTask -TaskName "Trading Signals Daily Automation" -Action $action -Trigger $trigger -Settings $settings -Principal $principal
```

---

## 📁 Output Files & Locations

### Log Files
- **Location**: `logs/daily_automation_YYYYMMDD_HHMMSS.log`
- **Content**: Detailed execution logs with timestamps
- **Retention**: Keep last 30 days

### Daily Reports
- **Location**: `daily_reports/daily_summary_YYYYMMDD.json`
- **Content**: JSON summary of daily automation results
- **Use**: Programmatic monitoring and alerts

### CSV Exports
- **Location**: `results/`
- **Files Generated**:
  - `trading_signals_YYYYMMDD_HHMMSS.csv` - Standard predictions
  - `trading_signals_enhanced_YYYYMMDD_HHMMSS.csv` - Comprehensive analysis
  - `trading_signals_summary_YYYYMMDD_HHMMSS.csv` - Compact format
  - `trading_signals_trading_YYYYMMDD_HHMMSS.csv` - Trading optimized
  - Segmented files by confidence levels

---

## 🔧 Configuration & Customization

### Retraining Frequency
Edit the `should_retrain()` function in `daily_automation.py`:
```python
# Current: Retrain if data is 0-2 days old
if data_age_days <= 2:
    return True, f"Data is fresh ({data_age_days} days old)"

# More frequent: Retrain if data is 0-1 days old
if data_age_days <= 1:
    return True, f"Data is fresh ({data_age_days} days old)"

# Less frequent: Retrain if data is same day only
if data_age_days == 0:
    return True, f"Data is fresh (same day)"
```

### Scheduling Time Recommendations
- **6:00 AM**: Before market open (good for pre-market analysis)
- **7:00 PM**: After market close (includes latest trading day)
- **11:00 PM**: Late night processing (less system load)

### Environment Variables (Optional)
Create a `.env` file for configuration:
```env
# Database configuration
DB_SERVER=localhost\MSSQLSERVER01
DB_DATABASE=stockdata_db
DB_TIMEOUT=30

# Automation settings
RETRAIN_THRESHOLD_DAYS=2
LOG_RETENTION_DAYS=30
CSV_RETENTION_DAYS=7
```

---

## 🚨 Monitoring & Alerts

### Check Automation Status
```bash
# View latest log
python -c "import os; logs = sorted([f for f in os.listdir('logs') if f.startswith('daily_automation')]); print(f'Latest log: logs/{logs[-1]}') if logs else print('No logs found')"

# View latest daily report
python -c "import os, json; reports = sorted([f for f in os.listdir('daily_reports') if f.endswith('.json')]); print(json.dumps(json.load(open(f'daily_reports/{reports[-1]}')), indent=2)) if reports else print('No reports found')"
```

### PowerShell Monitoring Script
```powershell
# Check if automation ran successfully today
$today = Get-Date -Format "yyyyMMdd"
$logFile = Get-ChildItem "logs" -Filter "*$today*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($logFile) {
    $content = Get-Content $logFile.FullName -Tail 20
    if ($content -match "completed successfully") {
        Write-Host "✅ Daily automation completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Daily automation may have failed" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ No automation log found for today" -ForegroundColor Yellow
}
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Task Scheduler Not Running
```cmd
# Check if Task Scheduler service is running
sc query schedule

# Start Task Scheduler service if stopped
sc start schedule
```

#### 2. Python Path Issues
- Edit `run_daily_automation.bat`
- Change `set PYTHON_EXE=python` to full path: `set PYTHON_EXE=C:\Python\python.exe`

#### 3. Database Connection Issues
```bash
# Test database connection manually
python check_data_status_fixed.py
```

#### 4. Permission Issues
- Run Task Scheduler as Administrator
- Set task to "Run with highest privileges"

#### 5. Long Running Tasks
- Check timeout settings in Task Scheduler (default: 1 hour)
- Monitor logs for bottlenecks

### Manual Recovery
```bash
# If automation fails, run components individually:

# 1. Check data status
python check_data_status_fixed.py

# 2. Force retrain if needed
python retrain_model.py --quick

# 3. Generate CSV exports
python predict_trading_signals.py --batch --export-csv
python export_results.py
```

---

## 📊 Performance Metrics

### Expected Execution Times
- **Data Status Check**: 10-30 seconds
- **Model Retraining**: 5-15 minutes (quick mode)
- **CSV Generation**: 1-3 minutes
- **Total Time**: 6-18 minutes (typical: 8-12 minutes)

### Success Criteria
- ✅ Database connectivity established
- ✅ Data age ≤ 2 days (or retraining skipped intentionally)
- ✅ Model artifacts saved (if retrained)
- ✅ CSV files generated successfully
- ✅ No critical errors in logs

---

## 🔄 Maintenance

### Weekly Tasks
- Review logs for errors or warnings
- Clean up old log files (>30 days)
- Verify CSV exports are being used

### Monthly Tasks
- Review model performance metrics
- Update retraining thresholds if needed
- Archive old CSV exports

### Quarterly Tasks
- Review and optimize feature engineering
- Update technical indicators if needed
- Evaluate automation schedule effectiveness
