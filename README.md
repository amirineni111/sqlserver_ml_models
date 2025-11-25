# 🚀 SQL Server ML Trading Signals Automation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![SQL Server](https://img.shields.io/badge/SQL%20Server-2019+-red.svg)](https://www.microsoft.com/sql-server)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready, end-to-end machine learning automation system for generating daily trading signals from NASDAQ 100 stock data stored in SQL Server.

## 🎯 **Project Overview**

This system automatically:
- ✅ **Connects to SQL Server** and monitors data freshness
- ✅ **Retrains ML models** when new data is available  
- ✅ **Generates trading signals** using Gradient Boosting (66.31% accuracy)
- ✅ **Exports multiple CSV formats** optimized for trading decisions
- ✅ **Monitors system health** with comprehensive logging and alerts
- ✅ **Runs fully automated** via Windows Task Scheduler

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQL Server    │◄──►│  Daily Automation │───►│   CSV Exports   │
│ (NASDAQ 100)    │    │     System        │    │ (Trading Signals)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐            │
         └──────────────►│  ML Pipeline     │◄───────────┘
                        │ (Gradient Boost) │
                        └──────────────────┘
```

## 🚀 **Quick Start**

### 1. **System Requirements**
- Python 3.8+
- SQL Server with NASDAQ 100 data
- Windows (for Task Scheduler automation)

### 2. **Installation**
```powershell
# Clone the repository
git clone https://github.com/amirineni111/sqlserver_ml_models.git
cd sqlserver_ml_models

# Install dependencies
pip install -r requirements.txt

# Configure database connection (copy and edit)
cp .env.example .env
```

### 3. **Database Setup**
Edit `.env` file with your SQL Server connection details:
```env
DB_SERVER=localhost\SQLEXPRESS
DB_DATABASE=your_database
DB_TRUSTED_CONNECTION=yes
```

### 4. **Verify Setup**
```powershell
# Test database connection and data status
python check_data_status_console_safe.py

# Monitor system health
python monitor_automation.py
```

## 🤖 **Daily Automation Usage**

### **Option 1: Complete Automation (Recommended)**
```powershell
# Run complete daily workflow
python daily_automation.py

# Check system status  
python monitor_automation.py

# Set up Windows Task Scheduler for hands-free operation
# See DAILY_AUTOMATION_GUIDE.md for detailed instructions
```

### **Option 2: Individual Components**
```powershell
# Check data status and connectivity
python daily_automation.py --check-only

# Generate CSV exports only
python daily_automation.py --csv-only

# Force model retraining
python daily_automation.py --force-retrain

# Individual prediction
python predict_trading_signals.py --ticker AAPL --export-csv
```

## 📊 **System Components**

| Component | Description | Status |
|-----------|-------------|---------|
| `daily_automation.py` | Main automation orchestrator | ✅ Production Ready |
| `monitor_automation.py` | System health monitoring | ✅ Production Ready |
| `predict_trading_signals.py` | ML prediction engine | ✅ Production Ready |
| `retrain_model.py` | Automated model retraining | ✅ Production Ready |
| `export_results.py` | Advanced CSV export utility | ✅ Production Ready |
| `check_data_status_console_safe.py` | Database monitoring | ✅ Production Ready |

## 🎯 **ML Model Performance**

- **Algorithm**: Gradient Boosting Classifier
- **Overall Accuracy**: 66.31%
- **Sell Signal Accuracy**: 83.6%
- **Buy Signal Accuracy**: 47.5%
- **F1-Score**: 65.10%
- **Training Data**: 41,152+ NASDAQ 100 records
- **Features**: 18 engineered technical indicators

## 📁 **CSV Export Formats**

The system generates multiple CSV formats optimized for different use cases:

| Format | File Pattern | Use Case |
|--------|-------------|----------|
| **Standard** | `batch_predictions_YYYYMMDD_HHMMSS.csv` | Basic predictions |
| **Enhanced** | `predictions_enhanced_YYYYMMDD_HHMMSS.csv` | Comprehensive analysis |
| **Trading** | `trading_signals_YYYYMMDD_HHMMSS.csv` | Optimized for trading |
| **High Confidence** | `high_confidence_signals_YYYYMMDD_HHMMSS.csv` | >70% confidence only |
| **Summary** | `trading_signals_summary_YYYYMMDD_HHMMSS.csv` | Compact key information |

## 📈 **Current Data Status**

- **Database Records**: 41,152 NASDAQ 100 historical records
- **Date Range**: 2024-03-18 to 2025-11-24 
- **Unique Tickers**: 97 stocks
- **Data Freshness**: Automatically monitored (retrains when ≤2 days old)

## 🔧 **Monitoring & Maintenance**

### **Daily Health Check**
```powershell
python monitor_automation.py
```
Shows: Database connectivity, data age, last retraining status, CSV generation status

### **Historical Performance**
```powershell
python monitor_automation.py --history
```
7-day performance history with success rates

### **Issue Detection**  
```powershell
python monitor_automation.py --alerts
```
Identifies problems requiring attention

## 📋 **Project Structure**

```
├── 📁 src/                          # Core application modules
│   ├── 📁 database/                 # SQL Server connection management  
│   ├── 📁 data/                     # Feature engineering & preprocessing
│   └── 📁 models/                   # ML model implementations
├── 📁 notebooks/                    # Interactive development & analysis
│   ├── 01_database_connection.ipynb # Database setup & testing
│   ├── 02_data_exploration.ipynb    # EDA & feature analysis  
│   └── 03_model_development.ipynb   # Model training & evaluation
├── 📁 data/                         # Model artifacts & processed data
├── 📁 results/                      # Generated CSV exports
├── 📁 logs/                         # System execution logs
├── 📁 daily_reports/               # JSON automation summaries
├── daily_automation.py             # 🚀 Main automation script
├── monitor_automation.py           # 📊 System monitoring
├── predict_trading_signals.py      # 🎯 ML prediction engine
├── retrain_model.py                # 🔄 Model retraining
├── export_results.py              # 📁 CSV export utility
└── 📚 Documentation Files          # Comprehensive guides
```

## 📚 **Documentation**

- **[DAILY_AUTOMATION_GUIDE.md](DAILY_AUTOMATION_GUIDE.md)** - Complete automation setup & Windows Task Scheduler
- **[CSV_EXPORT_GUIDE.md](CSV_EXPORT_GUIDE.md)** - Export formats, filtering, and usage examples  
- **[RETRAINING_GUIDE.md](RETRAINING_GUIDE.md)** - Model maintenance & retraining procedures
- **[PROJECT_STATUS_FINAL.md](PROJECT_STATUS_FINAL.md)** - Implementation summary & achievements

## 🚨 **Production Deployment**

### **Windows Task Scheduler Setup**
1. Open Task Scheduler (`taskschd.msc`)
2. Create Basic Task: "Trading Signals Daily Automation"  
3. Schedule: Daily at 6:00 AM
4. Action: Run `run_daily_automation.bat`
5. Settings: Run with highest privileges, wake computer

See [DAILY_AUTOMATION_GUIDE.md](DAILY_AUTOMATION_GUIDE.md) for detailed setup instructions.

### **Monitoring Schedule**
- **Daily**: Check `monitor_automation.py` output
- **Weekly**: Review `monitor_automation.py --history` 
- **Monthly**: Clean up old logs and CSV files

## 🏆 **Key Achievements**

✅ **Complete ML Pipeline Automation** - From data ingestion to CSV generation  
✅ **Production-Ready Architecture** - Error handling, logging, monitoring  
✅ **Smart Resource Management** - Only retrains when new data is available  
✅ **Multiple Export Formats** - Optimized for different trading strategies  
✅ **Comprehensive Documentation** - Setup, usage, and maintenance guides  
✅ **Windows Integration** - Task Scheduler automation for hands-free operation  

## 🔗 **Related Projects**

- **Database Setup**: Scripts for populating SQL Server with NASDAQ 100 data
- **Feature Engineering**: Technical indicator calculations and transformations  
- **Model Evaluation**: Performance metrics and validation frameworks

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 **Support**

- **Documentation**: Comprehensive guides in the repository
- **Monitoring**: Use built-in `monitor_automation.py` for health tracking
- **Troubleshooting**: Check logs in `logs/` folder for detailed diagnostics
- **Issues**: Create GitHub issues for bugs or feature requests

---

**🎉 Built with ❤️ for automated ML trading signal generation**
