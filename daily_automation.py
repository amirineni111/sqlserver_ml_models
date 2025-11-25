#!/usr/bin/env python3
"""
Daily Automation Script for SQL Server ML Trading Signals System

This script provides a complete daily workflow that:
1. Checks data freshness and connectivity
2. Performs automated model retraining if needed
3. Generates CSV exports for trading decisions
4. Handles errors and creates logs

Usage:
    python daily_automation.py                    # Full automated daily process
    python daily_automation.py --force-retrain   # Force retraining regardless of data age
    python daily_automation.py --csv-only        # Only generate CSV exports (skip retraining)
    python daily_automation.py --check-only      # Only check data status
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json

# Setup logging
def setup_logging():
    """Setup comprehensive logging for daily automation."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"daily_automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging with UTF-8 encoding and proper formatting
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            formatted = super().format(record)
            # Replace emoji with text equivalents for console output
            emoji_replacements = {
                '🔍': '[INFO]',
                '🚀': '[START]',
                '📝': '[LOG]',
                '❌': '[ERROR]',
                '✅': '[SUCCESS]',
                '📊': '[DATA]',
                '🔄': '[RETRAIN]',
                '📁': '[FILES]',
                '⏰': '[TIMEOUT]',
                '📈': '[STATS]',
                '📅': '[DATE]',
                '⚠️': '[WARN]',
                '🎉': '[COMPLETE]',
                '🤔': '[DECISION]',
                '⏭️': '[SKIP]',
                '📋': '[SUMMARY]'
            }
            for emoji, replacement in emoji_replacements.items():
                formatted = formatted.replace(emoji, replacement)
            return formatted
    
    # File handler with UTF-8
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_filename

def check_data_status():
    """
    Check data status and connectivity.
    Returns: (is_connected, data_age_days, latest_date, total_records)
    """
    try:
        logging.info("🔍 Checking data status and SQL Server connectivity...")
          # Run data status check script
        result = subprocess.run([
            sys.executable, "check_data_status_console_safe.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logging.error(f"Data status check failed: {result.stderr}")
            return False, None, None, None
            
        # Parse output to extract data information
        output_lines = result.stdout.strip().split('\n')
          # Look for key information in output
        is_connected = "Database connection successful" in result.stdout
        data_age_days = None
        latest_date = None
        total_records = None
        
        for line in output_lines:
            if "Data is" in line and "days old" in line:
                # Extract age from line like "Data is 1 days old"
                try:
                    parts = line.split("Data is")[1].split("days old")[0].strip()
                    data_age_days = int(parts)
                except:
                    pass
            elif "Latest date:" in line:
                latest_date = line.split("Latest date:")[1].strip()
            elif "Total records:" in line:
                try:
                    total_records = int(line.split("Total records:")[1].strip().replace(",", ""))
                except:
                    pass
        
        logging.info(f"📊 Data Status: Connected={is_connected}, Age={data_age_days} days, Latest={latest_date}, Records={total_records:,}" if total_records else f"📊 Data Status: Connected={is_connected}")
        
        return is_connected, data_age_days, latest_date, total_records
        
    except subprocess.TimeoutExpired:
        logging.error("⏰ Data status check timed out after 60 seconds")
        return False, None, None, None
    except Exception as e:
        logging.error(f"❌ Error checking data status: {str(e)}")
        return False, None, None, None

def should_retrain(data_age_days, force_retrain=False):
    """
    Determine if model should be retrained based on data age.
    
    Args:
        data_age_days: Age of latest data in days
        force_retrain: Force retraining regardless of age
        
    Returns: (should_retrain, reason)
    """
    if force_retrain:
        return True, "Force retrain requested by user"
    
    if data_age_days is None:
        return False, "Unable to determine data age"
    
    # Retrain if data is fresh (0-2 days old)
    if data_age_days <= 2:
        return True, f"Data is fresh ({data_age_days} days old)"
    
    # Don't retrain if data is older than 2 days
    return False, f"Data is {data_age_days} days old (threshold: 2 days)"

def run_model_retraining():
    """
    Execute model retraining with backup.
    Returns: (success, training_time_minutes)
    """
    try:
        logging.info("🚀 Starting automated model retraining...")
        start_time = datetime.now()
        
        # Run retraining with backup
        result = subprocess.run([
            sys.executable, "retrain_model.py", "--backup-old", "--quick"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60
        
        if result.returncode == 0:
            logging.info(f"✅ Model retraining completed successfully in {training_time:.1f} minutes")
            
            # Log key information from retraining output
            if "Best model:" in result.stdout:
                best_model_line = [line for line in result.stdout.split('\n') if 'Best model:' in line][0]
                logging.info(f"📈 {best_model_line}")
            
            return True, training_time
        else:
            logging.error(f"❌ Model retraining failed after {training_time:.1f} minutes")
            logging.error(f"Error output: {result.stderr}")
            return False, training_time
            
    except subprocess.TimeoutExpired:
        logging.error("⏰ Model retraining timed out after 30 minutes")
        return False, 30.0
    except Exception as e:
        logging.error(f"❌ Error during model retraining: {str(e)}")
        return False, 0.0

def generate_csv_exports():
    """
    Generate all CSV exports for trading decisions.
    Returns: (success, csv_files_created)
    """
    try:
        logging.info("📊 Generating CSV exports for trading decisions...")
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate predictions with CSV export
        result = subprocess.run([
            sys.executable, "predict_trading_signals.py", 
            "--batch", "--export-csv", f"--results-dir={results_dir}"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            logging.error(f"❌ CSV generation failed: {result.stderr}")
            return False, []
        
        # Run advanced export utility
        export_result = subprocess.run([
            sys.executable, "export_results.py"
        ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        # Count generated CSV files
        csv_files = list(results_dir.glob("*.csv"))
        csv_files = [f.name for f in csv_files if f.stat().st_mtime > (datetime.now().timestamp() - 3600)]  # Files from last hour
        
        if export_result.returncode == 0:
            logging.info(f"✅ CSV exports completed successfully")
            logging.info(f"📁 Generated files: {', '.join(csv_files)}")
            return True, csv_files
        else:
            logging.warning(f"⚠️ Basic CSV export succeeded, but advanced export had issues: {export_result.stderr}")
            return True, csv_files
            
    except subprocess.TimeoutExpired:
        logging.error("⏰ CSV generation timed out")
        return False, []
    except Exception as e:
        logging.error(f"❌ Error generating CSV exports: {str(e)}")
        return False, []

def create_daily_summary(log_filename, data_status, retrain_info, csv_info):
    """Create a daily summary report."""
    summary_dir = Path("daily_reports")
    summary_dir.mkdir(exist_ok=True)
    
    summary_file = summary_dir / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
    
    summary = {
        "date": datetime.now().isoformat(),
        "log_file": str(log_filename),
        "data_status": {
            "connected": data_status[0],
            "age_days": data_status[1],
            "latest_date": data_status[2],
            "total_records": data_status[3]
        },
        "retraining": {
            "performed": retrain_info[0] if retrain_info else False,
            "success": retrain_info[1] if retrain_info and retrain_info[0] else None,
            "duration_minutes": retrain_info[2] if retrain_info and len(retrain_info) > 2 else None,
            "reason": retrain_info[3] if retrain_info and len(retrain_info) > 3 else None
        },
        "csv_exports": {
            "success": csv_info[0] if csv_info else False,
            "files_created": csv_info[1] if csv_info and len(csv_info) > 1 else []
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"📋 Daily summary saved to {summary_file}")
    return summary_file

def main():
    parser = argparse.ArgumentParser(description="Daily automation for SQL Server ML Trading Signals")
    parser.add_argument("--force-retrain", action="store_true", 
                       help="Force model retraining regardless of data age")
    parser.add_argument("--csv-only", action="store_true",
                       help="Only generate CSV exports (skip retraining check)")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check data status (no retraining or CSV)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_filename = setup_logging()
    
    logging.info("🚀 Starting daily automation for SQL Server ML Trading Signals System")
    logging.info(f"📝 Log file: {log_filename}")
    
    # Track results
    data_status = None
    retrain_info = None
    csv_info = None
    
    try:
        # Step 1: Check data status
        logging.info("=" * 60)
        logging.info("STEP 1: DATA STATUS CHECK")
        logging.info("=" * 60)
        
        data_status = check_data_status()
        is_connected, data_age_days, latest_date, total_records = data_status
        
        if not is_connected:
            logging.error("❌ Cannot connect to SQL Server database. Aborting automation.")
            return 1
        
        if args.check_only:
            logging.info("🏁 Check-only mode completed")
            create_daily_summary(log_filename, data_status, retrain_info, csv_info)
            return 0
        
        # Step 2: Model retraining decision
        if not args.csv_only:
            logging.info("=" * 60)
            logging.info("STEP 2: MODEL RETRAINING DECISION")
            logging.info("=" * 60)
            
            should_train, reason = should_retrain(data_age_days, args.force_retrain)
            logging.info(f"🤔 Retraining decision: {'YES' if should_train else 'NO'} - {reason}")
            
            if should_train:
                logging.info("=" * 60)
                logging.info("STEP 3: MODEL RETRAINING")
                logging.info("=" * 60)
                
                retrain_success, training_time = run_model_retraining()
                retrain_info = (True, retrain_success, training_time, reason)
                
                if not retrain_success:
                    logging.error("❌ Model retraining failed. Continuing with CSV generation using existing model.")
            else:
                logging.info("⏭️ Skipping model retraining")
                retrain_info = (False, None, None, reason)
        
        # Step 3/4: CSV Generation
        logging.info("=" * 60)
        logging.info("STEP 4: CSV EXPORT GENERATION" if not args.csv_only else "CSV EXPORT GENERATION")
        logging.info("=" * 60)
        
        csv_success, csv_files = generate_csv_exports()
        csv_info = (csv_success, csv_files)
        
        if not csv_success:
            logging.error("❌ CSV generation failed")
        
        # Final summary
        logging.info("=" * 60)
        logging.info("DAILY AUTOMATION SUMMARY")
        logging.info("=" * 60)
        
        logging.info(f"📊 Data Status: {'✅ Connected' if is_connected else '❌ Disconnected'}")
        if data_age_days is not None:
            logging.info(f"📅 Data Age: {data_age_days} days (Latest: {latest_date})")
        if total_records:
            logging.info(f"📈 Records: {total_records:,}")
        
        if retrain_info:
            if retrain_info[0]:  # Retraining was performed
                status = "✅ Success" if retrain_info[1] else "❌ Failed"
                logging.info(f"🔄 Retraining: {status} ({retrain_info[2]:.1f} min)")
            else:
                logging.info("⏭️ Retraining: Skipped")
        
        if csv_info:
            status = "✅ Success" if csv_info[0] else "❌ Failed"
            file_count = len(csv_info[1]) if csv_info[1] else 0
            logging.info(f"📁 CSV Exports: {status} ({file_count} files)")
        
        # Create summary report
        summary_file = create_daily_summary(log_filename, data_status, retrain_info, csv_info)
        
        # Determine exit code
        if is_connected and (csv_info is None or csv_info[0]):
            logging.info("🎉 Daily automation completed successfully!")
            return 0
        else:
            logging.error("❌ Daily automation completed with errors")
            return 1
            
    except KeyboardInterrupt:
        logging.warning("⚠️ Daily automation interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"❌ Unexpected error during daily automation: {str(e)}")
        return 1
    finally:
        logging.info(f"📝 Full log available at: {log_filename}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
