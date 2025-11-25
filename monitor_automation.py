#!/usr/bin/env python3
"""
Automation Monitoring Dashboard

This script provides a quick overview of the daily automation system status.
Use it to check if automation is running properly and view recent results.

Usage:
    python monitor_automation.py           # Show current status
    python monitor_automation.py --history # Show historical performance
    python monitor_automation.py --alerts  # Show any issues requiring attention
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import glob

def get_latest_log():
    """Get the most recent log file."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    log_files = list(logs_dir.glob("daily_automation_*.log"))
    if not log_files:
        return None
    
    return max(log_files, key=lambda f: f.stat().st_mtime)

def get_latest_report():
    """Get the most recent daily report."""
    reports_dir = Path("daily_reports")
    if not reports_dir.exists():
        return None
    
    report_files = list(reports_dir.glob("daily_summary_*.json"))
    if not report_files:
        return None
    
    latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_report, 'r') as f:
            return json.load(f)
    except:
        return None

def get_recent_csv_files():
    """Get CSV files created in the last 24 hours."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    
    cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)  # 24 hours ago
    csv_files = []
    
    for csv_file in results_dir.glob("*.csv"):
        if csv_file.stat().st_mtime > cutoff_time:
            csv_files.append({
                'name': csv_file.name,
                'size_mb': csv_file.stat().st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(csv_file.stat().st_mtime)
            })
    
    return sorted(csv_files, key=lambda x: x['created'], reverse=True)

def show_current_status():
    """Display current automation status."""
    print("🔍 SQL Server ML Trading Signals - Automation Status")
    print("=" * 60)
    
    # Check latest report
    latest_report = get_latest_report()
    if latest_report:
        report_date = datetime.fromisoformat(latest_report['date'].replace('Z', '+00:00'))
        hours_ago = (datetime.now() - report_date.replace(tzinfo=None)).total_seconds() / 3600
        
        print(f"📅 Last Run: {report_date.strftime('%Y-%m-%d %H:%M:%S')} ({hours_ago:.1f} hours ago)")
        
        # Data status
        data_status = latest_report.get('data_status', {})
        if data_status.get('connected'):
            age_days = data_status.get('age_days')
            print(f"📊 Database: ✅ Connected (Data age: {age_days} days)")
            print(f"📈 Records: {data_status.get('total_records', 'Unknown'):,}")
            print(f"📅 Latest Date: {data_status.get('latest_date', 'Unknown')}")
        else:
            print("📊 Database: ❌ Connection failed")
        
        # Retraining status
        retrain_info = latest_report.get('retraining', {})
        if retrain_info.get('performed'):
            if retrain_info.get('success'):
                duration = retrain_info.get('duration_minutes', 0)
                print(f"🔄 Retraining: ✅ Successful ({duration:.1f} minutes)")
            else:
                print("🔄 Retraining: ❌ Failed")
        else:
            reason = retrain_info.get('reason', 'Unknown reason')
            print(f"🔄 Retraining: ⏭️ Skipped ({reason})")
        
        # CSV exports
        csv_info = latest_report.get('csv_exports', {})
        if csv_info.get('success'):
            file_count = len(csv_info.get('files_created', []))
            print(f"📁 CSV Exports: ✅ Success ({file_count} files)")
        else:
            print("📁 CSV Exports: ❌ Failed")
    else:
        print("📅 Last Run: ⚠️ No automation reports found")
    
    print()
    
    # Recent CSV files
    recent_csvs = get_recent_csv_files()
    if recent_csvs:
        print("📁 Recent CSV Files (Last 24 hours):")
        for csv in recent_csvs[:5]:  # Show top 5
            print(f"   • {csv['name']} ({csv['size_mb']:.1f} MB) - {csv['created'].strftime('%H:%M:%S')}")
        if len(recent_csvs) > 5:
            print(f"   ... and {len(recent_csvs) - 5} more files")
    else:
        print("📁 Recent CSV Files: None found")
    
    print()
    
    # Latest log snippet
    latest_log = get_latest_log()
    if latest_log:
        print("📝 Latest Log Snippet:")
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                # Show last 5 non-empty lines
                relevant_lines = [line.strip() for line in lines if line.strip()][-5:]
                for line in relevant_lines:
                    print(f"   {line}")
        except:
            print("   ⚠️ Unable to read log file")
    else:
        print("📝 Latest Log: No log files found")

def show_history():
    """Show historical performance over the last week."""
    print("📊 Automation History (Last 7 Days)")
    print("=" * 60)
    
    reports_dir = Path("daily_reports")
    if not reports_dir.exists():
        print("No historical reports found")
        return
    
    # Get reports from last 7 days
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_reports = []
    
    for report_file in reports_dir.glob("daily_summary_*.json"):
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
                report_date = datetime.fromisoformat(report['date'].replace('Z', '+00:00'))
                if report_date.replace(tzinfo=None) >= cutoff_date:
                    recent_reports.append((report_date, report))
        except:
            continue
    
    recent_reports.sort(key=lambda x: x[0])
    
    if not recent_reports:
        print("No reports found for the last 7 days")
        return
    
    print("Date       | DB  | Retrain | CSV | Duration")
    print("-" * 45)
    
    for report_date, report in recent_reports:
        date_str = report_date.strftime("%Y-%m-%d")
        
        # Database status
        db_status = "✅" if report.get('data_status', {}).get('connected') else "❌"
        
        # Retraining status
        retrain_info = report.get('retraining', {})
        if retrain_info.get('performed'):
            retrain_status = "✅" if retrain_info.get('success') else "❌"
        else:
            retrain_status = "⏭️"
        
        # CSV status
        csv_status = "✅" if report.get('csv_exports', {}).get('success') else "❌"
        
        # Duration
        duration = retrain_info.get('duration_minutes')
        duration_str = f"{duration:.1f}m" if duration else "N/A"
        
        print(f"{date_str} |  {db_status}  |    {retrain_status}    |  {csv_status}  |  {duration_str:>6}")

def show_alerts():
    """Show any issues that require attention."""
    print("🚨 Automation Alerts & Issues")
    print("=" * 60)
    
    alerts = []
    
    # Check if automation ran recently
    latest_report = get_latest_report()
    if latest_report:
        report_date = datetime.fromisoformat(latest_report['date'].replace('Z', '+00:00'))
        hours_since = (datetime.now() - report_date.replace(tzinfo=None)).total_seconds() / 3600
        
        if hours_since > 30:  # More than 30 hours since last run
            alerts.append(f"⚠️ Automation hasn't run in {hours_since:.1f} hours")
        
        # Check for failures
        if not latest_report.get('data_status', {}).get('connected'):
            alerts.append("❌ Database connection failed in last run")
        
        retrain_info = latest_report.get('retraining', {})
        if retrain_info.get('performed') and not retrain_info.get('success'):
            alerts.append("❌ Model retraining failed in last run")
        
        if not latest_report.get('csv_exports', {}).get('success'):
            alerts.append("❌ CSV export generation failed in last run")
        
        # Check data age
        data_age = latest_report.get('data_status', {}).get('age_days')
        if data_age is not None and data_age > 5:
            alerts.append(f"⚠️ Data is {data_age} days old (may be stale)")
    else:
        alerts.append("❌ No automation reports found")
    
    # Check disk space for results
    results_dir = Path("results")
    if results_dir.exists():
        total_size = sum(f.stat().st_size for f in results_dir.glob("*.csv"))
        if total_size > 100 * 1024 * 1024:  # More than 100MB
            alerts.append(f"💾 Results folder is large ({total_size / (1024*1024):.1f} MB) - consider cleanup")
    
    # Check log file count
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_count = len(list(logs_dir.glob("*.log")))
        if log_count > 30:
            alerts.append(f"📝 Many log files ({log_count}) - consider cleanup")
    
    if alerts:
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("✅ No issues detected - automation appears to be running normally")
    
    print()
    
    # Show recommendations
    print("💡 Recommendations:")
    print("   • Schedule daily automation using Task Scheduler")
    print("   • Monitor this dashboard weekly")
    print("   • Clean up old CSV files monthly")
    print("   • Review model performance quarterly")

def main():
    parser = argparse.ArgumentParser(description="Monitor automation system status")
    parser.add_argument("--history", action="store_true", help="Show historical performance")
    parser.add_argument("--alerts", action="store_true", help="Show alerts and issues")
    
    args = parser.parse_args()
    
    if args.history:
        show_history()
    elif args.alerts:
        show_alerts()
    else:
        show_current_status()

if __name__ == "__main__":
    main()
