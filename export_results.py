"""
CSV Export Utility for Trading Signal Results

This script provides advanced CSV export functionality for trading signal predictions.
It can export results with various filtering and formatting options.

Usage:
    python export_results.py --batch --export-csv
    python export_results.py --ticker AAPL --export-csv --format enhanced
    python export_results.py --batch --filter high-confidence --export-csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

# Import the predictor
from predict_trading_signals import TradingSignalPredictor

class ResultsExporter:
    """Advanced CSV export functionality for trading signals"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.predictor = TradingSignalPredictor()
        
    def export_predictions(self, ticker=None, date=None, confidence_threshold=0.7, 
                          export_format='standard', filter_type='all'):
        """Export predictions to CSV with various options"""
        
        print("[PROCESSING] Generating predictions for export...")
        
        # Get predictions
        results = self.predictor.predict_signals(
            ticker=ticker, 
            date=date, 
            confidence_threshold=confidence_threshold
        )
        
        if results is None or results.empty:
            print("[ERROR] No predictions available for export")
            return None
        
        # Apply filters
        filtered_results = self._apply_filters(results, filter_type)
        
        if filtered_results.empty:
            print(f"[ERROR] No predictions match the filter: {filter_type}")
            return None
        
        # Format based on export format
        if export_format == 'enhanced':
            export_df = self._create_enhanced_format(filtered_results)
        elif export_format == 'summary':
            export_df = self._create_summary_format(filtered_results)
        elif export_format == 'trading':
            export_df = self._create_trading_format(filtered_results)
        else:  # standard
            export_df = self._create_standard_format(filtered_results)
        
        # Generate filename
        filename = self._generate_filename(ticker, date, filter_type, export_format)
        filepath = self.results_dir / filename
        
        # Export to CSV
        export_df.to_csv(filepath, index=False)
        
        print(f"[SUCCESS] Results exported to: {filepath}")
        print(f"[DATA] Exported {len(export_df)} predictions")
        
        # Print summary
        self._print_export_summary(export_df, filter_type)
        
        return filepath
    
    def _apply_filters(self, results, filter_type):
        """Apply filtering based on filter type"""
        if filter_type == 'high-confidence':
            return results[results['high_confidence']]
        elif filter_type == 'buy-signals':
            return results[results['predicted_signal'].str.contains('Buy', na=False)]
        elif filter_type == 'sell-signals':
            return results[results['predicted_signal'].str.contains('Sell', na=False)]
        elif filter_type == 'medium-high':
            return results[results['confidence'] > 0.6]
        else:  # 'all'
            return results
    
    def _create_standard_format(self, results):
        """Create standard CSV format"""
        df = results.copy()
        df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['confidence_level'] = df['confidence'].apply(
            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.6 else 'Low'
        )
        
        column_order = [
            'export_timestamp', 'trading_date', 'ticker', 'company',
            'predicted_signal', 'confidence', 'confidence_level',
            'close_price', 'RSI', 'high_confidence'
        ]
        
        return df.reindex(columns=[col for col in column_order if col in df.columns])
    
    def _create_enhanced_format(self, results):
        """Create enhanced CSV format with additional calculated fields"""
        df = results.copy()
        df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['confidence_level'] = df['confidence'].apply(
            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.6 else 'Low'
        )
        df['confidence_percentage'] = (df['confidence'] * 100).round(1)
        df['rsi_category'] = df['RSI'].apply(
            lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral'
        )
        df['signal_strength'] = df.apply(
            lambda row: 'Strong' if row['confidence'] > 0.8 else 'Moderate' if row['confidence'] > 0.6 else 'Weak',
            axis=1
        )
        
        # Price-based risk assessment
        df['price_risk'] = df['close_price'].apply(
            lambda x: 'High' if x > 500 else 'Medium' if x > 100 else 'Low'
        )
        
        column_order = [
            'export_timestamp', 'trading_date', 'ticker', 'company',
            'predicted_signal', 'signal_strength', 'confidence', 'confidence_percentage', 'confidence_level',
            'close_price', 'price_risk', 'RSI', 'rsi_category',
            'sell_probability', 'buy_probability', 'high_confidence'
        ]
        
        return df.reindex(columns=[col for col in column_order if col in df.columns])
    
    def _create_summary_format(self, results):
        """Create summary CSV format with key information only"""
        df = results[['ticker', 'company', 'predicted_signal', 'confidence', 'close_price', 'RSI']].copy()
        df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['confidence_pct'] = (df['confidence'] * 100).round(1)
        
        # Reorder for summary view
        column_order = [
            'export_timestamp', 'ticker', 'company', 'predicted_signal', 
            'confidence_pct', 'close_price', 'RSI'
        ]
        
        return df.reindex(columns=column_order)
    
    def _create_trading_format(self, results):
        """Create trading-focused CSV format"""
        df = results.copy()
        df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['action'] = df['predicted_signal'].apply(
            lambda x: 'SELL' if 'Sell' in str(x) else 'BUY' if 'Buy' in str(x) else 'HOLD'
        )
        df['priority'] = df['confidence'].apply(
            lambda x: 1 if x > 0.8 else 2 if x > 0.7 else 3 if x > 0.6 else 4
        )
        df['risk_level'] = df.apply(
            lambda row: 'Low' if row['confidence'] > 0.8 else 'Medium' if row['confidence'] > 0.6 else 'High',
            axis=1
        )
        
        column_order = [
            'export_timestamp', 'ticker', 'action', 'priority', 'risk_level',
            'confidence', 'close_price', 'RSI', 'predicted_signal'
        ]
        
        return df.reindex(columns=[col for col in column_order if col in df.columns])
    
    def _generate_filename(self, ticker, date, filter_type, export_format):
        """Generate appropriate filename for the export"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        parts = ['predictions']
        
        if ticker:
            parts.append(ticker)
        
        if date:
            parts.append(date.replace('-', ''))
        
        if filter_type != 'all':
            parts.append(filter_type.replace('-', '_'))
        
        if export_format != 'standard':
            parts.append(export_format)
        
        parts.append(timestamp)
        
        return '_'.join(parts) + '.csv'
    
    def _print_export_summary(self, export_df, filter_type):
        """Print summary of exported data"""
        if 'predicted_signal' in export_df.columns:
            buy_count = len(export_df[export_df['predicted_signal'].str.contains('Buy', na=False)])
            sell_count = len(export_df[export_df['predicted_signal'].str.contains('Sell', na=False)])
            
            print(f"📈 Buy Signals: {buy_count}")
            print(f"📉 Sell Signals: {sell_count}")
        
        if 'confidence' in export_df.columns:
            avg_confidence = export_df['confidence'].mean()
            print(f"🎯 Average Confidence: {avg_confidence:.1%}")
    
    def export_batch_with_segments(self, confidence_threshold=0.7):
        """Export batch results segmented by confidence levels"""
        print("[PROCESSING] Generating segmented batch export...")
        
        # Get all predictions
        results = self.predictor.predict_signals(confidence_threshold=0.5)  # Lower threshold to get all
        
        if results is None or results.empty:
            print("[ERROR] No predictions available")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export high confidence
        high_conf = results[results['confidence'] > 0.7]
        if not high_conf.empty:
            filename = f"high_confidence_signals_{timestamp}.csv"
            filepath = self.results_dir / filename
            enhanced_high = self._create_enhanced_format(high_conf)
            enhanced_high.to_csv(filepath, index=False)
            print(f"[SUCCESS] High confidence signals exported: {filepath} ({len(high_conf)} signals)")
        
        # Export medium confidence
        medium_conf = results[(results['confidence'] > 0.6) & (results['confidence'] <= 0.7)]
        if not medium_conf.empty:
            filename = f"medium_confidence_signals_{timestamp}.csv"
            filepath = self.results_dir / filename
            enhanced_medium = self._create_enhanced_format(medium_conf)
            enhanced_medium.to_csv(filepath, index=False)
            print(f"[SUCCESS] Medium confidence signals exported: {filepath} ({len(medium_conf)} signals)")
        
        # Export trading summary
        trading_signals = results[results['confidence'] > confidence_threshold]
        if not trading_signals.empty:
            filename = f"trading_signals_summary_{timestamp}.csv"
            filepath = self.results_dir / filename
            trading_format = self._create_trading_format(trading_signals)
            trading_format.to_csv(filepath, index=False)
            print(f"[SUCCESS] Trading signals summary exported: {filepath} ({len(trading_signals)} signals)")

def main():
    """Main CLI interface for CSV export utility"""
    parser = argparse.ArgumentParser(description='Advanced CSV Export for Trading Signals')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)')
    parser.add_argument('--batch', action='store_true', help='Export batch predictions')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--export-csv', action='store_true', help='Export to CSV')
    parser.add_argument('--format', choices=['standard', 'enhanced', 'summary', 'trading'], 
                       default='standard', help='CSV format type')
    parser.add_argument('--filter', choices=['all', 'high-confidence', 'buy-signals', 'sell-signals', 'medium-high'],
                       default='all', help='Filter type for export')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--segmented', action='store_true', help='Export segmented by confidence levels')
    
    args = parser.parse_args()
    
    if not args.export_csv and not args.segmented:
        print("[ERROR] Please specify --export-csv or --segmented to export results")
        return 1
    
    # Initialize exporter
    exporter = ResultsExporter(args.results_dir)
    
    try:
        if args.segmented:
            exporter.export_batch_with_segments(args.confidence)
        elif args.batch:
            exporter.export_predictions(
                confidence_threshold=args.confidence,
                export_format=args.format,
                filter_type=args.filter
            )
        else:
            exporter.export_predictions(
                ticker=args.ticker,
                date=args.date,
                confidence_threshold=args.confidence,
                export_format=args.format,
                filter_type=args.filter
            )
        
        print(f"\n[FILES] All exports saved to: {args.results_dir}/")
        return 0
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
