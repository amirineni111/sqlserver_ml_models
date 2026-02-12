"""
Database Export Utility for Trading Signal Results

This script exports trading signal predictions and technical indicators 
directly to SQL Server tables for analysis and visualization.

Creates tables:
- trading_predictions: All predictions with confidence scores
- technical_indicators: Detailed MACD/SMA/EMA values for each prediction
- prediction_summary: Daily summary statistics

Usage:
    python export_to_database.py --batch
    python export_to_database.py --ticker AAPL
    python export_to_database.py --create-tables  # First time setup
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path
import pyodbc
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, Float, String, DateTime, Boolean

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

# Import the predictor
from predict_trading_signals import TradingSignalPredictor


class DatabaseExporter:
    """Export trading signals and technical indicators to SQL Server"""
    
    def __init__(self):
        self.db = SQLServerConnection()
        self.predictor = TradingSignalPredictor()
        self.engine = self.db.get_sqlalchemy_engine()
        
        # Table names
        self.predictions_table = 'ml_trading_predictions'
        self.technical_table = 'ml_technical_indicators'
        self.summary_table = 'ml_prediction_summary'
        
    def create_tables(self):
        """Create SQL Server tables for storing predictions"""
        print("[PROCESSING] Creating database tables...")
        
        # Create predictions table
        predictions_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{self.predictions_table}')
        BEGIN
            CREATE TABLE {self.predictions_table} (
                prediction_id INT IDENTITY(1,1) PRIMARY KEY,
                run_timestamp DATETIME NOT NULL,
                trading_date DATE NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                company VARCHAR(200),
                predicted_signal VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                confidence_percentage FLOAT,
                signal_strength VARCHAR(20),
                close_price FLOAT,
                RSI FLOAT,
                rsi_category VARCHAR(20),
                high_confidence BIT,
                sell_probability FLOAT,
                buy_probability FLOAT,
                created_at DATETIME DEFAULT GETDATE(),
                INDEX IDX_ticker_date (ticker, trading_date),
                INDEX IDX_run_timestamp (run_timestamp),
                INDEX IDX_confidence (confidence)
            )
            PRINT 'Table {self.predictions_table} created successfully'
        END
        ELSE
            PRINT 'Table {self.predictions_table} already exists'
        """
        
        # Create technical indicators table
        technical_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{self.technical_table}')
        BEGIN
            CREATE TABLE {self.technical_table} (
                indicator_id INT IDENTITY(1,1) PRIMARY KEY,
                run_timestamp DATETIME NOT NULL,
                trading_date DATE NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                -- Moving Averages
                sma_5 FLOAT,
                sma_10 FLOAT,
                sma_20 FLOAT,
                sma_50 FLOAT,
                ema_5 FLOAT,
                ema_10 FLOAT,
                ema_20 FLOAT,
                ema_50 FLOAT,
                -- MACD
                macd FLOAT,
                macd_signal FLOAT,
                macd_histogram FLOAT,
                macd_trend VARCHAR(20),
                -- Price Relationships
                price_vs_sma20 FLOAT,
                price_vs_sma20_pct FLOAT,
                price_vs_sma50 FLOAT,
                price_vs_sma50_pct FLOAT,
                price_vs_ema20 FLOAT,
                -- Trend Indicators
                sma20_vs_sma50 FLOAT,
                ema20_vs_ema50 FLOAT,
                trend_direction VARCHAR(20),
                sma5_vs_sma20 FLOAT,
                -- Volume
                volume_sma_20 FLOAT,
                volume_sma_ratio FLOAT,
                -- Momentum
                price_momentum_5 FLOAT,
                price_momentum_10 FLOAT,
                rsi_momentum FLOAT,
                daily_volatility FLOAT,
                created_at DATETIME DEFAULT GETDATE(),
                INDEX IDX_ticker_date_tech (ticker, trading_date),
                INDEX IDX_run_timestamp_tech (run_timestamp)
            )
            PRINT 'Table {self.technical_table} created successfully'
        END
        ELSE
            PRINT 'Table {self.technical_table} already exists'
        """
        
        # Create summary table
        summary_sql = f"""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{self.summary_table}')
        BEGIN
            CREATE TABLE {self.summary_table} (
                summary_id INT IDENTITY(1,1) PRIMARY KEY,
                run_timestamp DATETIME NOT NULL UNIQUE,
                run_date DATE NOT NULL,
                total_predictions INT,
                high_confidence_count INT,
                medium_confidence_count INT,
                buy_signals INT,
                sell_signals INT,
                avg_confidence FLOAT,
                avg_rsi FLOAT,
                bullish_macd_count INT,
                bearish_macd_count INT,
                uptrend_count INT,
                downtrend_count INT,
                sideways_count INT,
                created_at DATETIME DEFAULT GETDATE(),
                INDEX IDX_run_date (run_date)
            )
            PRINT 'Table {self.summary_table} created successfully'
        END
        ELSE
            PRINT 'Table {self.summary_table} already exists'
        """
        
        # Execute table creation
        try:
            with self.engine.connect() as conn:
                conn.execute(text(predictions_sql))
                conn.execute(text(technical_sql))
                conn.execute(text(summary_sql))
                conn.commit()
            print("[SUCCESS] All tables created successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create tables: {e}")
            return False
    
    def export_predictions_to_db(self, ticker=None, confidence_threshold=0.5):
        """Export predictions to database"""
        print("[PROCESSING] Generating predictions for database export...")
        
        # Generate run timestamp
        run_timestamp = datetime.now()
        
        # Get predictions
        results = self.predictor.predict_signals(
            ticker=ticker,
            confidence_threshold=confidence_threshold
        )
        
        if results is None or results.empty:
            print("[ERROR] No predictions available")
            return False
        
        # Prepare predictions data
        predictions_df = self._prepare_predictions_data(results, run_timestamp)
        
        # Get technical indicators data
        technical_df = self._prepare_technical_data(results, run_timestamp)
        
        # Generate summary
        summary_df = self._generate_summary(results, technical_df, run_timestamp)
        
        # Export to database
        try:
            print(f"[DATABASE] Inserting {len(predictions_df)} predictions...")
            predictions_df.to_sql(
                self.predictions_table,
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=100
            )
            print(f"[SUCCESS] Predictions inserted into {self.predictions_table}")
            
            print(f"[DATABASE] Inserting {len(technical_df)} technical indicators...")
            technical_df.to_sql(
                self.technical_table,
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=10  # Smaller chunks for complex data
            )
            print(f"[SUCCESS] Technical indicators inserted into {self.technical_table}")
            
            print(f"[DATABASE] Inserting summary record...")
            summary_df.to_sql(
                self.summary_table,
                self.engine,
                if_exists='append',
                index=False
            )
            print(f"[SUCCESS] Summary inserted into {self.summary_table}")
            
            print(f"\n[COMPLETE] Database export completed successfully!")
            print(f"Run Timestamp: {run_timestamp}")
            print(f"Total Predictions: {len(predictions_df)}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Database export failed: {e}")
            return False
    
    def _prepare_predictions_data(self, results, run_timestamp):
        """Prepare predictions data for database"""
        df = results.copy()
        
        # Add run timestamp
        df['run_timestamp'] = run_timestamp
        
        # Add calculated fields
        df['confidence_percentage'] = (df['confidence'] * 100).round(1)
        # Thresholds aligned with calibrated model performance:
        # Strong >= 65% (calibration shows 67-75% accuracy at this level)
        # Moderate >= 55% (above random chance, actionable with caution)
        # Weak < 55% (near coin-flip, not actionable)
        df['signal_strength'] = df['confidence'].apply(
            lambda x: 'Strong' if x >= 0.65 else 'Moderate' if x >= 0.55 else 'Weak'
        )
        df['rsi_category'] = df['RSI'].apply(
            lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral'
        )
        
        # Select columns for database
        columns = [
            'run_timestamp', 'trading_date', 'ticker', 'company',
            'predicted_signal', 'confidence', 'confidence_percentage', 'signal_strength',
            'close_price', 'RSI', 'rsi_category', 'high_confidence',
            'sell_probability', 'buy_probability'
        ]
        
        return df[[col for col in columns if col in df.columns]]
    
    def _prepare_technical_data(self, base_results, run_timestamp):
        """Prepare technical indicators data for database"""
        try:
            # Get technical indicators
            tickers = base_results['ticker'].unique().tolist()
            recent_data = self.predictor.get_latest_data(days_back=80)
            
            if tickers:
                recent_data = recent_data[recent_data['ticker'].isin(tickers)]
            
            # Calculate features
            feature_data = self.predictor.engineer_features(recent_data)
            feature_data_latest = feature_data.groupby('ticker').last().reset_index()
            
            # Add run timestamp
            feature_data_latest['run_timestamp'] = run_timestamp
            
            # Add MACD trend analysis
            if 'macd' in feature_data_latest.columns and 'macd_signal' in feature_data_latest.columns:
                feature_data_latest['macd_trend'] = feature_data_latest.apply(
                    lambda row: 'Bullish' if row['macd'] > row['macd_signal'] else 'Bearish',
                    axis=1
                )
            
            # Add trend direction analysis
            if 'sma20_vs_sma50' in feature_data_latest.columns:
                feature_data_latest['trend_direction'] = feature_data_latest['sma20_vs_sma50'].apply(
                    lambda x: 'Uptrend' if x > 1.02 else 'Downtrend' if x < 0.98 else 'Sideways'
                )
            
            # Add price vs MA percentages
            if 'price_vs_sma20' in feature_data_latest.columns:
                feature_data_latest['price_vs_sma20_pct'] = ((feature_data_latest['price_vs_sma20'] - 1) * 100).round(2)
            if 'price_vs_sma50' in feature_data_latest.columns:
                feature_data_latest['price_vs_sma50_pct'] = ((feature_data_latest['price_vs_sma50'] - 1) * 100).round(2)
            
            # Select columns for database
            columns = [
                'run_timestamp', 'trading_date', 'ticker',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'macd', 'macd_signal', 'macd_histogram', 'macd_trend',
                'price_vs_sma20', 'price_vs_sma20_pct',
                'price_vs_sma50', 'price_vs_sma50_pct',
                'price_vs_ema20',
                'sma20_vs_sma50', 'ema20_vs_ema50', 'trend_direction',
                'sma5_vs_sma20',
                'volume_sma_20', 'volume_sma_ratio',
                'price_momentum_5', 'price_momentum_10',
                'rsi_momentum', 'daily_volatility'
            ]
            
            return feature_data_latest[[col for col in columns if col in feature_data_latest.columns]]
            
        except Exception as e:
            print(f"[WARNING] Could not prepare technical data: {e}")
            return pd.DataFrame()
    
    def _generate_summary(self, predictions_df, technical_df, run_timestamp):
        """Generate summary statistics"""
        summary = {
            'run_timestamp': run_timestamp,
            'run_date': run_timestamp.date(),
            'total_predictions': len(predictions_df),
            'high_confidence_count': len(predictions_df[predictions_df['confidence'] > 0.7]),
            'medium_confidence_count': len(predictions_df[(predictions_df['confidence'] > 0.6) & (predictions_df['confidence'] <= 0.7)]),
            'buy_signals': len(predictions_df[predictions_df['predicted_signal'].str.contains('Buy', na=False)]),
            'sell_signals': len(predictions_df[predictions_df['predicted_signal'].str.contains('Sell', na=False)]),
            'avg_confidence': predictions_df['confidence'].mean(),
            'avg_rsi': predictions_df['RSI'].mean() if 'RSI' in predictions_df.columns else None
        }
        
        # Add technical summary if available
        if not technical_df.empty:
            if 'macd_trend' in technical_df.columns:
                summary['bullish_macd_count'] = len(technical_df[technical_df['macd_trend'] == 'Bullish'])
                summary['bearish_macd_count'] = len(technical_df[technical_df['macd_trend'] == 'Bearish'])
            
            if 'trend_direction' in technical_df.columns:
                summary['uptrend_count'] = len(technical_df[technical_df['trend_direction'] == 'Uptrend'])
                summary['downtrend_count'] = len(technical_df[technical_df['trend_direction'] == 'Downtrend'])
                summary['sideways_count'] = len(technical_df[technical_df['trend_direction'] == 'Sideways'])
        
        return pd.DataFrame([summary])
    
    def query_predictions(self, start_date=None, end_date=None, ticker=None, min_confidence=None):
        """Query predictions from database"""
        query = f"SELECT * FROM {self.predictions_table} WHERE 1=1"
        
        if start_date:
            query += f" AND trading_date >= '{start_date}'"
        if end_date:
            query += f" AND trading_date <= '{end_date}'"
        if ticker:
            query += f" AND ticker = '{ticker}'"
        if min_confidence:
            query += f" AND confidence >= {min_confidence}"
        
        query += " ORDER BY run_timestamp DESC, confidence DESC"
        
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None
    
    def get_latest_run_summary(self):
        """Get summary of latest prediction run"""
        query = f"""
        SELECT TOP 1 * 
        FROM {self.summary_table} 
        ORDER BY run_timestamp DESC
        """
        
        try:
            df = pd.read_sql(query, self.engine)
            if not df.empty:
                print("\n[SUMMARY] Latest Run Statistics:")
                print("=" * 60)
                for col in df.columns:
                    if col not in ['summary_id', 'created_at']:
                        print(f"{col}: {df[col].iloc[0]}")
                print("=" * 60)
            return df
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return None


def main():
    """Main CLI interface for database export"""
    parser = argparse.ArgumentParser(description='Export Trading Signals to SQL Server Database')
    parser.add_argument('--create-tables', action='store_true', help='Create database tables (first time setup)')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--batch', action='store_true', help='Export all predictions')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--query', action='store_true', help='Query existing predictions')
    parser.add_argument('--summary', action='store_true', help='Show latest run summary')
    parser.add_argument('--start-date', type=str, help='Start date for query (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for query (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = DatabaseExporter()
    
    try:
        if args.create_tables:
            exporter.create_tables()
            return 0
        
        if args.summary:
            exporter.get_latest_run_summary()
            return 0
        
        if args.query:
            results = exporter.query_predictions(
                start_date=args.start_date,
                end_date=args.end_date,
                ticker=args.ticker,
                min_confidence=args.confidence
            )
            if results is not None:
                print(f"\n[RESULTS] Found {len(results)} predictions")
                print(results.head(10))
            return 0
        
        # Default: Export to database
        success = exporter.export_predictions_to_db(
            ticker=args.ticker,
            confidence_threshold=args.confidence
        )
        
        if success:
            exporter.get_latest_run_summary()
            return 0
        else:
            return 1
        
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
