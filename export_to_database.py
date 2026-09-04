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

# Single source of truth for what counts as "high confidence". Before Sep 2026 this
# module hardcoded its own thresholds (0.5 for the flag, 0.67/0.55 for signal_strength,
# 0.7 for the summary count) while nasdaq_config said 0.58 — four disagreeing gates.
# The 0.5 default in particular made high_confidence fire on every row, because
# confidence is the probability of the *called* direction and is >= 0.5 by construction.
from nasdaq_config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD
from model_version import load_model_version


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
                is_actionable BIT,
                suppression_reason VARCHAR(100),
                model_version VARCHAR(50),
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
                actionable_predictions INT,
                suppressed_predictions INT,
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
            self._ensure_predictions_schema()
            print("[SUCCESS] All tables created successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to create tables: {e}")
            return False

    def _ensure_predictions_schema(self):
        """Add suppression-flag columns to a pre-existing predictions table.

        Suppressed signals are now written with is_actionable=0 instead of
        being dropped before export. model_version (Sep 2026) records which model
        produced the row, so drift and calibration can be attributed to a specific
        retrain instead of inferred from run_timestamp. Idempotent; safe to run
        every export.
        """
        migration_sql = f"""
        IF COL_LENGTH('{self.predictions_table}', 'is_actionable') IS NULL
            ALTER TABLE {self.predictions_table} ADD is_actionable BIT;
        IF COL_LENGTH('{self.predictions_table}', 'suppression_reason') IS NULL
            ALTER TABLE {self.predictions_table} ADD suppression_reason VARCHAR(100);
        IF COL_LENGTH('{self.predictions_table}', 'model_version') IS NULL
            ALTER TABLE {self.predictions_table} ADD model_version VARCHAR(50);
        IF COL_LENGTH('{self.summary_table}', 'actionable_predictions') IS NULL
            ALTER TABLE {self.summary_table} ADD actionable_predictions INT;
        IF COL_LENGTH('{self.summary_table}', 'suppressed_predictions') IS NULL
            ALTER TABLE {self.summary_table} ADD suppressed_predictions INT;
        """
        with self.engine.connect() as conn:
            conn.execute(text(migration_sql))
            conn.commit()


    def export_predictions_to_db(self, ticker=None, confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
                                 skip_on_holiday=True):
        """Export predictions to database.

        ``confidence_threshold`` is the gate for the high_confidence flag and MUST
        stay tied to HIGH_CONFIDENCE_THRESHOLD. It used to default to 0.5, which is
        below the floor of the confidence scale, so the flag fired on every row and
        collapsed into a duplicate of is_actionable (Sep 2026 fix).

        When ``skip_on_holiday`` is True (default), the export is skipped entirely if
        NASDAQ is closed today (market holiday or weekend) per dbo.market_calendar,
        so we never insert predictions for a non-trading day. Fails open if the
        calendar has no entry for today.
        """
        if confidence_threshold < MEDIUM_CONFIDENCE_THRESHOLD:
            print(f"[WARN] high-confidence gate {confidence_threshold} is below the "
                  f"medium threshold {MEDIUM_CONFIDENCE_THRESHOLD} — the flag will be "
                  f"near-meaningless. Intended value is {HIGH_CONFIDENCE_THRESHOLD}.")
        if skip_on_holiday:
            try:
                from market_calendar_check import get_nasdaq_calendar_status
                cal = get_nasdaq_calendar_status()
                if not cal.is_trading_day:
                    print(f"[SKIP] {cal.reason} — not inserting predictions for a non-trading day.")
                    return True  # clean no-op, not a failure
                print(f"[CALENDAR] {cal.reason}")
            except Exception as e:
                print(f"[WARN] Market-calendar check failed ({e}); proceeding as a trading day.")

        print("[PROCESSING] Generating predictions for database export...")

        # Guarantee the flag/versioning columns exist before the insert — the export
        # runs unattended from daily_automation.py and never calls create_tables().
        self._ensure_predictions_schema()

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
            self._ensure_predictions_schema()
            if 'is_actionable' in predictions_df.columns:
                actionable_count = int(predictions_df['is_actionable'].astype(bool).sum())
                print(f"[DATABASE] Inserting {len(predictions_df)} predictions "
                      f"({actionable_count} actionable, {len(predictions_df) - actionable_count} suppressed)...")
            else:
                print(f"[DATABASE] Inserting {len(predictions_df)} predictions...")
            predictions_df.to_sql(
                self.predictions_table,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=100
            )
            print(f"[SUCCESS] Predictions inserted into {self.predictions_table}")
            
            print(f"[DATABASE] Inserting {len(technical_df)} technical indicators...")
            technical_df.to_sql(
                self.technical_table,
                self.engine,
                if_exists='append',
                index=False,
                chunksize=100
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
            print(f"{len(predictions_df)} records inserted to database")
            
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
        # Bands are derived from nasdaq_config, not hardcoded (Sep 2026). The old
        # 0.67/0.55 bands were tuned for the pre-June isotonic model's wider spread
        # and disagreed with the high_confidence gate, producing rows that read
        # "high_confidence=True, signal_strength='Weak'". Invariant now:
        #   signal_strength == 'Strong'  <=>  high_confidence == 1  (on actionable rows)
        df['signal_strength'] = df['confidence'].apply(
            lambda x: 'Strong' if x > HIGH_CONFIDENCE_THRESHOLD
            else 'Moderate' if x >= MEDIUM_CONFIDENCE_THRESHOLD else 'Weak'
        )
        # Suppressed rows are stored for the outcomes feedback loop but are not
        # tradeable — label them so they can't be mistaken for strong signals
        if 'is_actionable' in df.columns:
            df.loc[~df['is_actionable'].astype(bool), 'signal_strength'] = 'Suppressed'
        df['rsi_category'] = df['RSI'].apply(
            lambda x: 'Oversold' if x < 30 else 'Overbought' if x > 70 else 'Neutral'
        )
        
        # Select columns for database
        # Model outputs Up/Down signals + up_probability/down_probability
        # Map to DB schema (Buy/Sell + buy_probability/sell_probability) for backward compatibility
        signal_map = {'Up': 'Buy', 'Down': 'Sell'}
        if 'predicted_signal' in df.columns:
            df['predicted_signal'] = df['predicted_signal'].map(signal_map).fillna(df['predicted_signal'])
        
        # Map probability columns to DB column names
        if 'up_probability' in df.columns:
            df['buy_probability'] = df['up_probability']
        if 'down_probability' in df.columns:
            df['sell_probability'] = df['down_probability']
        
        # Normalize suppression flags (bool for BIT, None stays NULL).
        # Suppressed rows must never read as high-confidence: downstream
        # consumers (dashboard, agentic AI) filter on high_confidence=1 and
        # don't know about is_actionable yet.
        if 'is_actionable' in df.columns:
            df['is_actionable'] = df['is_actionable'].astype(bool)
            df['high_confidence'] = df['high_confidence'].astype(bool) & df['is_actionable']

        # Stamp the producing model at write time. Attribution is then exact —
        # evaluation no longer has to infer it from run_timestamp, and
        # derive_thresholds.py can select current-model outcomes directly.
        _, version = load_model_version()
        if version is None:
            print("[WARN] data/training_metadata.pkl unreadable — writing NULL model_version")
        df['model_version'] = version

        columns = [
            'run_timestamp', 'trading_date', 'ticker', 'company',
            'predicted_signal', 'confidence', 'confidence_percentage', 'signal_strength',
            'close_price', 'RSI', 'rsi_category', 'high_confidence',
            'sell_probability', 'buy_probability',
            'is_actionable', 'suppression_reason', 'model_version'
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
            
            # Add trend direction analysis (price vs SMA50: stock price relative to its 50-day average)
            if 'price_vs_sma50' in feature_data_latest.columns:
                feature_data_latest['trend_direction'] = feature_data_latest['price_vs_sma50'].apply(
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
        """Generate summary statistics

        Signal counts use only actionable rows so the summary keeps its
        pre-flag semantics (suppressed rows are stored but not tradeable).
        """
        if 'is_actionable' in predictions_df.columns:
            actionable_df = predictions_df[predictions_df['is_actionable'].astype(bool)]
        else:
            actionable_df = predictions_df

        summary = {
            'run_timestamp': run_timestamp,
            'run_date': run_timestamp.date(),
            # buy_signals + sell_signals count ACTIONABLE rows while total_predictions
            # counts every row, so the two never added up and the table looked broken.
            # actionable/suppressed make the arithmetic explicit:
            #   total = actionable + suppressed,  actionable = buy_signals + sell_signals
            'total_predictions': len(predictions_df),
            'actionable_predictions': len(actionable_df),
            'suppressed_predictions': len(predictions_df) - len(actionable_df),
            # Counts must agree with the high_confidence bit on the rows themselves;
            # these used to hardcode 0.7/0.6 and disagreed with both the flag and
            # nasdaq_config (Sep 2026 fix).
            'high_confidence_count': int(actionable_df['high_confidence'].sum())
                if 'high_confidence' in actionable_df.columns
                else len(actionable_df[actionable_df['confidence'] > HIGH_CONFIDENCE_THRESHOLD]),
            'medium_confidence_count': len(actionable_df[
                (actionable_df['confidence'] >= MEDIUM_CONFIDENCE_THRESHOLD) &
                (actionable_df['confidence'] <= HIGH_CONFIDENCE_THRESHOLD)]),
            'buy_signals': len(actionable_df[actionable_df['predicted_signal'].str.contains('Buy|Up', na=False)]),
            'sell_signals': len(actionable_df[actionable_df['predicted_signal'].str.contains('Sell|Down', na=False)]),
            'avg_confidence': actionable_df['confidence'].mean() if not actionable_df.empty else None,
            'avg_rsi': actionable_df['RSI'].mean() if 'RSI' in actionable_df.columns and not actionable_df.empty else None
        }
        
        # Guardrail: a populated prediction set with ~0 actionable buys is the signature
        # of the Jun 2026 bug (gates drifted out of the model's confidence range). Warn
        # loudly so it can't silently persist for weeks again.
        total = summary['total_predictions']
        if total > 50 and summary['buy_signals'] == 0:
            print(f"[GUARDRAIL][WARN] 0 actionable BUY signals out of {total} predictions. "
                  f"Likely a confidence-threshold/calibration mismatch (see nasdaq_config.py / "
                  f"derive_thresholds.py). avg_confidence={summary['avg_confidence']}.")

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
    # --confidence used to mean two different things: the high_confidence gate on
    # export and a minimum-confidence filter on --query. Split into two flags so the
    # export gate can never be silently lowered again (it was 0.5, hence the flag
    # firing on 76% of all rows historically).
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Minimum confidence filter for --query only (default: 0.5)')
    parser.add_argument('--high-conf-threshold', type=float, default=HIGH_CONFIDENCE_THRESHOLD,
                        help=f'Gate for the high_confidence flag on export '
                             f'(default: {HIGH_CONFIDENCE_THRESHOLD} from nasdaq_config)')
    parser.add_argument('--query', action='store_true', help='Query existing predictions')
    parser.add_argument('--summary', action='store_true', help='Show latest run summary')
    parser.add_argument('--start-date', type=str, help='Start date for query (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for query (YYYY-MM-DD)')
    parser.add_argument('--ignore-holiday', action='store_true',
                        help='Insert even if NASDAQ is closed today (bypass market_calendar gate)')
    
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
            confidence_threshold=args.high_conf_threshold,
            skip_on_holiday=not args.ignore_holiday
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
