#!/usr/bin/env python3
"""
Optimized Weekly Model Retraining Script

Streamlined weekly retraining process optimized for speed and efficiency.
Designed to complete in minutes instead of hours.

Usage:
    python weekly_retrain_optimized.py              # Full optimized weekly retrain
    python weekly_retrain_optimized.py --no-backup  # Skip backup for faster execution
    python weekly_retrain_optimized.py --model gradient_boosting  # Train specific model only
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# ML imports - only what we need
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

class OptimizedWeeklyRetrainer:
    """Streamlined weekly model retraining - optimized for speed"""
    
    def __init__(self, backup_old=True, model_type='gradient_boosting'):
        self.backup_old = backup_old
        self.model_type = model_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Paths
        self.data_dir = Path('data')
        self.backup_dir = Path('data/backups') if backup_old else None
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(exist_ok=True)
        
        print(f"[INIT] Optimized Weekly Retrainer initialized")
        print(f"  Model: {model_type}")
        print(f"  Backup: {'Enabled' if backup_old else 'Disabled'}")
    
    def backup_existing_model(self):
        """Quick backup of critical model files only"""
        if not self.backup_old:
            return
        
        print("[BACKUP] Backing up existing model...")
        
        # Only backup essential files
        files_to_backup = [
            f'best_model_{self.model_type}.joblib',
            'scaler.joblib',
            'target_encoder.joblib'
        ]
        
        backup_count = 0
        for file_name in files_to_backup:
            src_path = self.data_dir / file_name
            if src_path.exists():
                backup_path = self.backup_dir / f"{self.timestamp}_{file_name}"
                shutil.copy2(src_path, backup_path)
                backup_count += 1
        
        print(f"[BACKUP] {backup_count} files backed up")
    
    def load_training_data(self):
        """Load training data with optimized query"""
        print("[DATA] Loading training data...")
        
        # Optimized query - only essential columns, balanced date range
        query = """
        SELECT 
            h.trading_date,
            h.ticker,
            CAST(h.open_price AS FLOAT) as open_price,
            CAST(h.high_price AS FLOAT) as high_price,
            CAST(h.low_price AS FLOAT) as low_price,
            CAST(h.close_price AS FLOAT) as close_price,
            CAST(h.volume AS BIGINT) as volume,
            r.RSI,
            r.rsi_trade_signal
        FROM dbo.nasdaq_100_hist_data h
        INNER JOIN dbo.nasdaq_100_rsi_signals r 
            ON h.ticker = r.ticker AND h.trading_date = r.trading_date
        WHERE h.trading_date >= '2024-01-01' 
          AND h.trading_date <= '2025-10-31'
          AND r.rsi_trade_signal IS NOT NULL
        ORDER BY h.trading_date DESC, h.ticker
        """
        
        try:
            df = self.db.execute_query(query)
            print(f"[DATA] Loaded {df.shape[0]:,} records from {df['trading_date'].min()} to {df['trading_date'].max()}")
            
            if df.empty:
                raise ValueError("No data found in database")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            raise
    
    def engineer_features_fast(self, df):
        """Optimized feature engineering - only proven high-impact features"""
        print("[FEATURES] Engineering features...")
        
        df_features = df.copy()
        
        # Basic calculated features
        df_features['daily_volatility'] = ((df_features['high_price'] - df_features['low_price']) / df_features['close_price']) * 100
        df_features['daily_return'] = ((df_features['close_price'] - df_features['open_price']) / df_features['open_price']) * 100
        df_features['volume_millions'] = df_features['volume'] / 1000000.0
        df_features['price_range'] = df_features['high_price'] - df_features['low_price']
        df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
        
        # Sort for time-series features
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        # RSI features
        df_features['rsi_oversold'] = (df_features['RSI'] < 30).astype(int)
        df_features['rsi_overbought'] = (df_features['RSI'] > 70).astype(int)
        df_features['rsi_momentum'] = df_features.groupby('ticker')['RSI'].diff()
        
        # Critical technical indicators (proven high-impact)
        print("[FEATURES] Adding technical indicators...")
        df_features = df_features.groupby('ticker', group_keys=False).apply(self._add_technical_indicators)
        
        # Time features
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
        df_features['month'] = df_features['trading_date'].dt.month
        
        # Handle NaN values efficiently
        df_features = df_features.bfill().fillna(0)
        
        print(f"[FEATURES] Complete - {df_features.shape[1]} features")
        return df_features
    
    def _add_technical_indicators(self, group_df):
        """Add only high-impact technical indicators"""
        df = group_df.copy()
        price_col = 'close_price'
        volume_col = 'volume'
        
        # Moving Averages (critical features)
        df['sma_20'] = df[price_col].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df[price_col].rolling(window=50, min_periods=1).mean()
        df['ema_20'] = df[price_col].ewm(span=20, min_periods=1).mean()
        
        # MACD (proven high-impact)
        ema_12 = df[price_col].ewm(span=12, min_periods=1).mean()
        ema_26 = df[price_col].ewm(span=26, min_periods=1).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price vs MA ratios (critical features)
        df['price_vs_sma20'] = df[price_col] / df['sma_20']
        df['price_vs_sma50'] = df[price_col] / df['sma_50']
        df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50']
        
        # Volume indicators
        df['volume_sma_20'] = df[volume_col].rolling(window=20, min_periods=1).mean()
        df['volume_sma_ratio'] = df[volume_col] / df['volume_sma_20']
        
        # Momentum
        df['price_momentum_10'] = df[price_col] / df[price_col].shift(10)
        
        # Volatility
        df['price_volatility_10'] = df[price_col].pct_change().rolling(window=10, min_periods=1).std()
        
        return df
    
    def prepare_ml_dataset(self, df_features):
        """Prepare dataset for ML training"""
        print("[PREP] Preparing ML dataset...")
        
        target_column = 'rsi_trade_signal'
        exclude_cols = ['trading_date', 'ticker', target_column]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols].copy()
        y = df_features[target_column].copy()
        
        # Filter valid targets
        valid_classes = ['Overbought (Sell)', 'Oversold (Buy)']
        valid_mask = y.isin(valid_classes)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) == 0:
            raise ValueError("No valid target data found")
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        print(f"[PREP] Dataset ready: {X.shape[0]:,} samples, {X.shape[1]} features")
        print(f"  Target classes: {list(target_encoder.classes_)}")
        
        return X, y_encoded, target_encoder, feature_cols
    
    def train_model_fast(self, X, y, feature_cols):
        """Train single best model (Gradient Boosting) - no cross-validation"""
        print(f"[TRAIN] Training {self.model_type} model...")
        
        # Time-aware split (80/20)
        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize best model only
        if self.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=6, 
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train
        print("[TRAIN] Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"[RESULTS] Model Performance:")
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        
        return {
            'model': model,
            'scaler': scaler,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'feature_columns': feature_cols
        }
    
    def save_model_artifacts(self, training_results, target_encoder):
        """Save trained model and artifacts"""
        print("[SAVE] Saving model artifacts...")
        
        # Save model
        model_path = self.data_dir / f'best_model_{self.model_type}.joblib'
        joblib.dump(training_results['model'], model_path)
        
        # Save preprocessing artifacts
        joblib.dump(training_results['scaler'], self.data_dir / 'scaler.joblib')
        joblib.dump(target_encoder, self.data_dir / 'target_encoder.joblib')
        
        print(f"[SAVE] Model saved to {model_path}")
    
    def run_optimized_retrain(self):
        """Execute optimized weekly retraining"""
        start_time = datetime.now()
        
        print("=" * 70)
        print(f"[START] Optimized Weekly Retraining - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        try:
            # Step 1: Backup
            if self.backup_old:
                self.backup_existing_model()
            
            # Step 2: Load data
            df = self.load_training_data()
            
            # Step 3: Engineer features
            df_features = self.engineer_features_fast(df)
            
            # Step 4: Prepare dataset
            X, y_encoded, target_encoder, feature_cols = self.prepare_ml_dataset(df_features)
            
            # Step 5: Train model
            training_results = self.train_model_fast(X, y_encoded, feature_cols)
            
            # Step 6: Save artifacts
            self.save_model_artifacts(training_results, target_encoder)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            print("=" * 70)
            print("[SUCCESS] WEEKLY RETRAINING COMPLETE!")
            print(f"  Duration: {duration:.1f} minutes")
            print(f"  F1-Score: {training_results['metrics']['f1_score']:.3f}")
            print(f"  Model: {self.model_type}")
            print("=" * 70)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='Optimized Weekly Model Retraining')
    parser.add_argument('--no-backup', action='store_true', 
                       help='Skip backup for faster execution')
    parser.add_argument('--model', type=str, default='gradient_boosting',
                       choices=['gradient_boosting'],
                       help='Model type to train (default: gradient_boosting)')
    
    args = parser.parse_args()
    
    # Initialize retrainer
    retrainer = OptimizedWeeklyRetrainer(
        backup_old=not args.no_backup,
        model_type=args.model
    )
    
    # Run retraining
    success = retrainer.run_optimized_retrain()
    
    if success:
        print("\n[NEXT STEPS]")
        print("  1. Test model: python predict_trading_signals.py --batch")
        print("  2. Export results: python export_results.py")
    else:
        print("\n[ERROR] Retraining failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
