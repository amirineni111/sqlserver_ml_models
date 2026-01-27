"""
Model Retraining Script

This script automates the complete process of retraining the ML model with the latest data.
It performs all the steps from data loading to model deployment.

Usage:
    python retrain_model.py                    # Full retrain with latest data
    python retrain_model.py --quick            # Quick retrain (skip EDA)
    python retrain_model.py --backup-old       # Backup current model before retrain
    python retrain_model.py --compare-models   # Compare new vs old model performance
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from datetime import datetime, timedelta
import sys
import os
import shutil
import joblib
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

warnings.filterwarnings('ignore')

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace emoji with text equivalents for Windows console
        emoji_replacements = {
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📊': '[DATA]',
            '💡': '[TIP]',
            '🎯': '[TARGET]',
            '📈': '[PREDICTION]',
            '⚠️': '[WARN]',
            '🔍': '[INFO]',
            '📋': '[RESULTS]',
            '📁': '[FILE]',
            '🚀': '[START]',
            '🔄': '[PROCESSING]',
            '💾': '[SAVE]',
            '📦': '[BACKUP]',
            '🎉': '[COMPLETE]',
            '⏩': '[SKIP]',
            '🏆': '[BEST]',
            '📅': '[DATE]',
            '📉': '[DECLINE]',
            '⚙️': '[CONFIG]',
            '⚙': '[CONFIG]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

class ModelRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, backup_old=False, quick_mode=False):
        self.backup_old = backup_old
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Paths
        self.data_dir = Path('data')
        self.reports_dir = Path('reports')
        self.backup_dir = Path('data/backups') if backup_old else None
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(exist_ok=True)
    
    def backup_existing_model(self):
        """Backup existing model files before retraining"""
        if not self.backup_old:
            return
        
        safe_print("📦 Backing up existing model artifacts...")
        
        files_to_backup = [
            'best_model_gradient_boosting.joblib',
            'scaler.joblib',
            'target_encoder.joblib',
            'model_results.pkl',
            'exploration_results.pkl'
        ]
        
        backup_count = 0
        for file_name in files_to_backup:
            src_path = self.data_dir / file_name
            if src_path.exists():
                backup_path = self.backup_dir / f"{self.timestamp}_{file_name}"
                shutil.copy2(src_path, backup_path)
                backup_count += 1
                safe_print(f"  ✅ Backed up {file_name}")
        
        safe_print(f"📦 Backup complete: {backup_count} files backed up to {self.backup_dir}")
    
    def load_latest_data(self, days_back=None):
        """Load the latest data from SQL Server"""
        print("[DATA] Loading latest data from SQL Server...")
        
        # Determine date range
        if days_back:
            date_filter = f"WHERE h.trading_date >= DATEADD(day, -{days_back}, CAST(GETDATE() AS DATE))"
        else:
            # Load balanced training data - exclude recent heavily biased period (Nov 2025)
            # Nov 2025: 64.5% Buy vs 35.5% Sell - too skewed for good training
            date_filter = "WHERE h.trading_date >= '2024-01-01' AND h.trading_date <= '2025-10-31'"
        
        query = f"""
        SELECT 
            h.trading_date,
            h.ticker,
            h.company,
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
        {date_filter}
        ORDER BY h.trading_date DESC, h.ticker
        """
        
        try:
            df = self.db.execute_query(query)
            print(f"[SUCCESS] Data loaded: {df.shape[0]:,} records from {df['trading_date'].min()} to {df['trading_date'].max()}")
            
            # Check for new data
            if df.empty:
                raise ValueError("No data found in database")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            raise
    
    def compare_with_previous_data(self, new_df):
        """Compare new data with previously processed data"""
        try:
            with open('data/exploration_results.pkl', 'rb') as f:
                old_results = pickle.load(f)
            
            old_shape = old_results['data_shape']
            new_shape = new_df.shape
            
            safe_print(f"📈 Data comparison:")
            print(f"  Previous: {old_shape[0]:,} records")
            print(f"  Current:  {new_shape[0]:,} records")
            print(f"  New data: {new_shape[0] - old_shape[0]:,} records")
            
            if new_shape[0] <= old_shape[0]:
                safe_print("⚠️  Warning: No new data found. Consider checking data sources.")
            
        except FileNotFoundError:
            print("[DATA] No previous data found - performing fresh analysis")
    
    def perform_eda(self, df):
        """Perform exploratory data analysis"""
        if self.quick_mode:
            safe_print("⏩ Skipping detailed EDA (quick mode)")
            return {'data_shape': df.shape, 'target_column': 'rsi_trade_signal', 'target_exists': True}
        
        safe_print("🔍 Performing exploratory data analysis...")
        
        # Basic statistics
        print(f"  Data shape: {df.shape}")
        print(f"  Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        
        # Target analysis
        target_column = 'rsi_trade_signal'
        if target_column in df.columns:
            target_dist = df[target_column].value_counts()
            print(f"  Target distribution:")
            for signal, count in target_dist.items():
                pct = (count / len(df)) * 100
                print(f"    {signal}: {count:,} ({pct:.1f}%)")
        
        # Missing values
        missing_summary = df.isnull().sum()
        missing_data = missing_summary[missing_summary > 0].to_dict()
        
        if missing_data:
            print(f"  Missing values detected: {missing_data}")
        else:
            print("  [SUCCESS] No missing values detected")
        
        # Save exploration results
        exploration_results = {
            'data_shape': df.shape,
            'target_column': target_column,
            'target_exists': target_column in df.columns,
            'target_distribution': target_dist.to_dict() if target_column in df.columns else None,
            'missing_values_summary': missing_data,
            'date_range': (str(df['trading_date'].min()), str(df['trading_date'].max())),
            'unique_tickers': df['ticker'].nunique(),
            'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'timestamp': self.timestamp
        }
        
        with open('data/exploration_results.pkl', 'wb') as f:
            pickle.dump(exploration_results, f)
        
        print("[SUCCESS] EDA complete and results saved")
        return exploration_results
    
    def engineer_features(self, df):
        """Apply feature engineering"""
        safe_print("⚙️  Performing feature engineering...")
        
        df_features = df.copy()
        
        # Basic calculated features
        df_features['daily_volatility'] = ((df_features['high_price'] - df_features['low_price']) / df_features['close_price']) * 100
        df_features['daily_return'] = ((df_features['close_price'] - df_features['open_price']) / df_features['open_price']) * 100
        df_features['volume_millions'] = df_features['volume'] / 1000000.0
        
        # Additional features
        df_features['price_range'] = df_features['high_price'] - df_features['low_price']
        df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
        
        # Sort by ticker and date for proper time-series calculations
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        df_features['gap'] = df_features.groupby('ticker')['open_price'].diff() - df_features.groupby('ticker')['close_price'].shift(1)
        df_features['volume_price_trend'] = df_features['volume'] * df_features['daily_return']
        df_features['rsi_oversold'] = (df_features['RSI'] < 30).astype(int)
        df_features['rsi_overbought'] = (df_features['RSI'] > 70).astype(int)
        df_features['rsi_momentum'] = df_features.groupby('ticker')['RSI'].diff()
        
        # Enhanced technical indicators (proven to improve model accuracy)
        df_features = self.add_enhanced_features(df_features)
        
        # Time features
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
        df_features['month'] = df_features['trading_date'].dt.month
        
        # Handle NaN values
        df_features = df_features.fillna(method='bfill').fillna(0)
        
        print(f"[SUCCESS] Feature engineering complete: {df_features.shape[1]} total features")
        return df_features
    
    def add_enhanced_features(self, df):
        """Add enhanced technical indicators (MACD, SMA, EMA)"""
        safe_print("📈 Adding enhanced technical indicators...")
        df_copy = df.copy()
        
        # Apply enhanced feature engineering per ticker
        df_copy = df_copy.groupby('ticker').apply(self._calculate_technical_indicators).reset_index(drop=True)
        
        return df_copy
    
    def _calculate_technical_indicators(self, group_df):
        """Calculate technical indicators for a single ticker"""
        df = group_df.copy()
        
        # Use close_price column name (database naming convention)
        price_col = 'close_price'
        volume_col = 'volume'
        
        # Simple Moving Averages
        df['sma_5'] = df[price_col].rolling(window=5).mean()
        df['sma_10'] = df[price_col].rolling(window=10).mean()
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df[price_col].ewm(span=5).mean()
        df['ema_10'] = df[price_col].ewm(span=10).mean()
        df['ema_20'] = df[price_col].ewm(span=20).mean()
        df['ema_50'] = df[price_col].ewm(span=50).mean()
        
        # MACD Calculation
        ema_12 = df[price_col].ewm(span=12).mean()
        ema_26 = df[price_col].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price vs Moving Average ratios (proven high impact features)
        df['price_vs_sma20'] = df[price_col] / df['sma_20']
        df['price_vs_sma50'] = df[price_col] / df['sma_50']
        df['price_vs_ema20'] = df[price_col] / df['ema_20']
        
        # Moving Average relationships
        df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50']
        df['ema20_vs_ema50'] = df['ema_20'] / df['ema_50']
        df['sma5_vs_sma20'] = df['sma_5'] / df['sma_20']
        
        # Volume indicators
        df['volume_sma_20'] = df[volume_col].rolling(window=20).mean()
        df['volume_sma_ratio'] = df[volume_col] / df['volume_sma_20']
        
        # Price momentum features
        df['price_momentum_5'] = df[price_col] / df[price_col].shift(5)
        df['price_momentum_10'] = df[price_col] / df[price_col].shift(10)
        
        # Volatility features
        df['price_volatility_10'] = df[price_col].pct_change().rolling(window=10).std()
        df['price_volatility_20'] = df[price_col].pct_change().rolling(window=20).std()
        
        # Trend strength indicators
        df['trend_strength_10'] = df[price_col].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() != 0 else 0
        )
        
        return df
    
    def prepare_ml_dataset(self, df_features):
        """Prepare dataset for machine learning"""
        print("🧮 Preparing ML dataset...")
        
        target_column = 'rsi_trade_signal'
        exclude_cols = ['trading_date', 'ticker', 'company', target_column]
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[feature_cols].copy()
        y = df_features[target_column].copy()
        
        # Remove rows with missing target (handle both NaN and None values)
        valid_mask = y.notna() & (y != 'None') & (y.astype(str) != 'None')
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Convert any remaining None values to a default class if needed
        y = y.astype(str)
        
        # Remove any remaining problematic values
        valid_classes = ['Overbought (Sell)', 'Oversold (Buy)']
        class_mask = y.isin(valid_classes)
        X = X[class_mask]
        y = y[class_mask]
        
        if len(y) == 0:
            raise ValueError("No valid target data found after filtering")
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        print(f"[SUCCESS] ML dataset prepared:")
        print(f"  Features: {X.shape}")
        print(f"  Target classes: {list(target_encoder.classes_)}")
        print(f"  Valid samples: {len(X):,}")
        
        return X, y_encoded, target_encoder, feature_cols
    
    def train_models(self, X, y, feature_cols):
        """Train and compare ML models"""
        print("🤖 Training machine learning models...")
        
        # Time-aware split
        df_temp = pd.DataFrame({'y': y}, index=X.index)
        date_sorted_idx = X.index.to_series().sort_values().index
        X_sorted = X.loc[date_sorted_idx]
        y_sorted = y[date_sorted_idx]
        
        split_idx = int(0.8 * len(X_sorted))
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted[:split_idx]
        y_test = y_sorted[split_idx:]
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize models with better class balancing and calibration
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced_subsample',  # More aggressive balancing per tree
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000,
                C=0.1,  # Stronger regularization to prevent overfitting
                solver='liblinear'  # Better for small datasets
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, max_depth=10, 
                class_weight='balanced_subsample',  # More aggressive balancing
                random_state=42, n_jobs=-1
            )
        }
        
        # Train models
        model_results = {}
        trained_models = {}
        cv_splitter = TimeSeriesSplit(n_splits=3)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            try:
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=cv_splitter, scoring='accuracy')
                
                model_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                trained_models[model_name] = model
                
                print(f"    F1: {f1:.3f}, CV: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                
            except Exception as e:
                safe_print(f"    ❌ Error: {e}")
        
        # Find best model
        best_model_name = max(model_results.keys(), 
                             key=lambda k: model_results[k]['f1_score'])
        best_model = trained_models[best_model_name]
        
        safe_print(f"🏆 Best model: {best_model_name} (F1: {model_results[best_model_name]['f1_score']:.3f})")
        
        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_cols
        }
    
    def save_model_artifacts(self, training_results, target_encoder):
        """Save trained model and preprocessing artifacts"""
        safe_print("💾 Saving model artifacts...")
        
        # Save best model
        model_path = f"data/best_model_{training_results['best_model_name'].lower().replace(' ', '_')}.joblib"
        joblib.dump(training_results['best_model'], model_path)
        
        # Save preprocessing artifacts
        joblib.dump(training_results['scaler'], 'data/scaler.joblib')
        joblib.dump(target_encoder, 'data/target_encoder.joblib')
        
        # Save complete results
        results_to_save = {
            'model_results': training_results['model_results'],
            'best_model_name': training_results['best_model_name'],
            'feature_columns': training_results['feature_columns'],
            'training_timestamp': self.timestamp,
            'data_summary': {
                'train_samples': len(training_results['y_train']),
                'test_samples': len(training_results['y_test']),
                'features': len(training_results['feature_columns'])
            }
        }
        
        with open('data/model_results.pkl', 'wb') as f:
            pickle.dump(results_to_save, f)
        
        print("[SUCCESS] Model artifacts saved successfully:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: data/scaler.joblib")
        print(f"  Encoder: data/target_encoder.joblib")
        print(f"  Results: data/model_results.pkl")
    
    def compare_model_performance(self, training_results):
        """Compare new model with previous model if available"""
        try:
            # Try to load previous results
            with open('data/model_results.pkl', 'rb') as f:
                old_results = pickle.load(f)
            
            if 'training_timestamp' in old_results and old_results['training_timestamp'] != self.timestamp:
                old_best = old_results['best_model_name']
                old_f1 = old_results['model_results'][old_best]['f1_score']
                
                new_best = training_results['best_model_name']
                new_f1 = training_results['model_results'][new_best]['f1_score']
                
                safe_print(f"\n📈 Model Performance Comparison:")
                print(f"  Previous: {old_best} (F1: {old_f1:.3f})")
                print(f"  Current:  {new_best} (F1: {new_f1:.3f})")
                
                improvement = new_f1 - old_f1
                if improvement > 0:
                    print(f"  [IMPROVEMENT] Improvement: +{improvement:.3f} ({improvement/old_f1*100:.1f}%)")
                else:
                    safe_print(f"  📉 Decline: {improvement:.3f} ({improvement/old_f1*100:.1f}%)")
            
        except (FileNotFoundError, KeyError):
            print("[DATA] No previous model found for comparison")
    
    def run_full_retrain(self):
        """Execute complete retraining process"""
        print(f"[START] Starting model retraining - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Mode: {'Quick' if self.quick_mode else 'Full'}")
        print(f"   Backup: {'Yes' if self.backup_old else 'No'}")
        print("=" * 80)
        
        try:
            # Step 1: Backup existing model
            if self.backup_old:
                self.backup_existing_model()
            
            # Step 2: Load latest data
            df = self.load_latest_data()
            
            # Step 3: Compare with previous data
            self.compare_with_previous_data(df)
            
            # Step 4: Perform EDA
            eda_results = self.perform_eda(df)
            
            # Step 5: Feature engineering
            df_features = self.engineer_features(df)
            
            # Step 6: Prepare ML dataset
            X, y_encoded, target_encoder, feature_cols = self.prepare_ml_dataset(df_features)
            
            # Step 7: Train models
            training_results = self.train_models(X, y_encoded, feature_cols)
            
            # Step 8: Save artifacts
            self.save_model_artifacts(training_results, target_encoder)
            
            # Step 9: Compare performance
            self.compare_model_performance(training_results)
            
            print("=" * 80)
            print("[SUCCESS] RETRAINING COMPLETE!")
            safe_print(f"🏆 Best Model: {training_results['best_model_name']}")
            print(f"[DATA] F1-Score: {training_results['model_results'][training_results['best_model_name']]['f1_score']:.3f}")
            safe_print(f"📅 Timestamp: {self.timestamp}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Retraining failed: {e}")
            print("Check logs and data availability")
            return False

def main():
    parser = argparse.ArgumentParser(description='ML Model Retraining System')
    parser.add_argument('--quick', action='store_true', help='Quick retrain (skip detailed EDA)')
    parser.add_argument('--backup-old', action='store_true', help='Backup existing model files')
    parser.add_argument('--compare-models', action='store_true', help='Compare new vs old model performance')
    
    args = parser.parse_args()
    
    # Initialize retrainer
    retrainer = ModelRetrainer(backup_old=args.backup_old, quick_mode=args.quick)
    
    # Run retraining
    success = retrainer.run_full_retrain()
    
    if success:
        safe_print("\n🎯 Next Steps:")
        print("1. Test the new model: python predict_trading_signals.py --batch")
        print("2. Review performance: Check reports/ folder")
        print("3. Deploy if satisfied with results")
    else:
        print("\n[ERROR] Retraining failed. Please check the logs and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
