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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purge gap to prevent data leakage.
    
    Unlike standard TimeSeriesSplit, this adds a gap between train and validation
    sets to prevent information from the future leaking into training data.
    With 5-day prediction targets, we need at least 5 samples gap to prevent
    the target (which uses future prices) from leaking between folds.
    """
    
    def __init__(self, n_splits=5, purge_gap=5):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        min_train_size = max(50, n_samples // (self.n_splits + 2))
        fold_size = max(20, (n_samples - min_train_size) // self.n_splits)
        
        for i in range(self.n_splits):
            train_end = min_train_size + i * fold_size
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)
            
            if val_start >= n_samples or val_end <= val_start:
                continue
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

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
            '⚙': '[CONFIG]',
            '🧮': '[CALC]',
            '🤖': '[ML]'
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
        
        # Sector encoder (fitted during feature engineering, saved with model)
        self.sector_encoder = None
        
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
            
            # Load fundamentals separately and merge (avoids slow SQL JOINs)
            df = self._merge_fundamentals(df)
            
            # Load market context (VIX, indices, sector ETFs, treasury) and merge
            df = self._merge_market_context(df)
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            raise
    
    def _merge_fundamentals(self, df):
        """Load fundamentals & sector data separately and merge into main DataFrame"""
        print("[DATA] Loading fundamental data...")
        
        try:
            # Load latest fundamentals per ticker
            fund_query = """
            SELECT f1.ticker, f1.beta, f1.forward_pe, f1.trailing_pe,
                   f1.profit_margin, f1.revenue_growth, f1.earnings_growth,
                   f1.debt_to_equity, f1.return_on_equity, f1.current_ratio,
                   f1.dividend_yield, f1.fifty_two_week_high, f1.fifty_two_week_low,
                   f1.two_hundred_day_avg
            FROM dbo.nasdaq_100_fundamentals f1
            INNER JOIN (
                SELECT ticker, MAX(fetch_date) as max_date
                FROM dbo.nasdaq_100_fundamentals
                GROUP BY ticker
            ) f2 ON f1.ticker = f2.ticker AND f1.fetch_date = f2.max_date
            """
            df_fund = self.db.execute_query(fund_query)
            print(f"  Fundamentals loaded: {len(df_fund)} tickers")
            
            # Merge fundamentals on ticker
            if not df_fund.empty:
                df = df.merge(df_fund, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load fundamentals: {e}")
        
        try:
            # Load sector data
            sector_query = "SELECT ticker, sector FROM dbo.nasdaq_top100"
            df_sector = self.db.execute_query(sector_query)
            print(f"  Sectors loaded: {len(df_sector)} tickers")
            
            # Merge sector on ticker
            if not df_sector.empty:
                df = df.merge(df_sector, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load sector data: {e}")
        
        return df
    
    def _merge_market_context(self, df):
        """Load market context data (VIX, indices, sector ETFs, treasury) and merge on trading_date.
        
        Adds market regime and sector rotation features from the shared market_context_daily table.
        Each row broadcasts to all tickers for the same date (market-wide features).
        """
        print("[DATA] Loading market context data...")
        
        try:
            context_query = """
            SELECT trading_date,
                   vix_close, vix_change_pct,
                   sp500_close, sp500_return_1d,
                   nasdaq_comp_close, nasdaq_comp_return_1d,
                   dxy_close, dxy_return_1d,
                   us_10y_yield_close, us_10y_yield_change,
                   xlk_return_1d, xlf_return_1d, xle_return_1d,
                   xlv_return_1d, xli_return_1d, xlc_return_1d,
                   xly_return_1d, xlp_return_1d, xlb_return_1d,
                   xlre_return_1d, xlu_return_1d
            FROM dbo.market_context_daily
            ORDER BY trading_date
            """
            df_context = self.db.execute_query(context_query)
            
            if not df_context.empty:
                # Ensure date types match for merge
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_context['trading_date'] = pd.to_datetime(df_context['trading_date'])
                
                df = df.merge(df_context, on='trading_date', how='left')
                matched = df['vix_close'].notna().sum()
                print(f"  Market context merged: {len(df_context)} dates, {matched} matched rows")
            else:
                print("  [WARN] No market context data found — run get_market_context_daily.py --backfill")
        except Exception as e:
            print(f"  [WARN] Could not load market context: {e}")
        
        return df
    
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
            safe_print("Skipping detailed EDA (quick mode)")
            return {'data_shape': df.shape, 'target_column': 'direction_5d', 'target_exists': True}
        
        safe_print("Performing exploratory data analysis...")
        
        # Basic statistics
        print(f"  Data shape: {df.shape}")
        print(f"  Date range: {df['trading_date'].min()} to {df['trading_date'].max()}")
        print(f"  Unique tickers: {df['ticker'].nunique()}")
        
        # Target analysis (5-day direction will be created in prepare_ml_dataset)
        target_column = 'rsi_trade_signal'
        if target_column in df.columns:
            target_dist = df[target_column].value_counts()
            print(f"  RSI signal distribution (reference only - not used as target):")
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
        # Normalized gap as percentage of close price (ticker-independent)
        df_features['gap_pct'] = np.where(
            df_features['close_price'] > 0,
            df_features['gap'] / df_features['close_price'] * 100,
            0
        )
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
        
        # ================================================================
        # PHASE 3: Fundamental features from nasdaq_100_fundamentals
        # ================================================================
        print("[DATA] Adding fundamental features...")
        
        # Computed price vs key fundamental levels (relative/normalized)
        if 'fifty_two_week_high' in df_features.columns:
            df_features['price_vs_52wk_high'] = np.where(
                df_features['fifty_two_week_high'] > 0,
                df_features['close_price'] / df_features['fifty_two_week_high'],
                0
            )
        if 'fifty_two_week_low' in df_features.columns:
            df_features['price_vs_52wk_low'] = np.where(
                df_features['fifty_two_week_low'] > 0,
                df_features['close_price'] / df_features['fifty_two_week_low'],
                0
            )
        if 'two_hundred_day_avg' in df_features.columns:
            df_features['price_vs_200d_avg'] = np.where(
                df_features['two_hundred_day_avg'] > 0,
                df_features['close_price'] / df_features['two_hundred_day_avg'],
                0
            )
        
        # Sector encoding (label encoded - works well with tree-based models)
        if 'sector' in df_features.columns:
            self.sector_encoder = LabelEncoder()
            df_features['sector_encoded'] = self.sector_encoder.fit_transform(
                df_features['sector'].fillna('Unknown')
            )
            print(f"  Sectors found: {len(self.sector_encoder.classes_)} unique")
        else:
            df_features['sector_encoded'] = 0
        
        # ================================================================
        # PHASE 4: Market context — sector-specific ETF return mapping
        # ================================================================
        # Map each stock's sector to its corresponding sector ETF daily return
        SECTOR_TO_ETF = {
            'Technology': 'xlk_return_1d',
            'Financial Services': 'xlf_return_1d',
            'Energy': 'xle_return_1d',
            'Healthcare': 'xlv_return_1d',
            'Industrials': 'xli_return_1d',
            'Communication Services': 'xlc_return_1d',
            'Consumer Cyclical': 'xly_return_1d',
            'Consumer Defensive': 'xlp_return_1d',
            'Basic Materials': 'xlb_return_1d',
            'Real Estate': 'xlre_return_1d',
            'Utilities': 'xlu_return_1d',
        }
        if 'sector' in df_features.columns and 'xlk_return_1d' in df_features.columns:
            df_features['sector_etf_return_1d'] = df_features['sector'].map(
                lambda s: SECTOR_TO_ETF.get(s, 'sp500_return_1d')  # fallback to S&P 500
            )
            # Resolve column names to actual values
            df_features['sector_etf_return_1d'] = df_features.apply(
                lambda row: row.get(row['sector_etf_return_1d'], 0) if pd.notna(row.get('sector_etf_return_1d')) else 0,
                axis=1
            )
            print(f"  Sector ETF return mapped for {df_features['sector_etf_return_1d'].notna().sum()} rows")
        
        # Handle NaN values - use FORWARD fill only (bfill causes data leakage in time series)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
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
        high_col = 'high_price'
        low_col = 'low_price'
        
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
        
        # ================================================================
        # NEW PHASE 2: Additional technical indicators for improved accuracy
        # ================================================================
        
        # --- Bollinger Bands (20-period, 2 std) ---
        bb_sma = df[price_col].rolling(window=20).mean()
        bb_std = df[price_col].rolling(window=20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        # %B: where price sits within the bands (0=lower, 1=upper)
        df['bollinger_pctb'] = np.where(bb_range > 0, (df[price_col] - bb_lower) / bb_range, 0.5)
        # Bandwidth: width of bands relative to SMA (volatility measure)
        df['bollinger_bandwidth'] = np.where(bb_sma > 0, bb_range / bb_sma, 0)
        
        # --- Stochastic Oscillator (14-period) ---
        low_14 = df[low_col].rolling(window=14).min()
        high_14 = df[high_col].rolling(window=14).max()
        stoch_range = high_14 - low_14
        df['stochastic_k'] = np.where(stoch_range > 0, (df[price_col] - low_14) / stoch_range * 100, 50)
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
        
        # --- ATR (Average True Range, 14-period) ---
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - df[price_col].shift(1)).abs()
        low_close = (df[low_col] - df[price_col].shift(1)).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = pd.Series(true_range, index=df.index).rolling(window=14).mean()
        # Normalized ATR as ratio of price (ticker-independent)
        df['atr_ratio'] = np.where(df[price_col] > 0, df['atr_14'] / df[price_col], 0)
        
        # --- Normalized MACD (percentage of price, ticker-independent) ---
        df['macd_normalized'] = np.where(df[price_col] > 0, df['macd'] / df[price_col] * 100, 0)
        df['macd_signal_normalized'] = np.where(df[price_col] > 0, df['macd_signal'] / df[price_col] * 100, 0)
        df['macd_histogram_normalized'] = np.where(df[price_col] > 0, df['macd_histogram'] / df[price_col] * 100, 0)
        
        # --- Lagged returns (percentage changes at different lookback periods) ---
        df['return_1d'] = df[price_col].pct_change(1)
        df['return_2d'] = df[price_col].pct_change(2)
        df['return_3d'] = df[price_col].pct_change(3)
        df['return_5d'] = df[price_col].pct_change(5)
        df['return_10d'] = df[price_col].pct_change(10)
        
        # --- RSI-Price divergence (RSI direction differs from price direction) ---
        if 'RSI' in df.columns:
            price_dir_5 = np.sign(df[price_col].pct_change(5))
            rsi_dir_5 = np.sign(df['RSI'].diff(5))
            df['rsi_price_divergence'] = (price_dir_5 != rsi_dir_5).astype(int)
        
        # ================================================================
        # PHASE 3: Market Regime Detection
        # Helps the model recognize trending vs mean-reverting environments
        # and suppress signals during volatile regime transitions.
        # ================================================================
        
        # --- Regime: SMA Trend Direction (20-day slope normalized) ---
        # Positive = uptrend, negative = downtrend, near-zero = sideways
        sma20 = df[price_col].rolling(window=20).mean()
        df['regime_sma20_slope'] = sma20.pct_change(5) * 100  # 5-day slope of 20-SMA
        
        # --- Regime: ADX-like trend strength (simplified) ---
        # Based on directional movement: |up moves| vs |down moves| over 14 days
        up_move = df[high_col].diff()
        down_move = -df[low_col].diff()
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        pos_dm_smooth = pd.Series(pos_dm, index=df.index).rolling(14).mean()
        neg_dm_smooth = pd.Series(neg_dm, index=df.index).rolling(14).mean()
        dm_sum = pos_dm_smooth + neg_dm_smooth
        # DX = |+DI - -DI| / (+DI + -DI), then smooth for ADX
        dx = np.where(dm_sum > 0, np.abs(pos_dm_smooth - neg_dm_smooth) / dm_sum * 100, 0)
        df['regime_adx'] = pd.Series(dx, index=df.index).rolling(14).mean()
        
        # --- Regime: Volatility regime (current vol vs long-term vol) ---
        # >1 = high-vol regime, <1 = low-vol regime
        vol_short = df[price_col].pct_change().rolling(10).std()
        vol_long = df[price_col].pct_change().rolling(60).std()
        df['regime_vol_ratio'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)
        
        # --- Regime: Mean reversion indicator (distance from 50-SMA / ATR) ---
        # Large positive = overbought relative to trend, large negative = oversold
        sma50 = df[price_col].rolling(window=50).mean()
        if 'atr_14' in df.columns and df['atr_14'].notna().any():
            df['regime_mean_reversion'] = np.where(
                df['atr_14'] > 0,
                (df[price_col] - sma50) / df['atr_14'],
                0
            )
        else:
            df['regime_mean_reversion'] = 0
        
        # --- Regime: Trend consistency (% of last 20 days that moved in same direction as overall) ---
        overall_dir = np.sign(df[price_col].diff(20))
        daily_dirs = np.sign(df[price_col].diff(1))
        consistent = (daily_dirs == overall_dir).astype(float)
        df['regime_trend_consistency'] = consistent.rolling(20).mean()
        
        return df
    
    def prepare_ml_dataset(self, df_features):
        """Prepare dataset for machine learning with 5-day direction target.
        
        Previously used rsi_trade_signal (RSI-based) which only fires at extremes
        and achieved ~50% accuracy (coin flip). Now uses 5-day forward price
        direction which:
        - Captures meaningful trends with less noise than 1-day
        - Uses all data points (not just RSI extremes)
        - Maps to actionable Buy/Sell signals
        """
        safe_print("Preparing ML dataset (5-day direction target)...")
        
        # --- Create 5-day forward direction target from price data ---
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        df_features['next_5d_close'] = df_features.groupby('ticker')['close_price'].shift(-5)
        df_features['next_5d_return'] = (
            (df_features['next_5d_close'] - df_features['close_price'])
            / df_features['close_price'] * 100
        )
        # Buy = price goes up over 5 days, Sell = price goes down
        df_features['direction_5d'] = np.where(
            df_features['next_5d_return'] > 0,
            'Oversold (Buy)',   # Keep legacy label format for downstream compatibility
            'Overbought (Sell)'
        )
        
        # Report target distribution
        valid_5d = df_features['next_5d_close'].notna()
        dist = df_features.loc[valid_5d, 'direction_5d'].value_counts()
        print(f"  5-day direction distribution:")
        for signal, count in dist.items():
            pct = (count / valid_5d.sum()) * 100
            print(f"    {signal}: {count:,} ({pct:.1f}%)")
        
        # Explicit feature list - only relative/normalized features (no raw prices)
        # MUST match predict_trading_signals.py feature_columns exactly (same order)
        feature_cols = [
            # Technical indicators (Phase 1 + 2)
            'RSI', 'daily_volatility', 'daily_return',
            'price_position', 'gap_pct',
            'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20',
            'sma20_vs_sma50', 'ema20_vs_ema50', 'sma5_vs_sma20',
            'volume_sma_ratio',
            'price_momentum_5', 'price_momentum_10',
            'price_volatility_10', 'price_volatility_20',
            'trend_strength_10',
            'day_of_week', 'month',
            'bollinger_pctb', 'bollinger_bandwidth',
            'stochastic_k', 'stochastic_d',
            'atr_ratio',
            'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
            'return_1d', 'return_2d', 'return_3d', 'return_5d', 'return_10d',
            'rsi_price_divergence',
            # Phase 3: Fundamental features
            'beta', 'forward_pe', 'trailing_pe',
            'profit_margin', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'return_on_equity', 'current_ratio',
            'dividend_yield',
            'price_vs_52wk_high', 'price_vs_52wk_low', 'price_vs_200d_avg',
            'sector_encoded',
            # Phase 3: Market regime features
            'regime_sma20_slope', 'regime_adx', 'regime_vol_ratio',
            'regime_mean_reversion', 'regime_trend_consistency',
            # Phase 4: Market context features (from market_context_daily)
            'vix_close', 'vix_change_pct',
            'sp500_return_1d', 'nasdaq_comp_return_1d',
            'dxy_return_1d', 'us_10y_yield_close', 'us_10y_yield_change',
            'sector_etf_return_1d',  # Mapped from sector → ETF (computed below)
        ]
        # Filter to only features that exist in the DataFrame
        feature_cols = [col for col in feature_cols if col in df_features.columns]
        
        X = df_features[feature_cols].copy()
        y = df_features['direction_5d'].copy()
        
        # Remove rows without 5-day target (last 5 rows per ticker)
        valid_mask = y.notna() & df_features['next_5d_close'].notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Remove rows with any NaN features
        feature_valid = X.notna().all(axis=1)
        X = X[feature_valid]
        y = y[feature_valid]
        
        if len(y) == 0:
            raise ValueError("No valid target data found after filtering")
        
        # Encode target (labels: 'Overbought (Sell)' and 'Oversold (Buy)')
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # --- Feature selection: keep top features by tree-based importance ---
        # Fundamentals (beta, PE ratios, margins) add noise without enough
        # time-series variation. Let the model tell us what matters.
        print(f"  All candidate features: {X.shape[1]}")
        
        from sklearn.ensemble import RandomForestClassifier as _RFC
        selector_model = _RFC(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        # Quick fit on a sample for speed
        sample_n = min(20000, len(X))
        sample_idx = np.random.RandomState(42).choice(len(X), sample_n, replace=False)
        selector_model.fit(
            X.iloc[sample_idx].fillna(0).values,
            y_encoded[sample_idx]
        )
        importances = pd.Series(selector_model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        
        # Keep top 20 features (sweet spot: enough signal, less noise)
        top_n = 20
        selected_features = importances.head(top_n).index.tolist()
        dropped_features = importances.tail(len(importances) - top_n).index.tolist()
        
        print(f"  Top {top_n} features selected by importance:")
        for i, (feat, imp) in enumerate(importances.head(top_n).items(), 1):
            print(f"    {i:2d}. {feat}: {imp:.4f}")
        print(f"  Dropped {len(dropped_features)} low-importance features: {dropped_features[:5]}...")
        
        # Filter to selected features
        X = X[selected_features]
        feature_cols = selected_features
        
        # Save selected feature list for prediction script
        import json
        feature_list_path = 'data/selected_features.json'
        with open(feature_list_path, 'w') as f:
            json.dump(selected_features, f, indent=2)
        print(f"  Saved feature list to {feature_list_path}")
        
        print(f"[SUCCESS] ML dataset prepared (5-day horizon):")
        print(f"  Features: {X.shape}")
        print(f"  Target classes: {list(target_encoder.classes_)}")
        print(f"  Valid samples: {len(X):,}")
        
        return X, y_encoded, target_encoder, feature_cols
    
    def train_models(self, X, y, feature_cols):
        """Train and compare ML models"""
        safe_print("🤖 Training machine learning models...")
        
        # Time-aware 3-way split: train (60%) / calibration (20%) / test (20%)
        # Calibrating on a separate set prevents overfitting the probability estimates
        df_temp = pd.DataFrame({'y': y}, index=X.index)
        date_sorted_idx = X.index.to_series().sort_values().index
        X_sorted = X.loc[date_sorted_idx]
        y_sorted = df_temp.loc[date_sorted_idx, 'y'].values  # Use pandas label indexing
        
        train_end = int(0.60 * len(X_sorted))
        cal_end = int(0.80 * len(X_sorted))
        X_train = X_sorted.iloc[:train_end]
        X_cal = X_sorted.iloc[train_end:cal_end]
        X_test = X_sorted.iloc[cal_end:]
        y_train = y_sorted[:train_end]
        y_cal = y_sorted[train_end:cal_end]
        y_test = y_sorted[cal_end:]
        
        print(f"  Split: Train={len(X_train):,}, Calibration={len(X_cal):,}, Test={len(X_test):,}")
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal_scaled = scaler.transform(X_cal)
        X_test_scaled = scaler.transform(X_test)
        
        # Compute sample weights: class balancing * time-based recency
        # Recent data is more relevant for prediction (market regimes change)
        class_weights = compute_sample_weight('balanced', y_train)
        
        # Exponential time weights: most recent sample = 1.0, oldest ~ 0.3
        # decay_rate controls how fast old data loses importance
        n_train = len(y_train)
        time_positions = np.arange(n_train) / n_train  # 0 to ~1 (oldest to newest)
        decay_rate = 1.2  # Higher = more emphasis on recent data
        time_weights = np.exp(decay_rate * (time_positions - 1))  # Ranges from ~0.3 to 1.0
        
        # Combine: class balance * time recency
        sample_weights = class_weights * time_weights
        sample_weights = sample_weights / sample_weights.mean()  # Normalize to mean=1
        
        print(f"  Time-weighted training: oldest weight={time_weights[0]:.3f}, "
              f"newest={time_weights[-1]:.3f}, ratio={time_weights[-1]/time_weights[0]:.1f}x")
        
        # Initialize models with better class balancing and calibration
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced_subsample',  # More aggressive balancing per tree
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                min_samples_split=5, subsample=0.8,  # Regularization to reduce overfitting
                random_state=42
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
        # Purged CV: 5-sample gap prevents 5-day target leakage between folds
        cv_splitter = PurgedTimeSeriesSplit(n_splits=5, purge_gap=5)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            try:
                # Train with time-weighted sample weights for all models
                # (class balance + recency emphasis)
                try:
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
                except TypeError:
                    # Fallback if model doesn't support sample_weight
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
        
        # Calibrate probabilities using dedicated calibration set (NOT test set)
        # Using isotonic regression which is more flexible than sigmoid (Platt scaling)
        # and works better when we have enough calibration data (20% of dataset)
        print("[CONFIG] Calibrating model probabilities on held-out calibration set...")
        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=best_model, cv='prefit', method='isotonic'
            )
            calibrated_model.fit(X_cal_scaled, y_cal)
            
            # Verify calibration improved: check that high-prob predictions are more accurate
            cal_probs = calibrated_model.predict_proba(X_test_scaled)
            cal_preds = calibrated_model.predict(X_test_scaled)
            cal_accuracy = accuracy_score(y_test, cal_preds)
            uncal_accuracy = model_results[best_model_name]['accuracy']
            
            print(f"  Pre-calibration test accuracy:  {uncal_accuracy:.3f}")
            print(f"  Post-calibration test accuracy: {cal_accuracy:.3f}")
            
            # Check if calibration helps: high-confidence should be more accurate
            max_probs = cal_probs.max(axis=1)
            high_mask = max_probs >= 0.65
            if high_mask.sum() > 10:
                high_acc = accuracy_score(y_test[high_mask], cal_preds[high_mask])
                low_acc = accuracy_score(y_test[~high_mask], cal_preds[~high_mask]) if (~high_mask).sum() > 0 else 0
                print(f"  High-confidence (>=65%) accuracy: {high_acc:.3f} ({high_mask.sum()} samples)")
                print(f"  Low-confidence (<65%) accuracy:   {low_acc:.3f} ({(~high_mask).sum()} samples)")
                
                if high_acc > low_acc:
                    print("[SUCCESS] Calibration confirmed: high confidence = higher accuracy")
                else:
                    print("[WARN] Calibration check: high confidence NOT more accurate - review needed")
            
            best_model = calibrated_model
            print("[SUCCESS] Probability calibration applied successfully")
        except Exception as e:
            print(f"[WARN] Calibration skipped: {e}")
        
        safe_print(f"🏆 Best model: {best_model_name} (F1: {model_results[best_model_name]['f1_score']:.3f})")
        
        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'scaler': scaler,
            'X_train': X_train,
            'X_cal': X_cal,
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
        
        # Save sector encoder (for consistent encoding during prediction)
        if self.sector_encoder is not None:
            joblib.dump(self.sector_encoder, 'data/sector_encoder.joblib')
            print(f"  Sector encoder: data/sector_encoder.joblib")
        
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
