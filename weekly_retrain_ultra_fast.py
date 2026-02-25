#!/usr/bin/env python3
"""
ULTRA-FAST Weekly Model Retraining Script (Enhanced — NSE-parity)

Combines vectorized operations (speed) with NSE-quality model architecture:
- 5-day price direction target (primary) instead of RSI trade signal
- 4-model ensemble + VotingClassifier with probability calibration
- 3-way split: 60% train / 20% calibration / 20% test
- Time-weighted training (recent data weighted higher)
- Purged time-series cross-validation (5-day gap prevents leakage)
- Market context features (VIX, S&P 500, NASDAQ, DXY, sector ETFs, treasury)
- Calendar features (holidays, short weeks, options expiry)
- Enriched DB indicators (Bollinger, ATR, MACD, SMA signals from DB views)
- Market regime detection (ADX, vol ratio, mean reversion, trend consistency)
- Regression model for 5-day return magnitude
- Feature selection via mutual_info_classif (top-N saved to selected_features.json)

Usage:
    python weekly_retrain_ultra_fast.py              # Full enhanced retrain
    python weekly_retrain_ultra_fast.py --no-backup  # Skip backup for maximum speed
    python weekly_retrain_ultra_fast.py --days-back 365  # Custom training window
"""

import argparse
import json
import pickle
import pandas as pd
import numpy as np
import joblib
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

# ML imports
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier,
    GradientBoostingRegressor, RandomForestRegressor,
    ExtraTreesRegressor, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection


class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purge gap to prevent data leakage.
    With 5-day prediction targets, we need at least 5 samples gap.
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


class UltraFastWeeklyRetrainer:
    """Ultra-fast weekly retraining with NSE-quality model architecture"""

    def __init__(self, backup_old=True, days_back=730):
        self.backup_old = backup_old
        self.days_back = days_back
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Database connection
        self.db = SQLServerConnection()

        # Paths
        self.data_dir = Path('data')
        self.backup_dir = Path('data/backups') if backup_old else None

        # Sector encoder (fitted during feature engineering, saved with model)
        self.sector_encoder = None

        # Feature columns (set during training, saved for prediction)
        self.feature_columns = []

        # Sector → ETF mapping for NASDAQ
        self.SECTOR_TO_ETF = {
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

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(exist_ok=True)

        print(f"[INIT] Ultra-Fast Weekly Retrainer (Enhanced) initialized")
        print(f"  Training window: {days_back} days")
        print(f"  Backup: {'Enabled' if backup_old else 'Disabled'}")

    def backup_existing_model(self):
        """Quick backup of critical model files only"""
        if not self.backup_old:
            return

        print("[BACKUP] Backing up existing model...")

        files_to_backup = [
            'best_model_gradient_boosting.joblib',
            'scaler.joblib',
            'target_encoder.joblib',
            'sector_encoder.joblib',
            'selected_features.json',
        ]

        backup_count = 0
        for file_name in files_to_backup:
            src_path = self.data_dir / file_name
            if src_path.exists():
                backup_path = self.backup_dir / f"{self.timestamp}_{file_name}"
                shutil.copy2(src_path, backup_path)
                backup_count += 1

        print(f"[BACKUP] {backup_count} files backed up")

    # ================================================================
    # DATA LOADING
    # ================================================================

    def load_training_data(self):
        """Load training data with 12-month rolling window"""
        print("[DATA] Loading training data...")

        query = f"""
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
        WHERE h.trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
          AND h.trading_date <= CAST(GETDATE() AS DATE)
          AND ISNUMERIC(h.close_price) = 1
          AND CAST(h.close_price AS FLOAT) > 0
          AND CAST(h.volume AS BIGINT) > 0
        ORDER BY h.ticker, h.trading_date
        """

        try:
            df = self.db.execute_query(query)
            print(f"[DATA] Loaded {df.shape[0]:,} records, "
                  f"{df['ticker'].nunique()} tickers, "
                  f"{df['trading_date'].min()} to {df['trading_date'].max()}")

            if df.empty:
                raise ValueError("No data found in database")

            # Merge enriched data sources
            df = self._merge_fundamentals(df)
            df = self._merge_enriched_db_features(df)
            df = self._merge_market_context(df)
            df = self._merge_calendar_features(df, market='NASDAQ')
            df = self._merge_sentiment(df)

            return df

        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
            raise

    def _merge_sentiment(self, df):
        """Load sector sentiment scores from nasdaq_sector_sentiment and merge.

        Adds 10 sentiment features per row:
        - sector_sentiment_score, sector_sentiment_confidence
        - sector_sentiment_momentum_3d, sector_sentiment_momentum_7d
        - sector_sentiment_vs_avg_30d
        - sector_positive_ratio, sector_negative_ratio
        - sector_news_volume (log-scaled news count)
        - market_sentiment_score
        - sentiment_price_divergence (computed later in engineer_features_vectorized)

        Merges on (trading_date, sector). Gracefully defaults to 0 if table missing.
        """
        print("[DATA] Loading sector sentiment data...")

        try:
            sent_query = """
            SELECT trading_date, sector,
                   sentiment_score, confidence,
                   sentiment_momentum_3d, sentiment_momentum_7d,
                   sentiment_vs_avg_30d,
                   positive_ratio, negative_ratio,
                   news_count,
                   market_sentiment_score
            FROM dbo.nasdaq_sector_sentiment
            ORDER BY trading_date
            """
            df_sent = self.db.execute_query(sent_query)

            if not df_sent.empty:
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_sent['trading_date'] = pd.to_datetime(df_sent['trading_date'])

                # Rename columns to avoid clashes
                df_sent = df_sent.rename(columns={
                    'sentiment_score': 'sector_sentiment_score',
                    'confidence': 'sector_sentiment_confidence',
                    'sentiment_momentum_3d': 'sector_sentiment_momentum_3d',
                    'sentiment_momentum_7d': 'sector_sentiment_momentum_7d',
                    'sentiment_vs_avg_30d': 'sector_sentiment_vs_avg_30d',
                    'positive_ratio': 'sector_positive_ratio',
                    'negative_ratio': 'sector_negative_ratio',
                    'news_count': 'sector_news_volume',
                })

                # Log-scale news volume
                df_sent['sector_news_volume'] = np.log1p(df_sent['sector_news_volume'])

                # Merge on (trading_date, sector)
                df = df.merge(df_sent, on=['trading_date', 'sector'], how='left')

                matched = df['sector_sentiment_score'].notna().sum()
                print(f"  Sentiment merged: {len(df_sent)} records, {matched} matched rows")
            else:
                print("  [WARN] No sentiment data found — run collect_sector_sentiment.py --backfill 30")
        except Exception as e:
            print(f"  [WARN] Could not load sentiment (table may not exist yet): {e}")

        # Ensure all sentiment columns exist with default 0
        for col in ['sector_sentiment_score', 'sector_sentiment_confidence',
                     'sector_sentiment_momentum_3d', 'sector_sentiment_momentum_7d',
                     'sector_sentiment_vs_avg_30d', 'sector_positive_ratio',
                     'sector_negative_ratio', 'sector_news_volume',
                     'market_sentiment_score']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)

        return df

    def _merge_fundamentals(self, df):
        """Load fundamentals & sector data separately and merge"""
        print("[DATA] Loading fundamental data...")

        try:
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
            if not df_fund.empty:
                df = df.merge(df_fund, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load fundamentals: {e}")

        try:
            sector_query = "SELECT ticker, sector FROM dbo.nasdaq_top100"
            df_sector = self.db.execute_query(sector_query)
            print(f"  Sectors loaded: {len(df_sector)} tickers")
            if not df_sector.empty:
                df = df.merge(df_sector, on='ticker', how='left')
        except Exception as e:
            print(f"  [WARN] Could not load sector data: {e}")

        return df

    def _merge_enriched_db_features(self, df):
        """Load enriched technical features from NASDAQ DB views and merge"""
        print("[DATA] Loading enriched features from DB views...")

        df['trading_date'] = pd.to_datetime(df['trading_date'])

        # Bollinger Bands
        try:
            bb_query = f"""
            SELECT ticker, trading_date,
                   CAST(Upper_Band AS FLOAT) as db_bb_upper,
                   CAST(Lower_Band AS FLOAT) as db_bb_lower,
                   CAST(SMA_20 AS FLOAT) as db_bb_sma20
            FROM dbo.nasdaq_100_bollingerband
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_bb = self.db.execute_query(bb_query)
            if not df_bb.empty:
                df_bb['trading_date'] = pd.to_datetime(df_bb['trading_date'])
                df_bb = df_bb.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_bb, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] Bollinger Bands: {len(df_bb):,} records")
        except Exception as e:
            print(f"  [WARN] Bollinger Bands load failed: {e}")

        # ATR
        try:
            atr_query = f"""
            SELECT ticker, trading_date,
                   CAST(ATR_14 AS FLOAT) as db_atr_14
            FROM dbo.nasdaq_100_atr
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_atr = self.db.execute_query(atr_query)
            if not df_atr.empty:
                df_atr['trading_date'] = pd.to_datetime(df_atr['trading_date'])
                df_atr = df_atr.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_atr, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] ATR: {len(df_atr):,} records")
        except Exception as e:
            print(f"  [WARN] ATR load failed: {e}")

        # MACD from DB
        try:
            macd_query = f"""
            SELECT ticker, trading_date,
                   CAST(MACD AS FLOAT) as db_macd,
                   CAST(Signal_Line AS FLOAT) as db_macd_signal
            FROM dbo.nasdaq_100_macd
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_macd = self.db.execute_query(macd_query)
            if not df_macd.empty:
                df_macd['trading_date'] = pd.to_datetime(df_macd['trading_date'])
                df_macd = df_macd.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_macd, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] MACD: {len(df_macd):,} records")
        except Exception as e:
            print(f"  [WARN] MACD load failed: {e}")

        # SMA Signals
        try:
            sma_query = f"""
            SELECT ticker, trading_date,
                   CAST(SMA_200 AS FLOAT) as sma_200,
                   CAST(SMA_100 AS FLOAT) as sma_100,
                   SMA_200_Flag, SMA_100_Flag, SMA_50_Flag, SMA_20_Flag
            FROM dbo.nasdaq_100_sma_signals
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_sma = self.db.execute_query(sma_query)
            if not df_sma.empty:
                df_sma['trading_date'] = pd.to_datetime(df_sma['trading_date'])
                df_sma = df_sma.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_sma, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] SMA Signals: {len(df_sma):,} records")
        except Exception as e:
            print(f"  [WARN] SMA Signals load failed: {e}")

        # MACD Signals (crossover)
        try:
            macd_sig_query = f"""
            SELECT ticker, trading_date,
                   MACD_Signal as macd_crossover_signal
            FROM dbo.nasdaq_100_macd_signals
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_macd_sig = self.db.execute_query(macd_sig_query)
            if not df_macd_sig.empty:
                df_macd_sig['trading_date'] = pd.to_datetime(df_macd_sig['trading_date'])
                df_macd_sig = df_macd_sig.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_macd_sig, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] MACD Signals: {len(df_macd_sig):,} records")
        except Exception as e:
            print(f"  [WARN] MACD Signals load failed: {e}")

        # Stochastic from DB
        try:
            stoch_query = f"""
            SELECT ticker, trading_date,
                   CAST(stoch_14d_k AS FLOAT) as db_stoch_k,
                   CAST(stoch_14d_d AS FLOAT) as db_stoch_d,
                   CAST(momentum_strength AS FLOAT) as db_stoch_momentum
            FROM dbo.nasdaq_100_stochastic
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_stoch = self.db.execute_query(stoch_query)
            if not df_stoch.empty:
                df_stoch['trading_date'] = pd.to_datetime(df_stoch['trading_date'])
                df_stoch = df_stoch.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_stoch, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] Stochastic: {len(df_stoch):,} records")
        except Exception as e:
            print(f"  [WARN] Stochastic load failed: {e}")

        # Fibonacci levels
        try:
            fib_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_nearest_fib_pct AS FLOAT) as fib_distance_pct,
                   fib_trade_signal
            FROM dbo.nasdaq_100_fibonacci
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_fib = self.db.execute_query(fib_query)
            if not df_fib.empty:
                df_fib['trading_date'] = pd.to_datetime(df_fib['trading_date'])
                df_fib = df_fib.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_fib, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] Fibonacci: {len(df_fib):,} records")
        except Exception as e:
            print(f"  [WARN] Fibonacci load failed: {e}")

        # Support/Resistance
        try:
            sr_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_s1_pct AS FLOAT) as sr_distance_to_support_pct,
                   CAST(distance_to_r1_pct AS FLOAT) as sr_distance_to_resistance_pct,
                   pivot_status,
                   sr_trade_signal
            FROM dbo.nasdaq_100_support_resistance
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_sr = self.db.execute_query(sr_query)
            if not df_sr.empty:
                df_sr['trading_date'] = pd.to_datetime(df_sr['trading_date'])
                df_sr = df_sr.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_sr, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] Support/Resistance: {len(df_sr):,} records")
        except Exception as e:
            print(f"  [WARN] Support/Resistance load failed: {e}")

        # Candlestick Patterns
        try:
            pattern_query = f"""
            SELECT ticker, trading_date,
                   pattern_signal,
                   CASE WHEN doji IS NOT NULL THEN 1 ELSE 0 END as has_doji,
                   CASE WHEN hammer IS NOT NULL THEN 1 ELSE 0 END as has_hammer,
                   CASE WHEN shooting_star IS NOT NULL THEN 1 ELSE 0 END as has_shooting_star,
                   CASE WHEN bullish_engulfing IS NOT NULL THEN 1 ELSE 0 END as has_bullish_engulfing,
                   CASE WHEN bearish_engulfing IS NOT NULL THEN 1 ELSE 0 END as has_bearish_engulfing,
                   CASE WHEN morning_star IS NOT NULL THEN 1 ELSE 0 END as has_morning_star,
                   CASE WHEN evening_star IS NOT NULL THEN 1 ELSE 0 END as has_evening_star
            FROM dbo.nasdaq_100_patterns
            WHERE trading_date >= DATEADD(DAY, -{self.days_back}, CAST(GETDATE() AS DATE))
            """
            df_pat = self.db.execute_query(pattern_query)
            if not df_pat.empty:
                df_pat['trading_date'] = pd.to_datetime(df_pat['trading_date'])
                df_pat = df_pat.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_pat, on=['ticker', 'trading_date'], how='left')
                print(f"  [OK] Patterns: {len(df_pat):,} records")
        except Exception as e:
            print(f"  [WARN] Patterns load failed: {e}")

        return df

    def _merge_market_context(self, df):
        """Load market context data (VIX, indices, sector ETFs, treasury) and merge"""
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
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_context['trading_date'] = pd.to_datetime(df_context['trading_date'])
                df = df.merge(df_context, on='trading_date', how='left')
                matched = df['vix_close'].notna().sum()
                print(f"  Market context merged: {len(df_context)} dates, {matched:,} matched rows")
            else:
                print("  [WARN] No market context data found")
        except Exception as e:
            print(f"  [WARN] Could not load market context: {e}")

        return df

    def _merge_calendar_features(self, df, market='NASDAQ'):
        """Load calendar features (holidays, short weeks, expiry) and merge"""
        print("[DATA] Loading market calendar features...")

        try:
            cal_query = f"""
            SELECT calendar_date,
                   is_pre_holiday, is_post_holiday, is_short_week,
                   trading_days_in_week, is_month_end, is_month_start,
                   is_quarter_end, is_options_expiry,
                   days_until_next_holiday, days_since_last_holiday,
                   other_market_closed
            FROM dbo.vw_market_calendar_features
            WHERE market = '{market}'
            """
            df_cal = self.db.execute_query(cal_query)

            if not df_cal.empty:
                df['trading_date'] = pd.to_datetime(df['trading_date'])
                df_cal['calendar_date'] = pd.to_datetime(df_cal['calendar_date'])
                df = df.merge(df_cal, left_on='trading_date', right_on='calendar_date', how='left')
                df = df.drop(columns=['calendar_date'], errors='ignore')
                matched = df['is_pre_holiday'].notna().sum()
                print(f"  Calendar features merged: {len(df_cal)} dates, {matched:,} matched rows")
            else:
                print("  [WARN] No calendar data found")
        except Exception as e:
            print(f"  [WARN] Could not load calendar features: {e}")

        return df

    # ================================================================
    # TARGET VARIABLE
    # ================================================================

    def create_target_variable(self, df):
        """
        Create target variable: 5-day price direction (Up/Down)

        Using 5-day forward returns because:
        - 1-day direction is essentially random noise (~50%)
        - 5-day captures meaningful trends with less noise
        - More actionable for swing trading
        - Matches NSE pipeline for consistency
        """
        print("[TARGET] Creating target variables (5-day horizon)...")

        df_target = df.copy()
        df_target = df_target.sort_values(['ticker', 'trading_date'])

        # 5-day targets (PRIMARY)
        df_target['next_5d_close'] = df_target.groupby('ticker')['close_price'].shift(-5)
        df_target['next_5d_return'] = (
            (df_target['next_5d_close'] - df_target['close_price'])
            / df_target['close_price'] * 100
        )
        df_target['direction_5d'] = np.where(df_target['next_5d_return'] > 0, 'Up', 'Down')

        # Remove rows without 5-day target (last 5 rows per ticker)
        valid_mask = df_target['next_5d_close'].notna()
        df_target = df_target[valid_mask]

        # Report target distribution
        dist = df_target['direction_5d'].value_counts()
        print(f"  Target distribution (5-day direction):")
        for direction, count in dist.items():
            pct = (count / len(df_target)) * 100
            print(f"    {direction}: {count:,} ({pct:.1f}%)")
        print(f"  Valid training samples: {len(df_target):,}")

        return df_target

    # ================================================================
    # FEATURE ENGINEERING (VECTORIZED)
    # ================================================================

    def engineer_features_vectorized(self, df):
        """ULTRA-FAST feature engineering using VECTORIZED operations"""
        print("[FEATURES] Engineering features (vectorized)...")

        df = df.copy()
        df = df.sort_values(['ticker', 'trading_date']).reset_index(drop=True)
        df['trading_date'] = pd.to_datetime(df['trading_date'])

        price_col = 'close_price'
        high_col = 'high_price'
        low_col = 'low_price'
        volume_col = 'volume'

        # === Basic Price Features ===
        df['daily_volatility'] = ((df[high_col] - df[low_col]) / df[price_col]) * 100
        df['daily_return'] = ((df[price_col] - df['open_price']) / df['open_price']) * 100
        df['volume_millions'] = df[volume_col] / 1_000_000
        df['price_range'] = df[high_col] - df[low_col]
        df['price_position'] = np.where(
            df['price_range'] > 0,
            (df[price_col] - df[low_col]) / df['price_range'],
            0.5
        )

        # Gap (normalized)
        gap_raw = (df.groupby('ticker')['open_price'].transform(lambda x: x.diff())
                   - df.groupby('ticker')[price_col].transform(lambda x: x.shift(1)))
        df['gap_pct'] = np.where(df[price_col] > 0, gap_raw / df[price_col] * 100, 0)

        # Volume-price trend
        df['volume_price_trend'] = df[volume_col] * df['daily_return']

        # === RSI Features ===
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        df['rsi_momentum'] = df.groupby('ticker')['RSI'].transform(lambda x: x.diff())

        # === Time Features ===
        df['day_of_week'] = df['trading_date'].dt.dayofweek
        df['month'] = df['trading_date'].dt.month

        # === Moving Averages (vectorized) ===
        print("[FEATURES] Adding technical indicators (vectorized)...")
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=period, min_periods=1).mean()
            )
            df[f'ema_{period}'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.ewm(span=period, min_periods=1).mean()
            )

        # === MACD (vectorized) ===
        ema_12 = df.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
        ema_26 = df.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # === Price vs MA Ratios (safe division) ===
        df['price_vs_sma20'] = np.where(df['sma_20'] > 0, df[price_col] / df['sma_20'], 1.0)
        df['price_vs_sma50'] = np.where(df['sma_50'] > 0, df[price_col] / df['sma_50'], 1.0)
        df['price_vs_ema20'] = np.where(df['ema_20'] > 0, df[price_col] / df['ema_20'], 1.0)
        df['sma20_vs_sma50'] = np.where(df['sma_50'] > 0, df['sma_20'] / df['sma_50'], 1.0)
        df['ema20_vs_ema50'] = np.where(df['ema_50'] > 0, df['ema_20'] / df['ema_50'], 1.0)
        df['sma5_vs_sma20'] = np.where(df['sma_20'] > 0, df['sma_5'] / df['sma_20'], 1.0)

        # === SMA 100/200 from DB (or calculate fallback) ===
        if 'sma_200' not in df.columns or df['sma_200'].isna().all():
            df['sma_200'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=200, min_periods=1).mean()
            )
        if 'sma_100' not in df.columns or df['sma_100'].isna().all():
            df['sma_100'] = df.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )
        df['price_vs_sma100'] = np.where(df['sma_100'] > 0, df[price_col] / df['sma_100'], 1.0)
        df['price_vs_sma200'] = np.where(df['sma_200'] > 0, df[price_col] / df['sma_200'], 1.0)

        # === SMA Flag encoding (Above/Below -> 1/-1) ===
        sma_flag_map = {'Above': 1, 'Below': -1, 'BULLISH': 1, 'BEARISH': -1}
        for flag_col in ['SMA_200_Flag', 'SMA_100_Flag', 'SMA_50_Flag', 'SMA_20_Flag']:
            target_col = flag_col.lower()
            if flag_col in df.columns:
                df[target_col] = df[flag_col].map(sma_flag_map).fillna(0).astype(float)
                df.drop(columns=[flag_col], inplace=True, errors='ignore')
            else:
                df[target_col] = 0

        # === MACD crossover signal encoding ===
        signal_strength_map = {
            'Bullish Crossover': 1, 'Bearish Crossover': -1,
            'BUY': 1, 'SELL': -1, 'NEUTRAL': 0, 'No Signal': 0,
        }
        if 'macd_crossover_signal' in df.columns:
            df['macd_signal_strength'] = df['macd_crossover_signal'].map(
                signal_strength_map
            ).fillna(0).astype(float)
            df.drop(columns=['macd_crossover_signal'], inplace=True, errors='ignore')
        else:
            df['macd_signal_strength'] = 0

        # === Volume Indicators ===
        df['volume_sma_20'] = df.groupby('ticker')[volume_col].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )
        df['volume_sma_ratio'] = np.where(df['volume_sma_20'] > 0, df[volume_col] / df['volume_sma_20'], 1.0)
        df['vol_change_ratio'] = df.groupby('ticker')[volume_col].transform(lambda x: x.pct_change())

        # === Momentum ===
        df['price_momentum_5'] = df.groupby('ticker')[price_col].transform(lambda x: x / x.shift(5))
        df['price_momentum_10'] = df.groupby('ticker')[price_col].transform(lambda x: x / x.shift(10))

        # === Volatility ===
        df['price_volatility_10'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.pct_change().rolling(window=10, min_periods=1).std()
        )
        df['price_volatility_20'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.pct_change().rolling(window=20, min_periods=1).std()
        )

        # === Trend Strength ===
        df['trend_strength_10'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=10, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / y.std() if len(y) > 1 and y.std() != 0 else 0
            )
        )

        # === Bollinger Bands (use DB if available, else calculate) ===
        if 'db_bb_upper' in df.columns and df['db_bb_upper'].notna().any():
            bb_upper = df['db_bb_upper']
            bb_lower = df['db_bb_lower']
        else:
            bb_sma = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
            bb_std = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).std())
            bb_upper = bb_sma + 2 * bb_std
            bb_lower = bb_sma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        df['bollinger_pctb'] = np.where(bb_range > 0, (df[price_col] - bb_lower) / bb_range, 0.5)
        df['bollinger_bandwidth'] = np.where(df[price_col] > 0, bb_range / df[price_col], 0)
        # Drop raw BB columns
        df.drop(columns=['db_bb_upper', 'db_bb_lower', 'db_bb_sma20'], inplace=True, errors='ignore')

        # === Stochastic Oscillator (use DB if available, else calculate 14-period) ===
        if 'db_stoch_k' in df.columns and df['db_stoch_k'].notna().any():
            low_14 = df.groupby('ticker')[low_col].transform(lambda x: x.rolling(window=14, min_periods=1).min())
            high_14 = df.groupby('ticker')[high_col].transform(lambda x: x.rolling(window=14, min_periods=1).max())
            stoch_range = high_14 - low_14
            calc_k = np.where(stoch_range > 0, (df[price_col] - low_14) / stoch_range * 100, 50)
            df['stochastic_k'] = df['db_stoch_k'].fillna(pd.Series(calc_k, index=df.index))
            calc_d = df.groupby('ticker')['stochastic_k'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df['stochastic_d'] = df['db_stoch_d'].fillna(calc_d) if 'db_stoch_d' in df.columns else calc_d
            df['stochastic_momentum'] = df['db_stoch_momentum'].fillna(0) if 'db_stoch_momentum' in df.columns else 0
            df.drop(columns=['db_stoch_k', 'db_stoch_d', 'db_stoch_momentum'], inplace=True, errors='ignore')
        else:
            low_14 = df.groupby('ticker')[low_col].transform(lambda x: x.rolling(window=14, min_periods=1).min())
            high_14 = df.groupby('ticker')[high_col].transform(lambda x: x.rolling(window=14, min_periods=1).max())
            stoch_range = high_14 - low_14
            df['stochastic_k'] = np.where(stoch_range > 0, (df[price_col] - low_14) / stoch_range * 100, 50)
            df['stochastic_d'] = df.groupby('ticker')['stochastic_k'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df['stochastic_momentum'] = df['stochastic_k'] - df['stochastic_d']

        # === ATR (use DB if available, else calculate) ===
        prev_close = df.groupby('ticker')[price_col].transform(lambda x: x.shift(1))
        high_low = df[high_col] - df[low_col]
        high_close = (df[high_col] - prev_close).abs()
        low_close = (df[low_col] - prev_close).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        if 'db_atr_14' in df.columns and df['db_atr_14'].notna().any():
            df['atr_14'] = df['db_atr_14'].fillna(
                pd.Series(true_range).groupby(df['ticker']).transform(
                    lambda x: x.rolling(window=14, min_periods=1).mean()
                )
            )
            df.drop(columns=['db_atr_14'], inplace=True, errors='ignore')
        else:
            df['_true_range'] = true_range
            df['atr_14'] = df.groupby('ticker')['_true_range'].transform(
                lambda x: x.rolling(window=14, min_periods=1).mean()
            )
            df.drop(columns=['_true_range'], inplace=True, errors='ignore')
        df['atr_ratio'] = np.where(df[price_col] > 0, df['atr_14'] / df[price_col], 0)

        # === Normalized MACD ===
        df['macd_normalized'] = np.where(df[price_col] > 0, df['macd'] / df[price_col] * 100, 0)
        df['macd_signal_normalized'] = np.where(df[price_col] > 0, df['macd_signal'] / df[price_col] * 100, 0)
        df['macd_histogram_normalized'] = np.where(df[price_col] > 0, df['macd_histogram'] / df[price_col] * 100, 0)

        # === Lagged Returns ===
        for period in [1, 2, 3, 5, 10]:
            df[f'return_{period}d'] = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change(period))

        # === RSI-Price Divergence ===
        price_dir_5 = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.pct_change(5)))
        rsi_dir_5 = df.groupby('ticker')['RSI'].transform(lambda x: np.sign(x.diff(5)))
        df['rsi_price_divergence'] = (price_dir_5 != rsi_dir_5).astype(int)

        # === Candlestick Features ===
        df['high_low_ratio'] = np.where(df[low_col] > 0, df[high_col] / df[low_col], 1.0)
        df['close_open_ratio'] = np.where(df['open_price'] > 0, df[price_col] / df['open_price'], 1.0)
        df['upper_shadow'] = np.where(
            df['price_range'] > 0,
            (df[high_col] - np.maximum(df['open_price'], df[price_col])) / df['price_range'],
            0
        )
        df['lower_shadow'] = np.where(
            df['price_range'] > 0,
            (np.minimum(df['open_price'], df[price_col]) - df[low_col]) / df['price_range'],
            0
        )

        # ================================================================
        # ENRICHED DB FEATURE ENCODING (Fibonacci, S/R, Patterns)
        # ================================================================
        print("[FEATURES] Encoding enriched DB signals...")

        # Signal encoding map (reusable across features)
        signal_strength_map_full = {
            'STRONG_BUY': 2, 'BUY': 1, 'BUY_FIB_500': 1, 'BUY_FIB_382': 1,
            'BUY_FIB_236': 1, 'BUY_BULLISH_CROSS': 1,
            'NEUTRAL': 0, 'NEUTRAL_WAIT': 0, 'No Signal': 0,
            'SELL': -1, 'SELL_BEARISH_CROSS': -1,
            'STRONG_SELL': -2,
            'NEAR_SUPPORT_BUY': 1, 'BULLISH_ZONE': 1,
            'NEAR_RESISTANCE_SELL': -1, 'BEARISH_ZONE': -1,
            'Bullish Crossover': 1, 'Bearish Crossover': -1,
        }
        position_map = {
            'ABOVE_PIVOT': 1, 'BELOW_PIVOT': -1,
            'Above': 1, 'Below': -1,
            'BULLISH': 1, 'BEARISH': -1,
        }

        # --- Fibonacci signal ---
        if 'fib_trade_signal' in df.columns:
            df['fib_signal_strength'] = df['fib_trade_signal'].map(
                signal_strength_map_full
            ).fillna(0).astype(float)
            df.drop(columns=['fib_trade_signal'], inplace=True, errors='ignore')
        else:
            df['fib_signal_strength'] = 0
        if 'fib_distance_pct' not in df.columns:
            df['fib_distance_pct'] = 0

        # --- Support/Resistance signals ---
        if 'pivot_status' in df.columns:
            df['sr_pivot_position'] = df['pivot_status'].map(
                position_map
            ).fillna(0).astype(float)
            df.drop(columns=['pivot_status'], inplace=True, errors='ignore')
        else:
            df['sr_pivot_position'] = 0

        if 'sr_trade_signal' in df.columns:
            df['sr_signal_strength'] = df['sr_trade_signal'].map(
                signal_strength_map_full
            ).fillna(0).astype(float)
            df.drop(columns=['sr_trade_signal'], inplace=True, errors='ignore')
        else:
            df['sr_signal_strength'] = 0

        for col in ['sr_distance_to_support_pct', 'sr_distance_to_resistance_pct']:
            if col not in df.columns:
                df[col] = 0

        # --- Pattern signals ---
        if 'pattern_signal' in df.columns:
            df['pattern_signal_strength'] = df['pattern_signal'].map(
                signal_strength_map_full
            ).fillna(0).astype(float)
            df.drop(columns=['pattern_signal'], inplace=True, errors='ignore')
        else:
            df['pattern_signal_strength'] = 0

        # Pattern binary flags (already 0/1 from SQL CASE)
        for col in ['has_doji', 'has_hammer', 'has_shooting_star',
                     'has_bullish_engulfing', 'has_bearish_engulfing',
                     'has_morning_star', 'has_evening_star']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(float)

        # ================================================================
        # MARKET REGIME DETECTION
        # ================================================================
        print("[FEATURES] Adding market regime features...")

        # SMA trend direction
        df['regime_sma20_slope'] = df.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean().pct_change(5) * 100
        )

        # ADX-like trend strength
        up_move = df.groupby('ticker')[high_col].transform(lambda x: x.diff())
        down_move = -df.groupby('ticker')[low_col].transform(lambda x: x.diff())
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df['_pos_dm'] = pos_dm
        df['_neg_dm'] = neg_dm
        pos_dm_smooth = df.groupby('ticker')['_pos_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        neg_dm_smooth = df.groupby('ticker')['_neg_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        dm_sum = pos_dm_smooth + neg_dm_smooth
        dx = np.where(dm_sum > 0, np.abs(pos_dm_smooth - neg_dm_smooth) / dm_sum * 100, 0)
        df['_dx'] = dx
        df['regime_adx'] = df.groupby('ticker')['_dx'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df.drop(columns=['_pos_dm', '_neg_dm', '_dx'], inplace=True, errors='ignore')

        # Volatility regime
        vol_short = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(10, min_periods=1).std())
        vol_long = df.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(60, min_periods=1).std())
        df['regime_vol_ratio'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)

        # Mean reversion indicator
        sma50 = df.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        df['regime_mean_reversion'] = np.where(
            df['atr_14'] > 0,
            (df[price_col] - sma50) / df['atr_14'],
            0
        )

        # Trend consistency
        overall_dir = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(20)))
        daily_dirs = df.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(1)))
        consistent = (daily_dirs == overall_dir).astype(float)
        df['regime_trend_consistency'] = consistent.groupby(df['ticker']).transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        # ================================================================
        # FUNDAMENTAL FEATURES
        # ================================================================
        print("[FEATURES] Adding fundamental features...")

        if 'fifty_two_week_high' in df.columns:
            df['price_vs_52wk_high'] = np.where(
                df['fifty_two_week_high'] > 0, df[price_col] / df['fifty_two_week_high'], 0
            )
        else:
            df['price_vs_52wk_high'] = 0

        if 'fifty_two_week_low' in df.columns:
            df['price_vs_52wk_low'] = np.where(
                df['fifty_two_week_low'] > 0, df[price_col] / df['fifty_two_week_low'], 0
            )
        else:
            df['price_vs_52wk_low'] = 0

        if 'two_hundred_day_avg' in df.columns:
            df['price_vs_200d_avg'] = np.where(
                df['two_hundred_day_avg'] > 0, df[price_col] / df['two_hundred_day_avg'], 0
            )
        else:
            df['price_vs_200d_avg'] = 0

        # Sector encoding
        if 'sector' in df.columns:
            self.sector_encoder = LabelEncoder()
            df['sector_encoded'] = self.sector_encoder.fit_transform(
                df['sector'].fillna('Unknown')
            )
            print(f"  Sectors found: {len(self.sector_encoder.classes_)} unique")
        else:
            df['sector_encoded'] = 0

        # === Sector-specific ETF return mapping ===
        if 'sector' in df.columns and 'xlk_return_1d' in df.columns:
            df['sector_etf_return_1d'] = df['sector'].map(
                lambda s: self.SECTOR_TO_ETF.get(s, 'sp500_return_1d')
            )
            df['sector_etf_return_1d'] = df.apply(
                lambda row: row.get(row['sector_etf_return_1d'], 0) if pd.notna(row.get('sector_etf_return_1d')) else 0,
                axis=1
            )
        else:
            df['sector_etf_return_1d'] = 0

        # ================================================================
        # SENTIMENT-PRICE DIVERGENCE
        # Detects when sentiment is negative but price looks bullish (or vice versa)
        # Key signal for catching sector-wide downturns the model might miss.
        # ================================================================
        print("[FEATURES] Adding sentiment-price divergence...")
        if 'sector_sentiment_score' in df.columns:
            # Use 5-day return for price direction
            if 'return_5d' in df.columns:
                _price_dir = df['return_5d']
            else:
                _price_dir = df.groupby('ticker')[price_col].transform(
                    lambda x: x.pct_change(5)
                )
            sent = df['sector_sentiment_score']
            price_dir_sign = np.sign(_price_dir)
            sent_dir_sign = np.sign(sent)
            # divergence = 1 when sentiment and price move opposite, magnitude-weighted
            df['sentiment_price_divergence'] = np.where(
                (price_dir_sign != sent_dir_sign) & (sent_dir_sign != 0),
                np.abs(sent) * 2,  # Amplify divergence signal
                0
            )
        else:
            df['sentiment_price_divergence'] = 0

        # Drop raw DB/merge columns not needed as features
        drop_cols = ['db_macd', 'db_macd_signal', 'company', 'rsi_trade_signal',
                     'next_5d_close', 'sector',
                     'fifty_two_week_high', 'fifty_two_week_low', 'two_hundred_day_avg',
                     'sp500_close', 'nasdaq_comp_close', 'dxy_close',
                     # Individual sector ETF columns (we use the mapped one)
                     'xlk_return_1d', 'xlf_return_1d', 'xle_return_1d',
                     'xlv_return_1d', 'xli_return_1d', 'xlc_return_1d',
                     'xly_return_1d', 'xlp_return_1d', 'xlb_return_1d',
                     'xlre_return_1d', 'xlu_return_1d',
                     # Raw pattern/signal categorical columns already encoded
                     'patterns_detected',
                     ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

        # Handle infinite values and NaN — forward fill only (no bfill to prevent leakage)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)

        print(f"[FEATURES] Complete — {df.shape[1]} columns, {df.shape[0]:,} rows")
        return df

    # ================================================================
    # ML DATASET PREPARATION
    # ================================================================

    def prepare_ml_dataset(self, df_features):
        """Prepare dataset for ML training with feature selection"""
        print("[PREP] Preparing ML dataset...")

        target_column = 'direction_5d'

        # Exclude non-feature columns
        exclude_cols = ['trading_date', 'ticker', target_column, 'next_5d_return',
                        'open_price', 'high_price', 'low_price', 'close_price', 'volume']

        feature_cols = [col for col in df_features.columns
                        if col not in exclude_cols
                        and df_features[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        X = df_features[feature_cols].copy()
        y_direction = df_features[target_column].copy()
        y_return = df_features['next_5d_return'].copy()

        # Remove any remaining NaN rows
        valid_mask = X.notna().all(axis=1) & y_direction.notna() & y_return.notna()
        X = X[valid_mask]
        y_direction = y_direction[valid_mask]
        y_return = y_return[valid_mask]

        # Encode target
        direction_encoder = LabelEncoder()
        y_direction_encoded = direction_encoder.fit_transform(y_direction)

        print(f"[PREP] Dataset ready:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]:,}")
        print(f"  Direction classes: {list(direction_encoder.classes_)}")
        balance = dict(zip(*np.unique(y_direction_encoded, return_counts=True)))
        print(f"  Balance: {balance}")

        # ================================================================
        # FEATURE SELECTION (STRATIFIED: market-wide + stock-specific)
        # ================================================================
        print("[PREP] Running STRATIFIED feature selection (mutual_info_classif)...")
        try:
            # Define market-wide feature patterns (same for all stocks on a given day)
            MARKET_WIDE_PATTERNS = [
                'vix_close', 'vix_change_pct', 'sp500_return_1d', 'sp500_close',
                'nasdaq_comp_return_1d', 'nasdaq_comp_close',
                'dxy_return_1d', 'dxy_close',
                'us_10y_yield_close', 'us_10y_yield_change',
                'xlk_return_1d', 'xlf_return_1d', 'xle_return_1d', 'xlv_return_1d',
                'xli_return_1d', 'xlc_return_1d', 'xly_return_1d', 'xlp_return_1d',
                'xlb_return_1d', 'xlre_return_1d', 'xlu_return_1d',
                'sector_etf_return_1d',
                'trading_days_in_week', 'day_of_week', 'month',
                'is_pre_holiday', 'is_post_holiday', 'is_short_week',
                'is_month_end', 'is_month_start', 'is_quarter_end',
                'is_options_expiry', 'days_until_next_holiday',
                'days_since_last_holiday', 'other_market_closed',
                # Sector sentiment features (from nasdaq_sector_sentiment)
                'sector_sentiment_score', 'sector_sentiment_confidence',
                'sector_sentiment_momentum_3d', 'sector_sentiment_momentum_7d',
                'sector_sentiment_vs_avg_30d',
                'sector_positive_ratio', 'sector_negative_ratio',
                'sector_news_volume', 'market_sentiment_score',
                'sentiment_price_divergence',
            ]
            market_features = [f for f in feature_cols if f in MARKET_WIDE_PATTERNS]
            stock_features = [f for f in feature_cols if f not in MARKET_WIDE_PATTERNS]
            print(f"  Market-wide features: {len(market_features)}")
            print(f"  Stock-specific features: {len(stock_features)}")

            # Compute MI scores for both groups separately
            mi_scores_all = mutual_info_classif(X, y_direction_encoded, random_state=42)
            mi_all = pd.Series(mi_scores_all, index=feature_cols).sort_values(ascending=False)

            mi_market = mi_all[mi_all.index.isin(market_features)].sort_values(ascending=False)
            mi_stock = mi_all[mi_all.index.isin(stock_features)].sort_values(ascending=False)

            # Stratified selection: top-8 market + top-22 stock = 30 features
            n_market = min(8, len(mi_market))
            n_stock = min(22, len(mi_stock))
            selected_market = mi_market.head(n_market).index.tolist()
            selected_stock = mi_stock.head(n_stock).index.tolist()
            selected_features = selected_stock + selected_market  # stock-specific first

            print(f"\n  Selected {len(selected_stock)} stock-specific features:")
            for i, feat in enumerate(selected_stock):
                print(f"    {i+1:2d}. {feat} (MI={mi_all[feat]:.4f})")
            print(f"\n  Selected {len(selected_market)} market-wide features:")
            for i, feat in enumerate(selected_market):
                print(f"    {i+1:2d}. {feat} (MI={mi_all[feat]:.4f})")

            # Save selected features for prediction consistency
            features_path = self.data_dir / 'selected_features.json'
            with open(features_path, 'w') as f:
                json.dump(selected_features, f, indent=2)
            print(f"\n  Total: {len(selected_features)} features saved to {features_path}")

            X = X[selected_features]
            feature_cols = selected_features
        except Exception as e:
            print(f"  [WARN] Feature selection failed, using all features: {e}")

        self.feature_columns = feature_cols

        return X, y_direction_encoded, y_return[valid_mask], direction_encoder, feature_cols

    # ================================================================
    # CLASSIFICATION MODELS (ENSEMBLE)
    # ================================================================

    def train_classification_models(self, X, y, feature_cols):
        """Train ensemble of classification models with calibration"""
        print("[TRAIN] Training classification models (ensemble)...")

        # 3-way split: train (60%) / calibration (20%) / test (20%)
        train_end = int(0.60 * len(X))
        cal_end = int(0.80 * len(X))
        X_train = X.iloc[:train_end]
        X_cal = X.iloc[train_end:cal_end]
        X_test = X.iloc[cal_end:]
        y_train = y[:train_end]
        y_cal = y[train_end:cal_end]
        y_test = y[cal_end:]

        print(f"  Split: Train={len(X_train):,}, Calibration={len(X_cal):,}, Test={len(X_test):,}")

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_cal_scaled = scaler.transform(X_cal)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_split=10, min_samples_leaf=5,
                class_weight='balanced', max_features='sqrt',
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, min_samples_split=10, min_samples_leaf=5,
                subsample=0.8, random_state=42
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200, max_depth=12,
                min_samples_split=10, min_samples_leaf=5,
                class_weight='balanced', max_features='sqrt',
                random_state=42, n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', C=0.1,
                solver='liblinear', max_iter=2000, random_state=42
            )
        }

        # Time-weighted sample weights (recent data more relevant)
        n_train = len(y_train)
        time_positions = np.arange(n_train) / n_train
        decay_rate = 1.2
        time_weights = np.exp(decay_rate * (time_positions - 1))
        time_weights = time_weights / time_weights.mean()

        print(f"  Time-weighted training: oldest={time_weights[0]:.3f}, "
              f"newest={time_weights[-1]:.3f}, ratio={time_weights[-1]/time_weights[0]:.1f}x")

        # Train and evaluate each model
        model_results = {}
        trained_models = {}
        cv_splitter = PurgedTimeSeriesSplit(n_splits=3, purge_gap=5)

        for model_name, model in models.items():
            print(f"  Training {model_name}...", flush=True)

            try:
                # Train with time-weighted sample weights
                try:
                    model.fit(X_train_scaled, y_train, sample_weight=time_weights)
                except TypeError:
                    model.fit(X_train_scaled, y_train)

                # Predict
                y_pred = model.predict(X_test_scaled)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=cv_splitter, scoring='accuracy'
                )

                model_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                }
                trained_models[model_name] = model

                print(f"    Acc={accuracy:.3f}, F1={f1:.3f}, "
                      f"CV={cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})")

            except Exception as e:
                print(f"    [ERROR] {model_name}: {e}")

        # Ensemble (Voting Classifier)
        print("  Training Ensemble (Voting Classifier)...")
        try:
            ensemble_estimators = [
                (name.lower().replace(' ', '_'), model)
                for name, model in trained_models.items()
            ]
            ensemble = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft',
                n_jobs=1  # n_jobs=-1 can deadlock on Windows
            )
            ensemble.fit(X_train_scaled, y_train)

            y_pred_ensemble = ensemble.predict(X_test_scaled)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted', zero_division=0)

            model_results['Ensemble'] = {
                'accuracy': ensemble_accuracy,
                'f1_score': ensemble_f1,
                'cv_mean': 0, 'cv_std': 0,
            }
            trained_models['Ensemble'] = ensemble

            print(f"    Ensemble Acc={ensemble_accuracy:.3f}, F1={ensemble_f1:.3f}")
        except Exception as e:
            print(f"    [ERROR] Ensemble: {e}")

        # Find best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['f1_score'])
        best_model = trained_models[best_model_name]

        print(f"\n[BEST] Best classifier: {best_model_name} "
              f"(F1={model_results[best_model_name]['f1_score']:.3f}, "
              f"Acc={model_results[best_model_name]['accuracy']:.1%})")

        # Calibrate probabilities on calibration set
        print("[CALIBRATE] Calibrating model probabilities on held-out calibration set...")
        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=best_model, cv='prefit', method='isotonic'
            )
            calibrated_model.fit(X_cal_scaled, y_cal)

            cal_preds = calibrated_model.predict(X_test_scaled)
            cal_probs = calibrated_model.predict_proba(X_test_scaled)
            cal_accuracy = accuracy_score(y_test, cal_preds)

            print(f"  Pre-calibration test accuracy:  {model_results[best_model_name]['accuracy']:.3f}")
            print(f"  Post-calibration test accuracy: {cal_accuracy:.3f}")

            # Verify calibration: high-confidence should be more accurate
            max_probs = cal_probs.max(axis=1)
            high_mask = max_probs >= 0.65
            if high_mask.sum() > 10:
                high_acc = accuracy_score(y_test[high_mask], cal_preds[high_mask])
                low_acc = accuracy_score(y_test[~high_mask], cal_preds[~high_mask]) if (~high_mask).sum() > 0 else 0
                print(f"  High-confidence (>=65%) accuracy: {high_acc:.3f} ({high_mask.sum()} samples)")
                print(f"  Low-confidence (<65%) accuracy:   {low_acc:.3f} ({(~high_mask).sum()} samples)")

            best_model = calibrated_model
            print("[CALIBRATE] Probability calibration applied successfully")
        except Exception as e:
            print(f"  [WARN] Calibration skipped: {e}")

        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'scaler': scaler,
            'feature_columns': feature_cols,
            'X_train': X_train,
            'X_cal': X_cal,
            'X_test': X_test,
            'y_train': y_train,
            'y_cal': y_cal,
            'y_test': y_test,
        }

    # ================================================================
    # REGRESSION MODELS (5-day return magnitude)
    # ================================================================

    def train_regression_models(self, X, y_return, feature_cols):
        """Train regression models for 5-day price change prediction"""
        print("[TRAIN] Training regression models (5-day return magnitude)...")

        split_idx = int(0.8 * len(X))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y_return.iloc[:split_idx]
        y_test = y_return.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clip extreme returns for stability
        y_train_clipped = np.clip(y_train.values, -10, 10)

        # Time-weighted
        n_reg = len(y_train)
        reg_time_pos = np.arange(n_reg) / n_reg
        reg_time_weights = np.exp(1.2 * (reg_time_pos - 1))
        reg_time_weights = reg_time_weights / reg_time_weights.mean()

        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=12,
                min_samples_split=10, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, min_samples_split=10, min_samples_leaf=5,
                subsample=0.8, random_state=42
            ),
            'Ridge': Ridge(alpha=1.0),
        }

        model_results = {}
        trained_models = {}

        for model_name, model in models.items():
            print(f"  Training {model_name} regressor...")
            try:
                try:
                    model.fit(X_train_scaled, y_train_clipped, sample_weight=reg_time_weights)
                except TypeError:
                    model.fit(X_train_scaled, y_train_clipped)
                y_pred = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                pred_dir = (y_pred > 0).astype(int)
                actual_dir = (y_test.values > 0).astype(int)
                dir_acc = accuracy_score(actual_dir, pred_dir)

                model_results[model_name] = {
                    'mae': mae, 'rmse': rmse, 'r2': r2,
                    'direction_accuracy': dir_acc,
                }
                trained_models[model_name] = model

                print(f"    MAE={mae:.4f}%, RMSE={rmse:.4f}%, Direction Acc={dir_acc:.1%}")
            except Exception as e:
                print(f"    [ERROR] {model_name}: {e}")

        # Regression ensemble — use pre-fitted models (skip expensive re-training)
        print("  Creating Regression Ensemble (from pre-fitted models)...")
        try:
            class PreFittedEnsemble:
                """Simple averaging ensemble from already-trained models."""
                def __init__(self, models_dict):
                    self.models = list(models_dict.values())
                    self.model_names = list(models_dict.keys())
                def predict(self, X):
                    preds = np.column_stack([m.predict(X) for m in self.models])
                    return preds.mean(axis=1)
                def get_params(self, deep=True):
                    return {'models_dict': dict(zip(self.model_names, self.models))}

            reg_ensemble = PreFittedEnsemble(trained_models)
            y_pred_ens = reg_ensemble.predict(X_test_scaled)
            ens_mae = mean_absolute_error(y_test, y_pred_ens)
            ens_dir = accuracy_score(
                (y_test.values > 0).astype(int),
                (y_pred_ens > 0).astype(int)
            )
            model_results['Ensemble'] = {'mae': ens_mae, 'direction_accuracy': ens_dir}
            trained_models['Ensemble'] = reg_ensemble

            print(f"    Ensemble MAE={ens_mae:.4f}%, Direction Acc={ens_dir:.1%}")
        except Exception as e:
            print(f"    [ERROR] Ensemble: {e}")

        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['direction_accuracy'])
        print(f"\n[BEST] Best regressor: {best_model_name} "
              f"(Direction Acc={model_results[best_model_name]['direction_accuracy']:.1%})")

        return {
            'model_results': model_results,
            'trained_models': trained_models,
            'best_model_name': best_model_name,
            'best_model': trained_models[best_model_name],
            'scaler': scaler,
        }

    # ================================================================
    # SAVE ARTIFACTS
    # ================================================================

    def save_model_artifacts(self, clf_results, reg_results, direction_encoder):
        """Save trained model and all artifacts"""
        print("[SAVE] Saving model artifacts...")

        # Save best classification model (this is what predict_trading_signals.py loads)
        model_path = self.data_dir / 'best_model_gradient_boosting.joblib'
        joblib.dump(clf_results['best_model'], model_path)
        print(f"  [OK] Best classifier -> {model_path}")

        # Save all classification models
        for name, model in clf_results['trained_models'].items():
            safe_name = name.lower().replace(' ', '_')
            path = self.data_dir / f'clf_{safe_name}.joblib'
            joblib.dump(model, path)

        # Save best regression model
        reg_path = self.data_dir / 'best_regressor.joblib'
        joblib.dump(reg_results['best_model'], reg_path)
        print(f"  [OK] Best regressor -> {reg_path}")

        # Save all regression models (skip unpicklable custom ensembles)
        for name, model in reg_results['trained_models'].items():
            safe_name = name.lower().replace(' ', '_')
            path = self.data_dir / f'reg_{safe_name}.joblib'
            try:
                joblib.dump(model, path)
            except Exception as e:
                print(f"  [SKIP] reg_{safe_name}: {e}")

        # Save preprocessing artifacts
        joblib.dump(clf_results['scaler'], self.data_dir / 'scaler.joblib')
        joblib.dump(direction_encoder, self.data_dir / 'target_encoder.joblib')
        joblib.dump(reg_results['scaler'], self.data_dir / 'reg_scaler.joblib')

        # Save sector encoder
        if self.sector_encoder is not None:
            joblib.dump(self.sector_encoder, self.data_dir / 'sector_encoder.joblib')

        # Save training metadata
        metadata = {
            'training_timestamp': self.timestamp,
            'feature_columns': self.feature_columns,
            'days_back': self.days_back,
            'target': '5-day price direction (Up/Down)',
            'best_clf_model': clf_results['best_model_name'],
            'best_reg_model': reg_results['best_model_name'],
            'direction_classes': list(direction_encoder.classes_),
            'training_samples': len(clf_results['X_train']),
            'calibration_samples': len(clf_results['X_cal']),
            'test_samples': len(clf_results['X_test']),
            'clf_results': {
                name: {k: v for k, v in res.items()}
                for name, res in clf_results['model_results'].items()
            },
            'reg_results': {
                name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in res.items()}
                for name, res in reg_results['model_results'].items()
            },
            'sector_to_etf_map': self.SECTOR_TO_ETF,
        }
        with open(self.data_dir / 'training_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"[SAVE] All artifacts saved to {self.data_dir}/")

    # ================================================================
    # MAIN PIPELINE
    # ================================================================

    def run_ultra_fast_retrain(self):
        """Execute ultra-fast weekly retraining with NSE-quality architecture"""
        start_time = datetime.now()

        print("=" * 70)
        print(f"[START] ULTRA-FAST Weekly Retraining (Enhanced) - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        try:
            # Step 1: Backup
            if self.backup_old:
                self.backup_existing_model()

            # Step 2: Load data + enriched features + market context + calendar
            df = self.load_training_data()

            # Step 3: Create 5-day target variable
            df = self.create_target_variable(df)

            # Step 4: Engineer features (VECTORIZED)
            df_features = self.engineer_features_vectorized(df)

            # Step 5: Prepare ML dataset + feature selection
            X, y_dir, y_return, direction_encoder, feature_cols = self.prepare_ml_dataset(df_features)

            # Step 6: Train classification ensemble
            clf_results = self.train_classification_models(X, y_dir, feature_cols)

            # Step 7: Train regression models
            reg_results = self.train_regression_models(X, y_return, feature_cols)

            # Step 8: Save all artifacts
            self.save_model_artifacts(clf_results, reg_results, direction_encoder)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60

            print("=" * 70)
            print("[SUCCESS] ULTRA-FAST WEEKLY RETRAINING COMPLETE!")
            print(f"  Duration: {duration:.1f} minutes")
            print(f"  Best Classifier: {clf_results['best_model_name']} "
                  f"(F1={clf_results['model_results'][clf_results['best_model_name']]['f1_score']:.3f})")
            print(f"  Best Regressor: {reg_results['best_model_name']} "
                  f"(Direction Acc={reg_results['model_results'][reg_results['best_model_name']]['direction_accuracy']:.1%})")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"[ERROR] Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Ultra-Fast Weekly Model Retraining (Enhanced)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip backup for maximum speed')
    parser.add_argument('--backup-old', action='store_true', default=True,
                        help='Backup existing model before retraining (default: enabled)')
    parser.add_argument('--quick', action='store_true',
                        help='Use shorter training window (365 days instead of 730)')
    parser.add_argument('--days-back', type=int, default=None,
                        help='Training data window in days (default: 730, or 365 with --quick)')

    args = parser.parse_args()

    # --quick sets 365-day window unless --days-back is explicitly provided
    if args.days_back is not None:
        days_back = args.days_back
    elif args.quick:
        days_back = 365
    else:
        days_back = 730

    # --no-backup overrides --backup-old
    backup = not args.no_backup

    retrainer = UltraFastWeeklyRetrainer(
        backup_old=backup,
        days_back=days_back
    )

    success = retrainer.run_ultra_fast_retrain()

    if success:
        print("\n[NEXT STEPS]")
        print("  1. Test model: python predict_trading_signals.py --batch")
        print("  2. Export results: python export_to_database.py")
    else:
        print("\n[ERROR] Retraining failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
