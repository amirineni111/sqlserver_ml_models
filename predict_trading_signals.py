"""
Trading Signal Prediction Deployment Script

This script provides a production-ready interface for making trading signal predictions
using the trained Gradient Boosting model.

Usage:
    python predict_trading_signals.py --ticker AAPL --date 2025-11-25
    python predict_trading_signals.py --batch --file tickers.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection
from nasdaq_config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD

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
            '🟢': '[BUY]',
            '🔴': '[SELL]',
            '🟡': '[MEDIUM]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

class TradingSignalPredictor:
    """Production trading signal prediction system"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None, sector_encoder_path=None):
        """Initialize the predictor with saved model artifacts (ensemble, 5-day direction)"""
        # Default paths (updated for ensemble model)
        self.model_path = model_path or 'data/best_model_gradient_boosting.joblib'
        self.scaler_path = scaler_path or 'data/scaler.joblib'
        self.encoder_path = encoder_path or 'data/target_encoder.joblib'
        self.sector_encoder_path = sector_encoder_path or 'data/sector_encoder.joblib'
        # Load model artifacts
        self.load_model_artifacts()
        # Database connection
        self.db = SQLServerConnection()
        # Feature columns - loaded dynamically from training or fallback to full list
        self.feature_columns = self._load_feature_columns()
    
    def _load_feature_columns(self):
        """Load selected feature columns from training, with fallback to full list."""
        import json
        feature_file = 'data/selected_features.json'
        try:
            with open(feature_file, 'r') as f:
                features = json.load(f)
            safe_print(f"Loaded {len(features)} selected features from {feature_file}")
            return features
        except FileNotFoundError:
            safe_print(f"No {feature_file} found, using legacy full feature list")
            return [
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
                'beta', 'forward_pe', 'trailing_pe',
                'profit_margin', 'revenue_growth', 'earnings_growth',
                'debt_to_equity', 'return_on_equity', 'current_ratio',
                'dividend_yield',
                'price_vs_52wk_high', 'price_vs_52wk_low', 'price_vs_200d_avg',
                'sector_encoded',
            ]
    
    def load_model_artifacts(self):
        """Load the trained ensemble model, scaler, and encoder (5-day direction)"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.target_encoder = joblib.load(self.encoder_path)
            safe_print("✅ Model artifacts loaded successfully (ensemble, 5-day direction)")
            # Get class names
            self.class_names = self.target_encoder.classes_
            safe_print(f"📊 Target classes: {list(self.class_names)}")
            # Load sector encoder (optional)
            try:
                self.sector_encoder = joblib.load(self.sector_encoder_path)
                safe_print(f"📊 Sector encoder loaded: {len(self.sector_encoder.classes_)} sectors")
            except FileNotFoundError:
                self.sector_encoder = None
                print("[INFO] No sector encoder found - sector features will default to 0")
        except FileNotFoundError as e:
            safe_print(f"❌ Error loading model artifacts: {e}")
            print("Please ensure the model has been trained and saved.")
            sys.exit(1)
    
    def get_latest_data(self, ticker=None, days_back=80):
        """Fetch latest stock data for prediction.
        
        Note: days_back defaults to 80 to ensure sufficient lookback for
        technical indicators (SMA-50 needs 50 days, ATR-14, Stochastic-14, etc.)
        """
        
        if ticker:
            ticker_filter = f"AND h.ticker = '{ticker}'"
        else:
            ticker_filter = ""
        
        query = f"""
        SELECT TOP 10000
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
        WHERE h.trading_date >= DATEADD(day, -{days_back}, CAST(GETDATE() AS DATE))
            {ticker_filter}
        ORDER BY h.trading_date DESC, h.ticker
        """
        
        try:
            df = self.db.execute_query(query)
            if df.empty:
                safe_print(f"⚠️  No data found for ticker: {ticker}")
                return None
            
            # Merge fundamentals & sector data (separate queries for speed)
            df = self._merge_fundamentals(df)
            # Merge enriched DB views (SMA signals, MACD signals, S/R, Fibonacci, Patterns, Stochastic)
            df = self._merge_enriched_db_features(df, days_back)
            # Merge market context (VIX, indices, sector ETFs, treasury)
            df = self._merge_market_context(df)
            # Merge calendar features (holidays, short weeks, expiry)
            df = self._merge_calendar_features(df, market='NASDAQ')
            # Merge sector sentiment (VADER + FinBERT)
            df = self._merge_sentiment(df)
            return df
        except Exception as e:
            safe_print(f"❌ Error fetching data: {e}")
            return None
    
    def _merge_fundamentals(self, df):
        """Load fundamentals & sector data separately and merge into main DataFrame"""
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
            if not df_fund.empty:
                df = df.merge(df_fund, on='ticker', how='left')
        except Exception as e:
            print(f"[WARN] Could not load fundamentals: {e}")
        
        try:
            sector_query = "SELECT ticker, sector FROM dbo.nasdaq_top100"
            df_sector = self.db.execute_query(sector_query)
            if not df_sector.empty:
                df = df.merge(df_sector, on='ticker', how='left')
        except Exception as e:
            print(f"[WARN] Could not load sector data: {e}")
        
        return df
    
    def _merge_enriched_db_features(self, df, days_back=80):
        """Load enriched features from DB views (SMA signals, MACD signals, S/R, Fibonacci, Patterns, Stochastic)"""
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        
        # SMA Signals (updated for new view schema — SMA_200/100/Flags removed, replaced by Trend_Status etc.)
        try:
            sma_query = f"""
            SELECT ticker, trading_date,
                   CAST(EMA_100 AS FLOAT) as db_ema_100,
                   CAST(EMA_200 AS FLOAT) as db_ema_200,
                   Trend_Status, SMA_Cross_Status, sma_trade_signal
            FROM dbo.nasdaq_100_sma_signals
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_sma = self.db.execute_query(sma_query)
            if not df_sma.empty:
                df_sma['trading_date'] = pd.to_datetime(df_sma['trading_date'])
                df_sma = df_sma.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_sma, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] SMA Signals load failed: {e}")
        
        # MACD Signals (crossover — column renamed from MACD_Signal to macd_trade_signal)
        try:
            macd_sig_query = f"""
            SELECT ticker, trading_date,
                   macd_trade_signal as macd_crossover_signal
            FROM dbo.nasdaq_100_macd_signals
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_macd_sig = self.db.execute_query(macd_sig_query)
            if not df_macd_sig.empty:
                df_macd_sig['trading_date'] = pd.to_datetime(df_macd_sig['trading_date'])
                df_macd_sig = df_macd_sig.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_macd_sig, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] MACD Signals load failed: {e}")
        
        # Stochastic
        try:
            stoch_query = f"""
            SELECT ticker, trading_date,
                   CAST(stoch_14d_k AS FLOAT) as db_stoch_k,
                   CAST(stoch_14d_d AS FLOAT) as db_stoch_d,
                   CAST(momentum_strength AS FLOAT) as db_stoch_momentum
            FROM dbo.nasdaq_100_stochastic
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_stoch = self.db.execute_query(stoch_query)
            if not df_stoch.empty:
                df_stoch['trading_date'] = pd.to_datetime(df_stoch['trading_date'])
                df_stoch = df_stoch.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_stoch, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] Stochastic load failed: {e}")
        
        # Fibonacci
        try:
            fib_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_nearest_fib_pct AS FLOAT) as fib_distance_pct,
                   fib_trade_signal
            FROM dbo.nasdaq_100_fibonacci
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_fib = self.db.execute_query(fib_query)
            if not df_fib.empty:
                df_fib['trading_date'] = pd.to_datetime(df_fib['trading_date'])
                df_fib = df_fib.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_fib, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] Fibonacci load failed: {e}")
        
        # Support/Resistance
        try:
            sr_query = f"""
            SELECT ticker, trading_date,
                   CAST(distance_to_s1_pct AS FLOAT) as sr_distance_to_support_pct,
                   CAST(distance_to_r1_pct AS FLOAT) as sr_distance_to_resistance_pct,
                   pivot_status,
                   sr_trade_signal
            FROM dbo.nasdaq_100_support_resistance
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_sr = self.db.execute_query(sr_query)
            if not df_sr.empty:
                df_sr['trading_date'] = pd.to_datetime(df_sr['trading_date'])
                df_sr = df_sr.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_sr, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] Support/Resistance load failed: {e}")
        
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
            WHERE trading_date >= DATEADD(DAY, -{days_back}, CAST(GETDATE() AS DATE))
            """
            df_pat = self.db.execute_query(pattern_query)
            if not df_pat.empty:
                df_pat['trading_date'] = pd.to_datetime(df_pat['trading_date'])
                df_pat = df_pat.drop_duplicates(subset=['ticker', 'trading_date'], keep='last')
                df = df.merge(df_pat, on=['ticker', 'trading_date'], how='left')
        except Exception as e:
            print(f"[WARN] Patterns load failed: {e}")
        
        return df
    
    def _merge_sentiment(self, df):
        """Load sector sentiment scores from nasdaq_sector_sentiment and merge.
        
        Mirrors the training pipeline's _merge_sentiment method.
        Merges on (trading_date, sector), defaults to 0 if table missing.
        """
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
        except Exception as e:
            print(f"[WARN] Could not load sentiment: {e}")
        
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
    
    def _merge_market_context(self, df):
        """Load market context data (VIX, indices, sector ETFs, treasury) and merge on trading_date."""
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
        except Exception as e:
            print(f"[WARN] Could not load market context: {e}")
        
        return df
    
    def _merge_calendar_features(self, df, market='NASDAQ'):
        """Load calendar features (holidays, short weeks, expiry) and merge on trading_date.
        
        Adds pre/post holiday flags, short week indicators, options expiry,
        and cross-market holiday awareness from the shared market_calendar table.
        """
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
        except Exception as e:
            print(f"[WARN] Could not load calendar features: {e}")
        
        return df
    
    def engineer_features(self, df):
        """Apply feature engineering to raw data"""
        if df is None or df.empty:
            return None
        
        df_features = df.copy()
        
        # Sort by ticker and date for proper calculation
        df_features = df_features.sort_values(['ticker', 'trading_date'])
        
        # Basic calculated features
        df_features['daily_volatility'] = ((df_features['high_price'] - df_features['low_price']) / df_features['close_price']) * 100
        df_features['daily_return'] = ((df_features['close_price'] - df_features['open_price']) / df_features['open_price']) * 100
        df_features['volume_millions'] = df_features['volume'] / 1000000.0
        
        # Additional features
        df_features['price_range'] = df_features['high_price'] - df_features['low_price']
        df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
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
        
        # ================================================================
        # ENRICHED DB FEATURE ENCODING (must match training pipeline)
        # ================================================================
        signal_strength_map = {
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
        
        # SMA Trend/Cross encoding (new view schema replaces old SMA_*_Flag columns)
        trend_map = {
            'STRONG_UPTREND': 2, 'UPTREND': 1, 'NEUTRAL': 0,
            'DOWNTREND': -1, 'STRONG_DOWNTREND': -2,
        }
        cross_map = {
            'GOLDEN_CROSS_ZONE': 1, 'NEUTRAL': 0, 'DEATH_CROSS_ZONE': -1,
        }
        sma_signal_map = {
            'Golden Cross': 1, 'Death Cross': -1,
        }
        if 'Trend_Status' in df_features.columns:
            df_features['sma_trend_strength'] = df_features['Trend_Status'].map(trend_map).fillna(0).astype(float)
            df_features.drop(columns=['Trend_Status'], inplace=True, errors='ignore')
        else:
            df_features['sma_trend_strength'] = 0
        if 'SMA_Cross_Status' in df_features.columns:
            df_features['sma_cross_signal'] = df_features['SMA_Cross_Status'].map(cross_map).fillna(0).astype(float)
            df_features.drop(columns=['SMA_Cross_Status'], inplace=True, errors='ignore')
        else:
            df_features['sma_cross_signal'] = 0
        if 'sma_trade_signal' in df_features.columns:
            df_features['sma_trade_strength'] = df_features['sma_trade_signal'].map(sma_signal_map).fillna(0).astype(float)
            df_features.drop(columns=['sma_trade_signal'], inplace=True, errors='ignore')
        else:
            df_features['sma_trade_strength'] = 0
        
        # MACD crossover signal
        if 'macd_crossover_signal' in df_features.columns:
            df_features['macd_signal_strength'] = df_features['macd_crossover_signal'].map(signal_strength_map).fillna(0).astype(float)
            df_features.drop(columns=['macd_crossover_signal'], inplace=True, errors='ignore')
        else:
            df_features['macd_signal_strength'] = 0
        
        # Fibonacci signal
        if 'fib_trade_signal' in df_features.columns:
            df_features['fib_signal_strength'] = df_features['fib_trade_signal'].map(signal_strength_map).fillna(0).astype(float)
            df_features.drop(columns=['fib_trade_signal'], inplace=True, errors='ignore')
        else:
            df_features['fib_signal_strength'] = 0
        if 'fib_distance_pct' not in df_features.columns:
            df_features['fib_distance_pct'] = 0
        
        # Support/Resistance
        if 'pivot_status' in df_features.columns:
            df_features['sr_pivot_position'] = df_features['pivot_status'].map(position_map).fillna(0).astype(float)
            df_features.drop(columns=['pivot_status'], inplace=True, errors='ignore')
        else:
            df_features['sr_pivot_position'] = 0
        if 'sr_trade_signal' in df_features.columns:
            df_features['sr_signal_strength'] = df_features['sr_trade_signal'].map(signal_strength_map).fillna(0).astype(float)
            df_features.drop(columns=['sr_trade_signal'], inplace=True, errors='ignore')
        else:
            df_features['sr_signal_strength'] = 0
        for col in ['sr_distance_to_support_pct', 'sr_distance_to_resistance_pct']:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Candlestick Pattern signals
        if 'pattern_signal' in df_features.columns:
            df_features['pattern_signal_strength'] = df_features['pattern_signal'].map(signal_strength_map).fillna(0).astype(float)
            df_features.drop(columns=['pattern_signal'], inplace=True, errors='ignore')
        else:
            df_features['pattern_signal_strength'] = 0
        for col in ['has_doji', 'has_hammer', 'has_shooting_star',
                     'has_bullish_engulfing', 'has_bearish_engulfing',
                     'has_morning_star', 'has_evening_star']:
            if col not in df_features.columns:
                df_features[col] = 0
            else:
                df_features[col] = df_features[col].fillna(0).astype(float)
        
        # Time features
        df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
        df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
        df_features['month'] = df_features['trading_date'].dt.month
        
        # ================================================================
        # PHASE 3: Fundamental features from nasdaq_100_fundamentals
        # ================================================================
        
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
        
        # Sector encoding (use saved encoder for consistency with training)
        if 'sector' in df_features.columns and self.sector_encoder is not None:
            sector_values = df_features['sector'].fillna('Unknown')
            # Handle unseen sectors gracefully
            known_classes = set(self.sector_encoder.classes_)
            sector_values = sector_values.apply(lambda x: x if x in known_classes else 'Unknown')
            df_features['sector_encoded'] = self.sector_encoder.transform(sector_values)
        else:
            df_features['sector_encoded'] = 0
        
        # Phase 4: Market context — sector-specific ETF return mapping
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
                lambda s: SECTOR_TO_ETF.get(s, 'sp500_return_1d')
            )
            df_features['sector_etf_return_1d'] = df_features.apply(
                lambda row: row.get(row['sector_etf_return_1d'], 0) if pd.notna(row.get('sector_etf_return_1d')) else 0,
                axis=1
            )
        
        # ================================================================
        # PHASE 6: Sentiment-price divergence
        # Detects when sentiment and price direction disagree.
        # ================================================================
        if 'sector_sentiment_score' in df_features.columns:
            _price_dir = df_features.groupby('ticker')['close_price'].transform(
                lambda x: x.pct_change(5)
            )
            sent = df_features['sector_sentiment_score']
            price_sign = np.sign(_price_dir)
            sent_sign = np.sign(sent)
            df_features['sentiment_price_divergence'] = np.where(
                (price_sign != sent_sign) & (sent_sign != 0),
                np.abs(sent) * 2,
                0
            )
        else:
            df_features['sentiment_price_divergence'] = 0
        
        # Handle NaN values - use FORWARD fill only (bfill causes data leakage in time series)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        return df_features
    
    def add_enhanced_features(self, df):
        """Add enhanced technical indicators (MACD, SMA, EMA) - VECTORIZED for speed"""
        df_copy = df.copy()
        
        price_col = 'close_price'
        volume_col = 'volume'
        
        # VECTORIZED Moving Averages using transform (100x faster than apply)
        df_copy['sma_5'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df_copy['sma_10'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        df_copy['sma_20'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        df_copy['sma_50'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        
        # VECTORIZED Exponential Moving Averages
        df_copy['ema_5'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=5, min_periods=1).mean())
        df_copy['ema_10'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=10, min_periods=1).mean())
        df_copy['ema_20'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=20, min_periods=1).mean())
        df_copy['ema_50'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=50, min_periods=1).mean())
        
        # VECTORIZED MACD Calculation
        df_copy['ema_12'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=12, min_periods=1).mean())
        df_copy['ema_26'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.ewm(span=26, min_periods=1).mean())
        df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy.groupby('ticker')['macd'].transform(lambda x: x.ewm(span=9, min_periods=1).mean())
        df_copy['macd_histogram'] = df_copy['macd'] - df_copy['macd_signal']
        
        # Drop temporary columns
        df_copy = df_copy.drop(['ema_12', 'ema_26'], axis=1)
        
        # SMA 100/200 from DB (or calculate fallback — matches training pipeline)
        if 'sma_200' not in df_copy.columns or df_copy['sma_200'].isna().all():
            df_copy['sma_200'] = df_copy.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=200, min_periods=1).mean()
            )
        if 'sma_100' not in df_copy.columns or df_copy['sma_100'].isna().all():
            df_copy['sma_100'] = df_copy.groupby('ticker')[price_col].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )
        df_copy['price_vs_sma100'] = np.where(df_copy['sma_100'] > 0, df_copy[price_col] / df_copy['sma_100'], 1.0)
        df_copy['price_vs_sma200'] = np.where(df_copy['sma_200'] > 0, df_copy[price_col] / df_copy['sma_200'], 1.0)
        
        # VECTORIZED Price vs MA ratios (safe division)
        df_copy['price_vs_sma20'] = np.where(df_copy['sma_20'] > 0, df_copy[price_col] / df_copy['sma_20'], 1.0)
        df_copy['price_vs_sma50'] = np.where(df_copy['sma_50'] > 0, df_copy[price_col] / df_copy['sma_50'], 1.0)
        df_copy['price_vs_ema20'] = np.where(df_copy['ema_20'] > 0, df_copy[price_col] / df_copy['ema_20'], 1.0)
        
        # VECTORIZED MA relationships
        df_copy['sma20_vs_sma50'] = np.where(df_copy['sma_50'] > 0, df_copy['sma_20'] / df_copy['sma_50'], 1.0)
        df_copy['ema20_vs_ema50'] = np.where(df_copy['ema_50'] > 0, df_copy['ema_20'] / df_copy['ema_50'], 1.0)
        df_copy['sma5_vs_sma20'] = np.where(df_copy['sma_20'] > 0, df_copy['sma_5'] / df_copy['sma_20'], 1.0)
        
        # VECTORIZED Volume indicators
        df_copy['volume_sma_20'] = df_copy.groupby('ticker')[volume_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        df_copy['volume_sma_ratio'] = np.where(df_copy['volume_sma_20'] > 0, df_copy[volume_col] / df_copy['volume_sma_20'], 1.0)
        
        # VECTORIZED Momentum features
        df_copy['price_momentum_5'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x / x.shift(5))
        df_copy['price_momentum_10'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x / x.shift(10))
        
        # VECTORIZED Volatility features
        df_copy['price_volatility_10'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(window=10, min_periods=1).std())
        df_copy['price_volatility_20'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(window=20, min_periods=1).std())
        
        # VECTORIZED Trend strength
        df_copy['trend_strength_10'] = df_copy.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=10, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / y.std() if len(y) > 1 and y.std() != 0 else 0
            )
        )
        
        # ================================================================
        # NEW PHASE 2: Additional technical indicators for improved accuracy
        # ================================================================
        
        high_col = 'high_price'
        low_col = 'low_price'
        
        # --- Bollinger Bands (20-period, 2 std) ---
        bb_sma = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        bb_std = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).std())
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        df_copy['bollinger_pctb'] = np.where(bb_range > 0, (df_copy[price_col] - bb_lower) / bb_range, 0.5)
        df_copy['bollinger_bandwidth'] = np.where(bb_sma > 0, bb_range / bb_sma, 0)
        
        # --- Stochastic Oscillator (14-period) — use DB values if available, else calculate ---
        low_14 = df_copy.groupby('ticker')[low_col].transform(lambda x: x.rolling(window=14, min_periods=1).min())
        high_14 = df_copy.groupby('ticker')[high_col].transform(lambda x: x.rolling(window=14, min_periods=1).max())
        stoch_range = high_14 - low_14
        calc_k = np.where(stoch_range > 0, (df_copy[price_col] - low_14) / stoch_range * 100, 50)
        if 'db_stoch_k' in df_copy.columns and df_copy['db_stoch_k'].notna().any():
            df_copy['stochastic_k'] = df_copy['db_stoch_k'].fillna(pd.Series(calc_k, index=df_copy.index))
            calc_d = df_copy.groupby('ticker')['stochastic_k'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df_copy['stochastic_d'] = df_copy['db_stoch_d'].fillna(calc_d) if 'db_stoch_d' in df_copy.columns else calc_d
            df_copy['stochastic_momentum'] = df_copy['db_stoch_momentum'].fillna(0) if 'db_stoch_momentum' in df_copy.columns else 0
            df_copy.drop(columns=['db_stoch_k', 'db_stoch_d', 'db_stoch_momentum'], inplace=True, errors='ignore')
        else:
            df_copy['stochastic_k'] = calc_k
            df_copy['stochastic_d'] = df_copy.groupby('ticker')['stochastic_k'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df_copy['stochastic_momentum'] = df_copy['stochastic_k'] - df_copy['stochastic_d']
        
        # --- ATR (Average True Range, 14-period) ---
        prev_close = df_copy.groupby('ticker')[price_col].transform(lambda x: x.shift(1))
        high_low = df_copy[high_col] - df_copy[low_col]
        high_close = (df_copy[high_col] - prev_close).abs()
        low_close = (df_copy[low_col] - prev_close).abs()
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df_copy['_true_range'] = true_range
        df_copy['atr_14'] = df_copy.groupby('ticker')['_true_range'].transform(lambda x: x.rolling(window=14, min_periods=1).mean())
        df_copy['atr_ratio'] = np.where(df_copy[price_col] > 0, df_copy['atr_14'] / df_copy[price_col], 0)
        df_copy = df_copy.drop(['_true_range'], axis=1)
        
        # --- Normalized MACD (percentage of price, ticker-independent) ---
        df_copy['macd_normalized'] = np.where(df_copy[price_col] > 0, df_copy['macd'] / df_copy[price_col] * 100, 0)
        df_copy['macd_signal_normalized'] = np.where(df_copy[price_col] > 0, df_copy['macd_signal'] / df_copy[price_col] * 100, 0)
        df_copy['macd_histogram_normalized'] = np.where(df_copy[price_col] > 0, df_copy['macd_histogram'] / df_copy[price_col] * 100, 0)
        
        # --- Lagged returns (percentage changes at different lookback periods) ---
        df_copy['return_1d'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change(1))
        df_copy['return_2d'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change(2))
        df_copy['return_3d'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change(3))
        df_copy['return_5d'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change(5))
        df_copy['return_10d'] = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change(10))
        
        # --- RSI-Price divergence ---
        price_dir_5 = df_copy.groupby('ticker')[price_col].transform(lambda x: np.sign(x.pct_change(5)))
        rsi_dir_5 = df_copy.groupby('ticker')['RSI'].transform(lambda x: np.sign(x.diff(5)))
        df_copy['rsi_price_divergence'] = (price_dir_5 != rsi_dir_5).astype(int)
        
        # ================================================================
        # PHASE 3: Market Regime Detection
        # ================================================================
        
        # --- Regime: SMA Trend Direction (20-day slope normalized) ---
        sma20 = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        df_copy['regime_sma20_slope'] = df_copy.groupby('ticker')[price_col].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean().pct_change(5) * 100
        )
        
        # --- Regime: ADX-like trend strength (simplified) ---
        up_move = df_copy.groupby('ticker')[high_col].transform(lambda x: x.diff())
        down_move = -df_copy.groupby('ticker')[low_col].transform(lambda x: x.diff())
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        df_copy['_pos_dm'] = pos_dm
        df_copy['_neg_dm'] = neg_dm
        pos_dm_smooth = df_copy.groupby('ticker')['_pos_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        neg_dm_smooth = df_copy.groupby('ticker')['_neg_dm'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        dm_sum = pos_dm_smooth + neg_dm_smooth
        dx = np.where(dm_sum > 0, np.abs(pos_dm_smooth - neg_dm_smooth) / dm_sum * 100, 0)
        df_copy['_dx'] = dx
        df_copy['regime_adx'] = df_copy.groupby('ticker')['_dx'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df_copy = df_copy.drop(['_pos_dm', '_neg_dm', '_dx'], axis=1)
        
        # --- Regime: Volatility regime (current vol vs long-term vol) ---
        vol_short = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(10, min_periods=1).std())
        vol_long = df_copy.groupby('ticker')[price_col].transform(lambda x: x.pct_change().rolling(60, min_periods=1).std())
        df_copy['regime_vol_ratio'] = np.where(vol_long > 0, vol_short / vol_long, 1.0)
        
        # --- Regime: Mean reversion indicator (distance from 50-SMA / ATR) ---
        sma50 = df_copy.groupby('ticker')[price_col].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
        df_copy['regime_mean_reversion'] = np.where(
            df_copy['atr_14'] > 0,
            (df_copy[price_col] - sma50) / df_copy['atr_14'],
            0
        )
        
        # --- Regime: Trend consistency (% of last 20 days moving in overall direction) ---
        overall_dir = df_copy.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(20)))
        daily_dirs = df_copy.groupby('ticker')[price_col].transform(lambda x: np.sign(x.diff(1)))
        consistent = (daily_dirs == overall_dir).astype(float)
        df_copy['regime_trend_consistency'] = df_copy.groupby('ticker')[price_col].transform(
            lambda x: pd.Series(index=x.index, dtype=float)
        )
        # Calculate trend consistency per ticker
        consistent_series = consistent.copy()
        df_copy['regime_trend_consistency'] = consistent_series.groupby(df_copy['ticker']).transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        
        return df_copy
    
    def predict_signals(self, ticker=None, date=None, confidence_threshold=HIGH_CONFIDENCE_THRESHOLD):
        """Make trading signal predictions (ensemble, 5-day direction)"""
        # Get data (80 days lookback for technical indicator warm-up)
        df = self.get_latest_data(ticker, days_back=80)
        if df is None:
            return None
        # Engineer features
        df_features = self.engineer_features(df)
        if df_features is None:
            return None
        # Filter by date if specified
        if date:
            target_date = pd.to_datetime(date)
            df_features = df_features[df_features['trading_date'].dt.date == target_date.date()]
            if df_features.empty:
                safe_print(f"⚠️  No data found for date: {date}")
                return None
        # Get latest data for each ticker
        latest_data = df_features.groupby('ticker').tail(1).copy()
        if latest_data.empty:
            safe_print("⚠️  No data available for prediction")
            return None
        # Prepare features
        X = latest_data[self.feature_columns].copy()
        # Scale features
        X_scaled = self.scaler.transform(X)
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        # Create results DataFrame
        results = latest_data[['trading_date', 'ticker', 'company', 'close_price', 'RSI']].copy()
        results['predicted_signal'] = self.target_encoder.inverse_transform(predictions)
        results['confidence'] = probabilities.max(axis=1)
        # Up/Down probabilities (class order from encoder)
        up_idx = np.where(self.target_encoder.classes_ == 'Up')[0]
        down_idx = np.where(self.target_encoder.classes_ == 'Down')[0]
        if len(up_idx) == 1 and len(down_idx) == 1:
            results['up_probability'] = probabilities[:, up_idx[0]]
            results['down_probability'] = probabilities[:, down_idx[0]]
        else:
            results['up_probability'] = np.nan
            results['down_probability'] = np.nan
        # Add confidence flag
        results['high_confidence'] = results['confidence'] > confidence_threshold
        return results
    
    def format_prediction_output(self, results, show_all=False):
        """Format prediction results for display (Up/Down signals)"""
        if results is None or results.empty:
            return "No predictions available"
        output = []
        output.append("=" * 80)
        output.append(f"[TARGET] TRADING SIGNAL PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        # Filter high confidence predictions
        high_conf = results[results['high_confidence']]
        medium_conf = results[(results['confidence'] > MEDIUM_CONFIDENCE_THRESHOLD) & (results['confidence'] <= HIGH_CONFIDENCE_THRESHOLD)]
        low_conf = results[results['confidence'] <= MEDIUM_CONFIDENCE_THRESHOLD]
        # High confidence predictions
        if not high_conf.empty:
            output.append(f"\n[UP] HIGH CONFIDENCE PREDICTIONS (>{HIGH_CONFIDENCE_THRESHOLD:.0%})")
            output.append("-" * 50)
            for _, row in high_conf.iterrows():
                signal_emoji = "[UP]" if row['predicted_signal'] == 'Up' else "[DOWN]"
                output.append(f"{signal_emoji} {row['ticker']} ({row['company'][:20]})")
                output.append(f"   Signal: {row['predicted_signal']}")
                output.append(f"   Confidence: {row['confidence']:.1%}")
                output.append(f"   Up Prob: {row.get('up_probability', np.nan):.1%}")
                output.append(f"   Down Prob: {row.get('down_probability', np.nan):.1%}")
                output.append(f"   Close: ${row['close_price']:.2f}")
                output.append(f"   RSI: {row['RSI']:.1f}")
                output.append(f"   Date: {row['trading_date'].strftime('%Y-%m-%d')}")
                output.append("")
        # Medium confidence predictions
        if not medium_conf.empty and show_all:
            output.append(f"\n[MEDIUM] MEDIUM CONFIDENCE PREDICTIONS ({MEDIUM_CONFIDENCE_THRESHOLD:.0%}-{HIGH_CONFIDENCE_THRESHOLD:.0%})")
            output.append("-" * 50)
            for _, row in medium_conf.iterrows():
                signal_emoji = "[UP]" if row['predicted_signal'] == 'Up' else "[DOWN]"
                output.append(f"{signal_emoji} {row['ticker']}: {row['predicted_signal']} ({row['confidence']:.1%})")
        # Summary statistics
        total_predictions = len(results)
        high_conf_count = len(high_conf)
        up_signals = len(results[results['predicted_signal'] == 'Up'])
        down_signals = len(results[results['predicted_signal'] == 'Down'])
        output.append("\n[DATA] PREDICTION SUMMARY")
        output.append("-" * 30)
        output.append(f"Total Predictions: {total_predictions}")
        output.append(f"High Confidence: {high_conf_count} ({high_conf_count/total_predictions:.1%})")
        output.append(f"Up Signals: {up_signals}")
        output.append(f"Down Signals: {down_signals}")
        output.append(f"Average Confidence: {results['confidence'].mean():.1%}")
        return "\n".join(output)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Trading Signal Prediction System')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)')
    parser.add_argument('--batch', action='store_true', help='Run batch predictions for all stocks')
    parser.add_argument('--confidence', type=float, default=HIGH_CONFIDENCE_THRESHOLD, help=f'Confidence threshold (default: {HIGH_CONFIDENCE_THRESHOLD:.1f})')
    parser.add_argument('--show-all', action='store_true', help='Show all predictions including medium/low confidence')
    
    args = parser.parse_args()    
    # Initialize predictor
    predictor = TradingSignalPredictor()
      # Make predictions
    if args.batch:
        safe_print("🔄 Running batch predictions for all available stocks...")
        results = predictor.predict_signals(confidence_threshold=args.confidence)
    else:
        results = predictor.predict_signals(
            ticker=args.ticker,
            date=args.date,
            confidence_threshold=args.confidence
        )
    
    # Display results
    if results is not None:
        output = predictor.format_prediction_output(results, show_all=args.show_all)
        print(output)
        
        # Return high confidence count for exit code
        high_conf_count = len(results[results['high_confidence']])
        safe_print(f"\n✅ Analysis complete. Found {high_conf_count} high-confidence signals.")
        
    else:
        safe_print("❌ No predictions could be generated.")
        sys.exit(1)

if __name__ == "__main__":
    main()
