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
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """Initialize the predictor with saved model artifacts"""
        
        # Default paths (updated for rebalanced model)
        self.model_path = model_path or 'data/best_model_extra_trees.joblib'
        self.scaler_path = scaler_path or 'data/scaler.joblib'
        self.encoder_path = encoder_path or 'data/target_encoder.joblib'
        
        # Load model artifacts
        self.load_model_artifacts()
        
        # Database connection
        self.db = SQLServerConnection()
        
        # Feature columns (must match training)
        self.feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 
            'RSI', 'daily_volatility', 'daily_return', 'volume_millions',
            'price_range', 'price_position', 'gap', 'volume_price_trend',
            'rsi_oversold', 'rsi_overbought', 'rsi_momentum',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_ema20',
            'sma20_vs_sma50', 'ema20_vs_ema50', 'sma5_vs_sma20',
            'volume_sma_20', 'volume_sma_ratio',
            'price_momentum_5', 'price_momentum_10',
            'price_volatility_10', 'price_volatility_20',
            'trend_strength_10',
            'day_of_week', 'month'
        ]
    
    def load_model_artifacts(self):
        """Load the trained model, scaler, and encoder"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.target_encoder = joblib.load(self.encoder_path)
            safe_print("✅ Model artifacts loaded successfully")
            
            # Get class names
            self.class_names = self.target_encoder.classes_
            safe_print(f"📊 Target classes: {list(self.class_names)}")
            
        except FileNotFoundError as e:
            safe_print(f"❌ Error loading model artifacts: {e}")
            print("Please ensure the model has been trained and saved.")
            sys.exit(1)
    
    def get_latest_data(self, ticker=None, days_back=5):
        """Fetch latest stock data for prediction"""
        
        if ticker:
            ticker_filter = f"AND h.ticker = '{ticker}'"
        else:
            ticker_filter = ""
        
        query = f"""
        SELECT TOP 1000
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
        ORDER BY h.trading_date DESC, h.ticker        """
        
        try:
            df = self.db.execute_query(query)
            if df.empty:
                safe_print(f"⚠️  No data found for ticker: {ticker}")
                return None
            return df
        except Exception as e:
            safe_print(f"❌ Error fetching data: {e}")
            return None
    
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
        
        return df_copy
    
    def predict_signals(self, ticker=None, date=None, confidence_threshold=0.7):
        """Make trading signal predictions"""
        
        # Get data
        df = self.get_latest_data(ticker, days_back=10)
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
        results['sell_probability'] = probabilities[:, 0]  # Overbought (Sell)
        results['buy_probability'] = probabilities[:, 1]   # Oversold (Buy)
        
        # Add confidence flag
        results['high_confidence'] = results['confidence'] > confidence_threshold
        
        return results
    
    def format_prediction_output(self, results, show_all=False):
        """Format prediction results for display"""
        if results is None or results.empty:
            return "No predictions available"
        
        output = []
        output.append("=" * 80)
        output.append(f"[TARGET] TRADING SIGNAL PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        # Filter high confidence predictions
        high_conf = results[results['high_confidence']]
        medium_conf = results[(results['confidence'] > 0.6) & (results['confidence'] <= 0.7)]
        low_conf = results[results['confidence'] <= 0.6]
        
        # High confidence predictions
        if not high_conf.empty:
            output.append("\n[BUY] HIGH CONFIDENCE PREDICTIONS (>70%)")
            output.append("-" * 50)
            for _, row in high_conf.iterrows():
                signal_emoji = "[SELL]" if "Sell" in row['predicted_signal'] else "[BUY]"
                output.append(f"{signal_emoji} {row['ticker']} ({row['company'][:20]})")
                output.append(f"   Signal: {row['predicted_signal']}")
                output.append(f"   Confidence: {row['confidence']:.1%}")
                output.append(f"   Close: ${row['close_price']:.2f}")
                output.append(f"   RSI: {row['RSI']:.1f}")
                output.append(f"   Date: {row['trading_date'].strftime('%Y-%m-%d')}")
                output.append("")
        
        # Medium confidence predictions
        if not medium_conf.empty and show_all:
            output.append("\n[MEDIUM] MEDIUM CONFIDENCE PREDICTIONS (60-70%)")
            output.append("-" * 50)
            for _, row in medium_conf.iterrows():
                signal_emoji = "[SELL]" if "Sell" in row['predicted_signal'] else "[BUY]"
                output.append(f"{signal_emoji} {row['ticker']}: {row['predicted_signal']} ({row['confidence']:.1%})")
        
        # Summary statistics
        total_predictions = len(results)
        high_conf_count = len(high_conf)
        buy_signals = len(results[results['predicted_signal'].str.contains('Buy')])
        sell_signals = len(results[results['predicted_signal'].str.contains('Sell')])
        
        output.append("\n[DATA] PREDICTION SUMMARY")
        output.append("-" * 30)
        output.append(f"Total Predictions: {total_predictions}")
        output.append(f"High Confidence: {high_conf_count} ({high_conf_count/total_predictions:.1%})")
        output.append(f"Buy Signals: {buy_signals}")
        output.append(f"Sell Signals: {sell_signals}")
        output.append(f"Average Confidence: {results['confidence'].mean():.1%}")
        
        return "\n".join(output)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Trading Signal Prediction System')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--date', type=str, help='Date for prediction (YYYY-MM-DD)')
    parser.add_argument('--batch', action='store_true', help='Run batch predictions for all stocks')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
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
