"""
Continue ML Model Analysis
Execute the remaining analysis from the notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from datetime import datetime

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib

# Set up environment
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

# Database connection
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def load_analysis_results():
    """Load the results from model training"""
    try:
        # Try to load saved results if they exist
        with open('data/model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print("📁 Loaded existing model results")
        return results
    except FileNotFoundError:
        print("🔄 Re-running model analysis...")
        return run_complete_analysis()

def run_complete_analysis():
    """Run complete model analysis"""
    print("=== LOADING DATA ===")
    
    # Load exploration results
    try:
        with open('data/exploration_results.pkl', 'rb') as f:
            exploration_results = pickle.load(f)
        target_column = exploration_results['target_column']
        use_class_weights = True  # Based on previous analysis
    except FileNotFoundError:
        target_column = 'rsi_trade_signal'
        use_class_weights = True
    
    # Load data from SQL Server
    db = SQLServerConnection()
    
    basic_query = """
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
    WHERE h.trading_date >= '2024-01-01'
    ORDER BY h.trading_date DESC, h.ticker
    """
    
    df = db.execute_query(basic_query)
    
    # Feature Engineering
    print("=== FEATURE ENGINEERING ===")
    df_features = df.copy()
    
    # Calculate features manually
    df_features['daily_volatility'] = ((df_features['high_price'] - df_features['low_price']) / df_features['close_price']) * 100
    df_features['daily_return'] = ((df_features['close_price'] - df_features['open_price']) / df_features['open_price']) * 100
    df_features['volume_millions'] = df_features['volume'] / 1000000.0
    
    # Additional features
    df_features['price_range'] = df_features['high_price'] - df_features['low_price']
    df_features['price_position'] = (df_features['close_price'] - df_features['low_price']) / df_features['price_range']
    df_features['gap'] = df_features['open_price'] - df_features['close_price'].shift(1)
    df_features['volume_price_trend'] = df_features['volume'] * df_features['daily_return']
    df_features['rsi_oversold'] = (df_features['RSI'] < 30).astype(int)
    df_features['rsi_overbought'] = (df_features['RSI'] > 70).astype(int)
    df_features['rsi_momentum'] = df_features['RSI'].diff()
    
    # Time features
    df_features['trading_date'] = pd.to_datetime(df_features['trading_date'])
    df_features['day_of_week'] = df_features['trading_date'].dt.dayofweek
    df_features['month'] = df_features['trading_date'].dt.month
    
    # Handle NaN values
    df_features = df_features.fillna(method='bfill').fillna(0)
    
    print(f"Features created: {df_features.shape}")
    
    # Prepare ML dataset
    print("=== PREPARING ML DATASET ===")
    exclude_cols = ['trading_date', 'ticker', 'company', target_column]
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols].copy()
    y = df_features[target_column].copy()
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Time-aware split
    print("=== TRAIN-TEST SPLIT ===")
    date_sorted_idx = df_features['trading_date'].argsort()
    X_sorted = X.iloc[date_sorted_idx]
    y_sorted = y_encoded[date_sorted_idx]
    
    split_idx = int(0.8 * len(X_sorted))
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted[:split_idx]
    y_test = y_sorted[split_idx:]
    
    # Feature scaling
    print("=== FEATURE SCALING ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Model training
    print("=== MODEL TRAINING ===")
    class_weight_param = 'balanced' if use_class_weights else None
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight=class_weight_param, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            class_weight=class_weight_param, random_state=42, max_iter=1000
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100, max_depth=10, class_weight=class_weight_param,
            random_state=42, n_jobs=-1
        )
    }
    
    model_results = {}
    trained_models = {}
    cv_splitter = TimeSeriesSplit(n_splits=3)
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train_scaled_df, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled_df)
            y_pred_proba = model.predict_proba(X_test_scaled_df)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled_df, y_train, 
                                      cv=cv_splitter, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            model_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            trained_models[model_name] = model
            
            print(f"  F1-Score: {f1:.3f}, CV: {cv_mean:.3f} (±{cv_std:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Package results
    results = {
        'model_results': model_results,
        'trained_models': trained_models,
        'target_encoder': target_encoder,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'X_test': X_test_scaled_df,
        'y_test': y_test,
        'X_train': X_train_scaled_df,
        'y_train': y_train
    }
    
    # Save results
    with open('data/model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

def analyze_models(results):
    """Analyze and compare models"""
    print("=== MODEL COMPARISON ===\n")
    
    model_results = results['model_results']
    trained_models = results['trained_models']
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, res in model_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1-Score': res['f1_score'],
            'CV Mean': res['cv_mean'],
            'CV Std': res['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data).round(4)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_results = model_results[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    
    return best_model_name, best_model, best_results, comparison_df

def detailed_analysis(best_model_name, best_model, best_results, results):
    """Perform detailed analysis of the best model"""
    print(f"\n=== DETAILED ANALYSIS: {best_model_name.upper()} ===\n")
    
    y_test = results['y_test']
    y_pred_best = best_results['predictions']
    target_encoder = results['target_encoder']
    
    # Classification report
    print("Classification Report:")
    class_names = target_encoder.classes_
    report = classification_report(y_test, y_pred_best, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Print formatted report
    for class_name in class_names:
        metrics = report[class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1-score']:.3f}")
        print(f"  Support:   {metrics['support']:,}")
    
    # Overall metrics
    print(f"\nOverall:")
    print(f"  Accuracy:     {report['accuracy']:.3f}")
    print(f"  Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted F1:  {report['weighted avg']['f1-score']:.3f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature Importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print("\n=== FEATURE IMPORTANCE ===")
        feature_names = results['feature_columns']
        importances = best_model.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top 15 Feature Importances - {best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save the best model
    model_path = f'data/best_model_{best_model_name.lower().replace(" ", "_")}.joblib'
    joblib.dump(best_model, model_path)
    print(f"\n💾 Best model saved to: {model_path}")
    
    # Save scaler
    joblib.dump(results['scaler'], 'data/scaler.joblib')
    joblib.dump(results['target_encoder'], 'data/target_encoder.joblib')
    print("💾 Scaler and encoder saved")
    
    return feature_importance_df if hasattr(best_model, 'feature_importances_') else None

def trading_performance_analysis(best_results, results):
    """Analyze trading performance"""
    print("\n=== TRADING PERFORMANCE ANALYSIS ===")
    
    y_test = results['y_test']
    y_pred = best_results['predictions']
    y_proba = best_results['probabilities']
    target_encoder = results['target_encoder']
    
    # Trading signals mapping
    # 0 = Overbought (Sell), 1 = Oversold (Buy)
    
    # Calculate trading metrics
    buy_signals_actual = (y_test == 1).sum()
    sell_signals_actual = (y_test == 0).sum()
    buy_signals_predicted = (y_pred == 1).sum()
    sell_signals_predicted = (y_pred == 0).sum()
    
    # Correct predictions
    correct_buys = ((y_test == 1) & (y_pred == 1)).sum()
    correct_sells = ((y_test == 0) & (y_pred == 0)).sum()
    
    print(f"Trading Signal Analysis:")
    print(f"  Actual Buy Signals:     {buy_signals_actual:,}")
    print(f"  Predicted Buy Signals:  {buy_signals_predicted:,}")
    print(f"  Correct Buy Predictions: {correct_buys:,} ({correct_buys/buy_signals_actual*100:.1f}%)")
    print()
    print(f"  Actual Sell Signals:     {sell_signals_actual:,}")
    print(f"  Predicted Sell Signals:  {sell_signals_predicted:,}")
    print(f"  Correct Sell Predictions: {correct_sells:,} ({correct_sells/sell_signals_actual*100:.1f}%)")
    
    # Prediction confidence analysis
    print(f"\n=== PREDICTION CONFIDENCE ===")
    max_proba = y_proba.max(axis=1)
    high_confidence = (max_proba > 0.7).sum()
    medium_confidence = ((max_proba > 0.6) & (max_proba <= 0.7)).sum()
    low_confidence = (max_proba <= 0.6).sum()
    
    print(f"High Confidence (>70%):   {high_confidence:,} ({high_confidence/len(y_pred)*100:.1f}%)")
    print(f"Medium Confidence (60-70%): {medium_confidence:,} ({medium_confidence/len(y_pred)*100:.1f}%)")
    print(f"Low Confidence (<=60%):   {low_confidence:,} ({low_confidence/len(y_pred)*100:.1f}%)")
    
    # Performance by confidence level
    high_conf_mask = max_proba > 0.7
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
        print(f"\nHigh Confidence Predictions Accuracy: {high_conf_accuracy:.3f}")

if __name__ == "__main__":
    print("🚀 Starting ML Model Analysis...")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load or run analysis
    results = load_analysis_results()
    
    # Compare models
    best_model_name, best_model, best_results, comparison_df = analyze_models(results)
    
    # Detailed analysis
    feature_importance = detailed_analysis(best_model_name, best_model, best_results, results)
    
    # Trading performance
    trading_performance_analysis(best_results, results)
    
    print("\n✅ Analysis Complete!")
    print("\n📊 Summary:")
    print(f"   Best Model: {best_model_name}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"\n📁 Results saved in:")
    print("   - data/model_results.pkl")
    print("   - data/best_model_*.joblib")
    print("   - reports/confusion_matrix.png")
    if feature_importance is not None:
        print("   - reports/feature_importance.png")
