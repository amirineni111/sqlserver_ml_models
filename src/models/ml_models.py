"""
Machine learning models for SQL Server data analysis.

This module provides classes and functions for building, training, and evaluating
machine learning models using data from SQL Server databases.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import pickle
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class MLModelManager:
    """
    A comprehensive machine learning model manager for SQL Server data.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the ML Model Manager.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type.lower()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        
        # Initialize default models based on task type
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize default models based on task type."""
        if self.task_type == 'classification':
            self.models = {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier(n_neighbors=5),
                'svm': SVC(random_state=42, probability=True)
            }
        elif self.task_type == 'regression':
            self.models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'knn': KNeighborsRegressor(n_neighbors=5),
                'svm': SVR()
            }
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
    
    def add_model(self, name: str, model: Any):
        """
        Add a custom model to the manager.
        
        Args:
            name: Name for the model
            model: Scikit-learn compatible model
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models and evaluate using cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model evaluation results
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Fit the model
                model.fit(X_train, y_train)
                
                # Cross-validation scoring
                if self.task_type == 'classification':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    results[name] = {
                        'cv_mean_accuracy': cv_scores.mean(),
                        'cv_std_accuracy': cv_scores.std()
                    }
                else:  # regression
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                    results[name] = {
                        'cv_mean_r2': cv_scores.mean(),
                        'cv_std_r2': cv_scores.std()
                    }
                
                logger.info(f"Completed training {name}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        self._find_best_model()
        return results
    
    def evaluate_model(
        self, 
        model_name: str, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a specific model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        else:  # regression
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        logger.info(f"Evaluation completed for {model_name}")
        return metrics
    
    def _find_best_model(self):
        """Find the best performing model based on cross-validation results."""
        if not self.results:
            return
        
        best_score = -np.inf
        best_name = None
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            if self.task_type == 'classification':
                score = result.get('cv_mean_accuracy', -np.inf)
            else:
                score = result.get('cv_mean_r2', -np.inf)
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models.get(best_name)
        
        if best_name:
            logger.info(f"Best model identified: {best_name} (score: {best_score:.4f})")
    
    def hyperparameter_tuning(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: Dict[str, List],
        cv_folds: int = 5,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with tuning results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        base_model = self.models[model_name]
        
        # Choose scoring metric
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv_folds, 
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[f"{model_name}_tuned"] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"Hyperparameter tuning completed for {model_name}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for better sklearn model support
        joblib.dump(model, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str, model_name: str):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            model_name: Name to assign to the loaded model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        self.models[model_name] = model
        logger.info(f"Model loaded from {filepath} as {model_name}")
    
    def plot_model_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot comparison of model performance.
        
        Args:
            figsize: Figure size for the plot
        """
        if not self.results:
            raise ValueError("No model results available. Train models first.")
        
        # Prepare data for plotting
        model_names = []
        scores = []
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            model_names.append(name)
            
            if self.task_type == 'classification':
                scores.append(result.get('cv_mean_accuracy', 0))
            else:
                scores.append(result.get('cv_mean_r2', 0))
        
        # Create plot
        plt.figure(figsize=figsize)
        bars = plt.bar(model_names, scores)
        
        # Highlight best model
        if self.best_model_name:
            best_index = model_names.index(self.best_model_name)
            bars[best_index].set_color('red')
        
        plt.title(f'Model Performance Comparison ({self.task_type.title()})')
        plt.ylabel('Accuracy' if self.task_type == 'classification' else 'R² Score')
        plt.xlabel('Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def get_default_hyperparameters() -> Dict[str, Dict[str, List]]:
    """
    Get default hyperparameter grids for common models.
    
    Returns:
        Dictionary with parameter grids for different models
    """
    param_grids = {
        'random_forest_classification': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'random_forest_regression': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'logistic_regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'lbfgs']
        },
        'svm_classification': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    }
    
    return param_grids
