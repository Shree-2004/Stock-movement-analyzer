"""
Machine Learning Model Module - Train and evaluate stock prediction models.

This module provides functions to:
1. Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
2. Compare performance with/without sentiment features
3. Generate predictions and evaluate results
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import os
from src.config import MODELS_DIR, RANDOM_STATE, N_SPLITS


class StockPredictionModel:
    """
    Wrapper class for stock prediction models.
    """
    
    def __init__(self, model_type='logistic', random_state=RANDOM_STATE):
        """
        Initialize model.
        
        Args:
            model_type (str): 'logistic', 'random_forest', or 'xgboost'
            random_state (int): Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.is_trained = False
    
    def _create_model(self):
        """
        Create the specified model.
        
        Returns:
            sklearn/xgboost model instance
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                scale_pos_weight=1,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        print(f"\n🔧 Training {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.model_type.upper()} training complete")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance sorted by importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type == 'logistic':
            # For logistic regression, use absolute coefficients
            importance = np.abs(self.model.coef_[0])
        else:
            # For tree-based models
            importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filename=None):
        """
        Save model to disk.
        
        Args:
            filename (str): Filename to save (default: model_type.pkl)
        """
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(self.model, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filename=None):
        """
        Load model from disk.
        
        Args:
            filename (str): Filename to load
        """
        if filename is None:
            filename = f"{self.model_type}_model.pkl"
        
        filepath = os.path.join(MODELS_DIR, filename)
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"📂 Model loaded from {filepath}")


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    
    Args:
        model (StockPredictionModel): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test labels
        model_name (str): Name for display
        
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 {model_name} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1-Score:  {f1:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Down  Up")
    print(f"Actual Down  [[{cm[0,0]:4d} {cm[0,1]:4d}]]")
    print(f"Actual Up    [[{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    # Classification report
    print(f"\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def compare_models_with_without_sentiment(stock_df, news_df, test_size=0.3):
    """
    Compare model performance with and without sentiment features.
    
    Args:
        stock_df (pd.DataFrame): Stock data
        news_df (pd.DataFrame): News data
        test_size (float): Test set proportion
        
    Returns:
        dict: Comparison results
    """
    from src.features import create_feature_matrix, prepare_train_test_split
    
    print("\n" + "="*60)
    print("🔬 COMPARING MODELS: WITH vs WITHOUT SENTIMENT")
    print("="*60)
    
    results = {}
    
    # Test WITHOUT sentiment
    print("\n" + "="*60)
    print("📊 EXPERIMENT 1: WITHOUT Sentiment Features")
    print("="*60)
    
    features_no_sent, target_no_sent, cols_no_sent = create_feature_matrix(
        stock_df, news_df, include_sentiment=False
    )
    
    X_train_no, X_test_no, y_train_no, y_test_no, scaler_no = prepare_train_test_split(
        features_no_sent, target_no_sent, test_size
    )
    
    # Train models WITHOUT sentiment
    models_no_sent = {}
    for model_type in ['logistic', 'random_forest', 'xgboost']:
        model = StockPredictionModel(model_type)
        model.train(X_train_no, y_train_no)
        eval_results = evaluate_model(model, X_test_no, y_test_no, 
                                      f"{model_type.upper()} (No Sentiment)")
        models_no_sent[model_type] = {
            'model': model,
            'results': eval_results
        }
    
    results['without_sentiment'] = models_no_sent
    
    # Test WITH sentiment
    print("\n\n" + "="*60)
    print("📊 EXPERIMENT 2: WITH Sentiment Features")
    print("="*60)
    
    features_with_sent, target_with_sent, cols_with_sent = create_feature_matrix(
        stock_df, news_df, include_sentiment=True
    )
    
    X_train_with, X_test_with, y_train_with, y_test_with, scaler_with = prepare_train_test_split(
        features_with_sent, target_with_sent, test_size
    )
    
    # Train models WITH sentiment
    models_with_sent = {}
    for model_type in ['logistic', 'random_forest', 'xgboost']:
        model = StockPredictionModel(model_type)
        model.train(X_train_with, y_train_with)
        eval_results = evaluate_model(model, X_test_with, y_test_with,
                                      f"{model_type.upper()} (With Sentiment)")
        models_with_sent[model_type] = {
            'model': model,
            'results': eval_results
        }
    
    results['with_sentiment'] = models_with_sent
    
    # Print comparison summary
    print("\n\n" + "="*60)
    print("📈 COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': [],
        'Without Sentiment': [],
        'With Sentiment': [],
        'Improvement': []
    })
    
    for model_type in ['logistic', 'random_forest', 'xgboost']:
        acc_without = models_no_sent[model_type]['results']['accuracy']
        acc_with = models_with_sent[model_type]['results']['accuracy']
        improvement = acc_with - acc_without
        
        comparison_df = pd.concat([comparison_df, pd.DataFrame({
            'Model': [model_type.upper()],
            'Without Sentiment': [f"{acc_without:.4f}"],
            'With Sentiment': [f"{acc_with:.4f}"],
            'Improvement': [f"{improvement:+.4f}"]
        })], ignore_index=True)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Determine best model
    best_model_type = max(models_with_sent.keys(), 
                         key=lambda k: models_with_sent[k]['results']['accuracy'])
    best_accuracy = models_with_sent[best_model_type]['results']['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_type.upper()} with {best_accuracy:.4f} accuracy")
    
    results['comparison_df'] = comparison_df
    results['best_model'] = best_model_type
    
    return results


if __name__ == '__main__':
    # Test ML models
    from src.data_loader import load_stock_and_news
    
    print("="*60)
    print("🧪 TESTING ML MODELS")
    print("="*60)
    
    ticker = 'AAPL'
    start = '2024-01-01'
    end = '2024-12-31'
    
    # Load data
    stock_df, news_df = load_stock_and_news(ticker, start, end)
    
    if not stock_df.empty:
        # Run comparison
        results = compare_models_with_without_sentiment(stock_df, news_df)
        
        print("\n" + "="*60)
        print("✅ ML MODEL TESTING COMPLETE")
        print("="*60)
    else:
        print("❌ Could not load data for testing")
