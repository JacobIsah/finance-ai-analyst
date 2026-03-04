"""
ML Models for Transaction Classification and Anomaly Detection

This module implements:
1. Rule-based + ML hybrid transaction classifier
2. Anomaly detection using Isolation Forest and statistical methods

DEEP DIVE LESSON: ML Model Selection and Evaluation
---------------------------------------------------
For financial transaction analysis, we face interesting ML challenges:

1. Classification: Categorizing transactions is tricky because:
   - Training data is limited (user's own history)
   - Categories are domain-specific
   - Solution: Hybrid approach - rule-based for high-confidence, ML for ambiguous
   
2. Anomaly Detection: Identifying unusual spending requires:
   - Understanding "normal" patterns per user
   - Multiple definitions of "anomaly" (amount, timing, category)
   - Solution: Ensemble of Isolation Forest + statistical Z-score

Why Isolation Forest for anomaly detection?
- Works well with small datasets (unlike deep learning)
- No assumption about data distribution
- Fast prediction time (important for real-time UI)
- Handles high-dimensional data naturally
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import re

from config import SPENDING_CATEGORIES, ANOMALY_CONTAMINATION, ANOMALY_ZSCORE_THRESHOLD


class TransactionClassifier:
    """
    Hybrid classifier combining rule-based keyword matching with 
    pattern learning for transaction categorization.
    """
    
    def __init__(self, categories: Dict[str, List[str]] = None):
        self.categories = categories or SPENDING_CATEGORIES
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for all category keywords."""
        self.patterns = {}
        for category, keywords in self.categories.items():
            if keywords:  # Skip empty lists
                pattern = "|".join(re.escape(kw) for kw in keywords)
                self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def classify_single(self, description: str, amount: float) -> Tuple[str, float]:
        """
        Classify a single transaction.
        
        Returns:
            - category: string
            - confidence: float (0-1)
        """
        description = str(description).lower()
        
        # Rule-based matching with confidence scores
        matches = []
        for category, pattern in self.patterns.items():
            match = pattern.search(description)
            if match:
                # Confidence based on match length relative to description
                match_len = len(match.group())
                confidence = min(0.9, 0.5 + (match_len / len(description)) * 0.5)
                matches.append((category, confidence))
        
        # If we have matches, return the highest confidence one
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0]
        
        # Fallback heuristics based on amount
        if amount > 0:
            # Positive amounts are likely income
            return ("income", 0.6)
        
        # Default to "other" for unclassified expenses
        return ("other", 0.3)
    
    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all transactions in a DataFrame.
        
        Adds columns:
        - predicted_category: string
        - category_confidence: float
        """
        results = df.apply(
            lambda row: self.classify_single(row["description"], row["amount"]),
            axis=1
        )
        
        df["predicted_category"] = results.apply(lambda x: x[0])
        df["category_confidence"] = results.apply(lambda x: x[1])
        
        return df


class AnomalyDetector:
    """
    Ensemble anomaly detector combining:
    1. Isolation Forest (ML-based, catches multivariate anomalies)
    2. Statistical Z-score (catches simple outliers in amount)
    3. Category-specific thresholds (catches unusual spending in categories)
    
    DEEP DIVE: Why an ensemble approach?
    - Isolation Forest: Good at finding transactions that are unusual
      in multiple dimensions (e.g., large grocery purchase on a Tuesday at 3am)
    - Z-score: Simple but effective for catching extremely large transactions
    - Category thresholds: Domain-aware (a $200 grocery bill might be normal,
      but $200 at a coffee shop is suspicious)
    """
    
    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.category_stats = {}
        self.global_stats = {}
        self.is_fitted = False
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for Isolation Forest."""
        feature_cols = [
            "amount_abs",
            "day_of_week", 
            "day_of_month",
            "is_weekend",
            "description_length"
        ]
        
        # Only use columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]
        return df[available_cols].values
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the anomaly detector on historical transaction data.
        
        This learns:
        - Normal patterns via Isolation Forest
        - Statistical baselines (mean, std) per category and globally
        """
        # Fit Isolation Forest
        features = self._prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(features_scaled)
        
        # Compute statistical baselines
        expenses = df[df["amount"] < 0]["amount"].abs()
        if len(expenses) > 0:
            self.global_stats = {
                "mean": expenses.mean(),
                "std": expenses.std(),
                "median": expenses.median(),
                "p95": expenses.quantile(0.95)
            }
        
        # Category-specific stats (if categories exist)
        if "predicted_category" in df.columns:
            for category in df["predicted_category"].unique():
                cat_expenses = df[
                    (df["predicted_category"] == category) & 
                    (df["amount"] < 0)
                ]["amount"].abs()
                
                if len(cat_expenses) >= 3:  # Need minimum samples
                    self.category_stats[category] = {
                        "mean": cat_expenses.mean(),
                        "std": cat_expenses.std(),
                        "p95": cat_expenses.quantile(0.95)
                    }
        
        self.is_fitted = True
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in transactions.
        
        Adds columns:
        - is_anomaly: boolean
        - anomaly_score: float (higher = more anomalous)
        - anomaly_reasons: list of strings explaining why
        """
        if not self.is_fitted:
            self.fit(df)
        
        # Initialize anomaly columns
        df["is_anomaly"] = False
        df["anomaly_score"] = 0.0
        df["anomaly_reasons"] = [[] for _ in range(len(df))]
        
        # 1. Isolation Forest scoring
        features = self._prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Isolation Forest: -1 = anomaly, 1 = normal
        if_predictions = self.isolation_forest.predict(features_scaled)
        if_scores = -self.isolation_forest.score_samples(features_scaled)  # Higher = more anomalous
        
        # Normalize IF scores to 0-1 range
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-10)
        
        # 2. Statistical Z-score for amounts
        z_scores = np.zeros(len(df))
        if self.global_stats.get("std", 0) > 0:
            expenses_mask = df["amount"] < 0
            z_scores[expenses_mask] = (
                df.loc[expenses_mask, "amount"].abs() - self.global_stats["mean"]
            ) / self.global_stats["std"]
        
        # 3. Combine signals
        for idx in range(len(df)):
            reasons = []
            score = 0.0
            
            # Isolation Forest signal
            if if_predictions[idx] == -1:
                score += 0.4
                reasons.append("Unusual pattern detected by ML model")
            score += if_scores_norm[idx] * 0.3
            
            # Z-score signal
            if z_scores[idx] > ANOMALY_ZSCORE_THRESHOLD:
                score += 0.3
                reasons.append(f"Amount is {z_scores[idx]:.1f} standard deviations above average")
            
            # Category-specific check
            if "predicted_category" in df.columns:
                category = df.iloc[idx]["predicted_category"]
                amount = abs(df.iloc[idx]["amount"])
                
                if category in self.category_stats:
                    cat_p95 = self.category_stats[category]["p95"]
                    if amount > cat_p95 * 1.5:
                        score += 0.2
                        reasons.append(f"Unusually high for {category} (typical max: ${cat_p95:.2f})")
            
            # Determine if anomaly
            df.at[df.index[idx], "anomaly_score"] = min(score, 1.0)
            df.at[df.index[idx], "anomaly_reasons"] = reasons
            df.at[df.index[idx], "is_anomaly"] = score >= 0.5
        
        return df


def analyze_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main analysis pipeline - classifies and detects anomalies.
    
    This is the main entry point that combines:
    1. Transaction classification
    2. Anomaly detection
    """
    # Step 1: Classify transactions
    classifier = TransactionClassifier()
    df = classifier.classify_batch(df)
    
    # Step 2: Detect anomalies
    detector = AnomalyDetector()
    df = detector.detect(df)
    
    return df


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics by category.
    """
    if "predicted_category" not in df.columns:
        return pd.DataFrame()
    
    # Only look at expenses for spending summary
    expenses = df[df["amount"] < 0].copy()
    expenses["amount_abs"] = expenses["amount"].abs()
    
    summary = expenses.groupby("predicted_category").agg({
        "amount_abs": ["sum", "mean", "count"],
        "is_anomaly": "sum"
    }).round(2)
    
    summary.columns = ["total_spent", "avg_transaction", "num_transactions", "anomalies"]
    summary = summary.sort_values("total_spent", ascending=False)
    
    return summary


def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of detected anomalies for reporting.
    """
    anomalies = df[df["is_anomaly"] == True].copy()
    
    if len(anomalies) == 0:
        return pd.DataFrame()
    
    return anomalies[[
        "date", "description", "amount", 
        "predicted_category", "anomaly_score", "anomaly_reasons"
    ]].sort_values("anomaly_score", ascending=False)
