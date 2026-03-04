"""
Data Preprocessing Pipeline for Bank Statements

This module handles:
1. Auto-detection of CSV column mappings (different bank formats)
2. Data cleaning and normalization
3. Feature extraction for ML models

DEEP DIVE LESSON: Data Preprocessing Pipelines
----------------------------------------------
Real-world financial data is messy. Banks export CSVs differently:
- Chase: "Transaction Date", "Description", "Amount"
- Bank of America: "Date", "Payee", "Amount", "Running Bal."
- Wells Fargo: "Date", "Description", "Withdrawals", "Deposits"

A robust pipeline must:
1. Detect column mappings through heuristics
2. Normalize data to a standard schema
3. Handle edge cases (missing values, different date formats)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser as date_parser
from typing import Tuple, Dict, Optional, List
import re


# Common column name patterns for auto-detection
COLUMN_PATTERNS = {
    "date": [
        r"date", r"trans.*date", r"posting.*date", r"transaction.*date",
        r"posted", r"when", r"time"
    ],
    "description": [
        r"description", r"desc", r"memo", r"payee", r"merchant",
        r"details", r"narrative", r"particulars", r"reference"
    ],
    "amount": [
        r"^amount$", r"transaction.*amount", r"sum", r"value"
    ],
    "debit": [
        r"debit", r"withdrawal", r"out", r"expense", r"payment"
    ],
    "credit": [
        r"credit", r"deposit", r"in", r"income", r"receipt"
    ],
    "balance": [
        r"balance", r"running.*bal", r"available"
    ],
    "category": [
        r"category", r"type", r"classification"
    ]
}


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect which columns map to our standard schema.
    
    Returns a mapping: {"date": "actual_col_name", "description": "actual_col_name", ...}
    """
    mapping = {}
    columns_lower = {col: col.lower().strip() for col in df.columns}
    
    for field, patterns in COLUMN_PATTERNS.items():
        for col, col_lower in columns_lower.items():
            if col in mapping.values():
                continue
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    mapping[field] = col
                    break
            if field in mapping:
                break
    
    # Validate we have minimum required columns
    if "date" not in mapping:
        # Try to find a column that looks like dates
        for col in df.columns:
            sample = df[col].dropna().head(5)
            if _looks_like_dates(sample):
                mapping["date"] = col
                break
    
    if "description" not in mapping:
        # Find the column with the most text variety (likely descriptions)
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            variety = {col: df[col].nunique() for col in text_cols if col not in mapping.values()}
            if variety:
                mapping["description"] = max(variety, key=variety.get)
    
    return mapping


def _looks_like_dates(series: pd.Series) -> bool:
    """Check if a series looks like it contains dates."""
    try:
        for val in series:
            if pd.isna(val):
                continue
            date_parser.parse(str(val))
        return True
    except:
        return False


def _parse_amount(value) -> float:
    """Parse amount from various formats: '$1,234.56', '(500.00)', '-500', etc."""
    if pd.isna(value):
        return 0.0
    
    val_str = str(value).strip()
    
    # Handle parentheses as negative (accounting format)
    is_negative = val_str.startswith('(') and val_str.endswith(')')
    if is_negative:
        val_str = val_str[1:-1]
    
    # Remove currency symbols and thousand separators
    val_str = re.sub(r'[£$€,\s]', '', val_str)
    
    # Handle explicit negative sign
    if val_str.startswith('-'):
        is_negative = True
        val_str = val_str[1:]
    
    try:
        amount = float(val_str)
        return -amount if is_negative else amount
    except ValueError:
        return 0.0


def normalize_transactions(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize transactions to standard schema:
    - date: datetime
    - description: string
    - amount: float (negative = expense, positive = income)
    - original_category: string (if present in source)
    """
    normalized = pd.DataFrame()
    
    # Parse dates
    if "date" in mapping:
        normalized["date"] = pd.to_datetime(
            df[mapping["date"]], 
            infer_datetime_format=True,
            errors='coerce'
        )
    else:
        raise ValueError("Could not detect date column in CSV")
    
    # Parse descriptions
    if "description" in mapping:
        normalized["description"] = df[mapping["description"]].astype(str).str.strip()
    else:
        normalized["description"] = "Unknown"
    
    # Parse amounts - handle different formats
    if "amount" in mapping:
        # Single amount column
        normalized["amount"] = df[mapping["amount"]].apply(_parse_amount)
    elif "debit" in mapping or "credit" in mapping:
        # Separate debit/credit columns
        debit = df[mapping.get("debit", pd.Series(0, index=df.index))].apply(_parse_amount) if "debit" in mapping else 0
        credit = df[mapping.get("credit", pd.Series(0, index=df.index))].apply(_parse_amount) if "credit" in mapping else 0
        
        # Convention: debits are expenses (negative), credits are income (positive)
        if isinstance(debit, pd.Series):
            debit = debit.abs() * -1
        if isinstance(credit, pd.Series):
            credit = credit.abs()
        
        normalized["amount"] = credit + debit
    else:
        raise ValueError("Could not detect amount column(s) in CSV")
    
    # Preserve original category if present
    if "category" in mapping:
        normalized["original_category"] = df[mapping["category"]].astype(str)
    
    # Drop rows with invalid dates
    normalized = normalized.dropna(subset=["date"])
    
    # Sort by date
    normalized = normalized.sort_values("date").reset_index(drop=True)
    
    return normalized


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for ML models:
    - Temporal features (day of week, month, etc.)
    - Transaction characteristics
    """
    features = df.copy()
    
    # Temporal features
    features["day_of_week"] = features["date"].dt.dayofweek
    features["day_of_month"] = features["date"].dt.day
    features["month"] = features["date"].dt.month
    features["year"] = features["date"].dt.year
    features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
    features["week_of_year"] = features["date"].dt.isocalendar().week.astype(int)
    
    # Amount characteristics  
    features["amount_abs"] = features["amount"].abs()
    features["is_expense"] = (features["amount"] < 0).astype(int)
    features["is_income"] = (features["amount"] > 0).astype(int)
    
    # Description features
    features["description_length"] = features["description"].str.len()
    features["description_words"] = features["description"].str.split().str.len()
    
    return features


def preprocess_pipeline(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    """
    Main preprocessing pipeline - takes uploaded CSV and returns processed DataFrame.
    
    Returns:
        - processed_df: Normalized and feature-enriched DataFrame
        - mapping: Detected column mapping
        - report: String report of preprocessing steps
    """
    report_lines = []
    
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        report_lines.append(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    # Detect column mapping
    mapping = detect_column_mapping(df)
    report_lines.append(f"✓ Detected column mapping: {mapping}")
    
    # Normalize to standard schema
    normalized = normalize_transactions(df, mapping)
    report_lines.append(f"✓ Normalized {len(normalized)} valid transactions")
    
    # Extract features
    processed = extract_features(normalized)
    report_lines.append(f"✓ Extracted {len([c for c in processed.columns if c not in normalized.columns])} ML features")
    
    # Summary stats
    date_range = f"{processed['date'].min().strftime('%Y-%m-%d')} to {processed['date'].max().strftime('%Y-%m-%d')}"
    total_income = processed[processed["amount"] > 0]["amount"].sum()
    total_expenses = processed[processed["amount"] < 0]["amount"].sum()
    
    report_lines.append(f"✓ Date range: {date_range}")
    report_lines.append(f"✓ Total income: ${total_income:,.2f}")
    report_lines.append(f"✓ Total expenses: ${abs(total_expenses):,.2f}")
    
    return processed, mapping, "\n".join(report_lines)
