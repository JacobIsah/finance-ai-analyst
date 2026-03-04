"""
Configuration settings for the Personal Finance AI Analyst
"""

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"  # Can be changed to any Ollama model

# Category definitions for transaction classification
SPENDING_CATEGORIES = {
    "groceries": ["grocery", "supermarket", "walmart", "costco", "trader joe", "whole foods", "aldi", "kroger", "safeway", "publix", "food", "market"],
    "dining": ["restaurant", "cafe", "coffee", "starbucks", "mcdonald", "burger", "pizza", "doordash", "uber eats", "grubhub", "dining", "eat"],
    "transportation": ["gas", "fuel", "uber", "lyft", "taxi", "parking", "toll", "transit", "metro", "bus", "train", "shell", "chevron", "exxon"],
    "utilities": ["electric", "water", "gas bill", "internet", "phone", "verizon", "at&t", "comcast", "utility", "power"],
    "entertainment": ["netflix", "spotify", "hulu", "disney", "amazon prime", "movie", "theater", "concert", "game", "steam", "playstation", "xbox"],
    "shopping": ["amazon", "target", "best buy", "clothing", "shoes", "mall", "store", "shop", "retail", "ebay"],
    "healthcare": ["pharmacy", "doctor", "hospital", "medical", "cvs", "walgreens", "health", "dental", "vision", "insurance"],
    "subscriptions": ["subscription", "membership", "annual", "monthly fee", "recurring"],
    "income": ["payroll", "salary", "deposit", "direct dep", "income", "payment received", "transfer in", "refund"],
    "transfer": ["transfer", "zelle", "venmo", "paypal", "wire"],
    "other": []
}

# Anomaly detection settings
ANOMALY_CONTAMINATION = 0.05  # Expected proportion of outliers (5%)
ANOMALY_ZSCORE_THRESHOLD = 3  # Standard deviations for statistical anomaly

# Visualization settings  
CHART_HEIGHT = 400
COLOR_SCHEME = "plotly"
