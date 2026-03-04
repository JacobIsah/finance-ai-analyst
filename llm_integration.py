"""
Local LLM Integration via Ollama

This module handles:
1. Connection to local Ollama server
2. Streaming responses for real-time UI
3. Prompt engineering for financial analysis
4. Fallback handling when Ollama unavailable

DEEP DIVE LESSON: Local LLM Integration for Data Interpretation
---------------------------------------------------------------
Why local LLMs for financial data?
1. PRIVACY: Bank data never leaves your machine
2. COST: No API fees, unlimited queries
3. LATENCY: No network round-trips for simple queries
4. OFFLINE: Works without internet

Architecture considerations:
- Ollama runs as a background service (localhost:11434)
- We use streaming for better UX (show tokens as they arrive)
- Context window limits require careful prompt design
- Smaller models (llama3.2:3b) work great for structured analysis

Streaming vs. Batch:
- Streaming: Better perceived latency, user sees response building
- Batch: Easier to handle, but feels slow for >2s responses
- Our approach: Stream to Streamlit's st.write_stream() for best UX
"""

import requests
import json
from typing import Generator, Dict, Any, Optional
import pandas as pd

from config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaClient:
    """
    Client for interacting with local Ollama server.
    
    Supports both streaming and non-streaming responses.
    """
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self._available = any(self.model.split(":")[0] in name for name in model_names)
                return self._available
        except requests.exceptions.RequestException:
            pass
        
        self._available = False
        return False
    
    def get_available_models(self) -> list:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return [m.get("name") for m in response.json().get("models", [])]
        except requests.exceptions.RequestException:
            pass
        return []
    
    def generate(self, prompt: str, stream: bool = True) -> Generator[str, None, None]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            stream: If True, yields tokens as they arrive
        
        Yields:
            Response text (chunk by chunk if streaming)
        """
        if not self.is_available():
            yield "⚠️ Ollama is not available. Please ensure Ollama is running with the model installed.\n\n"
            yield f"To install: `ollama pull {self.model}`"
            return
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream
                },
                stream=stream,
                timeout=120
            )
            
            if response.status_code != 200:
                yield f"⚠️ Error from Ollama: {response.status_code}"
                return
            
            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                yield data.get("response", "")
                
        except requests.exceptions.RequestException as e:
            yield f"⚠️ Connection error: {e}"
    
    def generate_sync(self, prompt: str) -> str:
        """
        Generate a complete response (non-streaming).
        
        Returns:
            Complete response text
        """
        return "".join(self.generate(prompt, stream=False))


def create_financial_summary_prompt(df: pd.DataFrame, analysis_summary: Dict[str, Any]) -> str:
    """
    Create a prompt for the LLM to analyze financial data.
    
    The prompt includes structured data summaries to work within context limits.
    """
    # Extract key statistics
    total_transactions = len(df)
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].sum()
    net = income + expenses
    
    # Category breakdown
    category_summary = ""
    if "predicted_category" in df.columns:
        cat_totals = df[df["amount"] < 0].groupby("predicted_category")["amount"].sum().abs()
        cat_totals = cat_totals.sort_values(ascending=False)
        category_summary = "\n".join([f"  - {cat}: ${amt:,.2f}" for cat, amt in cat_totals.items()])
    
    # Anomalies
    anomaly_summary = ""
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == True]
        if len(anomalies) > 0:
            anomaly_summary = f"Detected {len(anomalies)} unusual transactions:\n"
            for _, row in anomalies.head(5).iterrows():
                reasons = ", ".join(row.get("anomaly_reasons", []))
                anomaly_summary += f"  - {row['description'][:30]}: ${abs(row['amount']):,.2f} ({reasons})\n"
    
    prompt = f"""You are a personal finance analyst assistant. Analyze the following financial summary and provide actionable insights.

## Financial Overview
- Period: {date_range}
- Total Transactions: {total_transactions}
- Total Income: ${income:,.2f}
- Total Expenses: ${abs(expenses):,.2f}
- Net Savings: ${net:,.2f}
- Savings Rate: {(net/income*100) if income > 0 else 0:.1f}%

## Spending by Category
{category_summary}

## Anomalies Detected
{anomaly_summary if anomaly_summary else "No significant anomalies detected."}

## Analysis Request
Based on this data, please provide:
1. **Key Observations**: 2-3 main takeaways about spending patterns
2. **Recommendations**: 2-3 specific, actionable suggestions for improving finances
3. **Alerts**: Any concerning patterns that need attention
4. **Positive Trends**: Anything the user is doing well

Keep your response concise and focused on actionable insights. Use bullet points for readability.
"""
    
    return prompt


def create_category_insight_prompt(df: pd.DataFrame, category: str) -> str:
    """
    Create a prompt for detailed analysis of a specific category.
    """
    cat_data = df[(df["predicted_category"] == category) & (df["amount"] < 0)]
    
    if len(cat_data) == 0:
        return f"No spending data available for category: {category}"
    
    total = cat_data["amount"].sum()
    avg = cat_data["amount"].mean()
    count = len(cat_data)
    
    # Recent transactions
    recent = cat_data.sort_values("date", ascending=False).head(10)
    recent_list = "\n".join([
        f"  - {row['date'].strftime('%Y-%m-%d')}: {row['description'][:30]} - ${abs(row['amount']):,.2f}"
        for _, row in recent.iterrows()
    ])
    
    prompt = f"""Analyze spending in the "{category}" category:

## Category Overview
- Total Spent: ${abs(total):,.2f}
- Average Transaction: ${abs(avg):,.2f}
- Number of Transactions: {count}

## Recent Transactions
{recent_list}

Please provide:
1. Is this spending level typical/reasonable for this category?
2. Any patterns in timing or amounts?
3. Specific suggestions to optimize spending in this category

Keep response brief and actionable.
"""
    
    return prompt


def create_anomaly_explanation_prompt(transaction: pd.Series) -> str:
    """
    Create a prompt to explain why a transaction was flagged as anomalous.
    """
    reasons = transaction.get("anomaly_reasons", [])
    reasons_text = "\n".join([f"  - {r}" for r in reasons]) if reasons else "  - Unusual pattern detected"
    
    prompt = f"""A transaction was flagged as potentially anomalous:

## Transaction Details
- Date: {transaction['date'].strftime('%Y-%m-%d')}
- Description: {transaction['description']}
- Amount: ${abs(transaction['amount']):,.2f}
- Category: {transaction.get('predicted_category', 'Unknown')}

## Detection Reasons
{reasons_text}

Please explain:
1. Why this might be flagged as unusual
2. Whether this seems like a legitimate concern or likely a false positive
3. What action (if any) the user should take

Be concise and helpful.
"""
    
    return prompt


class FinancialAnalysisLLM:
    """
    High-level interface for LLM-powered financial analysis.
    """
    
    def __init__(self, model: str = OLLAMA_MODEL):
        self.client = OllamaClient(model=model)
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client.is_available()
    
    def get_overall_insights(self, df: pd.DataFrame) -> Generator[str, None, None]:
        """
        Generate overall financial insights from the data.
        
        Yields response tokens for streaming display.
        """
        prompt = create_financial_summary_prompt(df, {})
        yield from self.client.generate(prompt, stream=True)
    
    def get_category_insights(self, df: pd.DataFrame, category: str) -> Generator[str, None, None]:
        """
        Generate insights for a specific spending category.
        """
        prompt = create_category_insight_prompt(df, category)
        yield from self.client.generate(prompt, stream=True)
    
    def explain_anomaly(self, transaction: pd.Series) -> Generator[str, None, None]:
        """
        Explain why a transaction was flagged as anomalous.
        """
        prompt = create_anomaly_explanation_prompt(transaction)
        yield from self.client.generate(prompt, stream=True)
    
    def ask_question(self, df: pd.DataFrame, question: str) -> Generator[str, None, None]:
        """
        Answer a custom question about the financial data.
        """
        # Build context from data
        context = create_financial_summary_prompt(df, {})
        
        prompt = f"""{context}

## User Question
{question}

Please answer the question based on the financial data provided above. Be specific and reference actual numbers from the data when relevant.
"""
        yield from self.client.generate(prompt, stream=True)
