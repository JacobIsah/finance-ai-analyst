"""
Visualization Module for Financial Data Analysis

This module creates interactive visualizations using Plotly:
1. Spending by category (pie/bar charts)
2. Spending over time (line charts)
3. Income vs expenses comparison
4. Anomaly highlights on timeline
5. Monthly/weekly summaries

DEEP DIVE LESSON: Visualization Libraries
-----------------------------------------
For financial dashboards, Plotly offers key advantages:
- Interactive by default (hover, zoom, pan)
- Native Streamlit integration via st.plotly_chart()
- Consistent styling API
- Client-side rendering (privacy - data never leaves browser)

Design principles for financial visualizations:
1. Use consistent color coding (red=expenses, green=income)
2. Always show context (time periods, comparisons)
3. Highlight anomalies without overwhelming
4. Enable drill-down through interactivity
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

from config import CHART_HEIGHT


# Color scheme for categories
CATEGORY_COLORS = {
    "groceries": "#2E86AB",
    "dining": "#A23B72",
    "transportation": "#F18F01",
    "utilities": "#C73E1D",
    "entertainment": "#592E83",
    "shopping": "#1B998B",
    "healthcare": "#E63946",
    "subscriptions": "#457B9D",
    "income": "#2A9D8F",
    "transfer": "#8D99AE",
    "other": "#6C757D"
}


def create_spending_by_category_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing spending distribution by category.
    """
    # Get expenses only
    expenses = df[df["amount"] < 0].copy()
    if len(expenses) == 0:
        return _empty_chart("No expense data available")
    
    expenses["amount_abs"] = expenses["amount"].abs()
    
    # Aggregate by category
    category_totals = expenses.groupby("predicted_category")["amount_abs"].sum().reset_index()
    category_totals.columns = ["Category", "Amount"]
    category_totals = category_totals.sort_values("Amount", ascending=False)
    
    # Create pie chart
    fig = px.pie(
        category_totals,
        values="Amount",
        names="Category",
        title="Spending by Category",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        hole=0.4  # Donut chart
    )
    
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>"
    )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    
    return fig


def create_spending_by_category_bar(df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart showing spending by category.
    """
    expenses = df[df["amount"] < 0].copy()
    if len(expenses) == 0:
        return _empty_chart("No expense data available")
    
    expenses["amount_abs"] = expenses["amount"].abs()
    
    category_totals = expenses.groupby("predicted_category")["amount_abs"].sum().reset_index()
    category_totals.columns = ["Category", "Amount"]
    category_totals = category_totals.sort_values("Amount", ascending=True)
    
    colors = [CATEGORY_COLORS.get(cat, "#6C757D") for cat in category_totals["Category"]]
    
    fig = go.Figure(go.Bar(
        x=category_totals["Amount"],
        y=category_totals["Category"],
        orientation="h",
        marker_color=colors,
        text=category_totals["Amount"].apply(lambda x: f"${x:,.2f}"),
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Spending by Category",
        xaxis_title="Amount ($)",
        yaxis_title="",
        height=CHART_HEIGHT,
        showlegend=False
    )
    
    return fig


def create_spending_over_time(df: pd.DataFrame, period: str = "daily") -> go.Figure:
    """
    Create a line chart showing spending over time.
    
    Args:
        period: "daily", "weekly", or "monthly"
    """
    expenses = df[df["amount"] < 0].copy()
    if len(expenses) == 0:
        return _empty_chart("No expense data available")
    
    expenses["amount_abs"] = expenses["amount"].abs()
    
    # Group by period
    if period == "weekly":
        expenses["period"] = expenses["date"].dt.to_period("W").dt.start_time
    elif period == "monthly":
        expenses["period"] = expenses["date"].dt.to_period("M").dt.start_time
    else:  # daily
        expenses["period"] = expenses["date"].dt.date
    
    time_series = expenses.groupby("period")["amount_abs"].sum().reset_index()
    time_series.columns = ["Date", "Amount"]
    time_series["Date"] = pd.to_datetime(time_series["Date"])
    
    fig = px.line(
        time_series,
        x="Date",
        y="Amount",
        title=f"Spending Over Time ({period.title()})",
        markers=True
    )
    
    fig.update_traces(
        line_color="#E63946",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>"
    )
    
    # Add trend line
    if len(time_series) > 2:
        z = np.polyfit(range(len(time_series)), time_series["Amount"], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=time_series["Date"],
            y=p(range(len(time_series))),
            mode="lines",
            name="Trend",
            line=dict(dash="dash", color="gray"),
            hoverinfo="skip"
        ))
    
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        hovermode="x unified"
    )
    
    return fig


def create_income_vs_expenses(df: pd.DataFrame, period: str = "monthly") -> go.Figure:
    """
    Create a comparison chart of income vs expenses over time.
    """
    df_copy = df.copy()
    
    # Group by period
    if period == "weekly":
        df_copy["period"] = df_copy["date"].dt.to_period("W").dt.start_time
    elif period == "monthly":
        df_copy["period"] = df_copy["date"].dt.to_period("M").dt.start_time
    else:
        df_copy["period"] = df_copy["date"].dt.date
    
    # Calculate income and expenses per period
    income = df_copy[df_copy["amount"] > 0].groupby("period")["amount"].sum()
    expenses = df_copy[df_copy["amount"] < 0].groupby("period")["amount"].sum().abs()
    
    # Combine into single DataFrame
    comparison = pd.DataFrame({
        "Income": income,
        "Expenses": expenses
    }).fillna(0).reset_index()
    comparison.columns = ["Date", "Income", "Expenses"]
    comparison["Date"] = pd.to_datetime(comparison["Date"])
    comparison["Net"] = comparison["Income"] - comparison["Expenses"]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison["Date"],
        y=comparison["Income"],
        name="Income",
        marker_color="#2A9D8F",
        hovertemplate="Income: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=comparison["Date"],
        y=comparison["Expenses"],
        name="Expenses",
        marker_color="#E63946",
        hovertemplate="Expenses: $%{y:,.2f}<extra></extra>"
    ))
    
    # Add net savings line
    fig.add_trace(go.Scatter(
        x=comparison["Date"],
        y=comparison["Net"],
        name="Net Savings",
        mode="lines+markers",
        line=dict(color="#457B9D", width=3),
        hovertemplate="Net: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Income vs Expenses ({period.title()})",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        barmode="group",
        height=CHART_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig


def create_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create a timeline highlighting anomalous transactions.
    """
    if "is_anomaly" not in df.columns:
        return _empty_chart("No anomaly data available")
    
    expenses = df[df["amount"] < 0].copy()
    if len(expenses) == 0:
        return _empty_chart("No expense data available")
    
    expenses["amount_abs"] = expenses["amount"].abs()
    
    # Separate normal and anomaly transactions
    normal = expenses[expenses["is_anomaly"] == False]
    anomalies = expenses[expenses["is_anomaly"] == True]
    
    fig = go.Figure()
    
    # Normal transactions
    fig.add_trace(go.Scatter(
        x=normal["date"],
        y=normal["amount_abs"],
        mode="markers",
        name="Normal",
        marker=dict(
            size=8,
            color="#457B9D",
            opacity=0.6
        ),
        text=normal["description"],
        hovertemplate="<b>%{text}</b><br>$%{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>"
    ))
    
    # Anomalous transactions
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies["date"],
            y=anomalies["amount_abs"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=15,
                color="#E63946",
                symbol="diamond",
                line=dict(width=2, color="white")
            ),
            text=anomalies["description"],
            customdata=anomalies["anomaly_reasons"],
            hovertemplate="<b>⚠️ %{text}</b><br>$%{y:,.2f}<br>%{x|%Y-%m-%d}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Transaction Timeline with Anomalies",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=CHART_HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest"
    )
    
    return fig


def create_weekly_summary(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing spending patterns by day of week and category.
    """
    expenses = df[df["amount"] < 0].copy()
    if len(expenses) == 0:
        return _empty_chart("No expense data available")
    
    expenses["amount_abs"] = expenses["amount"].abs()
    
    # Create pivot table
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    expenses["day_name"] = expenses["day_of_week"].map(lambda x: day_names[x])
    
    pivot = expenses.pivot_table(
        values="amount_abs",
        index="predicted_category",
        columns="day_name",
        aggfunc="sum",
        fill_value=0
    )
    
    # Reorder columns
    pivot = pivot.reindex(columns=[d for d in day_names if d in pivot.columns])
    
    fig = px.imshow(
        pivot,
        title="Spending Heatmap by Day & Category",
        labels=dict(x="Day of Week", y="Category", color="Spending ($)"),
        color_continuous_scale="Reds",
        aspect="auto"
    )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Day of Week",
        yaxis_title="Category"
    )
    
    return fig


def create_monthly_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a monthly summary table showing key metrics.
    """
    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].dt.to_period("M")
    
    summary = []
    for month in df_copy["month"].unique():
        month_data = df_copy[df_copy["month"] == month]
        
        income = month_data[month_data["amount"] > 0]["amount"].sum()
        expenses = month_data[month_data["amount"] < 0]["amount"].sum()
        net = income + expenses
        anomaly_count = month_data["is_anomaly"].sum() if "is_anomaly" in month_data.columns else 0
        
        summary.append({
            "Month": str(month),
            "Income": f"${income:,.2f}",
            "Expenses": f"${abs(expenses):,.2f}",
            "Net": f"${net:,.2f}",
            "Transactions": len(month_data),
            "Anomalies": int(anomaly_count)
        })
    
    return pd.DataFrame(summary)


def create_category_trend(df: pd.DataFrame, category: str) -> go.Figure:
    """
    Create a trend chart for a specific category over time.
    """
    cat_data = df[
        (df["predicted_category"] == category) & 
        (df["amount"] < 0)
    ].copy()
    
    if len(cat_data) == 0:
        return _empty_chart(f"No data for category: {category}")
    
    cat_data["amount_abs"] = cat_data["amount"].abs()
    cat_data["month"] = cat_data["date"].dt.to_period("M").dt.start_time
    
    monthly = cat_data.groupby("month")["amount_abs"].sum().reset_index()
    monthly.columns = ["Date", "Amount"]
    
    fig = px.bar(
        monthly,
        x="Date",
        y="Amount",
        title=f"{category.title()} Spending Over Time",
        color_discrete_sequence=[CATEGORY_COLORS.get(category, "#6C757D")]
    )
    
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Month",
        yaxis_title="Amount ($)"
    )
    
    return fig


def _empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig
