"""
Personal Finance AI Analyst - Main Streamlit Application

A privacy-first, local AI-powered financial data analyzer.
No signup required - just upload your bank statement CSV.

Features:
- Auto-detect CSV format from various banks
- Transaction classification by category
- Anomaly detection for unusual spending
- Interactive visualizations
- Local LLM insights via Ollama (data never leaves your machine)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from io import StringIO

# Import our modules
from preprocessing import preprocess_pipeline, detect_column_mapping
from ml_models import analyze_transactions, get_category_summary, get_anomaly_summary
from visualizations import (
    create_spending_by_category_pie,
    create_spending_by_category_bar,
    create_spending_over_time,
    create_income_vs_expenses,
    create_anomaly_timeline,
    create_weekly_summary,
    create_monthly_summary_table,
    create_category_trend
)
from llm_integration import FinancialAnalysisLLM
from config import OLLAMA_MODEL


# Page configuration
st.set_page_config(
    page_title="Personal Finance AI Analyst",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6C757D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None
    if "column_mapping" not in st.session_state:
        st.session_state.column_mapping = None
    if "preprocessing_report" not in st.session_state:
        st.session_state.preprocessing_report = None
    if "llm" not in st.session_state:
        st.session_state.llm = FinancialAnalysisLLM()


def render_header():
    """Render the app header."""
    st.markdown('<p class="main-header">💰 Personal Finance AI Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload your bank statement CSV for instant AI-powered analysis. '
        '100% local - your data never leaves your machine.</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with app info and settings."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app analyzes your bank statements locally using:
        - **ML Classification**: Automatically categorizes transactions
        - **Anomaly Detection**: Flags unusual spending patterns
        - **Local LLM**: AI insights via Ollama (privacy-first)
        """)
        
        st.divider()
        
        # LLM Status
        st.header("🤖 AI Assistant Status")
        llm = st.session_state.llm
        
        if llm.is_available():
            st.success(f"✅ Ollama connected ({OLLAMA_MODEL})")
        else:
            st.warning("⚠️ Ollama not available")
            st.markdown("""
            To enable AI insights:
            1. Install [Ollama](https://ollama.ai)
            2. Run: `ollama pull llama3.2`
            3. Refresh this page
            """)
        
        st.divider()
        
        # Supported formats
        st.header("📄 Supported CSV Formats")
        st.markdown("""
        Works with exports from:
        - Chase Bank
        - Bank of America
        - Wells Fargo
        - Capital One
        - Most other banks
        
        Auto-detects columns like:
        - Date, Transaction Date
        - Description, Memo, Payee
        - Amount, Debit, Credit
        """)


def render_upload_section():
    """Render the file upload section."""
    st.header("📤 Upload Bank Statement")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your bank statement in CSV format. Most banks allow exporting transactions as CSV."
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing your data..."):
            try:
                # Process the uploaded file
                processed_df, mapping, report = preprocess_pipeline(uploaded_file)
                
                # Run ML analysis
                analyzed_df = analyze_transactions(processed_df)
                
                # Store in session state
                st.session_state.processed_data = analyzed_df
                st.session_state.column_mapping = mapping
                st.session_state.preprocessing_report = report
                
                st.success("✅ Data processed successfully!")
                
                # Show preprocessing report
                with st.expander("📋 Preprocessing Report", expanded=False):
                    st.code(report)
                    st.caption("Column Mapping Detected:")
                    st.json(mapping)
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.info("Please ensure your CSV has date, description, and amount columns.")
    
    return st.session_state.processed_data is not None


def render_metrics(df: pd.DataFrame):
    """Render key financial metrics."""
    st.header("📊 Financial Overview")
    
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].sum()
    net = income + expenses
    num_transactions = len(df)
    anomaly_count = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Income", f"${income:,.2f}", delta=None)
    
    with col2:
        st.metric("Total Expenses", f"${abs(expenses):,.2f}", delta=None)
    
    with col3:
        delta_color = "normal" if net >= 0 else "inverse"
        st.metric("Net Balance", f"${net:,.2f}", 
                  delta=f"{(net/income*100) if income > 0 else 0:.1f}% saved" if net >= 0 else "Deficit",
                  delta_color=delta_color)
    
    with col4:
        st.metric("Transactions", f"{num_transactions:,}")
    
    with col5:
        st.metric("Anomalies", f"{int(anomaly_count)}", 
                  delta="flagged" if anomaly_count > 0 else None,
                  delta_color="inverse" if anomaly_count > 0 else "off")


def render_visualizations(df: pd.DataFrame):
    """Render the visualization dashboard."""
    st.header("📈 Visualizations")
    
    # Tab layout for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Categories", 
        "📈 Over Time", 
        "💵 Income vs Expenses",
        "⚠️ Anomalies",
        "📅 Patterns"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_spending_by_category_pie(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_spending_by_category_bar(df), use_container_width=True)
        
        # Category summary table
        st.subheader("Category Breakdown")
        summary = get_category_summary(df)
        if len(summary) > 0:
            st.dataframe(summary, use_container_width=True)
    
    with tab2:
        # Time period selector
        period = st.selectbox("Time Period", ["Daily", "Weekly", "Monthly"], index=1)
        st.plotly_chart(
            create_spending_over_time(df, period.lower()), 
            use_container_width=True
        )
    
    with tab3:
        period = st.selectbox("Aggregation", ["Weekly", "Monthly"], index=1, key="income_period")
        st.plotly_chart(
            create_income_vs_expenses(df, period.lower()),
            use_container_width=True
        )
        
        # Monthly summary table
        st.subheader("Monthly Summary")
        monthly_table = create_monthly_summary_table(df)
        st.dataframe(monthly_table, use_container_width=True)
    
    with tab4:
        st.plotly_chart(create_anomaly_timeline(df), use_container_width=True)
        
        # Anomaly details
        anomalies = get_anomaly_summary(df)
        if len(anomalies) > 0:
            st.subheader("🚨 Detected Anomalies")
            for _, row in anomalies.iterrows():
                with st.expander(f"⚠️ {row['description'][:40]}... - ${abs(row['amount']):,.2f}"):
                    st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Category:** {row['predicted_category']}")
                    st.write(f"**Anomaly Score:** {row['anomaly_score']:.2f}")
                    st.write("**Reasons:**")
                    for reason in row['anomaly_reasons']:
                        st.write(f"  - {reason}")
                    
                    # LLM explanation button
                    if st.session_state.llm.is_available():
                        if st.button("🤖 Explain this anomaly", key=f"explain_{row.name}"):
                            with st.spinner("Analyzing..."):
                                st.write_stream(st.session_state.llm.explain_anomaly(row))
        else:
            st.info("✅ No anomalies detected in your transactions!")
    
    with tab5:
        st.plotly_chart(create_weekly_summary(df), use_container_width=True)
        
        # Category-specific trend
        if "predicted_category" in df.columns:
            categories = df[df["amount"] < 0]["predicted_category"].unique().tolist()
            if categories:
                selected_cat = st.selectbox("View category trend", categories)
                st.plotly_chart(create_category_trend(df, selected_cat), use_container_width=True)


def render_ai_insights(df: pd.DataFrame):
    """Render the AI insights section."""
    st.header("🤖 AI-Powered Insights")
    
    llm = st.session_state.llm
    
    if not llm.is_available():
        st.warning("""
        ⚠️ **Ollama not available**
        
        To enable AI insights:
        1. Install [Ollama](https://ollama.ai)
        2. Run: `ollama pull llama3.2`
        3. Refresh this page
        
        The ML analysis (classification & anomaly detection) still works without Ollama!
        """)
        return
    
    # Overall insights
    st.subheader("📋 Overall Financial Analysis")
    
    if st.button("🔍 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing your financial data..."):
            insight_container = st.empty()
            full_response = ""
            for token in llm.get_overall_insights(df):
                full_response += token
                insight_container.markdown(full_response)
    
    st.divider()
    
    # Category deep dive
    st.subheader("🔎 Category Deep Dive")
    
    if "predicted_category" in df.columns:
        categories = df[df["amount"] < 0]["predicted_category"].unique().tolist()
        selected_category = st.selectbox("Select a category to analyze", categories)
        
        if st.button("📊 Analyze Category"):
            with st.spinner(f"Analyzing {selected_category} spending..."):
                cat_container = st.empty()
                full_response = ""
                for token in llm.get_category_insights(df, selected_category):
                    full_response += token
                    cat_container.markdown(full_response)
    
    st.divider()
    
    # Custom question
    st.subheader("💬 Ask a Question")
    user_question = st.text_input(
        "Ask anything about your finances",
        placeholder="e.g., What's my biggest expense category? Am I spending too much on dining?"
    )
    
    if user_question:
        if st.button("Ask AI"):
            with st.spinner("Thinking..."):
                answer_container = st.empty()
                full_response = ""
                for token in llm.ask_question(df, user_question):
                    full_response += token
                    answer_container.markdown(full_response)


def render_data_explorer(df: pd.DataFrame):
    """Render a data exploration section."""
    st.header("🔍 Data Explorer")
    
    with st.expander("📋 View All Transactions", expanded=False):
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "predicted_category" in df.columns:
                categories = ["All"] + df["predicted_category"].unique().tolist()
                filter_cat = st.selectbox("Filter by Category", categories)
        
        with col2:
            filter_type = st.selectbox("Transaction Type", ["All", "Income", "Expenses"])
        
        with col3:
            show_anomalies = st.checkbox("Show only anomalies")
        
        # Apply filters
        filtered = df.copy()
        
        if filter_cat != "All":
            filtered = filtered[filtered["predicted_category"] == filter_cat]
        
        if filter_type == "Income":
            filtered = filtered[filtered["amount"] > 0]
        elif filter_type == "Expenses":
            filtered = filtered[filtered["amount"] < 0]
        
        if show_anomalies:
            filtered = filtered[filtered["is_anomaly"] == True]
        
        # Display columns
        display_cols = ["date", "description", "amount", "predicted_category", "is_anomaly"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        st.dataframe(
            filtered[display_cols].sort_values("date", ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            "filtered_transactions.csv",
            "text/csv"
        )


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content
    has_data = render_upload_section()
    
    if has_data:
        df = st.session_state.processed_data
        
        st.divider()
        render_metrics(df)
        
        st.divider()
        render_visualizations(df)
        
        st.divider()
        render_ai_insights(df)
        
        st.divider()
        render_data_explorer(df)
    
    else:
        # Show demo/instructions when no data uploaded
        st.info("""
        👆 **Upload a bank statement CSV to get started!**
        
        This app will:
        1. Auto-detect your bank's CSV format
        2. Classify transactions into categories (groceries, dining, etc.)
        3. Detect unusual spending patterns
        4. Generate interactive visualizations
        5. Provide AI-powered insights (requires Ollama)
        
        **Your data stays local** - nothing is sent to external servers.
        """)
        
        # Sample CSV format
        with st.expander("📝 Example CSV Format"):
            st.code("""
Date,Description,Amount
2024-01-15,WALMART GROCERY,-85.43
2024-01-15,PAYROLL DEPOSIT,2500.00
2024-01-16,STARBUCKS COFFEE,-6.45
2024-01-17,SHELL GAS STATION,-45.00
2024-01-18,NETFLIX SUBSCRIPTION,-15.99
            """, language="csv")


if __name__ == "__main__":
    main()
