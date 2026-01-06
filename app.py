"""
Simple Momentum Strategy - Streamlit App
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from simple_momentum import SimpleMomentumStrategy


st.set_page_config(page_title="Momentum Strategy", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(tickers, start_date, end_date):
    """Download data."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    prices = prices.dropna(axis=1, how='all')
    return prices


@st.cache_data
def run_backtest(prices, n_long, n_short, rebalance_days, lookback_days):
    """Run momentum backtest."""
    strategy = SimpleMomentumStrategy(prices, lookback_days=lookback_days)
    results = strategy.run_backtest(n_long, n_short, rebalance_days, train_test_split=0.7)
    return results


def main():
    st.markdown('<div class="main-header">📈 Simple Momentum Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on Jegadeesh & Titman (1993) | Duke University</div>', unsafe_allow_html=True)
    
    # Info
    with st.expander("ℹ️ About This Strategy", expanded=False):
        st.markdown("""
        ### What is Momentum?
        
        **Momentum** is one of the most robust patterns in financial markets:
        - **Stocks that went up recently tend to keep going up**
        - **Stocks that went down recently tend to keep going down**
        
        This pattern has been documented in:
        - 200+ academic papers since 1993
        - Real money mutual funds and hedge funds
        - Works across markets, countries, asset classes
        
        ### This Strategy
        
        **Methodology:**
        1. Calculate 12-month return for each stock
        2. Rank all stocks by their momentum
        3. **Buy (long)** the top 10 stocks (highest momentum)
        4. **Sell short** the bottom 10 stocks (lowest momentum)
        5. Rebalance monthly
        6. Include realistic trading costs (15 bps)
        
        **Based on:**
        - Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
        - Published in Journal of Finance (top academic journal)
        - Most replicated factor in finance
        
        ### Data
        - **Source:** Yahoo Finance (real market data)
        - **Universe:** 20 large-cap US stocks
        - **Period:** 3 years of daily prices
        
        ### Validation
        - **70/30 train/test split** (prevent overfitting)
        - **Out-of-sample results** shown (the ones that matter!)
        - **Realistic costs** included
        
        ### Why This Works
        - **Behavioral:** Investors under-react to news
        - **Institutional:** Slow capital flows
        - **Risk-based:** Compensation for crash risk
        
        This is a **real, academically-validated strategy** - not data mining!
        """)
    
    # Sidebar
    st.sidebar.header("⚙️ Parameters")
    
    n_long = st.sidebar.slider("Long Positions", 5, 15, 10, help="Number of stocks to buy")
    n_short = st.sidebar.slider("Short Positions", 5, 15, 10, help="Number of stocks to short")
    rebalance_days = st.sidebar.slider("Rebalance (days)", 5, 30, 21, help="Monthly = 21 days")
    lookback_days = st.sidebar.slider("Momentum Period (days)", 60, 252, 252, help="252 days = 1 year")
    
    run = st.sidebar.button("🚀 Run Backtest", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Research Reference:**
        
        Jegadeesh, N., & Titman, S. (1993).
        "Returns to Buying Winners and Selling Losers:
        Implications for Stock Market Efficiency."
        *Journal of Finance*, 48(1), 65-91.
        
        **Author:** Soham Gugale  
        **School:** Duke University
    """)
    
    if run:
        # Tickers
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'MA', 'UNH', 'HD',
            'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM'
        ]
        
        with st.spinner("📥 Downloading data from Yahoo Finance..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            
            prices = load_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            st.success(f"✓ Downloaded {len(prices)} days for {len(prices.columns)} stocks")
        
        with st.spinner("🔄 Running backtest..."):
            results = run_backtest(prices, n_long, n_short, rebalance_days, lookback_days)
            st.success("✓ Backtest complete!")
        
        # Results
        tab1, tab2, tab3 = st.tabs(["📊 Performance", "💼 Holdings", "📖 Methodology"])
        
        with tab1:
            st.header("Performance Analysis")
            
            oos = results['out_sample_metrics']
            ins = results['in_sample_metrics']
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Sharpe Ratio", f"{oos['Sharpe Ratio']:.3f}")
                st.metric("Total Return", f"{oos['Total Return (%)']:.2f}%")
            
            with col2:
                st.metric("Ann. Return", f"{oos['Annualized Return (%)']:.2f}%")
                st.metric("Ann. Volatility", f"{oos['Annualized Volatility (%)']:.2f}%")
            
            with col3:
                st.metric("Max Drawdown", f"{oos['Max Drawdown (%)']:.2f}%")
                st.metric("Calmar Ratio", f"{oos['Calmar Ratio']:.3f}")
            
            with col4:
                st.metric("Win Rate", f"{oos['Win Rate (%)']:.1f}%")
                st.metric("Sortino Ratio", f"{oos['Sortino Ratio']:.3f}")
            
            st.markdown("---")
            
            # Equity curve
            fig = go.Figure()
            
            equity = results['equity_curve']
            split_date = results['split_date']
            
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name='Strategy',
                line=dict(color='#1E88E5', width=2)
            ))
            
            # Add split line
            split_str = pd.Timestamp(split_date).strftime('%Y-%m-%d')
            fig.add_shape(
                type="line", x0=split_str, x1=split_str, y0=0, y1=1, yref='paper',
                line=dict(color="green", width=2, dash="dot")
            )
            fig.add_annotation(
                x=split_str, y=1.05, yref='paper', text="Train/Test Split",
                showarrow=False, font=dict(color="green")
            )
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison
            st.subheader("In-Sample vs Out-of-Sample")
            
            comparison = pd.DataFrame({
                'Metric': ['Ann. Return (%)', 'Sharpe Ratio', 'Max DD (%)', 'Win Rate (%)'],
                'In-Sample (Training)': [
                    ins['Annualized Return (%)'],
                    ins['Sharpe Ratio'],
                    ins['Max Drawdown (%)'],
                    ins['Win Rate (%)']
                ],
                'Out-of-Sample (Testing)': [
                    oos['Annualized Return (%)'],
                    oos['Sharpe Ratio'],
                    oos['Max Drawdown (%)'],
                    oos['Win Rate (%)']
                ]
            })
            
            st.dataframe(comparison, use_container_width=True)
        
        with tab2:
            st.header("Current Holdings")
            
            last_pos = results['positions'].iloc[-1]
            longs = last_pos[last_pos > 0].sort_values(ascending=False)
            shorts = last_pos[last_pos < 0].sort_values()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🟢 Long Positions")
                if len(longs) > 0:
                    long_df = pd.DataFrame({
                        'Stock': longs.index,
                        'Weight': [f"{w:.2%}" for w in longs.values]
                    })
                    st.dataframe(long_df, use_container_width=True)
                else:
                    st.info("No long positions")
            
            with col2:
                st.subheader("🔴 Short Positions")
                if len(shorts) > 0:
                    short_df = pd.DataFrame({
                        'Stock': shorts.index,
                        'Weight': [f"{w:.2%}" for w in shorts.values]
                    })
                    st.dataframe(short_df, use_container_width=True)
                else:
                    st.info("No short positions")
            
            # Top movers
            st.subheader("📈 Current Momentum Rankings")
            
            last_mom = results['momentum_scores'].iloc[-1].dropna().sort_values(ascending=False)
            
            mom_df = pd.DataFrame({
                'Stock': last_mom.index,
                '12-Month Return': [f"{m:.2%}" for m in last_mom.values]
            })
            
            st.dataframe(mom_df, use_container_width=True)
        
        with tab3:
            st.header("Academic Foundation")
            
            st.markdown("""
            ### The Momentum Effect
            
            **Discovery:**
            - First documented by Jegadeesh & Titman in 1993
            - Published in *Journal of Finance* (top-tier journal)
            - One of the most replicated findings in finance
            
            **Key Finding:**
            > "Strategies which buy stocks that have performed well in the past and sell stocks 
            > that have performed poorly generate significant positive returns over 3- to 12-month holding periods."
            
            **Magnitude:**
            - Original paper: ~1% per month (12%+ annually)
            - Continues to work 30+ years later
            - Works across countries, asset classes
            
            ### Why It Works
            
            **Behavioral Explanations:**
            1. **Under-reaction:** Investors slow to recognize new information
            2. **Herding:** Trend-following creates self-reinforcing patterns
            3. **Confirmation bias:** Winners attract more buyers
            
            **Institutional Explanations:**
            1. **Slow capital flows:** Large funds can't react instantly
            2. **Constraints:** Short-selling restrictions limit arbitrage
            3. **Risk management:** Forced selling during drawdowns
            
            ### Implementation Notes
            
            **This Implementation:**
            - 252-day (1-year) lookback period
            - Monthly rebalancing (21 trading days)
            - Equal-weighted long/short portfolio
            - 15 bps transaction costs
            - 70/30 train/test validation
            
            **Industry Practice:**
            - Most funds use 6-12 month momentum
            - Some skip the most recent month (avoid mean reversion)
            - Often combined with other factors (value, quality)
            
            ### Further Reading
            
            1. **Original Paper:**
               Jegadeesh & Titman (1993), Journal of Finance
            
            2. **Review:**
               Asness, Moskowitz, & Pedersen (2013), "Value and Momentum Everywhere"
            
            3. **Practitioner Guide:**
               Antonacci (2014), "Dual Momentum Investing"
            """)
    
    else:
        st.info("""
        ### 👈 Configure parameters and click "Run Backtest"
        
        This implements a **simple, proven momentum strategy** based on academic research.
        
        **Key Features:**
        - Uses real Yahoo Finance data
        - Based on published research (Jegadeesh & Titman, 1993)
        - Actually generates long/short positions
        - Includes realistic transaction costs
        - Validated with out-of-sample testing
        """)


if __name__ == "__main__":
    main()
