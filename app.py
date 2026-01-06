"""
Simple Momentum Strategy - Streamlit App
FIXED VERSION with robust data loading and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from simple_momentum import SimpleMomentumStrategy
import time


st.set_page_config(page_title="Momentum Strategy", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E88E5; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #666; text-align: center; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data_robust(tickers, start_date, end_date, max_retries=3):
    """
    Robustly download data with retries and validation.
    """
    
    for attempt in range(max_retries):
        try:
            st.info(f"📥 Downloading data (attempt {attempt + 1}/{max_retries})...")
            
            # Download with explicit parameters
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
            
            # Handle single vs multiple tickers
            if len(tickers) == 1:
                if 'Close' in data.columns:
                    prices = data[['Close']].copy()
                    prices.columns = [tickers[0]]
                else:
                    st.warning(f"⚠️ No data for {tickers[0]}")
                    continue
            else:
                # Multiple tickers - extract Close prices
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-level columns: ('AAPL', 'Close'), ('MSFT', 'Close'), ...
                    prices = data.xs('Close', axis=1, level=1)
                elif 'Close' in data.columns:
                    prices = data['Close']
                else:
                    st.error("❌ Unexpected data format from yfinance")
                    continue
            
            # Validate data
            if prices.empty:
                st.warning(f"⚠️ Downloaded data is empty (attempt {attempt + 1})")
                time.sleep(2)  # Wait before retry
                continue
            
            # Drop columns with all NaN
            initial_cols = len(prices.columns)
            prices = prices.dropna(axis=1, how='all')
            dropped = initial_cols - len(prices.columns)
            
            if dropped > 0:
                st.warning(f"⚠️ Dropped {dropped} stocks with no data")
            
            # Check we have enough data
            if len(prices.columns) < 10:
                st.error(f"❌ Only {len(prices.columns)} stocks have data. Need at least 10.")
                if attempt < max_retries - 1:
                    st.info("Retrying with different parameters...")
                    time.sleep(2)
                    continue
                else:
                    st.error("Failed to download sufficient data. Using fallback...")
                    return generate_fallback_data(tickers, start_date, end_date)
            
            # Drop rows with any NaN (forward fill first)
            prices = prices.fillna(method='ffill').fillna(method='bfill')
            prices = prices.dropna()
            
            if len(prices) < 100:
                st.error(f"❌ Only {len(prices)} days of data. Need at least 100.")
                continue
            
            st.success(f"✓ Downloaded {len(prices)} days for {len(prices.columns)} stocks")
            
            # Log which stocks we got
            stock_list = ', '.join(prices.columns.tolist())
            if len(prices.columns) <= 15:
                st.info(f"📊 Stocks: {stock_list}")
            else:
                st.info(f"📊 Stocks: {', '.join(prices.columns[:10].tolist())}... (+{len(prices.columns)-10} more)")
            
            return prices
            
        except Exception as e:
            st.error(f"❌ Download error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                st.info("Retrying...")
                time.sleep(2)
            else:
                st.error("All download attempts failed. Using fallback data...")
                return generate_fallback_data(tickers, start_date, end_date)
    
    # If we get here, all retries failed
    return generate_fallback_data(tickers, start_date, end_date)


def generate_fallback_data(tickers, start_date, end_date):
    """
    Generate synthetic data as fallback if yfinance fails.
    """
    st.warning("⚠️ Using synthetic data (yfinance unavailable)")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:750]
    
    prices = pd.DataFrame(index=dates)
    
    np.random.seed(42)
    
    for ticker in tickers[:20]:  # Limit to 20
        # Generate realistic-ish prices
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * (1 + returns).cumprod()
        prices[ticker] = price
    
    st.info(f"✓ Generated synthetic data: {len(prices)} days, {len(prices.columns)} stocks")
    
    return prices


@st.cache_data
def run_backtest(prices, n_long, n_short, rebalance_days, lookback_days):
    """Run momentum backtest."""
    try:
        strategy = SimpleMomentumStrategy(prices, lookback_days=lookback_days)
        results = strategy.run_backtest(n_long, n_short, rebalance_days, train_test_split=0.7)
        return results
    except Exception as e:
        st.error(f"❌ Backtest error: {str(e)}")
        st.exception(e)
        return None


def main():
    st.markdown('<div class="main-header">📈 Simple Momentum Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Based on Jegadeesh & Titman (1993) | Duke University</div>', unsafe_allow_html=True)
    
    # Info
    with st.expander("ℹ️ About This Strategy", expanded=False):
        st.markdown("""
        ### What is Momentum?
        
        **Momentum** is one of the most robust patterns in financial markets:
        - Stocks that went up recently tend to keep going up
        - Stocks that went down recently tend to keep going down
        
        ### This Strategy
        
        **Methodology:**
        1. Calculate 12-month return for each stock
        2. **Buy (long)** the top 10 stocks (highest momentum)
        3. **Sell short** the bottom 10 stocks (lowest momentum)
        4. Rebalance monthly
        5. Include trading costs (15 bps)
        
        **Based on:**
        - Jegadeesh & Titman (1993), *Journal of Finance*
        - One of the most replicated findings in finance (200+ papers)
        
        ### Data
        - **Source:** Yahoo Finance (real market data)
        - **Universe:** 20 large-cap US stocks
        - **Period:** ~3 years of daily data
        """)
    
    # Sidebar
    st.sidebar.header("⚙️ Parameters")
    
    n_long = st.sidebar.slider("Long Positions", 5, 15, 10)
    n_short = st.sidebar.slider("Short Positions", 5, 15, 10)
    rebalance_days = st.sidebar.slider("Rebalance (days)", 5, 30, 21)
    lookback_days = st.sidebar.slider("Momentum Period (days)", 60, 252, 252)
    
    run = st.sidebar.button("🚀 Run Backtest", type="primary")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        **Research:**
        Jegadeesh & Titman (1993)
        *Journal of Finance*, 48(1), 65-91
        
        **Author:** Soham Gugale  
        **School:** Duke University
    """)
    
    if run:
        # Tickers - Top 30 liquid large-caps (extra backups in case some fail)
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
            'JNJ', 'PG', 'MA', 'UNH', 'HD',
            'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM',
            'COST', 'ORCL', 'CSCO', 'INTC', 'AMD',
            'QCOM', 'TXN', 'IBM', 'AVGO', 'NOW'
        ]
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        # Download data with robust error handling
        with st.spinner("📥 Downloading data from Yahoo Finance..."):
            prices = load_data_robust(
                tickers,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if prices is None or prices.empty:
            st.error("❌ Failed to load data. Please try again or contact support.")
            return
        
        # Check minimum requirements
        min_required = n_long + n_short
        if len(prices.columns) < min_required:
            st.error(f"❌ Need at least {min_required} stocks for {n_long} longs + {n_short} shorts")
            st.info(f"Only have {len(prices.columns)} stocks. Reduce position counts to {len(prices.columns)//2} or fewer.")
            return
        elif len(prices.columns) < 20:
            st.warning(f"⚠️ Have {len(prices.columns)} stocks (less than 20). Strategy will still work!")
        
        # Run backtest
        with st.spinner("🔄 Running backtest..."):
            results = run_backtest(prices, n_long, n_short, rebalance_days, lookback_days)
        
        if results is None:
            st.error("❌ Backtest failed. Check logs above.")
            return
        
        st.success("✓ Backtest complete!")
        
        # Results tabs
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
                x=equity.index, y=equity.values,
                name='Strategy', line=dict(color='#1E88E5', width=2)
            ))
            
            # Split line
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
                title='Equity Curve', xaxis_title='Date', yaxis_title='Cumulative Return',
                hovermode='x unified', height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison
            st.subheader("In-Sample vs Out-of-Sample")
            
            comparison = pd.DataFrame({
                'Metric': ['Ann. Return (%)', 'Sharpe Ratio', 'Max DD (%)', 'Win Rate (%)'],
                'In-Sample': [
                    ins['Annualized Return (%)'], ins['Sharpe Ratio'],
                    ins['Max Drawdown (%)'], ins['Win Rate (%)']
                ],
                'Out-of-Sample': [
                    oos['Annualized Return (%)'], oos['Sharpe Ratio'],
                    oos['Max Drawdown (%)'], oos['Win Rate (%)']
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
            
            # Momentum rankings
            st.subheader("📈 Current Momentum Rankings")
            
            last_mom = results['momentum_scores'].iloc[-1].dropna().sort_values(ascending=False)
            
            if len(last_mom) > 0:
                mom_df = pd.DataFrame({
                    'Stock': last_mom.index,
                    f'{lookback_days}-Day Return': [f"{m:.2%}" for m in last_mom.values]
                })
                st.dataframe(mom_df, use_container_width=True)
            else:
                st.info("No momentum data available")
        
        with tab3:
            st.header("Academic Foundation")
            
            st.markdown("""
            ### The Momentum Effect
            
            **Discovery:**
            - Jegadeesh & Titman (1993) in *Journal of Finance*
            - One of the most replicated findings in finance
            
            **Key Finding:**
            > Strategies that buy past winners and sell past losers generate significant positive returns.
            
            **Why It Works:**
            1. **Behavioral:** Investors under-react to news
            2. **Institutional:** Slow capital flows
            3. **Risk-based:** Compensation for crash risk
            
            ### This Implementation
            
            - **Lookback:** {lookback_days} days ({lookback_days/21:.0f} months)
            - **Rebalance:** Every {rebalance_days} days
            - **Long/Short:** {n_long} longs, {n_short} shorts
            - **Costs:** 15 bps per trade
            - **Validation:** 70/30 train/test split
            
            ### Further Reading
            
            1. Jegadeesh & Titman (1993) - Original paper
            2. Asness et al. (2013) - "Value and Momentum Everywhere"
            3. Antonacci (2014) - "Dual Momentum Investing"
            """.format(
                lookback_days=lookback_days,
                rebalance_days=rebalance_days,
                n_long=n_long,
                n_short=n_short
            ))
    
    else:
        st.info("""
        ### 👈 Configure parameters and click "Run Backtest"
        
        This implements the **momentum strategy** from Jegadeesh & Titman's 1993 paper.
        
        **Features:**
        - Real Yahoo Finance data
        - Academically validated methodology
        - Out-of-sample testing
        - Realistic transaction costs
        """)


if __name__ == "__main__":
    main()
