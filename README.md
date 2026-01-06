# Simple Momentum Trading Strategy

**Quantitative Research Project | Duke University**

A clean implementation of the momentum trading strategy based on **Jegadeesh & Titman (1993)**, one of the most replicated findings in financial economics.

## 🎯 What This Is

A **simple, academically-validated trading strategy** that:
- Buys stocks with high recent returns (momentum)
- Shorts stocks with low recent returns
- Rebalances monthly
- Uses real market data from Yahoo Finance

**This is intentionally simple** - it works, it's proven, and it's easy to explain.

## 📊 The Strategy

### Concept
**Momentum**: Stocks that performed well recently tend to continue performing well (and vice versa).

### Implementation
1. Calculate 12-month return for each stock
2. Rank all stocks by momentum
3. **Long** top 10 stocks (highest momentum)
4. **Short** bottom 10 stocks (lowest momentum)
5. Rebalance every 21 days (monthly)
6. Include transaction costs (15 bps total)

### Academic Foundation
- **Paper**: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"
- **Journal**: *Journal of Finance* (top-tier academic journal)
- **Finding**: Momentum generates ~1% per month (12%+ annually)
- **Robustness**: Replicated in 200+ academic papers since 1993

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Backtest
```bash
python main_simple.py
```

### Launch Web App
```bash
streamlit run app_simple.py
```

## 📁 Project Structure

```
.
├── simple_momentum.py      # Core strategy implementation
├── main_simple.py          # Command-line backtest
├── app_simple.py           # Streamlit web interface
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 💡 Why This Strategy?

### Advantages
✅ **Academically validated** - 30+ years of research  
✅ **Actually works** - Real funds use variations of this  
✅ **Simple to implement** - ~200 lines of clean code  
✅ **Easy to explain** - "Buy winners, sell losers"  
✅ **Robust** - Works across markets and time periods  

### Limitations
⚠️ Can have large drawdowns (20-30%)  
⚠️ Performance varies by market regime  
⚠️ Transaction costs matter (use monthly rebalancing)  
⚠️ Crowding risk (many funds use momentum)  

## 📈 Expected Performance

Based on academic research and realistic assumptions:

| Metric | Expected Range |
|--------|----------------|
| Annual Return | 8-15% |
| Sharpe Ratio | 0.5-1.2 |
| Max Drawdown | -20% to -35% |
| Win Rate | 52-58% |

**Note**: Performance depends heavily on the time period tested.

## 🎓 Interview Talking Points

### What to Say
✅ "I implemented a momentum strategy based on Jegadeesh & Titman's 1993 paper in the *Journal of Finance*"  
✅ "Used 3 years of real market data from Yahoo Finance for 20 large-cap stocks"  
✅ "Validated with 70/30 train/test split to prevent overfitting"  
✅ "Included realistic transaction costs of 15 basis points"  
✅ "Momentum is one of the most robust anomalies in finance - replicated in hundreds of papers"  

### Technical Details
- **Lookback period**: 252 trading days (1 year)
- **Rebalance frequency**: 21 days (monthly)
- **Position sizing**: Equal-weight long/short
- **Universe**: 20 liquid large-cap US equities
- **Costs**: 10 bps transaction cost + 5 bps slippage

### Why Momentum Works
1. **Behavioral**: Investors under-react to news
2. **Institutional**: Slow capital flows in large funds
3. **Risk premium**: Compensation for crash risk

## 📚 Further Reading

### Academic Papers
1. **Jegadeesh & Titman (1993)** - Original momentum paper
2. **Asness, Moskowitz & Pedersen (2013)** - "Value and Momentum Everywhere"
3. **Novy-Marx (2012)** - "Is Momentum Really Momentum?"

### Books
1. **Antonacci (2014)** - "Dual Momentum Investing"
2. **Asness et al. (2015)** - "Fact, Fiction and Momentum Investing"

## 🛠️ Technical Implementation

### Core Logic
```python
# Calculate momentum
momentum = prices.pct_change(periods=252)

# Rank stocks
for each rebalance_date:
    long_stocks = momentum.nlargest(10)
    short_stocks = momentum.nsmallest(10)
    
    # Equal weight
    positions[long_stocks] = +0.1  # 10% each = 100% long
    positions[short_stocks] = -0.1  # 10% each = 100% short
```

### Key Features
- **Modular design**: Strategy class separate from data/execution
- **Efficient caching**: Streamlit caching for fast reruns
- **Robust error handling**: Handles missing data gracefully
- **Clean output**: Clear logging of all steps

## 📊 Data

### Source
**Yahoo Finance** via `yfinance` library

### Universe
```python
['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
 'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
 'JNJ', 'PG', 'MA', 'UNH', 'HD',
 'DIS', 'BAC', 'ADBE', 'NFLX', 'CRM']
```

### Period
- **Length**: 3 years
- **Frequency**: Daily
- **Fields**: Adjusted close prices

## 🎯 Project Goals

This project demonstrates:
1. **Understanding of quantitative finance** - Factor investing, backtesting
2. **Academic research skills** - Implementing published strategies
3. **Software engineering** - Clean, modular, documented code
4. **Statistical validation** - Out-of-sample testing, performance metrics
5. **Communication** - Clear explanation of complex concepts

## ⚖️ Disclaimer

This is an **educational project** for learning quantitative finance concepts.

**Not intended for real trading:**
- Simplified implementation
- Limited universe (20 stocks)
- No live trading infrastructure
- No risk management beyond basic sizing

For real trading, you would need:
- Larger universe (100+ stocks)
- More sophisticated risk management
- Better execution (limit orders, smart routing)
- Regular monitoring and rebalancing
- Proper backtesting infrastructure

## 👤 Author

**Soham Gugale**  
Duke University  
Master's in Computational Mechanics  
[GitHub](https://github.com/sohamgugale)

## 📝 License

MIT License - Feel free to use for learning and research.

---

**Built with**: Python, Pandas, Streamlit, yfinance  
**Inspired by**: 30+ years of momentum research in academic finance  
**Purpose**: Demonstrate quantitative research skills for recruiting
