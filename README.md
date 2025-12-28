# ETF Trend Monitor

A systematic ETF trend-monitoring tool that evaluates moving-average strategies
across a liquid ETF universe.

Features:
- Universe filtering by category, liquidity, and inception date
- Local per-ticker price caching (incremental updates from last cached date)
- Moving-average trend evaluation across multiple windows
- Risk-aware filtering (Sharpe, drawdown, time-in-market)
- Designed for research, monitoring, and portfolio signal generation
