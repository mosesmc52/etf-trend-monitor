# ETF Trend Monitor

A systematic ETF trend-monitoring tool that evaluates moving-average strategies
across a liquid ETF universe.

Features:
- Universe filtering by category, liquidity, and inception date
- Local per-ticker price caching (incremental updates from last cached date)
- Moving-average trend evaluation across multiple windows
- Risk-aware filtering (Sharpe, drawdown, time-in-market)
- Designed for research, monitoring, and portfolio signal generation

## DigitalOcean Functions

Create the Functions env file from the template and fill in the required values:

```bash
cp infra/do-functions/.env.example infra/do-functions/.env
```

Deploy the function:

```bash
make do-fn-deploy DO_FN_ENV=infra/do-functions/.env
```

Tail the log on the ephemeral droplet:

```bash
make do-droplet-log DROPLET_IP=<ip>
```

If the droplet uses a different SSH user:

```bash
make do-droplet-log DROPLET_IP=<ip> DROPLET_USER=<user>
```

- Tasks
    - Keep Track of last run
    - Add AI commentary ?
    - Think about how to categorize shorts
