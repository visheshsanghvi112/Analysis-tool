// ==============================================================================
// StockIQ Pro — Institutional Financial Dictionary for InfoBadges (ⓘ)
// Concise definitions, strategic institutional rationale, and interpretation rules.
// ==============================================================================

export const INFO_DICTIONARY = {
  // ── Dashboard Overview & Status Badges ─────────────────────────────────────
  live_prices: {
    title: 'Real-Time Price & Market Feeds',
    what: 'Live intraday market quotes reflecting current transactions on the NSE and BSE.',
    why: 'Enables precise trade timing, slippage minimization, and accurate mark-to-market valuation.',
    interpretation: 'Official exchange feeds operate with a standard ~15-minute delay for retail feeds.'
  },
  ml_predictions: {
    title: 'Machine Learning Directional Forecasting',
    what: 'A 6-model ensemble (Random Forest, Gradient Boosting, XGBoost, LightGBM, Extra Trees, Ridge) predicting 5-day price trajectories.',
    why: 'Captures non-linear feature interactions across price momentum, volume volatility, and sentiment.',
    interpretation: 'Look for ensemble agreement across models and high directional probability (>65%).'
  },
  news_intelligence: {
    title: 'Scrapling Live News & Corporate Catalyst Engine',
    what: 'Real-time scraping and deep-reading of financial articles with corporate catalyst extraction and sentiment analysis.',
    why: 'Fundamental price shocks originate from corporate announcements, order wins, and regulatory probes before appearing on price charts.',
    interpretation: 'Bullish catalysts (order wins, rating upgrades) drive positive momentum; regulatory flags warrant caution.'
  },
  risk_analytics: {
    title: 'Quantitative Risk & Tail-Risk Analytics',
    what: 'Stochastic risk assessment incorporating Value at Risk (VaR), Maximum Drawdown, and Beta sensitivity.',
    why: 'Institutional capital preservation requires understanding extreme downside loss potential during market dislocations.',
    interpretation: 'Lower VaR and beta values indicate lower vulnerability to broader market crashes.'
  },

  // ── Live Price & Intraday Metrics ──────────────────────────────────────────
  prev_close: {
    title: 'Previous Close',
    what: 'The final traded benchmark price of the asset at the end of the previous trading session.',
    why: 'Serves as the zero-baseline for calculating daily price change and intraday percentage gains.',
    interpretation: 'Opening substantially above previous close indicates overnight buying pressure (gap up).'
  },
  day_range: {
    title: 'Day High & Day Low (Intraday Range)',
    what: 'The maximum and minimum transaction prices recorded during the current trading session.',
    why: 'Measures intraday price volatility and identifies local support (low) and resistance (high) levels.',
    interpretation: 'Trading near the day high signifies persistent buyer strength throughout the session.'
  },
  volume: {
    title: 'Trading Volume',
    what: 'Total number of shares or units exchanged between buyers and sellers during the session.',
    why: 'Volume validates price trends: price moves on high volume indicate institutional participation.',
    interpretation: 'A breakout on 2x+ average volume confirms strong institutional conviction.'
  },
  range_52w: {
    title: '52-Week High & Low',
    what: 'The highest and lowest price points reached by the security over the trailing 12 months.',
    why: 'Represents critical long-term psychological resistance and value support zones.',
    interpretation: 'Stocks near 52-week highs frequently demonstrate strong relative momentum (52-week high momentum anomaly).'
  },
  market_cap: {
    title: 'Market Capitalization',
    what: 'Total rupee market value of a company’s outstanding shares (Share Price × Outstanding Shares).',
    why: 'Defines asset scale, liquidity profile, and index categorization (Large-Cap, Mid-Cap, Small-Cap).',
    interpretation: 'Large-caps (>₹20,000 Cr) offer stability; mid/small-caps offer higher growth potential with increased volatility.'
  },

  // ── Technical Analysis & Chart Indicators ──────────────────────────────────
  candlestick_chart: {
    title: 'Interactive Candlestick Terminal',
    what: 'Visual representation displaying Open, High, Low, and Close (OHLC) prices across discrete time frames.',
    why: 'Provides deep visual insights into market sentiment, buyer-seller battles, and chart patterns.',
    interpretation: 'Green bars indicate net accumulation; red bars indicate net distribution or selling.'
  },
  sma_indicators: {
    title: 'Simple Moving Averages (SMA 20 & 50)',
    what: 'Arithmetic average of closing prices over trailing 20 and 50 trading days.',
    why: 'Smooths out short-term price noise to reveal prevailing medium-term trend direction.',
    interpretation: 'Price trading above both SMAs indicates a strong uptrend; a Golden Cross (SMA 20 > SMA 50) is a bullish buy signal.'
  },
  bollinger_bands: {
    title: 'Bollinger Bands (20, 2σ)',
    what: 'A 20-period moving average flanked by upper and lower bands set 2 standard deviations away.',
    why: 'Dynamically adapts to market volatility, identifying statistically overbought and oversold conditions.',
    interpretation: 'Touching the upper band signals potential overextension; a squeeze (narrowing bands) precedes high volatility breakouts.'
  },
  rsi_indicator: {
    title: 'Relative Strength Index (RSI 14)',
    what: 'A momentum oscillator measuring the speed and velocity of recent price changes on a scale of 0 to 100.',
    why: 'Identifies overbought (>70) and oversold (<30) market conditions and bullish/bearish divergences.',
    interpretation: 'RSI > 70 suggests extended rally and potential pullback; RSI < 30 indicates deeply discounted selling.'
  },
  macd_indicator: {
    title: 'MACD (Moving Average Convergence Divergence)',
    what: 'Trend-following momentum indicator displaying the relationship between the 12-day and 26-day EMAs.',
    why: 'Detects changes in the strength, direction, momentum, and duration of an emerging trend.',
    interpretation: 'MACD line crossing above Signal line generates a bullish entry signal; histogram expansion confirms momentum.'
  },

  // ── Machine Learning & Regime Forecasting ──────────────────────────────────
  ml_ensemble: {
    title: '6-Model Machine Learning Ensemble',
    what: 'Aggregates predictions from Random Forest, Gradient Boosting, XGBoost, LightGBM, Extra Trees, and Ridge regression.',
    why: 'Ensemble models significantly outperform single models by reducing variance, eliminating bias, and preventing overfitting.',
    interpretation: 'High model consensus (e.g. 5/6 models predicting UP) indicates high predictive reliability.'
  },
  markov_regime: {
    title: 'Hidden Markov Model (HMM) Regime Detection',
    what: 'Probabilistic framework classifying the asset into Bullish, Bearish, or Sideways/Volatile regimes.',
    why: 'Trading strategies that work in trending bull markets fail during high-volatility sideways regimes.',
    interpretation: 'Align your strategy with the detected regime: trend-following in Bull regimes, capital defense in Volatile regimes.'
  },

  // ── Fundamental Valuation & Forensics ──────────────────────────────────────
  dcf_valuation: {
    title: 'Discounted Cash Flow (DCF) Model',
    what: 'Estimates intrinsic fair value by projecting Free Cash Flows to Firm (FCFF) and discounting them via WACC.',
    why: 'The gold standard of corporate finance: values a business based on its true cash generation power rather than market hype.',
    interpretation: 'Intrinsic Value > Current Price indicates an undervalued asset offering a Margin of Safety.'
  },
  piotroski_f_score: {
    title: 'Piotroski F-Score (0–9)',
    what: 'A 9-point fundamental financial strength score evaluating profitability, leverage, liquidity, and operating efficiency.',
    why: 'Identifies fundamentally improving value companies while filtering out value traps.',
    interpretation: 'Scores of 8–9 indicate stellar financial health; scores 0–2 indicate critical operational distress.'
  },
  altman_z_score: {
    title: 'Altman Z-Score (Credit & Solvency)',
    what: 'A multivariate formula combining working capital, retained earnings, EBIT, market value, and sales.',
    why: 'Predicts probability of corporate insolvency or bankruptcy within a 2-year horizon.',
    interpretation: 'Z > 2.99 = Safe Zone; 1.81–2.99 = Grey Zone; Z < 1.81 = High Distress / Bankruptcy Risk.'
  },
  beneish_m_score: {
    title: 'Beneish M-Score (Forensic Accounting)',
    what: 'Mathematical model detecting whether a company is actively manipulating its reported financial earnings.',
    why: 'Protects investors from accounting fraud, aggressive revenue recognition, and hidden corporate rot.',
    interpretation: 'M-Score < -1.78 suggests clean accounting; M-Score > -1.78 signals high probability of earnings manipulation.'
  },
  dupont_analysis: {
    title: 'DuPont 3-Stage ROE Decomposition',
    what: 'Decomposes Return on Equity (ROE) into Net Profit Margin × Asset Turnover × Financial Leverage.',
    why: 'Reveals whether profitability is driven by true operating efficiency, rapid asset turnover, or dangerous debt leverage.',
    interpretation: 'High ROE driven by margins and asset turnover is sustainable; high ROE driven purely by debt leverage is hazardous.'
  },

  // ── Portfolio Theory & Risk Simulation ─────────────────────────────────────
  monte_carlo_var: {
    title: 'Monte Carlo 10,000-Path VaR Simulation',
    what: 'Simulates 10,000 forward price paths using Geometric Brownian Motion (GBM) stochastic calculus.',
    why: 'Models non-linear risk outcomes and severe market shock probabilities beyond normal distributions.',
    interpretation: '99% VaR specifies the maximum loss expected over the horizon with 99% statistical confidence.'
  },
  cvar_expected_shortfall: {
    title: 'Conditional Value at Risk (CVaR / Expected Shortfall)',
    what: 'The average loss in the worst 1% or 5% tail of all simulated market paths.',
    why: 'Measures catastrophic "black swan" tail risk when extreme market crashes occur.',
    interpretation: 'CVaR reveals the true pain threshold of extreme unexpected crises (e.g. 2008 or March 2020).'
  },
  sharpe_sortino: {
    title: 'Sharpe & Sortino Ratios',
    what: 'Sharpe measures excess return per unit of total risk; Sortino penalizes only downside volatility.',
    why: 'Distinguishes genuine alpha generation from lucky returns achieved by taking reckless risks.',
    interpretation: 'Sharpe > 1.0 is considered good; > 2.0 is institutional quality; Sortino > 1.5 shows strong downside protection.'
  },
  beta_alpha: {
    title: 'Beta & Jensen’s Alpha',
    what: 'Beta measures systematic correlation to Nifty 50; Alpha measures genuine excess return generated beyond benchmark.',
    why: 'Quantifies whether the manager/stock beats the market on a risk-adjusted basis.',
    interpretation: 'Beta = 1.0 moves in tandem with market; Beta < 0.8 is defensive; Positive Alpha indicates true outperformance.'
  },
  max_drawdown: {
    title: 'Maximum Historical Drawdown',
    what: 'The largest observed peak-to-trough percentage drop in asset price before a new peak is attained.',
    why: 'Crucial for investor psychology and stress-testing capital recovery timeframes.',
    interpretation: 'Lower drawdowns preserve compounding; a 50% drawdown requires a 100% gain just to break even.'
  },
  mpt_efficient_frontier: {
    title: 'Markowitz Modern Portfolio Theory (MPT)',
    what: 'Mathematical framework calculating optimal asset allocations that maximize expected return for a target volatility.',
    why: 'Diversification is the only "free lunch" in finance: combining uncorrelated assets reduces risk without sacrificing return.',
    interpretation: 'Portfolios lying on the upper frontier curve are mathematically optimal.'
  },

  // ── Strategy Backtesting & Wealth Planning ─────────────────────────────────
  strategy_backtesting: {
    title: 'Algorithmic Strategy Backtester',
    what: 'Simulates rule-based quantitative trading strategies across historical market data.',
    why: 'Validates whether an edge truly exists before risking real capital in live market conditions.',
    interpretation: 'Look for Win Rate > 55%, Profit Factor > 1.5, and low drawdown relative to benchmark buy-and-hold.'
  },
  sip_calculator: {
    title: 'SIP & Wealth Compounding Calculator',
    what: 'Simulates periodic Systematic Investment Plan cash flows compounded across historical asset return distributions.',
    why: 'Disciplined dollar-cost averaging removes emotional market timing and harnesses exponential compounding.',
    interpretation: 'Adjust inflation to observe true real purchasing power of accumulated wealth over 5–20 years.'
  },
  peer_valuation: {
    title: 'Relative Valuation & Peer Matrix',
    what: 'Comparative multiple analysis benchmarking Price-to-Earnings (P/E), Price-to-Book (P/B), and EV/EBITDA against industry peers.',
    why: 'Prevents overpaying by establishing whether a stock trades at a justifiable premium or discount to competitors.',
    interpretation: 'Low P/E coupled with superior ROE indicates high-conviction value investment opportunity.'
  }
};
