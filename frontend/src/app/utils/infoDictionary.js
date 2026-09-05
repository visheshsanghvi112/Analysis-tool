// ==============================================================================
// StockIQ Pro — Institutional Financial Dictionary for InfoBadges (ⓘ)
// Fully audited and cross-verified against backend algorithms & frontend components.
// Crisp definitions, quantitative mathematical rationale, and actionable rules of thumb.
// ==============================================================================

export const INFO_DICTIONARY = {
  // ── 1. Dashboard Status Row & Overview ───────────────────────────────────────
  live_prices: {
    title: 'Real-Time Market Feeds & Intraday Quote',
    what: 'Live intraday market transaction feed for NSE & BSE securities fetched via high-frequency pricing proxies with ~15-minute standard exchange latency.',
    why: 'Precision trade timing, slippage minimization, and accurate mark-to-market portfolio valuation require current benchmark price awareness.',
    interpretation: 'Always assess intraday quotes in context with trading volume to confirm whether price swings reflect genuine institutional flow.'
  },
  ml_predictions: {
    title: 'Machine Learning Directional Forecasting',
    what: 'A 6-model diverse stacked ensemble (Random Forest, Gradient Boosting, Extra Trees, Bayesian Ridge, XGBoost, LightGBM) blended with a Ridge meta-model and news sentiment fusion to forecast 5-day directional returns.',
    why: 'Single models suffer from individual inductive bias; stacking diverse tree architectures with Bayesian linear regularizers eliminates overfitting and captures non-linear momentum patterns.',
    interpretation: 'High predictive reliability requires consensus (>65% directional probability) across both tree-based models and the meta-learner.'
  },
  news_intelligence: {
    title: 'Scrapling Live News & Corporate Catalyst Engine',
    what: 'Real-time full-text article reading powered by Scrapling, classifying financial sentiment and automatically isolating high-impact corporate catalysts (order wins, earnings revisions, regulatory probes).',
    why: 'Material fundamental price shifts originate from corporate announcements and executive actions well before appearing on quantitative price charts.',
    interpretation: 'An Impact Score > 75 coupled with Bullish Catalysts signals institutional re-rating; regulatory or governance flags warrant immediate defensive hedging.'
  },
  risk_analytics: {
    title: 'Quantitative Risk & Sensitivity Suite',
    what: 'Multi-factor risk diagnostic suite calculating 1-Day Historical Value at Risk (95% & 99%), Maximum Peak-to-Trough Drawdown, Beta vs. NIFTY 50, Tracking Error, and Black-Scholes Options Greeks.',
    why: 'Institutional risk management focuses on tail-risk preservation: knowing your maximum probable capital drawdowns under volatile market regimes.',
    interpretation: 'Target a Sharpe Ratio > 1.0, 95% Daily VaR < 3.0%, and a Beta aligned with your portfolio risk tolerance (<0.8 defensive, >1.2 aggressive).'
  },

  // ── 2. Live Price & Intraday Metrics ──────────────────────────────────────────
  prev_close: {
    title: 'Previous Close Benchmark',
    what: 'The final official traded price of the security recorded at the 3:30 PM IST market close of the preceding trading session.',
    why: 'Serves as the absolute zero-reference point for calculating daily rupee gain/loss, intraday percentage change, and market gap-ups/gap-downs.',
    interpretation: 'Opening substantially above previous close indicates overnight institutional accumulation; opening below signals overnight distribution.'
  },
  day_high: {
    title: 'Day High (Session Peak Resistance)',
    what: 'The highest traded price of the security recorded during the current market session.',
    why: 'Marks the session\'s psychological resistance ceiling where supply temporarily overwhelmed buyer demand.',
    interpretation: 'Consolidating within 0.5% of the day high signals aggressive institutional accumulation with high breakout probability.'
  },
  day_low: {
    title: 'Day Low (Session Floor Support)',
    what: 'The lowest traded price of the security recorded during the current market session.',
    why: 'Marks the session\'s primary support floor where value-oriented buyers stepped in to absorb selling pressure.',
    interpretation: 'A rapid recovery bounce off the day low confirms institutional bids defending key technical price levels.'
  },
  day_range: {
    title: 'Intraday Trading Range & Volatility Spread',
    what: 'The full percentage spread between the day\'s lowest and highest traded prices, along with current price position inside that band.',
    why: 'Measures intraday volatility expansion and reveals whether buyers or sellers are controlling the session close.',
    interpretation: 'Current price holding above the 70th percentile of the day range reflects persistent buyer dominance into the close.'
  },
  volume: {
    title: 'Intraday Trading Volume & Conviction',
    what: 'The cumulative count of shares transacted across all executed orders during the active trading day.',
    why: 'Volume validates price discovery: breakouts on 2× or higher average daily volume confirm institutional participation and sustainability.',
    interpretation: 'Price rallies on below-average volume often represent weak "low-liquidity drifts" vulnerable to sudden reversals.'
  },
  range_52w: {
    title: '52-Week High & Low Range',
    what: 'The extreme boundary prices registered by the security across the trailing 252 market trading days.',
    why: 'Represents fundamental long-term institutional valuation boundaries and key psychological breakout levels.',
    interpretation: 'Stocks breaking out to fresh 52-week highs frequently exhibit sustained momentum driven by structural business growth.'
  },

  // ── 3. Interactive Chart & Technical Indicators ──────────────────────────────
  candlestick_chart: {
    title: 'Interactive OHLC Candlestick Terminal',
    what: 'High-resolution price chart depicting Open, High, Low, and Close (OHLC) values across configurable intraday and multi-year timeframes.',
    why: 'Candlestick bodies and wicks reveal the real-time balance of power between institutional buyers and profit-taking sellers.',
    interpretation: 'Long lower wicks indicate strong dip-buying; full-bodied candles closing near highs signify decisive trend continuation.'
  },
  rsi_indicator: {
    title: 'Relative Strength Index (RSI 14)',
    what: 'A 14-period momentum oscillator that calculates the velocity and magnitude of recent price advances versus declines on a 0–100 scale.',
    why: 'Identifies overextended momentum conditions and detects early trend divergences between price action and momentum.',
    interpretation: 'RSI > 70 denotes overbought conditions (watch for mean reversion); RSI < 30 indicates oversold territory; divergence warns of trend reversal.'
  },
  macd_indicator: {
    title: 'MACD (Moving Average Convergence Divergence)',
    what: 'Trend-following momentum indicator displaying the divergence between the 12-day and 26-day EMAs, paired with a 9-day exponential signal line.',
    why: 'Filters out short-term market noise to isolate accelerating medium-term momentum and structural trend shifts.',
    interpretation: 'MACD crossing above Signal line generates a bullish entry; expanding histogram confirms accelerating momentum.'
  },
  adx_indicator: {
    title: 'Average Directional Index (ADX 14)',
    what: 'A 14-period non-directional trend strength oscillator measuring the absolute intensity of prevailing market direction from 0 to 100.',
    why: 'Distinguishes between genuine trending market phases and choppy sideways consolidations where momentum indicators fail.',
    interpretation: 'ADX < 15 indicates no trend; 15–25 reflects a developing trend; ADX > 25 confirms a strong trend optimal for breakout strategies.'
  },

  // ── 4. Machine Learning & Quantitative Regimes ─────────────────────────────
  ml_ensemble: {
    title: '6-Model Stacked ML Ensemble Architecture',
    what: 'A production machine learning pipeline combining Random Forest (200 trees), Gradient Boosting, Extra Trees, Bayesian Ridge, XGBoost, and LightGBM, fused via a Ridge meta-estimator and news sentiment weighting.',
    why: 'Ensembling non-correlated regressors reduces generalization variance, prevents single-model hallucination, and elevates directional forecast accuracy.',
    interpretation: 'Review the consensus breakdown: agreement across 5+ models with tight confidence bounds indicates strong statistical edge.'
  },
  markov_regime: {
    title: '3-State Gaussian Hidden Markov Model (HMM)',
    what: 'Unsupervised machine learning model fitting a 3-state Gaussian HMM to historical return variance to classify latent market states: Low Volatility, Medium Volatility, or High Volatility / Trending.',
    why: 'Quantitative strategies perform radically differently across regimes; trend-following works in low-to-medium volatility but suffers severe whipsaws in high-volatility regimes.',
    interpretation: 'In Low Volatility regimes, position sizing can be maximized; in High Volatility regimes, reduce capital allocation and widen trailing stops.'
  },

  // ── 5. Portfolio Risk & Market Sensitivity ──────────────────────────────────
  risk_assessment: {
    title: 'Empirical Volatility & Tail Risk Diagnostics',
    what: 'Rigorous quantitative downside risk diagnostics: 1-Day Value at Risk at 95% & 99% confidence, Conditional Expected Shortfall, Peak-to-Trough Drawdown, Annualized Volatility, and Return Skewness.',
    why: 'Standard volatility metrics assume normal Gaussian distributions; tail-risk metrics capture the true catastrophic impact of fat-tailed market crash events.',
    interpretation: 'A 95% 1-Day VaR < 2.5% indicates manageable daily downside; negative skewness warns of rare but severe downside shock events.'
  },
  market_relationship: {
    title: 'Benchmark Co-Movement & Sensitivity (vs. NIFTY 50)',
    what: 'Statistical co-movement metrics calculated against the benchmark NIFTY 50 (^NSEI) index over the trailing 252 trading days: Beta, Pearson Correlation, Tracking Error, and Information Ratio.',
    why: 'Identifies whether an asset acts as an aggressive market amplifier, a defensive diversifier, or generates genuine active alpha.',
    interpretation: 'Beta = 1.0 mirrors the index; Beta < 0.8 is defensive; an Information Ratio > 0.5 confirms active return generation relative to tracking error.'
  },
  options_pricing: {
    title: 'Black-Scholes Options Pricing & Greeks (30D ATM)',
    what: 'Closed-form analytical European option pricing using the Black-Scholes PDE solver: computes fair call/put premiums, Delta, Gamma, Theta, Vega, and implied volatility with a 6.5% RBI benchmark risk-free rate.',
    why: 'Essential for quantitative portfolio hedging: enables institutional investors to calculate the exact cost of protective put options and analyze delta sensitivity.',
    interpretation: 'ATM Call/Put Delta is ~0.50; elevated option prices relative to historical volatility indicate high market-expected volatility ahead.'
  },
  sharpe_sortino: {
    title: 'Risk-Adjusted Return Metrics (Sharpe & Sortino)',
    what: 'Sharpe measures excess return per unit of total risk above a 6.5% risk-free rate; Sortino penalizes only harmful downside volatility.',
    why: 'Distinguishes genuine portfolio alpha from lucky gains achieved by taking reckless tail risks.',
    interpretation: 'Sharpe > 1.0 is good; Sharpe > 2.0 is elite; Sortino significantly higher than Sharpe indicates favorable upside volatility.'
  },
  beta_alpha: {
    title: 'Beta & Jensen’s Alpha Benchmark',
    what: 'Beta quantifies systematic market sensitivity against NIFTY 50; Jensen’s Alpha measures annualized excess return above the Capital Asset Pricing Model (CAPM) expectation.',
    why: 'Active management fees and stock selection are only justified if the manager generates positive Jensen’s Alpha.',
    interpretation: 'Positive Alpha (>0%) indicates the security is outperforming its risk-adjusted benchmark; Beta < 1.0 provides downside protection.'
  },

  // ── 6. Quantitative Backtesting & Monte Carlo ──────────────────────────────
  strategy_backtesting: {
    title: 'Dual-Momentum (RSI + MACD) + Dynamic ATR Trailing Stop',
    what: 'Systematic algorithmic backtester: enters long on either trend resumption (MACD cross with healthy RSI 40–68) or oversold rebound (RSI > 35 with expanding MACD histogram). Exits on RSI > 70 overbought, MACD bearish cross, or dynamic 2.0× ATR trailing stop-loss, incorporating 15 bps roundtrip friction.',
    why: 'Validates whether systematic trading rules produce positive net expectancy after real-world slippage, STT/taxes, and execution latency before risking capital.',
    interpretation: 'Look for Strategy Alpha > 0%, Sortino Ratio > 1.0, and smaller peak-to-trough drawdown and underwater duration than the market benchmark.'
  },
  strategy_alpha: {
    title: 'Strategy Alpha vs. Buy & Hold Benchmark',
    what: 'The net percentage outperformance of the quantitative trading strategy relative to simply holding the underlying stock over the identical lookback window.',
    why: 'Active trading incurs execution friction and tax events; positive alpha proves the rules systematically bypassed bear markets and preserved capital.',
    interpretation: 'Alpha > 0% demonstrates the strategy avoided major drawdowns while capturing the core compound growth of the security.'
  },
  monte_carlo_var: {
    title: 'Monte Carlo Value-at-Risk (VaR 95% & 99%)',
    what: 'The maximum percentage loss expected over the chosen horizon at a 95% (or 99%) confidence level across 1,000 simulated stochastic price paths.',
    why: 'Required by institutional risk standards (Basel III / FRTB): establishes the capital reserve needed to survive 19 out of 20 market scenarios.',
    interpretation: 'A 95% Horizon VaR of 15% means there is only a 5% statistical probability of losing more than 15% over the investment horizon.'
  },
  expected_shortfall: {
    title: 'Conditional VaR (CVaR) / Expected Shortfall',
    what: 'The average percentage loss sustained specifically in the catastrophic worst 5% of simulated market outcomes (beyond the 95% VaR threshold).',
    why: 'Standard VaR ignores the severity of losses in the tail; Expected Shortfall measures the true severity of a Black Swan market crash or tail event.',
    interpretation: 'A large gap between VaR and Expected Shortfall indicates high fat-tail risk (kurtosis) where crashes are particularly devastating.'
  },
  historical_bootstrap: {
    title: 'Non-Parametric Historical Bootstrapping',
    what: 'Generates forward price paths by resampling actual historical daily returns with replacement, without assuming any theoretical mathematical distribution.',
    why: 'Stock markets exhibit real-world fat tails, skewness, and sudden jumps that standard Gaussian models fail to capture; bootstrapping inherits the true empirical distribution of the security.',
    interpretation: 'Use Historical Bootstrap alongside Geometric Brownian Motion (GBM) to contrast theoretical projections with empirical historical reality.'
  },
  drift_shrinkage: {
    title: 'Bayesian Drift Shrinkage (Merton Estimation Risk Control)',
    what: 'Intelligently regularizes the stock\'s noisy sample return toward a 12% long-term market equity equilibrium prior, scaling shrinkage intensity with horizon length.',
    why: 'Historical sample mean return has massive standard error (Merton, 1980). Unconstrained momentum causes runaway multi-year exponential absurdity or unwarranted bankruptcy projections.',
    interpretation: 'Higher shrinkage alpha on 3Y/5Y horizons stabilizes multi-year projections against short-term momentum overfitting.'
  },
  horizon_probabilities: {
    title: 'Horizon Outcome Probabilities (Empirical Distribution)',
    what: 'Outcome probabilities derived from 1,000 simulated forward paths, computing exact odds of the stock finishing UP, Gain ≥ 5%, 10%, 20%, or Loss ≥ 5%, 10%, 20%.',
    why: 'Enables asymmetric risk-to-reward profiling before entering swing positions or structuring option spreads.',
    interpretation: 'Prob(UP) > 60% with upside probabilities heavily outweighing downside loss odds confirms an attractive risk-reward profile.'
  },

  // ── 7. Peer-to-Peer Duel & Wealth Compounding ────────────────────────────────
  peer_valuation: {
    title: 'Head-to-Head Peer Performance & Risk Duel',
    what: 'Direct competitive comparison benchmarking multi-timeframe returns (1M, 3M, 6M, 1Y), Sharpe ratio, annualized volatility, and RSI momentum against industry sector rivals.',
    why: 'Isolates true sector alpha: reveals whether a stock\'s momentum is driven by internal operational excellence or simply floating on broad sector tailwinds.',
    interpretation: 'The superior peer displays higher trailing return and Sharpe ratio (>1.0) paired with lower annualized risk volatility.'
  },
  sip_calculator: {
    title: 'SIP Compounding & Newton-Raphson XIRR Engine',
    what: 'Simulates Systematic Investment Plan (SIP) monthly wealth accumulation and calculates the exact Extended Internal Rate of Return (XIRR) using numerical Newton-Raphson iteration.',
    why: 'Dollar-cost averaging systematically buys more units during market drawdowns, removing emotional market timing and unleashing geometric compounding.',
    interpretation: 'Wealth creation is heavily back-loaded: over a 15–20 year horizon, compound returns typically generate 2× to 4× the cumulative principal invested.'
  },
  step_up_sip: {
    title: 'Step-Up (Top-Up) Annual Increment Engine',
    what: 'Models annual percentage increases (e.g. +10%/yr) in your monthly SIP installments, synchronizing investment contributions with annual salary increments.',
    why: 'Fixed SIP contributions lose real purchasing power to inflation over time. Stepping up contributions by just 10% annually can more than double your terminal wealth over 15–20 years.',
    interpretation: 'A 10% annual step-up drastically accelerates the compounding curve, cutting the time to achieve target wealth milestones by 3 to 6 years.'
  },
  target_goal_planner: {
    title: 'Target Goal Planner & Cost of Delay Suite',
    what: 'Reverse-calculates the exact monthly SIP installment required to accumulate a predetermined financial corpus (e.g. ₹1 Crore or $1 Million) across target time horizons.',
    why: 'Helps investors anchor financial planning around tangible life goals (retirement, child education, financial independence) rather than arbitrary monthly guesses.',
    interpretation: 'Calculates the exponential "Cost of Delay": delaying your investment journey by even 5 years often requires more than double the monthly capital to reach the identical corpus.'
  },
  inflation_adjusted_corpus: {
    title: 'Real Purchasing Power vs. Nominal Corpus',
    what: 'Discounts nominal future wealth using compound annual inflation (default 6% p.a.) to reveal the actual purchasing power of your future corpus in today\'s money.',
    why: 'Nominal future crores or millions can be misleading due to purchasing power decay; evaluating real purchasing power ensures long-term lifestyle goals are genuinely met.',
    interpretation: 'At 6% inflation, prices double roughly every 12 years. ₹1 Crore 20 years from now will possess approximately ₹31 Lakh of today\'s purchasing power.'
  },
  mpt_efficient_frontier: {
    title: 'Markowitz Efficient Frontier & SLSQP Portfolio Optimizer',
    what: 'Solves constrained quadratic optimization problems (using SLSQP) across your holdings\' historical covariance matrix to locate the mathematical Maximum Sharpe (tangency) and Minimum Volatility portfolios.',
    why: 'Harry Markowitz proved diversification is the only "free lunch" in finance: combining non-correlated assets reduces portfolio variance without sacrificing expected return.',
    interpretation: 'Portfolios on the upper frontier curve dominate all sub-optimal allocations; align weights with the Max Sharpe portfolio to optimize risk-adjusted growth.'
  },

  // ── 8. Fundamental Valuation & Corporate Health ─────────────────────────────
  dcf_valuation: {
    title: 'Discounted Cash Flow (DCF) & Margin of Safety',
    what: 'Computes intrinsic enterprise value by forecasting 5-year Free Cash Flows, discounting via the Weighted Average Cost of Capital (WACC), and adding discounted terminal value.',
    why: 'The gold standard of institutional valuation: values a business based on its true cash generation power rather than speculative market multiples.',
    interpretation: 'Intrinsic Value > Current Price yields a positive Margin of Safety (>15% provides a protective buffer against forecast errors).'
  },
  dupont_analysis: {
    title: 'DuPont 3-Stage ROE Decomposition',
    what: 'Decomposes Return on Equity (ROE) into Net Profit Margin × Asset Turnover × Financial Leverage.',
    why: 'Uncovers whether profitability is powered by genuine operating margins and asset efficiency, or masked by dangerous debt leverage.',
    interpretation: 'High ROE driven by margins and asset turnover is resilient; high ROE driven predominantly by debt leverage (>3.0×) carries bankruptcy risk.'
  },
  graham_number: {
    title: 'Benjamin Graham Intrinsic Valuation Number',
    what: 'Computes conservative upper-bound acquisition price using Ben Graham\'s classic formula: √(22.5 × Trailing EPS × Book Value per Share).',
    why: 'Graham established that a defensive investor should never pay a price where P/E × P/B exceeds 22.5.',
    interpretation: 'Current Price < Graham Number represents deep value with an automatic margin of safety.'
  },
  peg_ratio: {
    title: 'PEG Ratio (Price/Earnings-to-Growth)',
    what: 'Normalized valuation multiple calculating Price-to-Earnings divided by expected annual EPS growth rate percentage.',
    why: 'Invented by legendary investor Peter Lynch. A P/E of 30x is cheap if earnings grow at 40%/yr (PEG = 0.75), whereas a P/E of 15x is expensive if growth is 5%/yr (PEG = 3.0).',
    interpretation: 'PEG < 1.0 indicates undervalued growth; 1.0–2.0 indicates fair value; > 2.0 indicates market is pricing in aggressive speculative growth.'
  },
  ev_ebitda: {
    title: 'EV / EBITDA (Enterprise Valuation Multiple)',
    what: 'Enterprise Value (Market Cap + Total Debt - Cash) divided by Earnings Before Interest, Taxes, Depreciation, and Amortization.',
    why: 'Unlike standard P/E, EV/EBITDA is capital-structure neutral: it accounts for debt burdens and allows clean comparisons between companies with different leverage.',
    interpretation: 'Lower values (< 10x–12x depending on sector) signal attractive valuation; > 20x indicates high growth expectations.'
  },
  net_debt: {
    title: 'Net Debt Position & Solvency Health',
    what: 'Total debt liabilities minus total cash and cash equivalents on the corporate balance sheet.',
    why: 'Companies with negative net debt (Net Cash) are virtually immune to bankruptcy, interest rate hikes, and credit crunches.',
    interpretation: 'Net Cash provides dry powder for buybacks, acquisitions, or dividends; high Net Debt (> 3x annual EBITDA) raises debt distress risk.'
  },
  pros_and_cons: {
    title: 'Automated Investment Strengths & Risks Digest',
    what: 'Algorithmic balance sheet, margin, valuation, and governance synthesis inspired by Screener.in, classifying raw corporate filings into clear strengths and vulnerabilities.',
    why: 'Prevents emotional bias and analysis paralysis: gives retail investors an instant, objective reality check before deploying hard-earned capital.',
    interpretation: 'A robust compounder exhibits 3+ structural pros (low leverage, pricing power, PEG < 1.0) with zero critical solvency or promoter dilution flags.'
  },
  sales_cagr: {
    title: 'Compounded Sales & Profit Growth (3Y CAGR)',
    what: 'Compound Annual Growth Rate (CAGR) measuring top-line revenue and bottom-line net profit expansion over a multi-year economic cycle.',
    why: 'Stock prices ultimately mirror corporate cash flow generation over 3–5 year horizons. Revenue growth validates customer demand and market share gains.',
    interpretation: 'Look for Profit CAGR matching or exceeding Sales CAGR, which indicates expanding operational efficiency and operating leverage.'
  },
  net_debt_to_ebitda: {
    title: 'Net Debt / EBITDA Payoff Horizon',
    what: 'Calculates the number of years of current annual operating profit (EBITDA) required to completely extinguish all outstanding net borrowings.',
    why: 'The gold standard solvency metric used by credit rating agencies and banks to evaluate real debt serviceability.',
    interpretation: '< 1.5x indicates debt is easily serviceable and low risk; 1.5x–3.0x is moderate; > 3.5x indicates heavy debt burden vulnerable to rate shocks.'
  },

  // ── 9. Intraday Trading Desk & Microstructure ────────────────────────────
  vwap: {
    title: 'Volume-Weighted Average Price (VWAP)',
    what: 'The cumulative benchmark price weighted by volume transacted across the active trading session: VWAP = Σ(Typical Price × Volume) / Σ(Volume).',
    why: 'Used by institutional execution desks and pension funds as the primary liquidity benchmark. Buying below VWAP represents an institutional discount.',
    interpretation: 'Holding firmly above VWAP signals buyer control (bullish regime); trading below VWAP indicates active seller absorption and resistance.'
  },
  vwap_bands: {
    title: 'VWAP Volatility Envelope (±1σ, ±2σ, ±3σ)',
    what: 'Standard deviation volatility bands plotted around the session VWAP curve, measuring statistical price dispersion across active intraday volume.',
    why: 'Institutional market makers treat ±2σ as dynamic statistical value boundaries; prices touching +2σ or -2σ without extreme volume often revert toward VWAP.',
    interpretation: 'Price tagging the lower -2σ band with bullish reversal candles provides high-probability mean-reversion long entries; breakouts beyond +2σ on high volume indicate trend runs.'
  },
  volume_profile: {
    title: 'Volume Profile (VPVR) & Point of Control (POC)',
    what: 'A horizontal histogram displaying cumulative trading volume executed at specific price intervals throughout the session, identifying the Point of Control (POC) and the 70% Value Area (VAH/VAL).',
    why: 'Reveals where institutions actually transacted capital rather than just where prices traveled. POC acts as an institutional liquidity magnet.',
    interpretation: 'Price testing POC or Value Area Low (VAL) from above typically finds strong responsive buying; trading outside the Value Area represents price discovery.'
  },
  camarilla_pivots: {
    title: 'Camarilla Equation & Institutional Breakout Levels',
    what: 'A mathematical price-action formula using previous session\'s High, Low, and Close to generate 8 critical inflection levels (H1–H4 and L1–L4).',
    why: 'Designed specifically for intraday scalping and mean-reversion: L3 and H3 act as institutional reversal bounds, while L4 and H4 trigger explosive trend breakouts.',
    interpretation: 'Look for long entries around L3 with targets at H3; if price breaks above H4 on heavy volume, trade breakout continuation toward H5.'
  },
  floor_pivots: {
    title: 'Classic Floor Pivot Points (P, R1–R3, S1–S3)',
    what: 'Standard floor trader pivot equations calculating the central pivot (P = [H+L+C]/3) and multi-tier geometric support and resistance levels.',
    why: 'Provides broad intraday reference points watched concurrently by algorithmic order books and human market makers globally.',
    interpretation: 'Trading above the Central Pivot establishes an intraday bullish bias; S1 and R1 serve as primary session targets and reaction nodes.'
  },
  orb_strategy: {
    title: 'Opening Range Breakout (ORB 15m / 30m)',
    what: 'The price envelope defined by the absolute High and Low established during the initial 15 or 30 minutes of market opening.',
    why: 'The opening auction absorbs overnight orders and establishes institutional direction for the remainder of the session.',
    interpretation: 'A decisive candle close above the 15m High on elevated volume triggers high-probability momentum long trades; failure to break either boundary indicates choppy range-bound trading.'
  },
  supertrend: {
    title: 'Institutional Intraday Supertrend (10, 3)',
    what: 'A trend-following volatility indicator that combines median price with an Average True Range (ATR) multiplier (10 periods, 3× multiplier) to construct a dynamic trailing stop line.',
    why: 'Eliminates emotional whipsaws by offering an unambiguous binary trend status (Green = Bullish Long, Red = Bearish Short) alongside exact risk stops.',
    interpretation: 'Stay long while price closes above the green trailing band; flip to short or tighten stops immediately when price violates the band to the downside.'
  },
  order_flow_delta: {
    title: 'Microstructure Order Flow & Cumulative Volume Delta (CVD)',
    what: 'Approximates buyer-initiated versus seller-initiated trade volume per candle, tracking the net cumulative delta (Aggressive Buy Volume minus Aggressive Sell Volume).',
    why: 'Reveals whether aggressive market orders are absorbing passive liquidity limit orders, uncovering hidden institutional accumulation or distribution.',
    interpretation: 'Rising prices accompanied by expanding positive CVD confirm genuine aggressive buying; price rising while CVD is falling warns of absorption and impending exhaustion.'
  },
  intraday_rvol: {
    title: 'Relative Volume (RVOL)',
    what: 'The ratio of current trading volume to the average historical volume expected at the identical time of the trading day.',
    why: 'Volume is the fuel of price action. Breakouts occurring on RVOL < 1.0 lack conviction; breakouts on RVOL > 2.0x signal major institutional sponsorship.',
    interpretation: 'Target intraday setups with RVOL > 1.5x for sustainable trend follow-through and reduced whipsaw risk.'
  },
  mis_leverage: {
    title: 'Intraday Margin Intraday Square-off (MIS) & Risk Rules',
    what: 'Exchange-regulated intraday margin offering up to 5× leverage on liquid equities, requiring positions to be squared off before the session close.',
    why: 'Leverage magnifies both returns and capital destruction; institutional money management strictly mandates risking no more than 1% to 2% of total capital on any single scalp.',
    interpretation: 'Calculate exact share quantities using your fixed rupee/dollar risk divided by the distance between entry and stop-loss: Never adjust stop-loss to match position size.'
  },
  intraday_quant_score: {
    title: 'Composite Intraday Quant Score (-100 to +100)',
    what: 'A multi-factor algorithmic index fusing VWAP alignment (±25 pts), Supertrend state (±25 pts), EMA ribbon momentum (±20 pts), ORB breakout status (±20 pts), and RSI velocity (±10 pts).',
    why: 'Synthesizes disparate price, volume, and volatility signals into an objective mathematical directional bias, eliminating subjective trader hesitation.',
    interpretation: 'Score ≥ +50 triggers Strong Buy bias; Score ≤ -50 triggers Strong Sell bias; Scores between -20 and +20 recommend waiting for clearer setup confirmation.'
  },
  session_phase_clock: {
    title: 'Market Session Phase Clock & MIS Auto-Square-Off',
    what: 'Divides the trading session into 5 distinct psychological and liquidity regimes (Opening Auction, Morning Drive, Midday Chop, Afternoon Breakout, and MIS Auto-Square-Off Panic at 3:15 PM IST).',
    why: 'Trading setups have drastically varying win rates across phases. Breakouts during the 11:30 AM – 1:30 PM lunch slump fail over 70% of the time due to institutional absence.',
    interpretation: 'Concentrate primary aggressive trend scalps in Phase 2 (9:45 – 11:30 AM); reduce risk by 50% during Phase 3 chop; and close all MIS intraday positions prior to 3:15 PM broker auto-square-off.'
  },
  brokerage_friction_breakeven: {
    title: 'Brokerage, STT & Friction Breakeven Spread',
    what: 'Calculates the exact total statutory deductions (flat ₹20 brokerage, 0.025% STT, 0.00297% NSE charges, 18% GST, SEBI fee, Stamp duty) and computes the exact price move required to reach net breakeven.',
    why: 'SEBI studies reveal that transaction friction consumes >30% of gross intraday retail profits. Entering setups where the breakeven move exceeds 0.15% guarantees structural losses.',
    interpretation: 'Ensure your target distance is at least 3× to 5× your breakeven spread: Never scalp micro-ticks where transaction costs exceed your net expected profit.'
  },
  pre_market_gap: {
    title: 'Pre-Market Gap Intelligence & Fill Probability',
    what: 'Classifies session opening gaps into Full Gap Up/Down, Partial Gap Up/Down, or Flat, tracking live whether the price has retested and filled the gap back to previous close.',
    why: 'Opening gaps represent overnight order imbalances. Full gap-ups that hold above VWAP exhibit Gap-and-Go continuation; gap-ups failing below VWAP trigger high-probability Gap-Fade reversals.',
    interpretation: 'Look for Gap-and-Go momentum when a full gap-up trades above VWAP with RVOL > 1.5x; initiate Gap-Fade short scalps targeting previous close when price slips below VWAP.'
  },
  institutional_trap_detector: {
    title: 'Institutional Trap & Liquidity Sweep Detector',
    what: 'Monitors the session high and low for false breakouts (Bull Traps and Bear Traps) caused by institutional liquidity sweeps into retail stop-loss clusters.',
    why: 'Market makers routinely push price slightly above Day High to trigger retail buy-stops, absorb the liquidity with passive sell orders, and aggressively reverse price downward.',
    interpretation: 'When a new Day High is printed on decaying volume or negative delta (divergence), avoid chasing the breakout and prepare for mean-reversion toward VWAP.'
  },
  triple_screen_confluence: {
    title: 'Triple-Screen Multi-Timeframe Confluence Matrix',
    what: 'Dr. Alexander Elder\'s institutional multi-timeframe methodology evaluating 5m (Setup), 15m (Wave), and 1h (Tide) trend, EMA alignment, and RSI simultaneously.',
    why: 'Trading exclusively in the direction of higher timeframes filters out misleading lower-timeframe noise and dramatically elevates trade win-rate.',
    interpretation: 'Confluence Score ≥ 75% indicates full multi-timeframe alignment; trade aggressively in the trend direction. When confluence is < 40%, market is in choppy counter-trend conflict.'
  },
  benchmark_relative_strength: {
    title: 'Benchmark Relative Strength & Alpha (vs. NIFTY / S&P)',
    what: 'Measures the intraday percentage outperformance or underperformance of a security relative to its benchmark index (^NSEI for India, ^GSPC for US).',
    why: 'Leading institutional stocks refuse to fall even when the broad market is tumbling. Buying high relative strength leaders ensures you have structural market sponsorship.',
    interpretation: 'Alpha > +1.0% identifies institutional leadership; avoid longing stocks with negative alpha (< -1.0%) even if their individual chart looks attractive.'
  },
  intraday_battle_plan: {
    title: 'Actionable Intraday Battle Plan & Execution Card',
    what: 'A structured, pre-calculated algorithmic trade card specifying the exact Entry Trigger, Hard Stop Loss, Target 1 (1.5R), Target 2 (2.5R), and net profit after all charges.',
    why: 'Eliminates emotional hesitation and panic. Real-world prop trading mandates having a fully written, quantified trade plan before transmitting any live order.',
    interpretation: 'Execute only when price meets the exact trigger rule; trail stop-loss to breakeven once Target 1 is achieved to guarantee a risk-free runner.'
  },
  pdh_pdl_levels: {
    title: 'Previous Day High (PDH), Low (PDL) & Close (PDC)',
    what: 'Critical horizontal reference price boundaries established during the previous trading day\'s regular session.',
    why: 'Floor traders, institutional market makers, and algorithmic execution algorithms treat PDH, PDL, and PDC as the most vital liquidity and rejection barriers of the active day.',
    interpretation: 'A clean 5m candle close above PDH indicates bullish trend continuation; rejection at PDH indicates an institutional liquidity sweep. Sustained trading below PDL signals heavy institutional liquidation.'
  },
  traders_scratchpad: {
    title: 'Trader\'s Execution Scratchpad & Real-Time Journal',
    what: 'A private, real-time tactical journal stored locally in browser storage for logging trade hypotheses, entry rules, stops, and psychological discipline.',
    why: 'Proprietary trading desk studies show that traders who document their setups prior to order submission maintain 40% tighter drawdown control and avoid revenge trades.',
    interpretation: 'Log your reason for entry, execution trigger price, and stop discipline prior to placing trades; add timestamps to track price action evolution.'
  },
  options_pcr: {
    title: 'Options Put-Call Ratio (PCR Open Interest)',
    what: 'The ratio of total Open Interest (OI) in Put options divided by total Open Interest in Call options across all strikes of the active derivatives series.',
    why: 'Acts as a contrarian intraday sentiment barometer. Heavy put writing (PCR > 1.2) creates an options support floor, while heavy call writing (PCR < 0.8) forms an overhead ceiling.',
    interpretation: 'PCR > 1.3 indicates oversold conditions ripe for a short-covering rally; PCR < 0.7 signals extreme bullish complacency and warns of an impending intraday long squeeze.'
  },
  block_deals: {
    title: 'Institutional Block & Bulk Deals Feed',
    what: 'Exchange-reported high-value transactions involving a minimum of 5 lakh shares or ₹5 crore executed through a single dedicated institutional window.',
    why: 'Reveals where sovereign wealth funds, domestic mutual funds (DIIs), and foreign portfolio investors (FIIs) are committing large-scale institutional liquidity.',
    interpretation: 'Clusters of large block buys near key support confirm institutional accumulation; large block sells below VWAP warn of institutional portfolio liquidation.'
  },
  trade_log: {
    title: 'Intraday Execution Log & Trade Journal',
    what: 'Real-time trade ledger tracking active and closed positions, entry/exit fills, gross P&L, statutory friction, and session win/loss distribution.',
    why: 'Professional trading firms enforce strict execution logging to prevent emotional revenge trading, enforce daily drawdown limits, and maintain risk symmetry.',
    interpretation: 'Review your gross vs net P&L after friction to ensure your average winning trade adequately covers brokerage, exchange turnover, and STT charges.'
  }
};
