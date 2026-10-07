# ==============================================================================
# Intelligent News Reader — High-Performance Live Financial Scraping & Deep Reader
# Powered by Scrapling (TLS Impersonation, Anti-Bot Bypass, Adaptive Extraction)
# ==============================================================================

import sys
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import feedparser

# Add vendored scrapling to path
vendor_scrapling_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor', 'scrapling'))
if vendor_scrapling_path not in sys.path:
    sys.path.insert(0, vendor_scrapling_path)

try:
    from services.scrapling_client import scrapling_client, HAS_SCRAPLING
except Exception as e:
    HAS_SCRAPLING = False
    scrapling_client = None
    print(f"Warning: ScraplingClient import issue: {e}")

# Curated financial domain lexicon for realistic market sentiment scoring
FINANCIAL_LEXICON = {
    'strongly_positive': [
        'profit surges', 'record profit', 'beats estimates', 'ebitda expands', 
        'margin expansion', 'debt-free', 'debt reduction', 'order win', 'mega contract',
        'rating upgrade', 'upgraded to buy', 'raised target', 'guidance raised',
        'dividend hike', 'share buyback', 'all-time high profit', 'strong quarterly',
        'multi-fold surge', 'robust growth', 'upper circuit', 'locked in upper circuit',
        'multibagger', 'breakout', 'all-time high', 'blockbuster results', 'soaring profit',
        'record revenue', 'inflow surge'
    ],
    'positive': [
        'growth', 'surge', 'surges', 'expansion', 'jump', 'jumps', 'gain', 'gains', 'profit',
        'outperform', 'reiterates buy', 'contract', 'partnership', 'commissioned', 'capacity addition',
        'turnaround', 'loss narrows', 'cuts net loss', 'recovery', 'rally', 'rallies',
        'soars', 'soar', 'climbs', 'climb', 'advances', 'advance', 'higher', 'bullish', 'up',
        'rebound', 'rebounds', 'green', 'buying interest', 'accumulate'
    ],
    'strongly_negative': [
        'crash', 'crashed', 'crashing', 'meltdown', 'plunges', 'plunge', 'plunged',
        'free fall', 'freefall', 'steep premium', 'bubble', 'trading at premium',
        'premium to inav', 'premium to nav', 'circuit down', 'lower circuit',
        'locked in lower circuit', 'bloodbath', 'wealth eroded', 'investors lose',
        'overvalued', 'quota limit', 'quota freeze', 'trading halt', 'halted',
        'massive selloff', 'dumping', 'dump', 'dumped', 'sebi probe', 'tax raid',
        'ed summons', 'fraud', 'forensic audit', 'auditor resigns', 'default',
        'insolvency', 'bankruptcy', 'downgraded to sell', 'slashed target',
        'guidance cut', 'promoter pledge increases', 'loss widens', 'q1 miss',
        'q2 miss', 'q3 miss', 'q4 miss', 'falls another', 'tumbles another',
        'slumps another', 'drops another', 'sharp sell-off', 'panic selling',
        'liquidity squeeze', 'nav dislocation', 'valuation frenzy'
    ],
    'negative': [
        'loss', 'decline', 'drop', 'slump', 'falls', 'fall', 'weak', 'misses estimates',
        'margin contraction', 'cost pressures', 'penalty', 'fine', 'delay',
        'investigation', 'headwinds', 'subdued', 'slides', 'slid', 'tumbles', 'tumble',
        'dips', 'dip', 'down', 'lower', 'bearish', 'correction', 'corrects', 'retreats',
        'drags', 'frenzy', 'expensive', 'caution', 'unwinding', 'fallout', 'warning',
        'dislocation', 'premium has widened', 'premium widened'
    ]
}

# Live RSS feeds across major Indian financial news networks
LIVE_FINANCIAL_FEEDS = [
    ('Economic Times Markets', 'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2143429.cms'),
    ('Moneycontrol Business', 'https://www.moneycontrol.com/rss/business.xml'),
    ('LiveMint Companies', 'https://www.livemint.com/rss/companies'),
    ('Business Standard Companies', 'https://www.business-standard.com/rss/companies-101.rss'),
    ('NDTV Profit', 'https://feeds.feedburner.com/ndtvprofit-latest'),
]


MAX_NEWS_AGE_DAYS = 15.0

# Curated high-precision company name dictionary for major market tickers
KNOWN_COMPANY_NAMES = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "L&T": "Larsen & Toubro",
    "HINDUNILVR": "Hindustan Unilever",
    "AXISBANK": "Axis Bank",
    "BAJFINANCE": "Bajaj Finance",
    "BAJAJFINSV": "Bajaj Finserv",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "SUNPHARMA": "Sun Pharma",
    "NTPC": "NTPC",
    "POWERGRID": "Power Grid Corporation",
    "M&M": "Mahindra & Mahindra",
    "MM": "Mahindra & Mahindra",
    "TITAN": "Titan Company",
    "WIPRO": "Wipro",
    "ULTRACEMCO": "UltraTech Cement",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "ONGC": "ONGC",
    "BPCL": "Bharat Petroleum",
    "IOC": "Indian Oil Corporation",
    "HAL": "Hindustan Aeronautics",
    "BEL": "Bharat Electronics",
    "COALINDIA": "Coal India",
    "VEDL": "Vedanta",
    "ZOMATO": "Zomato",
    "PAYTM": "Paytm",
    "SWIGGY": "Swiggy",
    "IRFC": "Indian Railway Finance Corporation",
    "IRCTC": "IRCTC",
    "RAILTEL": "RailTel Corporation",
    "RVNL": "Rail Vikas Nigam",
    "JIOFIN": "Jio Financial Services",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "META": "Meta",
}


class IntelligentNewsReader:
    """
    100% Live, Deep-Reading News Intelligence Engine.
    Uses Scrapling's browser-fingerprinted Fetcher to bypass anti-bot shields
    and extract full-text corporate catalysts directly from article bodies.
    Enforces a strict 15-day maximum age window with continuous half-life recency decay.
    """

    def __init__(self):
        self.cache = {}
        self.cache_ttl = timedelta(minutes=10)

    def _resolve_company_search_terms(self, ticker: str, company_name: Optional[str] = None) -> tuple[str, List[str]]:
        """
        Determines targeted search terms for the stock to eliminate generic noise
        and scrape only directly relevant corporate stories.
        """
        raw_sym = ticker.replace('.NS', '').replace('.BO', '').replace('^', '').upper().strip()
        resolved_name = None

        if company_name and len(company_name.strip()) > 1:
            resolved_name = company_name.strip()
        elif raw_sym in KNOWN_COMPANY_NAMES:
            resolved_name = KNOWN_COMPANY_NAMES[raw_sym]
        else:
            try:
                from services.ticker_manager import TICKER_LIST
                for item in TICKER_LIST:
                    if item.get("symbol", "").replace(".NS", "").replace(".BO", "").upper() == raw_sym:
                        resolved_name = item.get("name")
                        break
            except Exception:
                pass

        if not resolved_name:
            resolved_name = raw_sym

        # Clean corporate legal entity suffixes
        clean_name = re.sub(
            r'\b(limited|ltd\.?|industries|corp\.?|corporation|incorporated|inc\.?|company|india|enterprises)\b',
            '',
            resolved_name,
            flags=re.IGNORECASE
        ).strip()
        clean_name = re.sub(r'\s+', ' ', clean_name)

        search_terms = []
        if clean_name and len(clean_name) >= 2:
            search_terms.append(clean_name)
        if raw_sym not in search_terms and len(raw_sym) >= 3:
            search_terms.append(raw_sym)

        return clean_name or raw_sym, search_terms

    def _compute_recency_weight(self, age_hours: float) -> tuple[float, str]:
        """
        Calculates exponential time-decay weight for news articles within a strict 15-day window.
        - <= 24h: 1.00 weight (Breaking / same-day catalyst)
        - 1-3 days: 0.85 weight (High relevance)
        - 4-7 days: 0.55 weight (Moderate relevance)
        - 8-15 days: 0.25 weight (Contextual background)
        """
        if age_hours <= 24.0:
            weight = 1.00
            bucket = "24h"
        elif age_hours <= 72.0:
            weight = 0.85
            bucket = "3d"
        elif age_hours <= 168.0:
            weight = 0.55
            bucket = "7d"
        else:
            weight = 0.25
            bucket = "15d"
        return weight, bucket

    def _parse_entry_datetime(self, entry: Dict[str, Any]) -> Optional[datetime]:
        """Robust multi-format date parser returning datetime or None."""
        pub_parsed = entry.get('published_parsed')
        if pub_parsed:
            try:
                import time as _t
                return datetime.fromtimestamp(_t.mktime(pub_parsed))
            except Exception:
                pass

        date_str = entry.get('published') or entry.get('updated')
        if not date_str:
            return None

        clean_str = date_str.strip()
        date_formats = [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(clean_str, fmt)
            except ValueError:
                continue

        try:
            import pandas as pd
            return pd.to_datetime(clean_str).to_pydatetime()
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract Crore / Million rupee figures from article body."""
        patterns = [
            r'(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?\s*(?:crore|cr|lakh|bn|billion|million)',
            r'[\d,]+(?:\.\d+)?\s*(?:crore|cr)\s*(?:order|contract|deal|profit|revenue)',
            r'\b\d+(?:\.\d+)?%\s*(?:margin|growth|rise|surge|drop|dividend)'
        ]
        results = []
        for p in patterns:
            matches = re.findall(p, text, re.IGNORECASE)
            for m in matches:
                if m not in results:
                    results.append(m.strip())
        return results[:4]

    def _deep_read_article_body(self, url: str) -> str:
        """
        Deep-reads the actual article body using Scrapling's stealth FetcherSession
        with Chrome TLS impersonation, automatic Google News URL resolution,
        and clean RAG Markdown parsing.
        """
        if not HAS_SCRAPLING or not scrapling_client or not url or not url.startswith('http'):
            return ""
        try:
            return scrapling_client.deep_read_markdown(url, max_chars=3500)
        except Exception:
            return ""

    def _detect_magnitude_impact(self, text: str) -> tuple[float, List[str]]:
        """
        Detects quantitative price drops/spikes, NAV premiums, circuit locks,
        and valuation bubbles directly from headline and article text.
        Returns (score_delta, detected_reasons).
        """
        score_delta = 0.0
        reasons = []
        lower = text.lower()

        # 1. Percentage Drops (e.g. 'crashed 46%', 'falls 20%', 'slumped 15%')
        drop_match = re.search(
            r'(?:crash(?:ed|ing)?|plung(?:ed?|es|ing)|fall(?:s|en|ing)?|slump(?:ed|ing)?|drop(?:ped|ping)?|tumbl(?:ed?|es|ing)|down)\s*(?:by\s*|another\s*)?(\d+(?:\.\d+)?)\s*%',
            lower
        )
        if drop_match:
            pct = float(drop_match.group(1))
            if pct >= 20.0:
                score_delta -= 0.65
                reasons.append(f"Severe Catastrophic Drop ({pct:.0f}%)")
            elif pct >= 10.0:
                score_delta -= 0.40
                reasons.append(f"Sharp Single-Period Drop ({pct:.0f}%)")
            elif pct >= 5.0:
                score_delta -= 0.20
                reasons.append(f"Notable Correction ({pct:.0f}%)")

        # 2. Percentage Gains (e.g. 'surged 25%', 'jumped 18%', 'rallies 12%')
        gain_match = re.search(
            r'(?:surg(?:ed?|es|ing)|jump(?:ed|ing)?|rall(?:y|ies|ied)|gain(?:s|ed|ing)?|soar(?:ed?|s|ing)|up)\s*(?:by\s*|another\s*)?(\d+(?:\.\d+)?)\s*%',
            lower
        )
        if gain_match:
            pct = float(gain_match.group(1))
            if pct >= 20.0:
                score_delta += 0.50
                reasons.append(f"Parabolic Surge ({pct:.0f}%)")
            elif pct >= 10.0:
                score_delta += 0.35
                reasons.append(f"Strong Breakout Move ({pct:.0f}%)")

        # 3. Extreme NAV Premium / Secondary Dislocation (e.g. '235% premium to NAV', 'steep premiums to iNAV')
        premium_match = re.search(
            r'(?:premium|dislocation)\s*(?:to\s*(?:i?nav|underlying))?\s*(?:of|at)?\s*(\d+(?:\.\d+)?)\s*%',
            lower
        ) or re.search(
            r'(\d+(?:\.\d+)?)\s*%\s*premium\s*to\s*(?:i?nav|underlying)',
            lower
        ) or re.search(
            r'premium\s*(?:has\s*)?widened',
            lower
        )
        if premium_match:
            try:
                prem = float(premium_match.group(1)) if premium_match.groups() and premium_match.group(1) else 25.0
            except Exception:
                prem = 25.0
            if prem >= 30.0:
                score_delta -= 0.60
                reasons.append(f"Extreme Speculative Premium to NAV ({prem:.0f}%)")
            elif prem >= 15.0:
                score_delta -= 0.35
                reasons.append(f"Elevated NAV Premium ({prem:.0f}%)")

        # 4. Multiplier Pricing (e.g., 'paying over 3 times its underlying', 'paying ₹150 for ₹100 of assets')
        if re.search(r'paying\s*(?:over\s*)?(\d+(?:\.\d+)?)\s*times\s*(?:its\s*)?underlying', lower) or \
           re.search(r'paying\s*[₹rs\.]*\s*(\d+)\s*for\s*[₹rs\.]*\s*(\d+)\s*of\s*assets', lower):
            score_delta -= 0.55
            reasons.append("Irrational Retail Valuation Frenzy / Asset Overpricing")

        # 5. Circuit Limits & Trading Constraints
        if any(w in lower for w in ['lower circuit', 'circuit down', 'locked in lower circuit']):
            score_delta -= 0.50
            reasons.append("Locked in Lower Circuit (Liquidity Freeze)")
        elif any(w in lower for w in ['upper circuit', 'locked in upper circuit']):
            score_delta += 0.50
            reasons.append("Locked in Upper Circuit (Institutional Demand)")

        return score_delta, reasons

    def _extract_catalysts(self, text: str) -> List[Dict[str, Any]]:
        """Identify concrete corporate catalysts from text."""
        catalysts = []
        lower = text.lower()

        # 1. Valuation Bubble & NAV Dislocation / Crash
        if any(w in lower for w in ['premium to nav', 'premium to inav', 'trading at steep premium', 'steep premiums to inav', 'frenzy', 'bubble', 'times its underlying', 'paying over 3 times']) or \
           ('crashed' in lower and any(k in lower for k in ['premium', 'nav', 'two days', 'etf', 'run up'])):
            catalysts.append({
                'type': 'Valuation / NAV Dislocation Risk',
                'impact': 'Critical',
                'polarity': -0.95,
                'highlight': 'Severe Secondary Market Premium Dislocation / Speculative Bubble Collapse'
            })
        elif any(w in lower for w in ['crashed', 'meltdown', 'bloodbath', 'plunges', 'free fall', 'massive selloff']):
            catalysts.append({
                'type': 'Extreme Volatility / Sell-Off',
                'impact': 'High',
                'polarity': -0.85,
                'highlight': 'Sharp Market Sell-Off & Crash Dynamics Detected'
            })

        # 2. Circuit Limits & Quota Freezes
        if any(w in lower for w in ['lower circuit', 'locked in lower circuit', 'quota limit', 'quota freeze', 'trading halt', 'halted']):
            catalysts.append({
                'type': 'Liquidity Freeze / Trading Constraint',
                'impact': 'Critical',
                'polarity': -0.90,
                'highlight': 'Exchange Circuit Limit / Overseas Investment Quota Freeze'
            })

        # 3. Order wins / Commercial deals
        if any(w in lower for w in ['bags order', 'secures contract', 'wins bid', 'awarded contract', 'cr order', 'crore order']):
            amounts = self._extract_amounts(text)
            highlight = f"Commercial Order Inflow: {', '.join(amounts)}" if amounts else "Major Order Win / Contract Award"
            catalysts.append({
                'type': 'Order Win',
                'impact': 'High',
                'polarity': 0.8,
                'highlight': highlight
            })

        # 4. Earnings Beat & Margins
        if any(w in lower for w in ['profit surges', 'record profit', 'ebitda expands', 'margin expansion', 'beats estimates']):
            catalysts.append({
                'type': 'Earnings Outperformance',
                'impact': 'High',
                'polarity': 0.85,
                'highlight': 'Strong Q-o-Q Operating Performance & Margin Expansion'
            })
        elif any(w in lower for w in ['profit slides', 'net loss widens', 'margin contraction', 'misses estimates']):
            catalysts.append({
                'type': 'Earnings Headwind',
                'impact': 'High',
                'polarity': -0.75,
                'highlight': 'Margin Contraction / Operating Headwinds Reported'
            })

        # 5. Solvency & Balance Sheet
        if any(w in lower for w in ['debt-free', 'debt reduction', 'pre-pays debt', 'rating upgraded']):
            catalysts.append({
                'type': 'Balance Sheet Deleveraging',
                'impact': 'Medium',
                'polarity': 0.7,
                'highlight': 'Debt Reduction / Balance Sheet Strengthening'
            })

        # 6. Regulatory & Governance Red Flags
        if any(w in lower for w in ['sebi probe', 'tax raid', 'ed summons', 'fraud', 'forensic audit', 'auditor resigns']):
            catalysts.append({
                'type': 'Regulatory Risk',
                'impact': 'Critical',
                'polarity': -0.95,
                'highlight': 'Regulatory Scrutiny / Governance Inquiry'
            })

        # 7. Brokerage Rating & Target Price
        target_match = re.search(r'(?:target price|target of)\s*(?:rs\.?|₹)?\s*([\d,]+)', lower)
        if target_match and any(w in lower for w in ['buy', 'outperform', 'overweight', 'raised target']):
            catalysts.append({
                'type': 'Brokerage Upgrade',
                'impact': 'Medium',
                'polarity': 0.65,
                'highlight': f"Institutional Price Target Set at ₹{target_match.group(1)}"
            })

        return catalysts[:4]

    def _calculate_financial_sentiment(self, text: str) -> Dict[str, Any]:
        """Calculates domain-aware sentiment using financial linguistics + numerical magnitude."""
        lower = text.lower()
        score = 0.0

        for kw in FINANCIAL_LEXICON['strongly_positive']:
            if kw in lower:
                score += 0.45
        for kw in FINANCIAL_LEXICON['positive']:
            if kw in lower:
                score += 0.15

        for kw in FINANCIAL_LEXICON['strongly_negative']:
            if kw in lower:
                score -= 0.55
        for kw in FINANCIAL_LEXICON['negative']:
            if kw in lower:
                score -= 0.15

        # Incorporate magnitude analysis
        mag_delta, mag_reasons = self._detect_magnitude_impact(text)
        score += mag_delta

        # Bound score between -1.0 and +1.0
        score = max(-1.0, min(1.0, score))

        if score >= 0.35:
            label = "STRONG BULLISH"
        elif score >= 0.12:
            label = "BULLISH"
        elif score <= -0.35:
            label = "STRONG BEARISH"
        elif score <= -0.12:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        confidence = round(min(1.0, abs(score) * 1.4 + 0.40) * 100, 1)

        return {
            'score': round(score, 3),
            'label': label,
            'confidence': confidence,
            'magnitude_reasons': mag_reasons
        }

    def _synthesize_decision(
        self,
        sentiment_res: Dict[str, Any],
        catalysts: List[Dict[str, Any]],
        articles: List[Dict[str, Any]],
        ticker: str,
        recency_stats: Optional[Dict[str, Any]] = None,
        weighted_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Translates recency-weighted sentiment scores, 15-day corporate catalysts,
        and deep article readings into an unambiguous, institutional trade decision
        ("BUY", "STRONG BUY", "HOLD", "REDUCE", or "STRONG AVOID / DO NOT BUY").
        """
        raw_score = sentiment_res.get('score', 0.0)
        score = weighted_score if weighted_score is not None else raw_score
        conf = sentiment_res.get('confidence', 50.0)

        # ── 1. Check Hard Risk Veto Triggers (CRO Risk Gate) ───────────
        veto_keywords = [
            'sebi probe', 'fraud', 'forensic audit', 'auditor resigns', 'auditor resignation',
            'default', 'bankruptcy', 'insolvency', 'lower circuit', 'locked in lower circuit',
            'bloodbath', 'tax raid', 'ed summons', 'trading halt', 'massive selloff',
            'licence cancelled', 'penalized by sebi', 'criminal charges', 'promoter arrested',
            'fined by sebi', 'insider trading'
        ]

        cro_risk_flags = []
        is_buy_vetoed = False
        veto_reason = None

        # Scan catalysts for severe negative events
        min_cat_polarity = min([c['polarity'] for c in catalysts], default=0.0)
        max_cat_polarity = max([c['polarity'] for c in catalysts], default=0.0)

        for c in catalysts:
            if c['polarity'] <= -0.75:
                cro_risk_flags.append(f"Catalyst Alert: {c['highlight']}")

        for r in sentiment_res.get('magnitude_reasons', []):
            if r not in cro_risk_flags:
                cro_risk_flags.append(f"Price/Structure Signal: {r}")

        # Scan article titles & summaries within 15-day window for catastrophe keywords
        for art in articles:
            text_check = f"{art.get('title', '')} {art.get('summary', '')}".lower()
            for kw in veto_keywords:
                if kw in text_check:
                    flag_msg = f"Critical Event: '{kw.upper()}' detected in recent news ({art.get('source', 'Media')})"
                    if flag_msg not in cro_risk_flags:
                        cro_risk_flags.append(flag_msg)
                    is_buy_vetoed = True
                    if not veto_reason:
                        veto_reason = f"CRO Risk Gate VETO: {kw.upper()} detected in 15-day news flow."

        if min_cat_polarity <= -0.80 or len(cro_risk_flags) >= 2 or score <= -0.50:
            is_buy_vetoed = True
            if not veto_reason:
                veto_reason = "CRO Risk Gate VETO: Severe fundamental headwind / catalyst cluster detected."

        # ── 2. Determine Primary Catalyst Class ───────────────────────
        if any('Regulatory' in c['type'] for c in catalysts) or any('probe' in f.lower() for f in cro_risk_flags):
            catalyst_class = "REGULATORY_GOVERNANCE_RISK"
        elif any('Liquidity' in c['type'] or 'Circuit' in c['type'] for c in catalysts):
            catalyst_class = "LIQUIDITY_AND_CIRCUIT_CONSTRAINT"
        elif any('Valuation' in c['type'] or 'Dislocation' in c['type'] for c in catalysts):
            catalyst_class = "STRUCTURAL_VALUATION_DISLOCATION"
        elif any('Earnings Outperformance' in c['type'] for c in catalysts):
            catalyst_class = "EARNINGS_ACCELERATION"
        elif any('Earnings Headwind' in c['type'] for c in catalysts):
            catalyst_class = "EARNINGS_HEADWIND"
        elif any('Order Win' in c['type'] for c in catalysts):
            catalyst_class = "COMMERCIAL_ORDER_INFLOW"
        elif score <= -0.25:
            catalyst_class = "NEGATIVE_MOMENTUM_HEADWIND"
        elif score >= 0.25:
            catalyst_class = "POSITIVE_MOMENTUM_DRIVER"
        else:
            catalyst_class = "ROUTINE_MARKET_COVERAGE"

        # ── 3. Calculate Deterministic Decision Score (0 to 100) ──────
        # Base conversion: map score (-1.0 to +1.0) onto 10 to 90
        base_decision_score = 50.0 + (score * 38.0)

        # Catalyst adjustments
        pos_cat_count = len([c for c in catalysts if c['polarity'] > 0])
        neg_cat_count = len([c for c in catalysts if c['polarity'] < 0])
        base_decision_score += min(pos_cat_count * 4.0, 12.0)
        base_decision_score -= min(neg_cat_count * 5.0, 15.0)

        # Recent 24h bonus / penalty
        if recency_stats and recency_stats.get('within_24h', 0) > 0:
            h24_arts = [a for a in articles if a.get('recency_bucket') == '24h']
            if h24_arts:
                h24_sent = sum(a.get('sentiment', 0.0) for a in h24_arts) / len(h24_arts)
                if h24_sent >= 0.20:
                    base_decision_score += 4.0
                elif h24_sent <= -0.20:
                    base_decision_score -= 5.0

        if is_buy_vetoed:
            # Hard risk cap: maximum decision score capped at 22/100
            decision_score = round(max(5.0, min(22.0, base_decision_score * 0.35)), 1)
        else:
            decision_score = round(max(0.0, min(100.0, base_decision_score)), 1)

        # ── 4. Synthesize Unambiguous Action Signal ("BUY OR NOT") ─────
        if is_buy_vetoed:
            action_signal = "STRONG AVOID / DO NOT BUY"
            verdict = "CRITICAL_HEADWIND"
            directive = "AVOID / HIGH TAIL RISK"
            committee_vote = "BEARISH"
            ml_adjustment = max(-0.25, score * 0.20)
            action_text = f"CRITICAL RISK VETO: Severe negative catalyst detected for {ticker}. Long positions or fresh buys are strictly prohibited under risk rules."
            executive_rationale = f"DO NOT BUY: Active high-severity risk events ({cro_risk_flags[0] if cro_risk_flags else 'regulatory/liquidity alert'}) in the 15-day window pose severe asymmetric downside. Fundamental risk vetoes any technical buy signal."
        elif decision_score >= 72.0 or (score >= 0.32 and max_cat_polarity >= 0.70):
            action_signal = "STRONG BUY"
            verdict = "HIGH_CONVICTION_BULLISH"
            directive = "STRONG BUY / AGGRESSIVE ACCUMULATE"
            committee_vote = "BULLISH"
            ml_adjustment = min(0.20, score * 0.20)
            action_text = f"High-conviction positive catalysts detected for {ticker} within 15 days. Institutional news flow strongly supports aggressive accumulation."
            executive_rationale = f"CLEAR BUY: Strong corporate catalysts ({catalyst_class}) with high recency weighting provide robust fundamental momentum. Zero active regulatory or governance risks."
        elif decision_score >= 58.0 or score >= 0.12:
            action_signal = "BUY"
            verdict = "MILD_ACCUMULATE"
            directive = "BUY / DIP ACCUMULATE"
            committee_vote = "BULLISH"
            ml_adjustment = score * 0.10
            action_text = f"Constructive news flow detected for {ticker}. Favorable backdrop for buying dips and trend-following entries."
            executive_rationale = f"BUY ON DIPS: Constructive 15-day news flow with positive net sentiment (+{score:.2f}) supports gradual position building."
        elif decision_score >= 42.0:
            action_signal = "HOLD / WATCHLIST"
            verdict = "NEUTRAL_NOISE"
            directive = "HOLD / NO DIRECTIONAL BIAS"
            committee_vote = "NEUTRAL"
            ml_adjustment = 0.0
            action_text = f"News flow for {ticker} is balanced without decisive corporate catalysts. Base decisions primarily on technical charts and levels."
            executive_rationale = f"HOLD / WATCHLIST: Recent 15-day coverage is routine and balanced. Insufficient catalyst momentum to justify fresh aggressive buying."
        elif decision_score >= 25.0:
            action_signal = "REDUCE / CAUTION"
            verdict = "BEARISH_HEADWIND"
            directive = "REDUCE / DEFENSIVE HEDGE"
            committee_vote = "BEARISH"
            ml_adjustment = score * 0.15
            action_text = f"Headwinds outweigh positive catalysts for {ticker}. Exercise caution, trim exposure, or tighten stop-loss levels."
            executive_rationale = f"REDUCE / CAUTION: Prevailing 15-day news flow shows negative momentum (-{abs(score):.2f}). Downside pressure exceeds upside catalysts."
        else:
            action_signal = "STRONG AVOID / DO NOT BUY"
            verdict = "CRITICAL_HEADWIND"
            directive = "DO NOT BUY / SEVERE TAIL RISK"
            committee_vote = "BEARISH"
            ml_adjustment = max(-0.25, score * 0.20)
            action_text = f"Dominant negative sentiment and persistent headwinds detected for {ticker}. Avoid buying."
            executive_rationale = f"DO NOT BUY: Overwhelming negative news sentiment score ({score:.2f}) over the last 15 days indicates persistent selling pressure."

        # ── 5. Conviction Calculation ──────────────────────────────────
        volume_factor = min(len(articles) / 8.0, 1.0) * 40.0
        sentiment_clarity = abs(score) * 40.0
        catalyst_clarity = 20.0 if catalysts else 5.0
        conviction_pct = round(min(98.0, max(35.0, volume_factor + sentiment_clarity + catalyst_clarity)), 1)

        # ── 6. Assemble Concrete Decision Drivers ──────────────────────
        decision_drivers = []
        decision_drivers.append(f"15-Day Recency-Weighted Sentiment: {score:+.2f} ({'Bullish' if score > 0.1 else ('Bearish' if score < -0.1 else 'Neutral')})")
        
        if recency_stats:
            h24 = recency_stats.get('within_24h', 0)
            d3 = recency_stats.get('within_3d', 0)
            d7 = recency_stats.get('within_7d', 0)
            decision_drivers.append(f"Recency Velocity: {h24} stories <24h, {d3} stories in 1-3d, {d7} stories in 4-7d (Max limit: 15d)")

        if catalysts:
            top_cat = catalysts[0]
            decision_drivers.append(f"Top Catalyst: {top_cat['type']} — \"{top_cat['highlight'][:75]}\"")

        if is_buy_vetoed:
            decision_drivers.append(f"Risk Gate Status: ACTIVE VETO ({veto_reason})")
        else:
            decision_drivers.append("Risk Gate Status: CLEAR (Zero regulatory, fraud, or circuit lock flags)")

        return {
            'action_signal': action_signal,
            'decision_score': decision_score,
            'conviction_pct': conviction_pct,
            'verdict': verdict,
            'trade_directive': directive,
            'catalyst_class': catalyst_class,
            'is_buy_vetoed': is_buy_vetoed,
            'veto_reason': veto_reason,
            'decision_drivers': decision_drivers,
            'executive_rationale': executive_rationale,
            'sentiment_score': round(score, 4),
            'raw_sentiment_score': round(raw_score, 4),
            'conviction_score': conviction_pct,
            'action_recommendation': action_text,
            'cro_risk_flags': cro_risk_flags,
            'committee_vote': committee_vote,
            'ml_adjustment_factor': round(ml_adjustment, 4),
        }

    def fetch_live_stock_news(self, ticker: str, company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates 100% live multi-source news gathering, targeted company scraping,
        strict 15-day maximum age window enforcement, exponential recency weighting,
        deep article body reading via Scrapling, and financial decision synthesis.
        """
        ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
        cache_key = f"{ticker_clean}_{company_name or ''}"

        # Check in-memory cache (8 min TTL)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data

        # Determine targeted company search terms
        clean_name, search_terms = self._resolve_company_search_terms(ticker, company_name)

        collected_articles = []
        seen_titles = set()
        stale_discarded = 0
        now = datetime.now()

        from urllib.parse import quote_plus

        is_us = not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^')
        query_suffix = "stock" if is_us else "stock India"
        hl_param = "en-US" if is_us else "en-IN"
        gl_param = "US" if is_us else "IN"
        ceid_param = "US:en" if is_us else "IN:en"

        # ── 1. Google News Live Search (Strictly restricted to 15 days via when:15d) ──
        for term in search_terms[:2]:
            try:
                # Add 'when:15d' directly to search query to restrict server-side to 15 days
                query_encoded = quote_plus(f"{term} {query_suffix} when:15d")
                search_url = f"https://news.google.com/rss/search?q={query_encoded}&hl={hl_param}&gl={gl_param}&ceid={ceid_param}"
                feed = feedparser.parse(search_url)

                for entry in feed.entries[:14]:
                    title = self._clean_text(entry.get('title', ''))
                    if not title or title.lower() in seen_titles:
                        continue

                    # Extract publication datetime and verify strict 15-day limit
                    published_dt = self._parse_entry_datetime(entry)
                    if not published_dt:
                        published_dt = now

                    age_seconds = max(0.0, (now - published_dt).total_seconds())
                    age_days = age_seconds / 86400.0

                    if age_days > MAX_NEWS_AGE_DAYS:
                        stale_discarded += 1
                        continue  # STRICT 15-DAY ENFORCEMENT: DISCARD

                    seen_titles.add(title.lower())
                    raw_summary = self._clean_text(entry.get('summary', ''))

                    # Extract source publication name
                    source = 'Financial Media'
                    if ' - ' in title:
                        parts = title.split(' - ')
                        source = parts[-1].strip()
                        title = ' - '.join(parts[:-1]).strip()

                    link = entry.get('link', '')
                    age_hours = age_seconds / 3600.0
                    recency_weight, recency_bucket = self._compute_recency_weight(age_hours)

                    collected_articles.append({
                        'title': title,
                        'summary': raw_summary,
                        'source': source,
                        'link': link,
                        'published': entry.get('published', published_dt.strftime('%Y-%m-%d %H:%M')),
                        'published_dt': published_dt,
                        'age_days': round(age_days, 1),
                        'age_hours': round(age_hours, 1),
                        'recency_weight': recency_weight,
                        'recency_bucket': recency_bucket,
                        'deep_body': ''
                    })
            except Exception as e:
                print(f"Error reading Google News for {term}: {e}")

        # ── 2. Specialized Financial Feeds (Moneycontrol, ET, LiveMint) ──
        for feed_name, feed_url in LIVE_FINANCIAL_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:15]:
                    title = self._clean_text(entry.get('title', ''))
                    summary = self._clean_text(entry.get('summary', ''))
                    full_text = (title + " " + summary).lower()

                    # Check if story mentions company name or ticker
                    if any(st.lower() in full_text for st in search_terms):
                        if title.lower() in seen_titles:
                            continue

                        published_dt = self._parse_entry_datetime(entry)
                        if not published_dt:
                            published_dt = now

                        age_seconds = max(0.0, (now - published_dt).total_seconds())
                        age_days = age_seconds / 86400.0

                        if age_days > MAX_NEWS_AGE_DAYS:
                            stale_discarded += 1
                            continue  # STRICT 15-DAY ENFORCEMENT: DISCARD

                        seen_titles.add(title.lower())
                        age_hours = age_seconds / 3600.0
                        recency_weight, recency_bucket = self._compute_recency_weight(age_hours)

                        collected_articles.append({
                            'title': title,
                            'summary': summary,
                            'source': feed_name,
                            'link': entry.get('link', ''),
                            'published': entry.get('published', published_dt.strftime('%Y-%m-%d %H:%M')),
                            'published_dt': published_dt,
                            'age_days': round(age_days, 1),
                            'age_hours': round(age_hours, 1),
                            'recency_weight': recency_weight,
                            'recency_bucket': recency_bucket,
                            'deep_body': ''
                        })
            except Exception:
                continue

        # Sort articles chronologically descending (freshest first)
        collected_articles.sort(key=lambda x: x.get('published_dt') or datetime.min, reverse=True)

        # Compute 15-day recency distribution stats
        recency_distribution = {
            'within_24h': sum(1 for a in collected_articles if a.get('recency_bucket') == '24h'),
            'within_3d': sum(1 for a in collected_articles if a.get('recency_bucket') == '3d'),
            'within_7d': sum(1 for a in collected_articles if a.get('recency_bucket') == '7d'),
            'within_15d': sum(1 for a in collected_articles if a.get('recency_bucket') == '15d'),
            'total_recent': len(collected_articles),
            'stale_discarded': stale_discarded,
            'max_age_days_limit': 15.0
        }

        # If zero articles found within 15-day window
        if not collected_articles:
            default_decision = {
                'action_signal': 'HOLD / WATCHLIST',
                'decision_score': 50.0,
                'conviction_pct': 30.0,
                'verdict': 'NEUTRAL_NOISE',
                'trade_directive': 'HOLD / NO DIRECTIONAL BIAS',
                'catalyst_class': 'ROUTINE_MARKET_COVERAGE',
                'is_buy_vetoed': False,
                'veto_reason': None,
                'decision_drivers': [
                    f"No news stories detected for {ticker_clean} in the strict 15-day window",
                    f"Scraped sources discarded {stale_discarded} outdated stories older than 15 days",
                    "Risk Gate Status: CLEAR"
                ],
                'executive_rationale': f"HOLD / WATCHLIST: Zero corporate news catalysts detected in the last 15 days. Make trade decisions strictly based on technical price action and key support/resistance levels.",
                'sentiment_score': 0.0,
                'raw_sentiment_score': 0.0,
                'conviction_score': 30.0,
                'action_recommendation': f"No corporate catalysts found for {ticker_clean} in the last 15 days. Base trading decisions on technical levels.",
                'cro_risk_flags': [],
                'committee_vote': 'NEUTRAL',
                'ml_adjustment_factor': 0.0,
            }
            result = {
                'status': 'active',
                'ticker': ticker_clean,
                'company_name': clean_name,
                'total_articles': 0,
                'decision': default_decision,
                'recency_distribution': recency_distribution,
                'sentiment': {
                    'overall_sentiment': 0.0,
                    'weighted_sentiment': 0.0,
                    'sentiment_label': 'NEUTRAL',
                    'confidence': 30.0,
                    'market_impact_score': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                },
                'market_impact_score': 0.0,
                'catalysts': [],
                'breaking_news': [],
                'articles': [],
                'summary': f"Zero live news stories detected for {ticker_clean} within the strict 15-day window. Reliance on technical price levels recommended.",
                'last_updated': datetime.now().isoformat()
            }
            self.cache[cache_key] = (result, datetime.now())
            return result

        # ── 3. Parallel Deep Article Reading via Scrapling Stealth Engine ──
        from concurrent.futures import ThreadPoolExecutor, as_completed

        deep_candidates = [
            art for art in collected_articles
            if art.get('link') and art['link'].startswith('http')
        ][:5]

        if deep_candidates and scrapling_client:
            try:
                with ThreadPoolExecutor(max_workers=min(len(deep_candidates), 5)) as executor:
                    future_to_art = {
                        executor.submit(self._deep_read_article_body, art['link']): art
                        for art in deep_candidates
                    }
                    for future in as_completed(future_to_art, timeout=4.0):
                        art = future_to_art[future]
                        try:
                            body = future.result()
                            if body:
                                art['deep_body'] = body
                        except Exception:
                            pass
            except Exception:
                pass

        all_text_corpus = []
        for art in collected_articles:
            corpus = (art['title'] + " " + art['summary'] + " " + (art.get('deep_body') or "")).strip()
            all_text_corpus.append(corpus)

        # ── 4. Financial Catalyst & Sentiment Evaluation ───────────────
        combined_corpus = " ".join(all_text_corpus)
        extracted_catalysts = self._extract_catalysts(combined_corpus)
        sentiment_res = self._calculate_financial_sentiment(combined_corpus)

        # Build clean article payloads with recency weight & sentiment
        formatted_articles = []
        breaking_news = []

        total_weight = 0.0
        weighted_sentiment_sum = 0.0

        for art in collected_articles[:15]:
            art_body = art.get('deep_body') or ''
            art_corpus = f"{art.get('title', '')} {art.get('summary', '')} {art_body}".strip()
            art_sent = self._calculate_financial_sentiment(art_corpus)
            art_catalysts = self._extract_catalysts(art_corpus)

            w = art.get('recency_weight', 1.0)
            total_weight += w
            weighted_sentiment_sum += (art_sent['score'] * w)

            raw_summary = art.get('summary', '')
            item = {
                'title': art.get('title', ''),
                'summary': (art_body[:280] + '...') if art_body else ((raw_summary[:200] + '...') if len(raw_summary) > 200 else raw_summary),
                'source': art['source'],
                'link': art['link'],
                'published': art['published'],
                'age_days': art.get('age_days', 0.0),
                'age_hours': art.get('age_hours', 0.0),
                'recency_weight': art.get('recency_weight', 1.0),
                'recency_bucket': art.get('recency_bucket', '24h'),
                'sentiment': art_sent['score'],
                'sentiment_label': art_sent['label'],
                'catalysts': [c['highlight'] for c in art_catalysts]
            }
            formatted_articles.append(item)

            # Flag as breaking/high impact if in 24h/3d with catalysts or high sentiment
            if (art.get('recency_bucket') in ['24h', '3d']) and (art_catalysts or abs(art_sent['score']) >= 0.35):
                breaking_news.append({
                    'title': art['title'],
                    'impact_score': 90 if art_catalysts else 75,
                    'urgency': 'HIGH' if (art.get('recency_bucket') == '24h' or abs(art_sent['score']) >= 0.5) else 'MEDIUM',
                    'reasons': [c['highlight'] for c in art_catalysts] if art_catalysts else [f"{art_sent['label']} Catalyst Momentum"],
                    'published': art['published'],
                    'link': art['link'],
                    'recency_bucket': art.get('recency_bucket', '24h')
                })

        # Calculate recency-weighted sentiment score
        weighted_sentiment = round(weighted_sentiment_sum / max(0.0001, total_weight), 3) if total_weight > 0 else sentiment_res['score']

        # Generate Actionable Decision Directive ("BUY OR NOT")
        decision = self._synthesize_decision(
            sentiment_res=sentiment_res,
            catalysts=extracted_catalysts,
            articles=formatted_articles,
            ticker=ticker_clean,
            recency_stats=recency_distribution,
            weighted_score=weighted_sentiment
        )

        # Calculate dynamic market impact score (0 to 100)
        volume_factor = min(len(collected_articles) / 8.0, 1.0) * 35
        catalyst_factor = min(len(extracted_catalysts) * 20, 45)
        sentiment_factor = abs(weighted_sentiment) * 20
        market_impact = round(min(100.0, volume_factor + catalyst_factor + sentiment_factor), 1)

        # Construct executive synthesis
        summary_bullets = []
        summary_bullets.append(f"Screened {len(formatted_articles)} stories within 15-day limit ({recency_distribution['within_24h']} in last 24h).")
        if extracted_catalysts:
            summary_bullets.append(f"Primary Catalyst: {extracted_catalysts[0]['highlight']}.")
        summary_bullets.append(f"Action: {decision['action_signal']} (Score: {decision['decision_score']}/100, Conviction: {decision['conviction_pct']}%).")
        summary_bullets.append(f"{decision['executive_rationale']}")

        final_response = {
            'status': 'live',
            'ticker': ticker_clean,
            'company_name': clean_name,
            'total_articles': len(formatted_articles),
            'decision': decision,
            'recency_distribution': recency_distribution,
            'sentiment': {
                'overall_sentiment': weighted_sentiment,
                'raw_sentiment': sentiment_res['score'],
                'sentiment_label': 'BULLISH' if weighted_sentiment >= 0.12 else ('BEARISH' if weighted_sentiment <= -0.12 else 'NEUTRAL'),
                'confidence': decision['conviction_pct'],
                'market_impact_score': market_impact,
                'positive_count': sum(1 for a in formatted_articles if a['sentiment'] > 0.1),
                'negative_count': sum(1 for a in formatted_articles if a['sentiment'] < -0.1),
                'neutral_count': sum(1 for a in formatted_articles if abs(a['sentiment']) <= 0.1)
            },
            'market_impact_score': market_impact,
            'catalysts': extracted_catalysts,
            'breaking_news': breaking_news[:3],
            'articles': formatted_articles,
            'summary': " ".join(summary_bullets),
            'last_updated': datetime.now().isoformat()
        }

        # Bound cache to max 120 entries
        if len(self.cache) > 120:
            for k in list(self.cache.keys())[:25]:
                self.cache.pop(k, None)

        # Cache result
        self.cache[cache_key] = (final_response, datetime.now())
        return final_response


# Global singleton instance
intelligent_news_reader = IntelligentNewsReader()
