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


class IntelligentNewsReader:
    """
    100% Live, Deep-Reading News Intelligence Engine.
    Uses Scrapling's browser-fingerprinted Fetcher to bypass anti-bot shields
    and extract full-text corporate catalysts directly from article bodies.
    """

    def __init__(self):
        self.cache = {}
        self.cache_ttl = timedelta(minutes=10)

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
        ticker: str
    ) -> Dict[str, Any]:
        """
        Translates raw sentiment scores and extracted corporate/market catalysts into
        rigorous, deterministic trading and risk directives.
        """
        score = sentiment_res.get('score', 0.0)
        conf = sentiment_res.get('confidence', 50.0)

        # Identify critical catalyst polarities
        min_cat_polarity = min([c['polarity'] for c in catalysts], default=0.0)
        max_cat_polarity = max([c['polarity'] for c in catalysts], default=0.0)

        cro_risk_flags = []
        for c in catalysts:
            if c['polarity'] <= -0.75:
                cro_risk_flags.append(f"Catalyst Alert: {c['highlight']}")

        for r in sentiment_res.get('magnitude_reasons', []):
            if r not in cro_risk_flags:
                cro_risk_flags.append(f"Price/Structure Signal: {r}")

        # Determine Primary Catalyst Class
        if any('Valuation' in c['type'] or 'Dislocation' in c['type'] for c in catalysts):
            catalyst_class = "STRUCTURAL_VALUATION_DISLOCATION"
        elif any('Regulatory' in c['type'] for c in catalysts):
            catalyst_class = "REGULATORY_GOVERNANCE_RISK"
        elif any('Liquidity' in c['type'] or 'Circuit' in c['type'] for c in catalysts):
            catalyst_class = "LIQUIDITY_AND_CIRCUIT_CONSTRAINT"
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

        # Determine Verdict & Trade Directive
        if score <= -0.35 or min_cat_polarity <= -0.80 or len(cro_risk_flags) >= 2:
            verdict = "CRITICAL_HEADWIND"
            directive = "AVOID / HIGH TAIL RISK"
            committee_vote = "BEARISH"
            ml_adjustment = max(-0.25, score * 0.20)
            action_text = f"High-severity negative catalyst detected for {ticker}. Downside tail risk is elevated; secondary trading or long entries are strictly unfavorable."
        elif score <= -0.12 or min_cat_polarity <= -0.50:
            verdict = "BEARISH_HEADWIND"
            directive = "SHORT_BIAS / CAUTION"
            committee_vote = "BEARISH"
            ml_adjustment = score * 0.15
            action_text = f"Headwinds outweigh positive catalysts for {ticker}. Exercise defensive position sizing and tighten stops."
        elif score >= 0.35 or max_cat_polarity >= 0.80:
            verdict = "HIGH_CONVICTION_BULLISH"
            directive = "ACCUMULATE / CATALYST PLAY"
            committee_vote = "BULLISH"
            ml_adjustment = min(0.20, score * 0.20)
            action_text = f"Strong institutional catalysts identified for {ticker}. News sentiment strongly supports directional momentum."
        elif score >= 0.12:
            verdict = "MILD_ACCUMULATE"
            directive = "MILD BUY / DIP ACCUMULATE"
            committee_vote = "BULLISH"
            ml_adjustment = score * 0.10
            action_text = f"Constructive corporate news flow detected for {ticker}. Favorable backdrop for trend-following entries."
        else:
            verdict = "NEUTRAL_NOISE"
            directive = "NO DIRECTIONAL BIAS"
            committee_vote = "NEUTRAL"
            ml_adjustment = 0.0
            action_text = f"News flow for {ticker} is balanced without decisive corporate catalysts. Rely primarily on price action and technical levels."

        return {
            'verdict': verdict,
            'trade_directive': directive,
            'catalyst_class': catalyst_class,
            'conviction_score': round(conf, 1),
            'action_recommendation': action_text,
            'cro_risk_flags': cro_risk_flags,
            'committee_vote': committee_vote,
            'ml_adjustment_factor': round(ml_adjustment, 4),
        }

    def fetch_live_stock_news(self, ticker: str, company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates 100% live multi-source news gathering, deep article body
        reading via Scrapling, and financial catalyst extraction.
        """
        ticker_clean = ticker.replace('.NS', '').replace('.BO', '').upper()
        cache_key = f"{ticker_clean}_{company_name or ''}"

        # Check in-memory cache (10 min TTL)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data

        # Determine search keywords
        keywords = [ticker_clean]
        if company_name:
            clean_name = re.sub(r'\b(ltd|limited|industries|india|corp|corporation|bank)\b', '', company_name, flags=re.IGNORECASE).strip()
            if clean_name:
                keywords.append(clean_name)

        collected_articles = []
        seen_titles = set()

        from urllib.parse import quote_plus

        is_us = not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^')
        query_suffix = "stock" if is_us else "stock India"
        hl_param = "en-US" if is_us else "en-IN"
        gl_param = "US" if is_us else "IN"
        ceid_param = "US:en" if is_us else "IN:en"

        # ── 1. Google News Live Search (Primary Live Source) ───────────
        for kw in keywords[:2]:
            try:
                query_encoded = quote_plus(f"{kw} {query_suffix}")
                search_url = f"https://news.google.com/rss/search?q={query_encoded}&hl={hl_param}&gl={gl_param}&ceid={ceid_param}"
                feed = feedparser.parse(search_url)

                for entry in feed.entries[:12]:
                    title = self._clean_text(entry.get('title', ''))
                    if not title or title.lower() in seen_titles:
                        continue

                    seen_titles.add(title.lower())
                    raw_summary = self._clean_text(entry.get('summary', ''))
                    
                    # Extract source publication name
                    source = 'Financial Media'
                    if ' - ' in title:
                        parts = title.split(' - ')
                        source = parts[-1].strip()
                        title = ' - '.join(parts[:-1]).strip()

                    link = entry.get('link', '')

                    pub_parsed = entry.get('published_parsed')
                    published_dt = None
                    if pub_parsed:
                        try:
                            import time as _t
                            published_dt = datetime.fromtimestamp(_t.mktime(pub_parsed))
                        except Exception:
                            pass
                    if not published_dt:
                        published_dt = datetime.now()

                    collected_articles.append({
                        'title': title,
                        'summary': raw_summary,
                        'source': source,
                        'link': link,
                        'published': entry.get('published', published_dt.strftime('%Y-%m-%d %H:%M')),
                        'published_dt': published_dt,
                        'deep_body': ''
                    })
            except Exception as e:
                print(f"Error reading Google News for {kw}: {e}")

        # ── 2. Specialized Financial Feeds (Moneycontrol, ET, LiveMint) ──
        for feed_name, feed_url in LIVE_FINANCIAL_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:15]:
                    title = self._clean_text(entry.get('title', ''))
                    summary = self._clean_text(entry.get('summary', ''))
                    full_text = (title + " " + summary).lower()

                    # Check if story mentions company or ticker
                    if any(kw.lower() in full_text for kw in keywords):
                        if title.lower() not in seen_titles:
                            seen_titles.add(title.lower())
                            pub_parsed = entry.get('published_parsed')
                            published_dt = None
                            if pub_parsed:
                                try:
                                    import time as _t
                                    published_dt = datetime.fromtimestamp(_t.mktime(pub_parsed))
                                except Exception:
                                    pass
                            if not published_dt:
                                published_dt = datetime.now()

                            collected_articles.append({
                                'title': title,
                                'summary': summary,
                                'source': feed_name,
                                'link': entry.get('link', ''),
                                'published': entry.get('published', published_dt.strftime('%Y-%m-%d %H:%M')),
                                'published_dt': published_dt,
                                'deep_body': ''
                            })
            except Exception:
                continue

        # Sort articles chronologically descending (freshest first)
        collected_articles.sort(key=lambda x: x.get('published_dt') or datetime.min, reverse=True)

        # De-weight / filter out obsolete stories (>180 days) if fresher stories exist
        fresh_articles = [a for a in collected_articles if (datetime.now() - (a.get('published_dt') or datetime.now())).days <= 180]
        if fresh_articles:
            collected_articles = fresh_articles

        # If zero articles found in real-time
        if not collected_articles:
            default_decision = {
                'verdict': 'NEUTRAL_NOISE',
                'trade_directive': 'NO DIRECTIONAL BIAS',
                'catalyst_class': 'ROUTINE_MARKET_COVERAGE',
                'conviction_score': 40.0,
                'action_recommendation': f"No recent corporate catalysts found for {ticker_clean}. Base decisions on technical price levels.",
                'cro_risk_flags': [],
                'committee_vote': 'NEUTRAL',
                'ml_adjustment_factor': 0.0,
            }
            result = {
                'status': 'active',
                'ticker': ticker_clean,
                'total_articles': 0,
                'decision': default_decision,
                'sentiment': {'overall_sentiment': 0.0, 'sentiment_label': 'NEUTRAL', 'confidence': 50.0},
                'market_impact_score': 0.0,
                'catalysts': [],
                'breaking_news': [],
                'articles': [],
                'summary': f"No live news stories detected for {ticker_clean} across Indian financial media in the last 7 days.",
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

        # Build clean article payloads
        formatted_articles = []
        breaking_news = []

        for art in collected_articles[:12]:
            art_body = art.get('deep_body') or ''
            art_corpus = f"{art.get('title', '')} {art.get('summary', '')} {art_body}".strip()
            art_sent = self._calculate_financial_sentiment(art_corpus)
            art_catalysts = self._extract_catalysts(art_corpus)

            raw_summary = art.get('summary', '')
            item = {
                'title': art.get('title', ''),
                'summary': (art_body[:280] + '...') if art_body else ((raw_summary[:200] + '...') if len(raw_summary) > 200 else raw_summary),
                'source': art['source'],
                'link': art['link'],
                'published': art['published'],
                'sentiment': art_sent['score'],
                'sentiment_label': art_sent['label'],
                'catalysts': [c['highlight'] for c in art_catalysts]
            }
            formatted_articles.append(item)

            # Flag as breaking/high impact if it contains strong catalysts
            if art_catalysts or abs(art_sent['score']) >= 0.35:
                breaking_news.append({
                    'title': art['title'],
                    'impact_score': 90 if art_catalysts else 70,
                    'urgency': 'HIGH' if (art_catalysts or abs(art_sent['score']) >= 0.5) else 'MEDIUM',
                    'reasons': [c['highlight'] for c in art_catalysts] if art_catalysts else [f"{art_sent['label']} Catalyst Momentum"],
                    'published': art['published'],
                    'link': art['link']
                })

        # Generate Actionable Decision Directive
        decision = self._synthesize_decision(sentiment_res, extracted_catalysts, formatted_articles, ticker_clean)

        # Calculate dynamic market impact score (0 to 100)
        volume_factor = min(len(collected_articles) / 8.0, 1.0) * 35
        catalyst_factor = min(len(extracted_catalysts) * 20, 45)
        sentiment_factor = abs(sentiment_res['score']) * 20
        market_impact = round(min(100.0, volume_factor + catalyst_factor + sentiment_factor), 1)

        # Construct executive synthesis
        summary_bullets = []
        summary_bullets.append(f"Monitored {len(formatted_articles)} live financial stories.")
        if extracted_catalysts:
            summary_bullets.append(f"Primary Catalyst: {extracted_catalysts[0]['highlight']}.")
        summary_bullets.append(f"Directive: {decision['trade_directive']} ({decision['verdict']}).")
        summary_bullets.append(f"Market Bias: {sentiment_res['label']} ({sentiment_res['score']:+.2f}) with {sentiment_res['confidence']}% conviction.")

        final_response = {
            'status': 'live',
            'ticker': ticker_clean,
            'total_articles': len(formatted_articles),
            'decision': decision,
            'sentiment': {
                'overall_sentiment': sentiment_res['score'],
                'sentiment_label': sentiment_res['label'],
                'confidence': sentiment_res['confidence'],
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
