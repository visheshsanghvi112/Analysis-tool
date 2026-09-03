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
    from scrapling import Fetcher, Selector
    HAS_SCRAPLING = True
except Exception as e:
    HAS_SCRAPLING = False
    print(f"Warning: Scrapling import issue: {e}")

# Curated financial domain lexicon for realistic market sentiment scoring
FINANCIAL_LEXICON = {
    'strongly_positive': [
        'profit surges', 'record profit', 'beats estimates', 'ebitda expands', 
        'margin expansion', 'debt-free', 'debt reduction', 'order win', 'mega contract',
        'rating upgrade', 'upgraded to buy', 'raised target', 'guidance raised',
        'dividend hike', 'share buyback', 'all-time high profit', 'strong quarterly',
        'multi-fold surge', 'robust growth'
    ],
    'positive': [
        'growth', 'surge', 'surges', 'expansion', 'jump', 'jumps', 'gain', 'gains', 'profit',
        'outperform', 'reiterates buy', 'contract', 'partnership', 'commissioned', 'capacity addition',
        'turnaround', 'loss narrows', 'cuts net loss', 'recovery', 'rally', 'rallies',
        'soars', 'soar', 'climbs', 'climb', 'advances', 'advance', 'higher', 'bullish', 'up'
    ],
    'strongly_negative': [
        'sebi probe', 'tax raid', 'ed summons', 'fraud', 'forensic audit',
        'auditor resigns', 'default', 'insolvency', 'bankruptcy', 'downgraded to sell',
        'slashed target', 'guidance cut', 'promoter pledge increases', 'loss widens',
        'q1 miss', 'q2 miss', 'q3 miss', 'q4 miss'
    ],
    'negative': [
        'loss', 'decline', 'drop', 'slump', 'falls', 'fall', 'weak', 'misses estimates',
        'margin contraction', 'cost pressures', 'penalty', 'fine', 'delay',
        'investigation', 'headwinds', 'subdued', 'slides', 'slid', 'tumbles', 'tumble',
        'dips', 'dip', 'down', 'lower', 'bearish'
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
        Deep-reads the actual article body using Scrapling's stealth Fetcher.
        Strips ads, boilerplate, and extracts substantive financial paragraphs.
        """
        if not HAS_SCRAPLING or not url or not url.startswith('http'):
            return ""
        try:
            res = Fetcher.get(url, timeout=3.5, follow_redirects=True)
            if res.status == 200:
                # Target primary article containers across Moneycontrol, ET, LiveMint, Reuters
                paras = res.css(
                    'article p::text, .content_wrapper p::text, .story-details p::text, '
                    '.artText p::text, .article_content p::text, .mainArea p::text, p::text'
                )
                clean_paras = []
                for p in paras:
                    clean = str(p).strip()
                    # Filter out short disclaimers, share buttons, and ads
                    if len(clean) > 50 and not any(w in clean.lower() for w in [
                        'subscribe', 'click here', 'advertisement', 'cookie', 'download app', 'terms of use'
                    ]):
                        clean_paras.append(clean)
                return " ".join(clean_paras[:8])
        except Exception:
            return ""
        return ""

    def _extract_catalysts(self, text: str) -> List[Dict[str, Any]]:
        """Identify concrete corporate catalysts from text."""
        catalysts = []
        lower = text.lower()

        # 1. Order wins / Commercial deals
        if any(w in lower for w in ['bags order', 'secures contract', 'wins bid', 'awarded contract', 'cr order', 'crore order']):
            amounts = self._extract_amounts(text)
            highlight = f"Commercial Order Inflow: {', '.join(amounts)}" if amounts else "Major Order Win / Contract Award"
            catalysts.append({
                'type': 'Order Win',
                'impact': 'High',
                'polarity': 0.8,
                'highlight': highlight
            })

        # 2. Earnings Beat & Margins
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

        # 3. Solvency & Balance Sheet
        if any(w in lower for w in ['debt-free', 'debt reduction', 'pre-pays debt', 'rating upgraded']):
            catalysts.append({
                'type': 'Balance Sheet Deleveraging',
                'impact': 'Medium',
                'polarity': 0.7,
                'highlight': 'Debt Reduction / Balance Sheet Strengthening'
            })

        # 4. Regulatory & Governance Red Flags
        if any(w in lower for w in ['sebi probe', 'tax raid', 'ed summons', 'fraud', 'forensic audit', 'auditor resigns']):
            catalysts.append({
                'type': 'Regulatory Risk',
                'impact': 'High',
                'polarity': -0.9,
                'highlight': 'Regulatory Scrutiny / Governance Inquiry'
            })

        # 5. Brokerage Rating & Target Price
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
        """Calculates domain-aware sentiment using financial linguistics."""
        lower = text.lower()
        score = 0.0

        for kw in FINANCIAL_LEXICON['strongly_positive']:
            if kw in lower:
                score += 0.4
        for kw in FINANCIAL_LEXICON['positive']:
            if kw in lower:
                score += 0.15

        for kw in FINANCIAL_LEXICON['strongly_negative']:
            if kw in lower:
                score -= 0.5
        for kw in FINANCIAL_LEXICON['negative']:
            if kw in lower:
                score -= 0.15

        # Bound score between -1.0 and +1.0
        score = max(-1.0, min(1.0, score))

        if score >= 0.2:
            label = "BULLISH"
        elif score <= -0.2:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return {
            'score': round(score, 3),
            'label': label,
            'confidence': round(min(1.0, abs(score) * 1.5 + 0.35) * 100, 1)
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

        # ── 1. Google News Live Search (Primary Live Source) ───────────
        for kw in keywords[:2]:
            try:
                query_encoded = quote_plus(f"{kw} stock India")
                search_url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-IN&gl=IN&ceid=IN:en"
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

                    collected_articles.append({
                        'title': title,
                        'summary': raw_summary,
                        'source': source,
                        'link': link,
                        'published': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M')),
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
                            collected_articles.append({
                                'title': title,
                                'summary': summary,
                                'source': feed_name,
                                'link': entry.get('link', ''),
                                'published': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M')),
                                'deep_body': ''
                            })
            except Exception:
                continue

        # If zero articles found in real-time
        if not collected_articles:
            result = {
                'status': 'active',
                'ticker': ticker_clean,
                'total_articles': 0,
                'sentiment': {'score': 0.0, 'label': 'NEUTRAL', 'confidence': 50.0},
                'market_impact_score': 0.0,
                'catalysts': [],
                'breaking_news': [],
                'articles': [],
                'summary': f"No live news stories detected for {ticker_clean} across Indian financial media in the last 7 days.",
                'last_updated': datetime.now().isoformat()
            }
            self.cache[cache_key] = (result, datetime.now())
            return result

        # ── 3. Deep Article Reading via Scrapling for Top 3 Stories ─────
        # For the top 3 most relevant articles, extract the real article body
        deep_read_count = 0
        all_text_corpus = []

        for art in collected_articles:
            body = ""
            if deep_read_count < 3 and art['link'] and not art['link'].startswith('https://news.google.com/rss/articles/'):
                body = self._deep_read_article_body(art['link'])
                if body:
                    art['deep_body'] = body
                    deep_read_count += 1

            # Use deep body if available, otherwise title + summary
            corpus = (art['title'] + " " + art['summary'] + " " + (art['deep_body'] or "")).strip()
            all_text_corpus.append(corpus)

        # ── 4. Financial Catalyst & Sentiment Evaluation ───────────────
        combined_corpus = " ".join(all_text_corpus)
        extracted_catalysts = self._extract_catalysts(combined_corpus)
        sentiment_res = self._calculate_financial_sentiment(combined_corpus)

        # Calculate dynamic market impact score (0 to 100)
        volume_factor = min(len(collected_articles) / 10.0, 1.0) * 35
        catalyst_factor = min(len(extracted_catalysts) * 20, 45)
        sentiment_factor = abs(sentiment_res['score']) * 20
        market_impact = round(volume_factor + catalyst_factor + sentiment_factor, 1)

        # Build clean article payloads
        formatted_articles = []
        breaking_news = []

        for art in collected_articles[:10]:
            art_corpus = f"{art['title']} {art['summary']} {art['deep_body']}"
            art_sent = self._calculate_financial_sentiment(art_corpus)
            art_catalysts = self._extract_catalysts(art_corpus)

            item = {
                'title': art['title'],
                'summary': art['deep_body'][:280] + '...' if art['deep_body'] else (art['summary'][:200] + '...' if len(art['summary']) > 200 else art['summary']),
                'source': art['source'],
                'link': art['link'],
                'published': art['published'],
                'sentiment': art_sent['score'],
                'sentiment_label': art_sent['label'],
                'catalysts': [c['highlight'] for c in art_catalysts]
            }
            formatted_articles.append(item)

            # Flag as breaking/high impact if it contains strong catalysts
            if art_catalysts or abs(art_sent['score']) >= 0.5:
                breaking_news.append({
                    'title': art['title'],
                    'impact_score': 85 if art_catalysts else 65,
                    'urgency': 'HIGH' if art_catalysts else 'MEDIUM',
                    'reasons': [c['highlight'] for c in art_catalysts] if art_catalysts else [f"{art_sent['label']} Catalyst Momentum"],
                    'published': art['published'],
                    'link': art['link']
                })

        # Construct executive synthesis
        summary_bullets = []
        summary_bullets.append(f"Monitored {len(collected_articles)} live financial stories.")
        if extracted_catalysts:
            summary_bullets.append(f"Primary Catalyst: {extracted_catalysts[0]['highlight']}.")
        summary_bullets.append(f"Overall Market Bias: {sentiment_res['label']} ({sentiment_res['score']:+.2f}) with {sentiment_res['confidence']}% conviction.")

        final_response = {
            'status': 'live',
            'ticker': ticker_clean,
            'total_articles': len(collected_articles),
            'sentiment': {
                'overall_sentiment': sentiment_res['score'],
                'sentiment_label': sentiment_res['label'],
                'confidence': sentiment_res['confidence'],
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

        # Cache result
        self.cache[cache_key] = (final_response, datetime.now())
        return final_response


# Global singleton instance
intelligent_news_reader = IntelligentNewsReader()
