# ============================================================
# Advanced News Intelligence — by Vishesh Sanghvi
# ============================================================

import requests
import feedparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from urllib.parse import urlencode
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class NewsIntelligence:
    def __init__(self):
        self.news_sources = [
            'https://feeds.feedburner.com/ndtvprofit-latest',
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'http://www.business-standard.com/rss/markets-106.rss',
            'https://www.moneycontrol.com/rss/business.xml',
            'https://www.livemint.com/rss/markets',
        ]
        
    def clean_text(self, text):
        """Clean and normalize text for better analysis"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip()
    
    def fetch_stock_news(self, ticker, company_name=None):
        """Fetch news specifically for a stock ticker"""
        ticker_clean = ticker.replace('.NS', '').replace('.BO', '')
        search_terms = [ticker_clean]
        
        if company_name:
            # Extract key words from company name
            company_words = company_name.lower().replace('ltd', '').replace('limited', '').split()
            # Filter out common words
            filtered_words = [w for w in company_words if len(w) > 3 and w not in ['bank', 'company', 'corporation', 'industries']]
            search_terms.extend(filtered_words[:2])  # Add top 2 relevant words
        
        all_articles = []
        
        # Search Google News
        for term in search_terms[:3]:  # Limit to avoid rate limits
            try:
                google_url = f"https://news.google.com/rss/search?q={term}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
                feed = feedparser.parse(google_url)
                
                for entry in feed.entries[:10]:  # Top 10 per term
                    published = self._parse_date(entry.get('published', ''))
                    
                    # Only include news from last 7 days
                    if published and (datetime.now() - published).days <= 7:
                        article = {
                            'title': self.clean_text(entry.get('title', '')),
                            'summary': self.clean_text(entry.get('summary', '')),
                            'link': entry.get('link', ''),
                            'published': published,
                            'source': 'Google News',
                            'search_term': term,
                            'relevance_score': 0
                        }
                        all_articles.append(article)
            except Exception as e:
                print(f"Error fetching Google News for {term}: {e}")
                continue
        
        # Fetch from RSS feeds
        for source_url in self.news_sources:
            try:
                feed = feedparser.parse(source_url)
                for entry in feed.entries[:20]:
                    title = self.clean_text(entry.get('title', ''))
                    summary = self.clean_text(entry.get('summary', ''))
                    
                    # Check relevance to our stock
                    relevance = self._calculate_relevance(title + ' ' + summary, search_terms)
                    
                    if relevance > 0.3:  # Only include relevant articles
                        published = self._parse_date(entry.get('published', ''))
                        
                        if published and (datetime.now() - published).days <= 7:
                            article = {
                                'title': title,
                                'summary': summary,
                                'link': entry.get('link', ''),
                                'published': published,
                                'source': source_url.split('/')[2],  # Extract domain
                                'search_term': ticker_clean,
                                'relevance_score': relevance
                            }
                            all_articles.append(article)
            except Exception as e:
                continue
        
        # Remove duplicates and sort by relevance and recency
        unique_articles = self._deduplicate_articles(all_articles)
        
        return sorted(unique_articles, 
                     key=lambda x: (x['relevance_score'], x['published']), 
                     reverse=True)[:20]
    
    def analyze_sentiment_advanced(self, articles):
        """Advanced sentiment analysis with confidence scoring"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'market_impact_score': 0.0
            }
        
        sentiment_scores = []
        confidence_scores = []
        market_keywords = {
            'positive': ['buy', 'bullish', 'surge', 'gains', 'profit', 'growth', 'upgrade', 'outperform', 'strong', 'beat'],
            'negative': ['sell', 'bearish', 'fall', 'loss', 'decline', 'downgrade', 'underperform', 'weak', 'miss', 'drop']
        }
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Keyword-based sentiment boost
            pos_keywords = sum(1 for kw in market_keywords['positive'] if kw in text)
            neg_keywords = sum(1 for kw in market_keywords['negative'] if kw in text)
            
            # Adjust polarity based on financial keywords
            keyword_adjustment = (pos_keywords - neg_keywords) * 0.1
            adjusted_polarity = max(-1, min(1, polarity + keyword_adjustment))
            
            # Calculate confidence based on subjectivity and keyword presence
            confidence = (1 - blob.sentiment.subjectivity) * 0.7 + (pos_keywords + neg_keywords) * 0.1
            confidence = min(1.0, confidence)
            
            # Weight by relevance and recency
            recency_weight = max(0.1, 1 - (datetime.now() - article['published']).days / 7)
            relevance_weight = article['relevance_score']
            weight = recency_weight * relevance_weight
            
            sentiment_scores.append(adjusted_polarity * weight)
            confidence_scores.append(confidence * weight)
            
            # Count sentiment categories
            if adjusted_polarity > 0.1:
                positive_count += 1
            elif adjusted_polarity < -0.1:
                negative_count += 1
            else:
                neutral_count += 1
        
        # Calculate weighted averages
        total_weight = sum(confidence_scores) if confidence_scores else 1
        overall_sentiment = sum(sentiment_scores) / total_weight if total_weight > 0 else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Market impact score (how much this news might affect stock price)
        # Based on sentiment strength, confidence, and article count
        impact_factors = [
            abs(overall_sentiment) * 2,  # Sentiment strength
            avg_confidence,               # Confidence in sentiment
            min(len(articles) / 10, 1),   # News volume (capped at 10 articles = max impact)
            sum(1 for a in articles if any(kw in (a['title'] + a['summary']).lower() 
                for kw in market_keywords['positive'] + market_keywords['negative'])) / max(len(articles), 1)  # Keyword density
        ]
        
        market_impact_score = np.mean(impact_factors) * 100
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'confidence': round(avg_confidence * 100, 1),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'market_impact_score': round(market_impact_score, 1),
            'total_articles': len(articles),
            'sentiment_label': self._get_sentiment_label(overall_sentiment)
        }
    
    def get_breaking_news_impact(self, articles):
        """Identify breaking news that might significantly impact stock price"""
        breaking_news = []
        
        for article in articles:
            impact_score = 0
            reasons = []
            
            text = (article['title'] + ' ' + article['summary']).lower()
            
            # Check for breaking news indicators
            if any(term in article['title'].lower() for term in ['breaking', 'urgent', 'alert', 'just in']):
                impact_score += 30
                reasons.append('Breaking news indicator')
            
            # Check for high-impact financial events
            high_impact_terms = [
                'earnings', 'results', 'guidance', 'merger', 'acquisition', 'ipo', 'dividend',
                'split', 'buyback', 'insider trading', 'regulatory', 'investigation',
                'ceo', 'management', 'partnership', 'contract', 'approval', 'rejection'
            ]
            
            for term in high_impact_terms:
                if term in text:
                    impact_score += 15
                    reasons.append(f'Financial event: {term}')
            
            # Check recency (more recent = higher impact)
            hours_old = (datetime.now() - article['published']).total_seconds() / 3600
            if hours_old < 1:
                impact_score += 20
                reasons.append('Very recent (< 1 hour)')
            elif hours_old < 6:
                impact_score += 10
                reasons.append('Recent (< 6 hours)')
            
            # Check sentiment strength
            blob = TextBlob(text)
            if abs(blob.sentiment.polarity) > 0.5:
                impact_score += 15
                reasons.append(f'Strong sentiment ({blob.sentiment.polarity:.2f})')
            
            if impact_score > 30:
                breaking_news.append({
                    'article': article,
                    'impact_score': impact_score,
                    'impact_reasons': reasons,
                    'urgency': 'HIGH' if impact_score > 50 else 'MEDIUM'
                })
        
        return sorted(breaking_news, key=lambda x: x['impact_score'], reverse=True)[:5]
    
    def _parse_date(self, date_str):
        """Parse various date formats"""
        if not date_str:
            return None
        
        try:
            # Common RSS date formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %Z',
                       '%a, %d %b %Y %H:%M:%S %z',
                       '%Y-%m-%dT%H:%M:%S%z',
                       '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            # Fallback to pandas
            return pd.to_datetime(date_str).to_pydatetime()
        except:
            return datetime.now() - timedelta(days=1)  # Default to yesterday
    
    def _calculate_relevance(self, text, search_terms):
        """Calculate how relevant an article is to our stock"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        relevance = 0.0
        
        for term in search_terms:
            term_lower = term.lower()
            # Exact match gets highest score
            if term_lower in text_lower:
                relevance += 0.5
            
            # Partial match gets lower score
            words = term_lower.split()
            for word in words:
                if len(word) > 3 and word in text_lower:
                    relevance += 0.2
        
        return min(1.0, relevance)
    
    def _deduplicate_articles(self, articles):
        """Remove duplicate articles based on title similarity"""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title_words = set(article['title'].lower().split())
            
            # Check if this title is too similar to existing ones
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                similarity = len(title_words & seen_words) / len(title_words | seen_words) if title_words | seen_words else 0
                
                if similarity > 0.7:  # 70% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(article['title'].lower())
        
        return unique_articles
    
    def _get_sentiment_label(self, sentiment_score):
        """Convert sentiment score to human-readable label"""
        if sentiment_score > 0.2:
            return 'Very Positive'
        elif sentiment_score > 0.05:
            return 'Positive'
        elif sentiment_score < -0.2:
            return 'Very Negative'
        elif sentiment_score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

# Global news intelligence instance
news_intelligence = NewsIntelligence()

def get_advanced_news_analysis(ticker, company_name=None):
    """Get comprehensive news analysis for a stock"""
    try:
        # Fetch news articles
        articles = news_intelligence.fetch_stock_news(ticker, company_name)
        
        if not articles:
            return {
                'sentiment': news_intelligence.analyze_sentiment_advanced([]),
                'articles': [],
                'breaking_news': [],
                'summary': 'No recent news found for this stock.',
                'last_updated': datetime.now().isoformat()
            }
        
        # Analyze sentiment
        sentiment_analysis = news_intelligence.analyze_sentiment_advanced(articles)
        
        # Find breaking news
        breaking_news = news_intelligence.get_breaking_news_impact(articles)
        
        # Prepare article summaries
        article_summaries = []
        for article in articles[:10]:  # Top 10 articles
            blob = TextBlob(article['title'] + ' ' + article['summary'])
            article_summaries.append({
                'title': article['title'],
                'summary': article['summary'][:200] + '...' if len(article['summary']) > 200 else article['summary'],
                'link': article['link'],
                'published': article['published'].strftime('%Y-%m-%d %H:%M'),
                'source': article['source'],
                'sentiment': round(blob.sentiment.polarity, 2),
                'relevance': round(article['relevance_score'], 2)
            })
        
        # Generate summary
        summary_parts = []
        if sentiment_analysis['total_articles'] > 0:
            summary_parts.append(f"Found {sentiment_analysis['total_articles']} relevant articles.")
            summary_parts.append(f"Overall sentiment: {sentiment_analysis['sentiment_label']} ({sentiment_analysis['overall_sentiment']:.2f})")
            summary_parts.append(f"Market impact score: {sentiment_analysis['market_impact_score']}/100")
            
            if breaking_news:
                summary_parts.append(f"Found {len(breaking_news)} high-impact news items.")
        
        summary = ' '.join(summary_parts) if summary_parts else 'Limited news coverage found.'
        
        return {
            'sentiment': sentiment_analysis,
            'articles': article_summaries,
            'breaking_news': [
                {
                    'title': item['article']['title'],
                    'impact_score': item['impact_score'],
                    'urgency': item['urgency'],
                    'reasons': item['impact_reasons'],
                    'published': item['article']['published'].strftime('%Y-%m-%d %H:%M'),
                    'link': item['article']['link']
                }
                for item in breaking_news
            ],
            'summary': summary,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'error': f"News analysis failed: {str(e)}",
            'sentiment': news_intelligence.analyze_sentiment_advanced([]),
            'articles': [],
            'breaking_news': [],
            'summary': 'Error occurred while fetching news.',
            'last_updated': datetime.now().isoformat()
        }