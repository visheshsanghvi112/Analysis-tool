# ==============================================================================
# Scrapling Client — Unified Stealth Scraping & Deep Financial Intelligence Core
# Powered by Scrapling (TLS JA3 Impersonation, Anti-Bot Bypass, Markdown Engine)
# ==============================================================================

import sys
import os
import re
import json
import html
import threading
from typing import Optional

# Ensure vendored scrapling is in Python path
vendor_scrapling_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor', 'scrapling'))
if vendor_scrapling_path not in sys.path:
    sys.path.insert(0, vendor_scrapling_path)

try:
    from scrapling.fetchers import FetcherSession, Fetcher
    HAS_SCRAPLING = True
except Exception as e:
    HAS_SCRAPLING = False
    print(f"Warning: Scrapling import error in ScraplingClient: {e}")


class ScraplingClient:
    """
    High-performance, persistent Scrapling client with:
    - HTTP/2 connection pooling via FetcherSession
    - Chrome JA3 TLS fingerprint impersonation (evades Cloudflare/Akamai/Datadome)
    - Google News redirect unmasking (resolves encoded RSS links to canonical publishers)
    - Clean RAG markdown extraction via res.markdown()
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_session()
            return cls._instance

    def _init_session(self):
        self.session = None
        self._session_ctx = None
        if HAS_SCRAPLING:
            try:
                self._session_ctx = FetcherSession(
                    impersonate="chrome",
                    stealthy_headers=True,
                    follow_redirects=True,
                    timeout=8.0,
                    retries=2,
                    retry_delay=1
                )
                self.session = self._session_ctx.__enter__()
            except Exception as e:
                print(f"ScraplingClient: FetcherSession init failed, falling back to Fetcher: {e}")
                self.session = None

    def decode_google_news_url(self, google_news_url: str) -> Optional[str]:
        """
        Decodes a Google News RSS redirect link (https://news.google.com/rss/articles/...)
        to the original canonical publisher URL (Reuters, Economic Times, Moneycontrol, etc.)
        using Google's DotsSplashUi batchexecute endpoint.
        """
        if not google_news_url or not google_news_url.startswith('https://news.google.com/rss/articles/'):
            return google_news_url

        try:
            # Step 1: Fetch Google News landing page to extract signature
            if self.session:
                resp = self.session.get(google_news_url, timeout=5.0)
            else:
                resp = Fetcher.get(google_news_url, impersonate="chrome", stealthy_headers=True, timeout=5.0)
            html_text = getattr(resp, 'text', None) or (resp.body.decode('utf-8', errors='ignore') if hasattr(resp, 'body') else str(resp))

            match = re.search(r'<c-wiz[^>]*data-p=[\"\']([^\"\']+)[\"\']', html_text)
            if not match:
                return None

            data_p = html.unescape(match.group(1))
            obj = json.loads(data_p.replace('%.@.', '["garturlreq",'))

            payload = {
                'f.req': json.dumps([[["Fbv4je", json.dumps(obj[:-6] + obj[-2:]), "null", "generic"]]])
            }

            url = 'https://news.google.com/_/DotsSplashUi/data/batchexecute'
            post_headers = {
                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
            }

            if self.session:
                res = self.session.post(url, headers=post_headers, data=payload, timeout=5.0)
            else:
                res = Fetcher.post(url, headers=post_headers, data=payload, timeout=5.0)
            res_text = getattr(res, 'text', None) or (res.body.decode('utf-8', errors='ignore') if hasattr(res, 'body') else str(res))

            # Step 2: Parse batchexecute response for canonical publisher link
            found = re.findall(r'\"(https?://[^\"]+)\"', res_text)
            for u in found:
                clean_u = u.replace('\\', '').strip()
                if 'google' not in clean_u and 'gstatic' not in clean_u and clean_u.startswith('http'):
                    return clean_u

        except Exception:
            pass

        return None

    def deep_read_markdown(self, url: str, max_chars: int = 3500) -> str:
        """
        Deep-reads the target article body using Scrapling's stealth Chrome session
        and returns sanitized, ad-free article text/Markdown.
        """
        if not HAS_SCRAPLING or not url or not url.startswith('http'):
            return ""

        # Auto-resolve Google News redirect if encountered
        if 'news.google.com/rss/articles/' in url:
            resolved_url = self.decode_google_news_url(url)
            if resolved_url:
                url = resolved_url
            else:
                return ""

        try:
            if self.session:
                res = self.session.get(url, timeout=6.0)
            else:
                res = Fetcher.get(url, impersonate="chrome", stealthy_headers=True, timeout=6.0, follow_redirects=True)

            if res.status == 200:
                # 1. Target primary story body paragraphs for cleaner financial extraction
                paras = res.css(
                    'article p::text, [itemprop="articleBody"] p::text, .story-details p::text, '
                    '.artText p::text, .article_content p::text, .mainArea p::text, .content_wrapper p::text'
                )
                clean_paras = [str(p).strip() for p in paras if len(str(p).strip()) > 45]
                # Filter out promotional lines
                meaningful_paras = [
                    p for p in clean_paras 
                    if not any(ign in p.lower() for ign in ['subscribe to', 'download our app', 'click here to', 'terms of service', 'unlisted shares'])
                ]
                if len(meaningful_paras) >= 2:
                    return "\n\n".join(meaningful_paras[:8])[:max_chars].strip()

                # 2. Fallback to Scrapling native markdown() if structured selectors yield little
                if hasattr(res, 'markdown'):
                    md = res.markdown()
                    if md:
                        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', md)
                        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
                        return cleaned[:max_chars].strip()

                if clean_paras:
                    return "\n\n".join(clean_paras[:6])[:max_chars].strip()

        except Exception:
            return ""

        return ""


# Shared singleton instance
scrapling_client = ScraplingClient()
