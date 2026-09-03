import { API_BASE_URL } from '../config';

// Curated financial aliases mapping common names, nicknames, and concepts
export const SEARCH_ALIASES = {
  // Banking & NBFC
  'sbi': 'SBIN.NS', 'state bank': 'SBIN.NS', 'state bank of india': 'SBIN.NS',
  'hdfc': 'HDFCBANK.NS', 'hdfc bank': 'HDFCBANK.NS', 'hdfcbk': 'HDFCBANK.NS',
  'icici': 'ICICIBANK.NS', 'icici bank': 'ICICIBANK.NS',
  'kotak': 'KOTAKBANK.NS', 'kotak bank': 'KOTAKBANK.NS',
  'axis': 'AXISBANK.NS', 'axis bank': 'AXISBANK.NS',
  'indusind': 'INDUSINDBK.NS', 'pnb': 'PNB.NS', 'bob': 'BANKBARODA.NS',
  'bajaj finance': 'BAJFINANCE.NS', 'bajfinance': 'BAJFINANCE.NS',
  'bajaj finserv': 'BAJAJFINSV.NS', 'bajfinsv': 'BAJAJFINSV.NS',
  // Conglomerates & Auto
  'reliance': 'RELIANCE.NS', 'ril': 'RELIANCE.NS', 'jio': 'RELIANCE.NS',
  'l&t': 'LT.NS', 'larsen': 'LT.NS', 'larsen & toubro': 'LT.NS', 'larsen and toubro': 'LT.NS',
  'm&m': 'M&M.NS', 'mahindra': 'M&M.NS', 'mahindra & mahindra': 'M&M.NS',
  'tata motors': 'TMCV.NS', 'tatamotors': 'TMCV.NS', 'tata motor': 'TMCV.NS',
  'tata power': 'TATAPOWER.NS', 'tata steel': 'TATASTEEL.NS', 'tata tech': 'TATATECH.NS',
  'maruti': 'MARUTI.NS', 'maruti suzuki': 'MARUTI.NS',
  'bajaj auto': 'BAJAJ-AUTO.NS', 'hero': 'HEROMOTOCO.NS', 'hero motocorp': 'HEROMOTOCO.NS',
  'eicher': 'EICHERMOT.NS', 'royal enfield': 'EICHERMOT.NS',
  // IT & Tech
  'tcs': 'TCS.NS', 'tata consultancy': 'TCS.NS',
  'infy': 'INFY.NS', 'infosys': 'INFY.NS',
  'wipro': 'WIPRO.NS', 'hcl': 'HCLTECH.NS', 'hcl tech': 'HCLTECH.NS',
  'tech mahindra': 'TECHM.NS', 'techm': 'TECHM.NS', 'ltim': 'LTIM.NS', 'mindtree': 'LTIM.NS',
  // FMCG & Consumer
  'hul': 'HINDUNILVR.NS', 'unilever': 'HINDUNILVR.NS', 'hindustan unilever': 'HINDUNILVR.NS',
  'itc': 'ITC.NS', 'nestle': 'NESTLEIND.NS', 'britannia': 'BRITANNIA.NS',
  'asian paints': 'ASIANPAINT.NS', 'asian paint': 'ASIANPAINT.NS',
  'titan': 'TITAN.NS', 'pidilite': 'PIDILITIND.NS', 'fevicol': 'PIDILITIND.NS',
  'dabur': 'DABUR.NS', 'marico': 'MARICO.NS',
  // Pharma
  'sun pharma': 'SUNPHARMA.NS', 'dr reddy': 'DRREDDY.NS', 'cipla': 'CIPLA.NS',
  'divis': 'DIVISLAB.NS', 'apollo hospital': 'APOLLOHOSP.NS', 'zydus': 'ZYDUSLIFE.NS',
  // Energy, Metals & PSU
  'ongc': 'ONGC.NS', 'ntpc': 'NTPC.NS', 'powergrid': 'POWERGRID.NS', 'power grid': 'POWERGRID.NS',
  'coal india': 'COALINDIA.NS', 'bpcl': 'BPCL.NS', 'ioc': 'IOC.NS', 'indian oil': 'IOC.NS',
  'gail': 'GAIL.NS', 'sail': 'SAIL.NS', 'bhel': 'BHEL.NS', 'bel': 'BEL.NS', 'hal': 'HAL.NS',
  'irctc': 'IRCTC.NS', 'irfc': 'IRFC.NS', 'pfc': 'PFC.NS', 'rec': 'RECLTD.NS',
  'jsw steel': 'JSWSTEEL.NS', 'hindalco': 'HINDALCO.NS', 'vedanta': 'VEDL.NS',
  'adani ent': 'ADANIENT.NS', 'adani ports': 'ADANIPORTS.NS', 'adani power': 'ADANIPOWER.NS',
  // New-Age Tech & IPOs
  'zomato': 'ETERNAL.NS', 'eternal': 'ETERNAL.NS',
  'paytm': 'PAYTM.NS', 'swiggy': 'SWIGGY.NS', 'ola': 'OLAELEC.NS', 'ola electric': 'OLAELEC.NS',
  'nykaa': 'NYKAA.NS', 'policybazaar': 'POLICYBZR.NS', 'delhivery': 'DELHIVERY.NS',
  'hyundai': 'HYUNDAI.NS', 'waaree': 'WAAREEENER.NS', 'waaree energies': 'WAAREEENER.NS',
  'premier energies': 'PREMIERENE.NS', 'afcons': 'AFCONS.NS', 'ntpc green': 'NTPCGREEN.NS',
  // Core Benchmark Indices
  'nifty': '^NSEI', 'nifty 50': '^NSEI', 'nifty50': '^NSEI',
  'sensex': '^BSESN', 'bse sensex': '^BSESN',
  'bank nifty': '^NSEBANK', 'banknifty': '^NSEBANK', 'nifty bank': '^NSEBANK',
  'nifty it': '^CNXIT', 'it index': '^CNXIT',
  // ETFs & Commodities
  'gold etf': 'GOLDBEES.NS', 'gold bees': 'GOLDBEES.NS', 'gold': 'GOLDBEES.NS',
  'silver etf': 'SILVERBEES.NS', 'silver bees': 'SILVERBEES.NS', 'silver': 'SILVERBEES.NS',
  'nifty etf': 'NIFTYBEES.NS', 'nifty bees': 'NIFTYBEES.NS',
  'bank etf': 'BANKBEES.NS', 'bank bees': 'BANKBEES.NS',
  'it etf': 'ITBEES.NS', 'it bees': 'ITBEES.NS',
  'nasdaq': 'MON100.NS', 'nasdaq etf': 'MON100.NS', 'nasdaq 100': 'MON100.NS',
  'fang': 'MAFANG.NS', 'faang': 'MAFANG.NS', 'fang etf': 'MAFANG.NS',
  'cpse': 'CPSEETF.NS', 'psu etf': 'CPSEETF.NS',
  'liquid etf': 'LIQUIDBEES.NS', 'liquid bees': 'LIQUIDBEES.NS',
  // Global Mega-Caps
  'apple': 'AAPL', 'nvidia': 'NVDA', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
  'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
  'sp500': 'SPY', 's&p 500': 'SPY', 'spy': 'SPY', 'qqq': 'QQQ'
};

// Prominence weights for high-liquidity / bluechips
export const BLUECHIP_PROMINENCE = {
  'RELIANCE.NS': 100, 'TCS.NS': 99, 'HDFCBANK.NS': 98, 'INFY.NS': 97, 'ICICIBANK.NS': 96,
  'SBIN.NS': 95, 'BHARTIARTL.NS': 94, 'ITC.NS': 93, 'LT.NS': 92, 'HINDUNILVR.NS': 91,
  'TMCV.NS': 90, 'TATASTEEL.NS': 89, 'TATAPOWER.NS': 88, 'BAJFINANCE.NS': 87,
  'M&M.NS': 86, 'MARUTI.NS': 85, 'SUNPHARMA.NS': 84, 'KOTAKBANK.NS': 83, 'AXISBANK.NS': 82,
  'NTPC.NS': 81, 'POWERGRID.NS': 80, 'ONGC.NS': 79, 'COALINDIA.NS': 78, 'TITAN.NS': 77,
  'NIFTYBEES.NS': 95, 'BANKBEES.NS': 94, 'GOLDBEES.NS': 93, 'SILVERBEES.NS': 92, 'ITBEES.NS': 91,
  'MON100.NS': 90, 'CPSEETF.NS': 89, 'MAFANG.NS': 88, '^NSEI': 100, '^BSESN': 100, '^NSEBANK': 99,
  'SWIGGY.NS': 85, 'ETERNAL.NS': 88, 'WAAREEENER.NS': 86, 'HYUNDAI.NS': 85, 'OLAELEC.NS': 84
};

// Simple Levenshtein distance for typo matching
function levenshteinDistance(s1, s2) {
  const m = s1.length;
  const n = s2.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (s1[i - 1] === s2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,     // deletion
          dp[i][j - 1] + 1,     // insertion
          dp[i - 1][j - 1] + 1  // substitution
        );
      }
    }
  }
  return dp[m][n];
}

function stringSimilarity(s1, s2) {
  const longer = s1.length >= s2.length ? s1 : s2;
  const shorter = s1.length < s2.length ? s1 : s2;
  if (longer.length === 0) return 1.0;
  return (longer.length - levenshteinDistance(longer, shorter)) / longer.length;
}

/**
 * Intelligent Client-Side Search
 * Ranks instruments using exact matches, aliases, multi-tokens, prominence, and typo tolerance.
 */
export function smartSearch(items, rawQuery, options = {}) {
  if (!Array.isArray(items) || items.length === 0) return [];
  const q = (rawQuery || '').toLowerCase().trim();
  if (!q) return items.slice(0, options.limit || 30);

  const limit = options.limit || 30;
  const qClean = q.replace('.ns', '').replace('.bo', '').replace('^', '').replace('-', '').replace('&', '');
  const qTokens = q.split(/[\s\-_&]+/).filter(Boolean);

  const scored = [];

  // Check direct alias target
  const aliasTarget = SEARCH_ALIASES[q] || SEARCH_ALIASES[qClean];
  if (aliasTarget) {
    const aliasItem = items.find(t => t.symbol === aliasTarget);
    if (aliasItem) {
      scored.push({ score: 1000, item: aliasItem });
    }
  }

  for (let i = 0; i < items.length; i++) {
    const t = items[i];
    const sym = t.symbol || '';
    const name = (t.name || '').toLowerCase();
    const bse = t.bse_code || '';

    const symClean = sym.toLowerCase().replace('.ns', '').replace('.bo', '').replace('^', '').replace('-', '').replace('&', '');
    const prominence = BLUECHIP_PROMINENCE[sym] || 0;

    // BSE code lookup
    if (bse && q === bse) {
      scored.push({ score: 950, item: t });
      continue;
    }

    let score = 0;

    // 1. Exact symbol match
    if (symClean === qClean) {
      score = 850 + prominence;
    }
    // 2. Exact start of symbol
    else if (symClean.startsWith(qClean)) {
      score = 650 + prominence - symClean.length;
    }
    // 3. Substring in symbol
    else if (symClean.includes(qClean)) {
      score = 450 + prominence - symClean.length;
    }
    // 4. Multi-token match across name/symbol
    else if (qTokens.length > 0 && qTokens.every(tok => name.includes(tok) || symClean.includes(tok))) {
      score = 350 + prominence;
    }
    // 5. Name starts with query
    else if (name.startsWith(q)) {
      score = 280 + prominence;
    }
    // 6. Name contains query
    else if (name.includes(q)) {
      score = 200 + prominence;
    }
    // 7. Typo tolerance (if query >= 4 chars)
    else if (qClean.length >= 4) {
      const symRatio = stringSimilarity(qClean, symClean);
      if (symRatio >= 0.78) {
        score = Math.floor(180 * symRatio) + prominence;
      } else {
        const firstWord = name.split(/\s+/)[0] || '';
        const wordRatio = stringSimilarity(qClean, firstWord);
        if (wordRatio >= 0.80) {
          score = Math.floor(150 * wordRatio) + prominence;
        }
      }
    }

    // Tie-breaker bonus for NSE primary over BSE
    if (score > 0 && sym.endsWith('.NS')) {
      score += 15;
    }

    if (score > 0) {
      scored.push({ score, item: t });
    }
  }

  // Sort descending by score
  scored.sort((a, b) => b.score - a.score);

  // Deduplicate
  const result = [];
  const seen = new Set();
  for (let i = 0; i < scored.length; i++) {
    const sym = scored[i].item.symbol;
    if (!seen.has(sym)) {
      seen.add(sym);
      result.push(scored[i].item);
      if (result.length >= limit) break;
    }
  }

  return result;
}

/**
 * Universal Smart Search Fetcher
 * Tries the backend intelligent endpoint, with automatic instant fallback to local universe.
 */
export async function fetchSmartTickers(query, localFallbackList = [], limit = 30) {
  if (!query || !query.trim()) return [];
  const cleanQ = query.trim();

  try {
    const res = await fetch(`${API_BASE_URL}/api/tickers?q=${encodeURIComponent(cleanQ)}&limit=${limit}`);
    if (res.ok) {
      const json = await res.json();
      if (Array.isArray(json.tickers) && json.tickers.length > 0) {
        return json.tickers;
      }
    }
  } catch (_) {
    // Network failed, fall through to client search
  }

  // Fallback to local client smart search
  return smartSearch(localFallbackList, cleanQ, { limit });
}
