'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Header from '../components/Header';
import {
  Sparkles,
  TrendingUp,
  TrendingDown,
  Target,
  Shield,
  Activity,
  Filter,
  RefreshCw,
  Search,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Star,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Clock,
  Layers,
  ArrowRight,
  Info,
  SlidersHorizontal,
  Flame,
  Award,
  BarChart2,
  Briefcase,
  Calculator,
  Download,
  Copy,
  Check,
  Percent,
  Coins,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  Trash2,
  FileText,
  CheckSquare,
  Share2,
} from 'lucide-react';
import { API_BASE_URL } from '../config';

const PRICE_TIERS = [
  { label: 'Under ₹200 (Focus)', value: 200 },
  { label: 'Under ₹100', value: 100 },
  { label: 'Under ₹50', value: 50 },
  { label: 'Under ₹500', value: 500 },
  { label: 'All Scrips', value: null },
];

const SETUP_FILTERS = [
  { label: 'All Setups', value: 'all' },
  { label: '🚀 Breakouts', value: 'MOMENTUM_BREAKOUT' },
  { label: '⚡ BTST 3 PM Surge', value: 'BTST_SURGE' },
  { label: '✨ Golden Crossovers', value: 'GOLDEN_CROSSOVER' },
  { label: '🛡️ High-Dividend PSU', value: 'DIVIDEND_VALUE' },
  { label: '💎 Oversold Reversals', value: 'OVERSOLD_REVERSAL' },
  { label: '📈 50 DMA Pullbacks', value: 'SUPPORT_PULLBACK' },
  { label: '🗜️ Squeeze Bases', value: 'SQUEEZE_BASE' },
  { label: '🔥 High Conviction (≥80)', value: 'high_conviction' },
];

const THEME_PLAYBOOKS = [
  { id: 'all', label: 'All Under ₹200 Radar', icon: '⚡' },
  { id: 'momentum', label: 'High-Volume Breakouts', icon: '🚀' },
  { id: 'btst', label: 'BTST 3 PM Overnight Surge', icon: '⚡' },
  { id: 'oversold', label: 'Oversold Turnarounds', icon: '💎' },
  { id: 'dividend_yield', label: 'PSU High Dividend Yield', icon: '🛡️' },
  { id: 'clean_energy', label: 'Clean Energy & Capex', icon: '🌿' },
  { id: 'pullback', label: '50 DMA Trend Pullbacks', icon: '📈' },
  { id: 'psu_capex', label: 'PSU Capex Titans', icon: '🏛️' },
  { id: 'high_liquidity', label: 'Ultra-Liquid (Turnover > ₹50 Cr)', icon: '🛡️' },
];

const SORT_OPTIONS = [
  { label: 'Highest Conviction', value: 'conviction' },
  { label: 'Best Risk-to-Reward', value: 'risk_reward' },
  { label: 'Volume Surge (RVOL)', value: 'rvol' },
  { label: 'Daily Turnover (₹ Cr)', value: 'turnover' },
  { label: 'Circuit Headroom %', value: 'circuit_headroom' },
  { label: 'Delivery Accumulation %', value: 'delivery' },
  { label: 'Price: Low to High', value: 'price_asc' },
  { label: 'Price: High to Low', value: 'price_desc' },
  { label: 'Session Gain %', value: 'day_gain' },
  { label: 'Near 52-Week Low', value: 'proximity_low' },
];

const BASKET_THEMES = [
  { id: 'balanced', label: 'Balanced Alpha', desc: 'Diversified cross-sector leaders with max 1 stock per industry', icon: '⚖️' },
  { id: 'momentum', label: 'Momentum Breakouts', desc: 'Highest relative volume explosion and resistance breaks', icon: '🚀' },
  { id: 'btst', label: 'BTST 3 PM Surge', desc: 'Scrips closing near day high with high volume for overnight gap', icon: '⚡' },
  { id: 'clean_energy', label: 'Clean Energy & Infra', desc: 'Renewable energy, wind, solar, and railway transformation scrips', icon: '🌿' },
  { id: 'dividend_yield', label: 'High-Yield Dividends', desc: 'Sovereign PSUs and cash cows with 4%–8% dividend yields + capital upside', icon: '🛡️' },
  { id: 'value', label: 'Deep Value Reversion', desc: 'Severely oversold large turnarounds with asymmetric upside', icon: '💎' },
  { id: 'psu_capex', label: 'PSU Capex Titans', desc: 'Sovereign-backed power, defense, railway, and bank giants under ₹200', icon: '🏛️' },
  { id: 'high_liquidity', label: 'Institutional Ultra-Liquid', desc: 'Only scrips with ₹50+ Crore daily turnover and minimal operator risk', icon: '🛡️' },
];

export default function SuggestionsPage() {
  const router = useRouter();

  // Primary view tabs: 'radar' (Signal list) or 'basket' (Smart Basket Studio)
  const [activeTab, setActiveTab] = useState('radar');

  // Suggestions state
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Filters
  const [priceTier, setPriceTier] = useState(200);
  const [setupFilter, setSetupFilter] = useState('all');
  const [selectedTheme, setSelectedTheme] = useState('all');
  const [selectedSector, setSelectedSector] = useState('all');
  const [sortBy, setSortBy] = useState('conviction');
  const [searchQuery, setSearchQuery] = useState('');

  // Expandable reasoning state
  const [expandedCards, setExpandedCards] = useState(new Set());

  // Interactive Position Calculator state on card
  const [activeCalcSymbol, setActiveCalcSymbol] = useState(null);

  // Custom ticker evaluation
  const [customTicker, setCustomTicker] = useState('');
  const [customEvaluating, setCustomEvaluating] = useState(false);
  const [customResult, setCustomResult] = useState(null);
  const [customError, setCustomError] = useState(null);

  // Watchlist & Toast state
  const [watchlist, setWatchlist] = useState([]);
  const [toastMessage, setToastMessage] = useState('');

  // Smart Basket Studio state
  const [basketCapital, setBasketCapital] = useState(50000);
  const [basketTheme, setBasketTheme] = useState('balanced');
  const [basketScripCount, setBasketScripCount] = useState(4);
  const [basketData, setBasketData] = useState(null);
  const [basketLoading, setBasketLoading] = useState(false);
  const [copiedBasket, setCopiedBasket] = useState(false);
  const [activeBrokerFormat, setActiveBrokerFormat] = useState('zerodha');

  // Paper Trading Ledger state
  const [paperTrades, setPaperTrades] = useState([]);

  // Standalone Capital & Nifty Stress Calculator state
  const [calcTicker, setCalcTicker] = useState('PNB.NS');
  const [calcCapital, setCalcCapital] = useState(100000);
  const [calcRiskPct, setCalcRiskPct] = useState(1.0);
  const [calcNiftyScenario, setCalcNiftyScenario] = useState(-1.0);

  // Sync watchlist and paper trades
  useEffect(() => {
    try {
      const savedWatchlist = localStorage.getItem('stockiq_pro_watchlist');
      if (savedWatchlist) {
        setWatchlist(JSON.parse(savedWatchlist) || []);
      }
      const savedTrades = localStorage.getItem('stockiq_pro_paper_trades');
      if (savedTrades) {
        setPaperTrades(JSON.parse(savedTrades) || []);
      }
    } catch (_) {}
  }, []);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(''), 3200);
  };

  const addPaperTrade = (scrip, customShares) => {
    const shares = customShares || scrip.position_sizing_presets?.capital_50k?.shares_to_buy || 100;
    const price = scrip.technicals.current_price;
    const newTrade = {
      id: `${scrip.symbol}_${Date.now()}`,
      symbol: scrip.symbol,
      display_symbol: scrip.display_symbol,
      name: scrip.name,
      sector: scrip.sector,
      setup: scrip.setup_label,
      entry_price: price,
      current_price: price,
      shares: shares,
      outlay: Math.round(shares * price),
      target_1: scrip.trade_plan.target_1,
      target_2: scrip.trade_plan.target_2,
      stop_loss: scrip.trade_plan.stop_loss,
      entry_date: new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }),
      status: 'ACTIVE',
      realized_pnl: 0,
      partial_booked: false,
    };
    const updated = [newTrade, ...paperTrades.filter((t) => t.symbol !== scrip.symbol)];
    setPaperTrades(updated);
    try {
      localStorage.setItem('stockiq_pro_paper_trades', JSON.stringify(updated));
    } catch (_) {}
    showToast(`Added ${scrip.display_symbol} (${shares} shares) to Paper Trading Ledger`);
  };

  const updateTradeStatus = (tradeId, newStatus, pnl = 0) => {
    const updated = paperTrades.map((t) => {
      if (t.id === tradeId) {
        return {
          ...t,
          status: newStatus,
          realized_pnl: (t.realized_pnl || 0) + pnl,
          partial_booked: newStatus === 'T1_HIT' ? true : t.partial_booked,
        };
      }
      return t;
    });
    setPaperTrades(updated);
    try {
      localStorage.setItem('stockiq_pro_paper_trades', JSON.stringify(updated));
    } catch (_) {}
    showToast(`Trade status updated to ${newStatus}`);
  };

  const removePaperTrade = (tradeId) => {
    const updated = paperTrades.filter((t) => t.id !== tradeId);
    setPaperTrades(updated);
    try {
      localStorage.setItem('stockiq_pro_paper_trades', JSON.stringify(updated));
    } catch (_) {}
    showToast('Trade removed from ledger');
  };

  const toggleWatchlist = (symbol) => {
    try {
      const exists = watchlist.includes(symbol);
      const updated = exists ? watchlist.filter((s) => s !== symbol) : [...watchlist, symbol];
      setWatchlist(updated);
      localStorage.setItem('stockiq_pro_watchlist', JSON.stringify(updated));
      window.dispatchEvent(new CustomEvent('stockiq-watchlist-changed'));
      showToast(exists ? `Removed ${symbol} from watchlist` : `Added ${symbol} to watchlist`);
    } catch (_) {}
  };

  // Toggle card expansion
  const toggleCard = (symbol) => {
    setExpandedCards((prev) => {
      const next = new Set(prev);
      if (next.has(symbol)) next.delete(symbol);
      else next.add(symbol);
      return next;
    });
  };

  const expandAll = () => {
    if (!data?.suggestions) return;
    setExpandedCards(new Set(data.suggestions.map((s) => s.symbol)));
  };

  const collapseAll = () => {
    setExpandedCards(new Set());
  };

  // Fetch suggestions
  const fetchSuggestions = useCallback(
    async (isManualRefresh = false) => {
      if (isManualRefresh) setRefreshing(true);
      else setLoading(true);
      setError(null);

      try {
        const params = new URLSearchParams();
        if (priceTier !== null) params.append('max_price', priceTier);
        else params.append('max_price', '0');

        params.append('setup_type', setupFilter);
        params.append('sector', selectedSector);
        params.append('sort_by', sortBy);
        if (isManualRefresh) params.append('force_refresh', 'true');

        const res = await fetch(`${API_BASE_URL}/api/suggestions?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load suggestions`);
        const json = await res.json();
        setData(json);

        if (!isManualRefresh && json?.suggestions?.length > 0) {
          const topSymbols = json.suggestions.slice(0, 2).map((s) => s.symbol);
          setExpandedCards(new Set(topSymbols));
        }
      } catch (err) {
        setError(err.message || 'Error connecting to StockIQ Pro backend.');
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [priceTier, setupFilter, selectedSector, sortBy]
  );

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  // Fetch smart basket
  const fetchSmartBasket = useCallback(async () => {
    setBasketLoading(true);
    setCopiedBasket(false);
    try {
      const params = new URLSearchParams({
        capital: basketCapital.toString(),
        theme: basketTheme,
        max_price: '200',
        max_scrips: basketScripCount.toString(),
      });
      const res = await fetch(`${API_BASE_URL}/api/suggestions/basket?${params.toString()}`);
      if (!res.ok) throw new Error(`Failed to generate basket (HTTP ${res.status})`);
      const json = await res.json();
      setBasketData(json);
    } catch (err) {
      showToast(err.message || 'Basket generation failed.');
    } finally {
      setBasketLoading(false);
    }
  }, [basketCapital, basketTheme, basketScripCount]);

  useEffect(() => {
    if (activeTab === 'basket' && !basketData) {
      fetchSmartBasket();
    }
  }, [activeTab, basketData, fetchSmartBasket]);

  // Custom ticker evaluation
  const handleEvaluateCustom = async (e) => {
    e?.preventDefault();
    const query = customTicker.trim().toUpperCase();
    if (!query) return;

    setCustomEvaluating(true);
    setCustomError(null);
    setCustomResult(null);

    try {
      const cleanTicker = query.includes('.') ? query : `${query}.NS`;
      const res = await fetch(`${API_BASE_URL}/api/suggestions/evaluate/${encodeURIComponent(cleanTicker)}`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Scrip '${cleanTicker}' could not be evaluated.`);
      }
      const result = await res.json();
      setCustomResult(result);
      setExpandedCards((prev) => new Set([...prev, result.symbol]));
    } catch (err) {
      setCustomError(err.message || 'Evaluation failed. Verify ticker symbol.');
    } finally {
      setCustomEvaluating(false);
    }
  };

  // Multi-broker format content resolver
  const getBrokerExportContent = () => {
    if (!basketData) return '';
    if (basketData.broker_exports) {
      if (activeBrokerFormat === 'zerodha') return basketData.broker_exports.zerodha_kite_csv;
      if (activeBrokerFormat === 'angel_one') return basketData.broker_exports.angel_one_csv;
      if (activeBrokerFormat === 'groww') return basketData.broker_exports.groww_csv;
      if (activeBrokerFormat === 'dhan') return basketData.broker_exports.dhan_webhook_orders;
      if (activeBrokerFormat === 'telegram') return basketData.broker_exports.telegram_whatsapp_summary;
    }
    return basketData.broker_basket_csv || '';
  };

  // Copy selected broker format to clipboard
  const handleCopySelectedBrokerFormat = () => {
    const content = getBrokerExportContent();
    if (!content) return;
    navigator.clipboard.writeText(content);
    setCopiedBasket(true);
    showToast(`Copied ${activeBrokerFormat.toUpperCase()} format to clipboard!`);
    setTimeout(() => setCopiedBasket(false), 3000);
  };

  // Download selected broker format file
  const handleDownloadSelectedBrokerFormat = () => {
    const content = getBrokerExportContent();
    if (!content) return;
    const isJson = activeBrokerFormat === 'dhan';
    const isTxt = activeBrokerFormat === 'telegram';
    const ext = isJson ? 'json' : isTxt ? 'txt' : 'csv';
    const mime = isJson ? 'application/json' : isTxt ? 'text/plain' : 'text/csv';
    const blob = new Blob([content], { type: `${mime};charset=utf-8;` });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `stockiq_${activeBrokerFormat}_${basketTheme}_${Date.now()}.${ext}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast(`Downloaded ${activeBrokerFormat.toUpperCase()} export file`);
  };

  // Add all basket scrips to Paper Trading Ledger
  const handleAddBasketToPaperTrades = () => {
    if (!basketData?.allocations?.length) return;
    let count = 0;
    let updated = [...paperTrades];
    basketData.allocations.forEach((item) => {
      if (!updated.some((t) => t.symbol === item.symbol && t.status === 'ACTIVE')) {
        updated = [
          {
            id: `${item.symbol}_${Date.now()}_${Math.random()}`,
            symbol: item.symbol,
            display_symbol: item.display_symbol,
            name: item.name,
            sector: item.sector,
            setup: `Basket: ${basketTheme}`,
            entry_price: item.price,
            current_price: item.price,
            shares: item.shares,
            outlay: item.outlay,
            target_1: item.target_1,
            target_2: item.target_2,
            stop_loss: item.stop_loss,
            entry_date: new Date().toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }),
            status: 'ACTIVE',
            realized_pnl: 0,
            partial_booked: false,
          },
          ...updated,
        ];
        count++;
      }
    });
    setPaperTrades(updated);
    try {
      localStorage.setItem('stockiq_pro_paper_trades', JSON.stringify(updated));
    } catch (_) {}
    showToast(`Added ${count} basket scrips to Paper Trading Ledger!`);
  };

  // Export Paper Trading Ledger as CSV
  const exportPaperTradesCSV = () => {
    if (paperTrades.length === 0) return;
    const headers = ['Symbol', 'Name', 'Sector', 'Setup', 'Entry Price', 'Shares', 'Outlay', 'Target 1', 'Target 2', 'Stop Loss', 'Entry Date', 'Status', 'Realized PnL'];
    const rows = paperTrades.map((t) => [
      t.display_symbol,
      `"${(t.name || '').replace(/"/g, '""')}"`,
      `"${t.sector || ''}"`,
      `"${t.setup || ''}"`,
      t.entry_price,
      t.shares,
      t.outlay,
      t.target_1,
      t.target_2,
      t.stop_loss,
      t.entry_date,
      t.status,
      t.realized_pnl || 0,
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `stockiq_paper_trades_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Exported paper trading ledger to CSV');
  };

  // Clear Paper Trading Ledger
  const clearAllPaperTrades = () => {
    if (typeof window !== 'undefined' && window.confirm('Are you sure you want to clear your entire paper trading ledger?')) {
      setPaperTrades([]);
      try {
        localStorage.removeItem('stockiq_pro_paper_trades');
      } catch (_) {}
      showToast('Cleared paper trading ledger');
    }
  };

  // Live scrip lookup for Position Sizing & Nifty Stress Calculator tab
  const activeCalcScrip = useMemo(() => {
    if (!data?.suggestions) return null;
    return (
      data.suggestions.find((s) => s.symbol === calcTicker || s.display_symbol === calcTicker) ||
      (customResult?.symbol === calcTicker ? customResult : null) ||
      data.suggestions[0] ||
      null
    );
  }, [data, calcTicker, customResult]);

  // Live Position Sizing & Nifty Stress Computations
  const calcResults = useMemo(() => {
    if (!activeCalcScrip) return null;
    const price = activeCalcScrip.technicals.current_price;
    const sl = activeCalcScrip.trade_plan.stop_loss;
    const t1 = activeCalcScrip.trade_plan.target_1;
    const t2 = activeCalcScrip.trade_plan.target_2;
    const riskPerShare = Math.max(price - sl, 0.05);
    const riskBudget = calcCapital * (calcRiskPct / 100.0);
    const shares = Math.max(1, Math.floor(riskBudget / riskPerShare));
    const outlay = Math.round(shares * price);
    const capitalPct = ((outlay / calcCapital) * 100).toFixed(1);
    const maxLoss = Math.round(shares * riskPerShare);
    const t1Gain = Math.round(shares * (t1 - price));
    const t2Gain = Math.round(shares * (t2 - price));
    const t1GainPct = ((t1Gain / calcCapital) * 100).toFixed(2);
    const t2GainPct = ((t2Gain / calcCapital) * 100).toFixed(2);
    const beta = activeCalcScrip.nifty_beta_sensitivity?.beta || 1.25;
    const scenarioStockPct = Number((calcNiftyScenario * beta).toFixed(2));
    const simulatedPrice = Number((price * (1 + scenarioStockPct / 100)).toFixed(2));
    const rupeeImpact = Math.round(outlay * (scenarioStockPct / 100));
    const wouldHitSL = simulatedPrice <= sl;

    return {
      price,
      sl,
      t1,
      t2,
      riskPerShare,
      riskBudget,
      shares,
      outlay,
      capitalPct,
      maxLoss,
      t1Gain,
      t2Gain,
      t1GainPct,
      t2GainPct,
      beta,
      scenarioStockPct,
      simulatedPrice,
      rupeeImpact,
      wouldHitSL,
      rr: activeCalcScrip.trade_plan.risk_reward_ratio,
    };
  }, [activeCalcScrip, calcCapital, calcRiskPct, calcNiftyScenario]);

  // Paper Trading Aggregate Stats
  const paperStats = useMemo(() => {
    const active = paperTrades.filter((t) => t.status === 'ACTIVE' || t.status === 'T1_HIT');
    const closed = paperTrades.filter((t) => t.status === 'T2_HIT' || t.status === 'STOPPED_OUT' || t.status === 'CLOSED');
    const totalCapital = active.reduce((acc, t) => acc + (t.outlay || 0), 0);
    const totalRealized = paperTrades.reduce((acc, t) => acc + (t.realized_pnl || 0), 0);
    const wins = paperTrades.filter((t) => t.status === 'T1_HIT' || t.status === 'T2_HIT').length;
    const winRate = closed.length > 0 ? Math.round((wins / (wins + paperTrades.filter((t) => t.status === 'STOPPED_OUT').length || 1)) * 100) : 100;
    return {
      activeCount: active.length,
      closedCount: closed.length,
      totalCapital,
      totalRealized,
      winRate,
    };
  }, [paperTrades]);

  // Client-side text and theme filtering
  const displayedSuggestions = useMemo(() => {
    if (!data?.suggestions) return [];
    let list = data.suggestions;

    // Filter by Playbook Theme
    if (selectedTheme === 'momentum') {
      list = list.filter((s) => s.setup_type === 'MOMENTUM_BREAKOUT' || s.technicals.rvol >= 1.25);
    } else if (selectedTheme === 'oversold') {
      list = list.filter((s) => s.setup_type === 'OVERSOLD_REVERSAL' || s.technicals.rsi <= 40);
    } else if (selectedTheme === 'pullback') {
      list = list.filter((s) => s.setup_type === 'SUPPORT_PULLBACK');
    } else if (selectedTheme === 'psu_capex') {
      const psuSet = new Set([
        'SAIL.NS', 'IOC.NS', 'BEL.NS', 'NHPC.NS', 'IRFC.NS', 'PNB.NS', 'CANBK.NS',
        'UNIONBANK.NS', 'SJVN.NS', 'HUDCO.NS', 'IREDA.NS', 'MAHABANK.NS', 'GAIL.NS',
        'CPSEETF.NS', 'CENTRALBK.NS', 'IOB.NS', 'UCOBANK.NS'
      ]);
      list = list.filter((s) => psuSet.has(s.symbol) || s.sector.includes('PSU') || s.industry.includes('Public'));
    } else if (selectedTheme === 'high_liquidity') {
      list = list.filter((s) => s.liquidity_guardrails.avg_turnover_cr >= 25.0);
    }

    // Sort by turnover if selected
    if (sortBy === 'turnover') {
      list = [...list].sort(
        (a, b) => b.liquidity_guardrails.avg_turnover_cr - a.liquidity_guardrails.avg_turnover_cr
      );
    }

    // Text search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(
        (s) =>
          s.display_symbol.toLowerCase().includes(q) ||
          s.name.toLowerCase().includes(q) ||
          s.sector.toLowerCase().includes(q) ||
          s.setup_label.toLowerCase().includes(q)
      );
    }

    return list;
  }, [data, selectedTheme, sortBy, searchQuery]);

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 selection:bg-emerald-500/20 selection:text-emerald-300">
      {/* ── Main Navigation Header ────────────────────────────────────────── */}
      <Header />

      {/* ── Toast Notification ────────────────────────────────────────────── */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-500 text-black font-semibold text-xs shadow-2xl backdrop-blur-md animate-fade-in">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* ── Hero / Header Strip ───────────────────────────────────────────── */}
      <div className="border-b border-white/[0.08] bg-gradient-to-b from-purple-950/20 via-[#07090e] to-[#07090e] pt-8 pb-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2.5 flex-wrap mb-2">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-purple-500/15 border border-purple-500/30 text-purple-300 font-semibold text-xs tracking-wide">
                  <Sparkles className="w-3.5 h-3.5" />
                  QUANTITATIVE RADAR
                </span>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 font-semibold text-xs">
                  <Target className="w-3.5 h-3.5" />
                  Primary Focus: Scrips Under ₹200
                </span>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-semibold text-xs">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Operator Risk &amp; Liquidity Audited
                </span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
                Scrip Suggestions, Risk Sizing &amp; Portfolio Studio
              </h1>
              <p className="mt-1 text-sm text-slate-400 max-w-3xl">
                Rigorous, data-backed stock suggestions engineered for active retail swing traders and investors.
                Includes <strong>position sizing calculations</strong>, <strong>liquidity/operator trap guardrails</strong>,
                a <strong>3-stage trailing stop-loss roadmap</strong>, and <strong>1-click multi-stock basket building</strong>.
              </p>
            </div>

            {/* Live Scan Trigger */}
            <div className="flex items-center gap-2 self-start md:self-center">
              <button
                onClick={() => fetchSuggestions(true)}
                disabled={refreshing || loading}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 hover:border-emerald-500/50 text-emerald-400 font-semibold text-xs transition disabled:opacity-50 cursor-pointer shadow-lg shadow-emerald-950/20"
                title="Force a real-time fresh market scan"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
                <span>{refreshing ? 'Scanning NSE...' : 'Run Live Scan'}</span>
              </button>
            </div>
          </div>

          {/* ── Mode Switcher Tabs ───────────────────────────────────────────── */}
          <div className="mt-6 flex items-center gap-2 overflow-x-auto border-b border-white/[0.08] pb-1 scrollbar-none">
            <button
              onClick={() => setActiveTab('radar')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg font-bold text-xs transition cursor-pointer border-b-2 flex-shrink-0 ${
                activeTab === 'radar'
                  ? 'border-emerald-400 text-white bg-white/[0.05]'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>Scrip Signal Radar ({displayedSuggestions.length})</span>
            </button>

            <button
              onClick={() => setActiveTab('basket')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg font-bold text-xs transition cursor-pointer border-b-2 flex-shrink-0 ${
                activeTab === 'basket'
                  ? 'border-purple-400 text-white bg-white/[0.05]'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Briefcase className="w-4 h-4 text-purple-400" />
              <span>🧺 Smart Multi-Broker Basket Studio</span>
              <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 text-[10px] uppercase font-bold">
                Deploy Capital
              </span>
            </button>

            <button
              onClick={() => setActiveTab('calculator')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg font-bold text-xs transition cursor-pointer border-b-2 flex-shrink-0 ${
                activeTab === 'calculator'
                  ? 'border-amber-400 text-white bg-white/[0.05]'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Calculator className="w-4 h-4 text-amber-400" />
              <span>🧮 Live Sizing &amp; Nifty Stress Test</span>
            </button>

            <button
              onClick={() => setActiveTab('tracker')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg font-bold text-xs transition cursor-pointer border-b-2 flex-shrink-0 ${
                activeTab === 'tracker'
                  ? 'border-cyan-400 text-white bg-white/[0.05]'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <FileText className="w-4 h-4 text-cyan-400" />
              <span>📋 Paper Trading Ledger</span>
              {paperTrades.length > 0 && (
                <span className="px-1.5 py-0.2 rounded-full bg-cyan-500/20 text-cyan-300 font-mono text-[10px] font-bold">
                  {paperTrades.length}
                </span>
              )}
            </button>
          </div>

          {/* ── Summary KPI Dashboard ────────────────────────────────────────── */}
          {data && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-sm">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                  <span>Under ₹200 Scanned</span>
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                </div>
                <div className="text-xl font-bold text-white font-mono">{data.under_200_count}</div>
                <div className="text-[11px] text-slate-500 mt-0.5">Liquid NSE equities &amp; ETFs</div>
              </div>

              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-sm">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                  <span>High Conviction (≥80)</span>
                  <Flame className="w-3.5 h-3.5 text-emerald-400" />
                </div>
                <div className="text-xl font-bold text-emerald-400 font-mono">{data.high_conviction_count}</div>
                <div className="text-[11px] text-slate-500 mt-0.5">Multi-indicator confluence</div>
              </div>

              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-sm">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                  <span>Avg Risk : Reward</span>
                  <Award className="w-3.5 h-3.5 text-amber-400" />
                </div>
                <div className="text-xl font-bold text-amber-400 font-mono">1 : {data.avg_risk_reward}</div>
                <div className="text-[11px] text-slate-500 mt-0.5">Asymmetric upside skew</div>
              </div>

              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-sm">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                  <span>Top Conviction Pick</span>
                  <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                </div>
                {data.top_pick ? (
                  <div>
                    <div className="text-sm font-bold text-purple-300 font-mono">{data.top_pick.symbol}</div>
                    <div className="text-[11px] text-slate-400">
                      ₹{data.top_pick.price.toFixed(2)} · {data.top_pick.conviction}/100 score
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-slate-500">Scanning...</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Main Content Area ─────────────────────────────────────────────── */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 1: SMART DIVERSIFIED BASKET STUDIO
        ═════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'basket' && (
          <div className="space-y-6 animate-fade-in">
            {/* Basket Control Panel */}
            <div className="p-5 rounded-2xl bg-gradient-to-r from-purple-950/30 via-slate-900/60 to-purple-950/20 border border-purple-500/30 shadow-2xl backdrop-blur-md">
              <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Briefcase className="w-5 h-5 text-purple-400" />
                  <h3 className="text-base font-extrabold text-white">Smart Under-₹200 Diversified Basket Studio</h3>
                </div>
                <span className="text-xs text-slate-400">
                  Select your capital and theme — StockIQ Pro computes whole share allocations and risk budgets
                </span>
              </div>

              {/* Capital & Theme Selectors */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* 1. Capital selection */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
                    <Coins className="w-3.5 h-3.5 text-amber-400" />
                    Capital to Deploy:
                  </label>
                  <div className="flex items-center gap-1.5 flex-wrap mb-2">
                    {[25000, 50000, 100000, 200000].map((amt) => (
                      <button
                        key={amt}
                        onClick={() => setBasketCapital(amt)}
                        className={`px-2.5 py-1 rounded-lg text-xs font-bold transition cursor-pointer border ${
                          basketCapital === amt
                            ? 'bg-amber-500/20 border-amber-500/60 text-amber-300'
                            : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:text-white'
                        }`}
                      >
                        ₹{(amt / 1000).toFixed(0)}k
                      </button>
                    ))}
                  </div>
                  <div className="relative">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs font-mono text-slate-400">₹</span>
                    <input
                      type="number"
                      value={basketCapital}
                      onChange={(e) => setBasketCapital(Math.max(5000, Number(e.target.value)))}
                      className="w-full pl-7 pr-3 py-1.5 rounded-lg bg-black/60 border border-white/[0.12] text-xs font-mono text-white focus:outline-none focus:border-amber-500"
                      min={5000}
                      step={5000}
                    />
                  </div>
                </div>

                {/* 2. Theme selection */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-purple-400" />
                    Basket Strategy Theme:
                  </label>
                  <div className="relative">
                    <select
                      value={basketTheme}
                      onChange={(e) => setBasketTheme(e.target.value)}
                      className="w-full px-3 py-2 pr-8 rounded-lg bg-black/60 border border-white/[0.12] text-xs text-white focus:outline-none focus:border-purple-500 cursor-pointer appearance-none"
                    >
                      {BASKET_THEMES.map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.icon} {t.label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="w-3.5 h-3.5 text-slate-500 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1.5">
                    {BASKET_THEMES.find((t) => t.id === basketTheme)?.desc}
                  </p>
                </div>

                {/* 3. Scrip count & Trigger */}
                <div>
                  <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-cyan-400" />
                    Number of Diversified Scrips:
                  </label>
                  <div className="flex items-center gap-2 mb-2">
                    {[3, 4, 5, 6].map((num) => (
                      <button
                        key={num}
                        onClick={() => setBasketScripCount(num)}
                        className={`flex-1 py-1 rounded-lg text-xs font-bold transition cursor-pointer border ${
                          basketScripCount === num
                            ? 'bg-cyan-500/20 border-cyan-500/60 text-cyan-300'
                            : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:text-white'
                        }`}
                      >
                        {num} Scrips
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={fetchSmartBasket}
                    disabled={basketLoading}
                    className="w-full py-2 rounded-lg bg-purple-500 hover:bg-purple-600 disabled:opacity-50 text-white font-bold text-xs transition cursor-pointer flex items-center justify-center gap-2 shadow-lg shadow-purple-950/30"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${basketLoading ? 'animate-spin' : ''}`} />
                    <span>{basketLoading ? 'Re-balancing Basket...' : 'Generate Optimized Basket'}</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Basket Results Table & Stats */}
            {basketData && (
              <div className="space-y-4">
                {/* Aggregate Basket Stats Strip */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08]">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Capital Deployed</div>
                    <div className="text-base font-bold text-white font-mono mt-0.5">
                      ₹{basketData.capital_deployed.toLocaleString('en-IN')}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">
                      Cash buffer: ₹{basketData.cash_buffer.toLocaleString('en-IN')}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08]">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 1 Portfolio Gain</div>
                    <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
                      +₹{basketData.projected_profit_t1.toLocaleString('en-IN')}{' '}
                      <span className="text-xs font-semibold">(+{basketData.projected_return_t1_pct}%)</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">2x ATR expansion target</div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08]">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 2 Portfolio Gain</div>
                    <div className="text-base font-bold text-cyan-400 font-mono mt-0.5">
                      +₹{basketData.projected_profit_t2.toLocaleString('en-IN')}{' '}
                      <span className="text-xs font-semibold">(+{basketData.projected_return_t2_pct}%)</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">Full multi-week runner</div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08]">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Combined Max Risk</div>
                    <div className="text-base font-bold text-red-400 font-mono mt-0.5">
                      -₹{basketData.max_portfolio_loss.toLocaleString('en-IN')}{' '}
                      <span className="text-xs font-semibold">(-{basketData.max_portfolio_risk_pct}%)</span>
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">If all stop-losses hit</div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] col-span-2 sm:col-span-1">
                    <div className="text-[10px] text-slate-400 uppercase font-semibold">Basket Risk : Reward</div>
                    <div className="text-base font-bold text-amber-400 font-mono mt-0.5">
                      1 : {basketData.portfolio_risk_reward}
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">{basketData.allocations.length} distinct sectors</div>
                  </div>
                </div>

                {/* Sector Allocation Distribution Strip */}
                <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08] space-y-2.5">
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                    <span className="flex items-center gap-1.5">
                      <Layers className="w-4 h-4 text-cyan-400" />
                      <span>Portfolio Sector Diversification Weights</span>
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">100% Capital Allocated</span>
                  </div>
                  {/* Segmented bar */}
                  <div className="h-3 w-full rounded-full bg-slate-900/90 flex overflow-hidden border border-white/[0.08] p-0.5 shadow-inner">
                    {basketData.allocations.map((a, i) => {
                      const colors = [
                        'bg-emerald-500',
                        'bg-cyan-500',
                        'bg-purple-500',
                        'bg-amber-500',
                        'bg-blue-500',
                        'bg-pink-500',
                        'bg-teal-500',
                        'bg-indigo-500',
                      ];
                      const color = colors[i % colors.length];
                      return (
                        <div
                          key={a.symbol}
                          className={`${color} h-full first:rounded-l-full last:rounded-r-full transition-all duration-300 hover:brightness-125 cursor-pointer`}
                          style={{ width: `${a.allocation_pct}%` }}
                          title={`${a.display_symbol} (${a.sector}): ${a.allocation_pct}% (₹${a.outlay.toLocaleString('en-IN')})`}
                        />
                      );
                    })}
                  </div>
                  {/* Legend */}
                  <div className="flex items-center gap-3 overflow-x-auto text-[11px] font-mono scrollbar-none pt-1">
                    {basketData.allocations.map((a, i) => {
                      const dotColors = [
                        'bg-emerald-400',
                        'bg-cyan-400',
                        'bg-purple-400',
                        'bg-amber-400',
                        'bg-blue-400',
                        'bg-pink-400',
                        'bg-teal-400',
                        'bg-indigo-400',
                      ];
                      const dot = dotColors[i % dotColors.length];
                      return (
                        <div key={a.symbol} className="flex items-center gap-1.5 flex-shrink-0">
                          <span className={`w-2 h-2 rounded-full ${dot}`} />
                          <span className="text-slate-200 font-bold">{a.display_symbol}</span>
                          <span className="text-slate-400 text-[10px]">({a.allocation_pct}%)</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Basket Table */}
                <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-black/40">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-white/[0.08] bg-white/[0.02] text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
                        <th className="p-3.5">Scrip &amp; Sector</th>
                        <th className="p-3.5">Price</th>
                        <th className="p-3.5">Shares to Buy</th>
                        <th className="p-3.5">Outlay &amp; Weight</th>
                        <th className="p-3.5">Target 1 &amp; Profit</th>
                        <th className="p-3.5">Stop Loss &amp; Risk</th>
                        <th className="p-3.5">R : R</th>
                        <th className="p-3.5 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.04]">
                      {basketData.allocations.map((item) => (
                        <tr key={item.symbol} className="hover:bg-white/[0.02] transition">
                          <td className="p-3.5">
                            <div className="flex items-center gap-2">
                              <span className="font-extrabold text-white font-mono text-sm">{item.display_symbol}</span>
                              <span className="text-[10px] px-1.5 py-0.2 rounded bg-white/[0.06] text-slate-300">
                                {item.sector}
                              </span>
                            </div>
                            <div className="text-[11px] text-slate-400 mt-0.5">{item.name}</div>
                          </td>
                          <td className="p-3.5 font-mono font-bold text-white">₹{item.price.toFixed(2)}</td>
                          <td className="p-3.5">
                            <span className="px-2.5 py-1 rounded-md bg-purple-500/15 border border-purple-500/30 text-purple-300 font-mono font-extrabold">
                              {item.shares} shares
                            </span>
                          </td>
                          <td className="p-3.5 font-mono">
                            <div className="text-slate-200 font-semibold">₹{item.outlay.toLocaleString('en-IN')}</div>
                            <div className="text-[10px] text-slate-500">{item.allocation_pct}% of basket</div>
                          </td>
                          <td className="p-3.5 font-mono">
                            <div className="text-emerald-400 font-semibold">₹{item.target_1.toFixed(2)}</div>
                            <div className="text-[10px] text-emerald-500">
                              +₹{item.target_1_profit.toLocaleString('en-IN')}
                            </div>
                          </td>
                          <td className="p-3.5 font-mono">
                            <div className="text-red-400 font-semibold">₹{item.stop_loss.toFixed(2)}</div>
                            <div className="text-[10px] text-red-500">-₹{item.max_stop_loss.toLocaleString('en-IN')}</div>
                          </td>
                          <td className="p-3.5 font-mono text-amber-400 font-bold">1 : {item.risk_reward_ratio}</td>
                          <td className="p-3.5 text-right">
                            <Link
                              href={`/?ticker=${encodeURIComponent(item.symbol)}`}
                              className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-emerald-400 transition"
                            >
                              <span>Analyze</span>
                              <ExternalLink className="w-3 h-3" />
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* ── Multi-Broker Order Export Studio & Ledger Integration ───── */}
                <div className="p-5 rounded-2xl bg-gradient-to-r from-white/[0.04] via-black/40 to-white/[0.02] border border-white/[0.1] shadow-2xl space-y-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-extrabold text-white flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span>Multi-Broker Execution Studio &amp; Terminal Export</span>
                      </div>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Import all {basketData.allocations.length} positions instantly into your broker terminal without manual typing.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleAddBasketToPaperTrades}
                        className="px-3.5 py-2 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/40 text-cyan-300 font-bold text-xs transition cursor-pointer flex items-center gap-1.5 shadow-md"
                        title="Add all basket scrips to Paper Trading Ledger"
                      >
                        <FileText className="w-3.5 h-3.5" />
                        <span>Track All in Paper Ledger</span>
                      </button>
                    </div>
                  </div>

                  {/* Broker Format Selection Tabs */}
                  <div className="flex items-center gap-2 overflow-x-auto pb-1 border-b border-white/[0.06] scrollbar-none">
                    {[
                      { id: 'zerodha', label: 'Zerodha Kite Basket', icon: '🪁', ext: 'CSV' },
                      { id: 'angel_one', label: 'Angel One Bulk', icon: '👼', ext: 'CSV' },
                      { id: 'groww', label: 'Groww Orders', icon: '🌱', ext: 'CSV' },
                      { id: 'dhan', label: 'Dhan Webhook API', icon: '⚡', ext: 'JSON' },
                      { id: 'telegram', label: 'WhatsApp / Journal', icon: '📱', ext: 'TXT' },
                    ].map((fmt) => (
                      <button
                        key={fmt.id}
                        onClick={() => setActiveBrokerFormat(fmt.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer flex items-center gap-1.5 border flex-shrink-0 ${
                          activeBrokerFormat === fmt.id
                            ? 'bg-purple-500/20 border-purple-500/60 text-purple-200'
                            : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:text-white'
                        }`}
                      >
                        <span>{fmt.icon}</span>
                        <span>{fmt.label}</span>
                        <span className="text-[10px] px-1 py-0.2 rounded bg-black/40 text-slate-400 uppercase font-mono">
                          {fmt.ext}
                        </span>
                      </button>
                    ))}
                  </div>

                  {/* Format Payload Preview Box */}
                  <div className="relative">
                    <pre className="p-3.5 rounded-xl bg-black/80 border border-white/[0.08] text-[11px] font-mono text-slate-300 overflow-x-auto max-h-44 leading-relaxed scrollbar-thin">
                      {getBrokerExportContent()}
                    </pre>

                    <div className="absolute top-2 right-2 flex items-center gap-1.5">
                      <button
                        onClick={handleCopySelectedBrokerFormat}
                        className="px-2.5 py-1 rounded-md bg-white/[0.1] hover:bg-white/[0.2] text-white text-[11px] font-medium transition cursor-pointer flex items-center gap-1 backdrop-blur-sm"
                      >
                        {copiedBasket ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedBasket ? 'Copied!' : 'Copy'}</span>
                      </button>
                      <button
                        onClick={handleDownloadSelectedBrokerFormat}
                        className="px-2.5 py-1 rounded-md bg-emerald-500 hover:bg-emerald-600 text-black text-[11px] font-bold transition cursor-pointer flex items-center gap-1 shadow-md"
                      >
                        <Download className="w-3 h-3" />
                        <span>Download</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 3: LIVE POSITION SIZING & NIFTY STRESS TEST
        ═════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'calculator' && (
          <div className="space-y-6 animate-fade-in">
            {/* Header banner */}
            <div className="p-5 rounded-2xl bg-gradient-to-r from-amber-950/30 via-slate-900/60 to-amber-950/20 border border-amber-500/30 shadow-2xl backdrop-blur-md">
              <div className="flex items-center gap-2.5 mb-1.5">
                <Calculator className="w-5 h-5 text-amber-400" />
                <h3 className="text-base font-extrabold text-white">Live Retail Position Sizing &amp; Market Stress Simulator</h3>
              </div>
              <p className="text-xs text-slate-400 max-w-3xl">
                Never trade arbitrary quantities. Calculate whole-share order sizing mathematically matched to your hard stop-loss.
                Simulate your portfolio&apos;s sensitivity to Nifty 50 market drops before risking real capital.
              </p>
            </div>

            {/* Main Interactive Studio Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Sizing Controls (5 cols) */}
              <div className="lg:col-span-5 space-y-4">
                <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-4">
                  {/* 1. Scrip Selector */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
                      <Target className="w-3.5 h-3.5 text-cyan-400" />
                      Select Scrip to Size:
                    </label>
                    <div className="relative">
                      <select
                        value={calcTicker}
                        onChange={(e) => setCalcTicker(e.target.value)}
                        className="w-full px-3 py-2 pr-8 rounded-lg bg-black/60 border border-white/[0.12] text-xs font-mono text-white focus:outline-none focus:border-amber-500 cursor-pointer appearance-none"
                      >
                        {data?.suggestions?.map((s) => (
                          <option key={s.symbol} value={s.symbol}>
                            {s.display_symbol} · ₹{s.technicals.current_price.toFixed(2)} ({s.setup_label})
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="w-3.5 h-3.5 text-slate-500 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    </div>
                  </div>

                  {/* 2. Trading Capital Selection */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center gap-1.5">
                      <Coins className="w-3.5 h-3.5 text-amber-400" />
                      Total Trading Capital:
                    </label>
                    <div className="flex items-center gap-1.5 flex-wrap mb-2">
                      {[25000, 50000, 100000, 200000, 500000].map((amt) => (
                        <button
                          key={amt}
                          onClick={() => setCalcCapital(amt)}
                          className={`px-2.5 py-1 rounded-lg text-xs font-bold transition cursor-pointer border ${
                            calcCapital === amt
                              ? 'bg-amber-500/20 border-amber-500/60 text-amber-300'
                              : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:text-white'
                          }`}
                        >
                          ₹{(amt / 1000).toFixed(0)}k
                        </button>
                      ))}
                    </div>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-xs font-mono text-slate-400">₹</span>
                      <input
                        type="number"
                        value={calcCapital}
                        onChange={(e) => setCalcCapital(Math.max(5000, Number(e.target.value)))}
                        className="w-full pl-7 pr-3 py-1.5 rounded-lg bg-black/60 border border-white/[0.12] text-xs font-mono text-white focus:outline-none focus:border-amber-500"
                        min={5000}
                        step={5000}
                      />
                    </div>
                  </div>

                  {/* 3. Risk Tolerance Per Trade */}
                  <div>
                    <label className="text-xs font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
                      <span className="flex items-center gap-1.5">
                        <Shield className="w-3.5 h-3.5 text-red-400" />
                        Max Risk Per Trade (%):
                      </span>
                      <span className="font-mono text-amber-300 font-bold text-xs">{calcRiskPct}%</span>
                    </label>
                    <div className="flex items-center gap-1.5 mb-2">
                      {[0.5, 1.0, 1.5, 2.0].map((pct) => (
                        <button
                          key={pct}
                          onClick={() => setCalcRiskPct(pct)}
                          className={`flex-1 py-1 rounded-lg text-xs font-bold transition cursor-pointer border ${
                            calcRiskPct === pct
                              ? 'bg-red-500/20 border-red-500/60 text-red-300'
                              : 'bg-white/[0.04] border-white/[0.08] text-slate-400 hover:text-white'
                          }`}
                        >
                          {pct}%
                        </button>
                      ))}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Allowable Risk Budget:{' '}
                      <strong className="text-white font-mono">
                        ₹{(calcCapital * (calcRiskPct / 100)).toFixed(0)}
                      </strong>{' '}
                      (Golden rule: never exceed 1.5% on small/mid-caps)
                    </div>
                  </div>
                </div>

                {/* Scrip Quick Stats Card */}
                {activeCalcScrip && (
                  <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-white font-mono">{activeCalcScrip.display_symbol}</span>
                      <span className="text-xs px-2 py-0.5 rounded bg-purple-500/15 text-purple-300 font-medium">
                        {activeCalcScrip.setup_label}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/[0.06] text-xs font-mono">
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Current</div>
                        <div className="font-bold text-white">₹{activeCalcScrip.technicals.current_price.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Target 1</div>
                        <div className="font-bold text-emerald-400">₹{activeCalcScrip.trade_plan.target_1.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Hard SL</div>
                        <div className="font-bold text-red-400">₹{activeCalcScrip.trade_plan.stop_loss.toFixed(2)}</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Execution Blueprint & Stress Test (7 cols) */}
              <div className="lg:col-span-7 space-y-4">
                {calcResults && (
                  <>
                    {/* Execution Output Blueprint */}
                    <div className="p-5 rounded-2xl bg-white/[0.03] border border-white/[0.08] space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-extrabold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          Computed Order Blueprint
                        </span>
                        <span className="text-xs font-mono text-amber-400 font-bold">
                          Risk-Reward 1 : {calcResults.rr}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div className="p-3 rounded-xl bg-black/50 border border-purple-500/30">
                          <div className="text-[10px] text-purple-300 uppercase font-semibold">Buy Quantity</div>
                          <div className="text-xl font-extrabold text-white font-mono mt-0.5">
                            {calcResults.shares} shares
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">Whole equity shares</div>
                        </div>

                        <div className="p-3 rounded-xl bg-black/50 border border-white/[0.08]">
                          <div className="text-[10px] text-slate-400 uppercase font-semibold">Capital Outlay</div>
                          <div className="text-xl font-extrabold text-white font-mono mt-0.5">
                            ₹{calcResults.outlay.toLocaleString('en-IN')}
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">{calcResults.capitalPct}% of capital</div>
                        </div>

                        <div className="p-3 rounded-xl bg-black/50 border border-red-500/30">
                          <div className="text-[10px] text-red-400 uppercase font-semibold">Max Downside Loss</div>
                          <div className="text-xl font-extrabold text-red-400 font-mono mt-0.5">
                            -₹{calcResults.maxLoss.toLocaleString('en-IN')}
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">At hard SL breach</div>
                        </div>

                        <div className="p-3 rounded-xl bg-black/50 border border-emerald-500/30">
                          <div className="text-[10px] text-emerald-400 uppercase font-semibold">Target 1 Profit</div>
                          <div className="text-xl font-extrabold text-emerald-400 font-mono mt-0.5">
                            +₹{calcResults.t1Gain.toLocaleString('en-IN')}
                          </div>
                          <div className="text-[10px] text-emerald-500 mt-0.5">+{calcResults.t1GainPct}% portfolio</div>
                        </div>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-white/[0.06] flex-wrap gap-2">
                        <div className="text-xs text-slate-400">
                          Target 2 Max Gain:{' '}
                          <strong className="text-cyan-400 font-mono">
                            +₹{calcResults.t2Gain.toLocaleString('en-IN')} (+{calcResults.t2GainPct}% portfolio)
                          </strong>
                        </div>

                        <button
                          onClick={() => activeCalcScrip && addPaperTrade(activeCalcScrip, calcResults.shares)}
                          className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black font-bold text-xs transition cursor-pointer flex items-center gap-1.5 shadow-lg shadow-emerald-950/30"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          <span>Track in Paper Trading Ledger</span>
                        </button>
                      </div>
                    </div>

                    {/* Nifty 50 Stress Simulator */}
                    <div className="p-5 rounded-2xl bg-gradient-to-r from-cyan-950/20 via-black/40 to-cyan-950/10 border border-cyan-500/30 space-y-4">
                      <div className="flex items-center justify-between flex-wrap gap-2">
                        <div>
                          <div className="text-sm font-extrabold text-white flex items-center gap-2">
                            <Activity className="w-4 h-4 text-cyan-400" />
                            <span>Nifty 50 Shock Stress Simulator</span>
                          </div>
                          <p className="text-xs text-slate-400 mt-0.5">
                            Sub-₹200 scrips have elevated market sensitivity ({calcResults.beta}x Nifty Beta). Test what happens in a sudden index move.
                          </p>
                        </div>
                        <span className="px-2.5 py-1 rounded-md bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-mono text-xs font-bold">
                          Stock Beta: {calcResults.beta}x
                        </span>
                      </div>

                      {/* Scenario Selector Pills */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {[
                          { val: -2.0, label: '-2.0% Flash Crash', color: 'red' },
                          { val: -1.0, label: '-1.0% Market Dip', color: 'amber' },
                          { val: 1.0, label: '+1.0% Bull Rally', color: 'emerald' },
                          { val: 2.0, label: '+2.0% Index Breakout', color: 'cyan' },
                        ].map((sc) => (
                          <button
                            key={sc.val}
                            onClick={() => setCalcNiftyScenario(sc.val)}
                            className={`p-2.5 rounded-xl text-xs font-bold transition cursor-pointer border text-center ${
                              calcNiftyScenario === sc.val
                                ? 'bg-white/[0.12] border-white/40 text-white shadow-md'
                                : 'bg-black/40 border-white/[0.08] text-slate-400 hover:text-white'
                            }`}
                          >
                            <div className="font-mono text-sm">{sc.val > 0 ? `+${sc.val}%` : `${sc.val}%`}</div>
                            <div className="text-[10px] text-slate-400 mt-0.5">{sc.label}</div>
                          </button>
                        ))}
                      </div>

                      {/* Stress Test Simulation Card */}
                      <div className="p-4 rounded-xl bg-black/60 border border-white/[0.08] grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div>
                          <div className="text-[10px] text-slate-400 uppercase">Projected Stock Move</div>
                          <div
                            className={`text-base font-extrabold font-mono mt-0.5 ${
                              calcResults.scenarioStockPct >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}
                          >
                            {calcResults.scenarioStockPct > 0 ? '+' : ''}
                            {calcResults.scenarioStockPct}%
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Simulated Price: ₹{calcResults.simulatedPrice.toFixed(2)}
                          </div>
                        </div>

                        <div>
                          <div className="text-[10px] text-slate-400 uppercase">P&amp;L Shock Impact</div>
                          <div
                            className={`text-base font-extrabold font-mono mt-0.5 ${
                              calcResults.rupeeImpact >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }`}
                          >
                            {calcResults.rupeeImpact > 0 ? '+' : ''}₹{calcResults.rupeeImpact.toLocaleString('en-IN')}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">On current position outlay</div>
                        </div>

                        <div>
                          <div className="text-[10px] text-slate-400 uppercase">Stop Loss Safety</div>
                          <div
                            className={`text-xs font-bold mt-1 ${
                              calcResults.wouldHitSL ? 'text-red-400' : 'text-emerald-400'
                            }`}
                          >
                            {calcResults.wouldHitSL
                              ? '⚠️ Breaches Hard SL! Reduce size'
                              : '🛡️ Within Safe Stop-Loss Buffer'}
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            Hard SL is ₹{calcResults.sl.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 4: PAPER TRADING LEDGER & PERFORMANCE TRACKER
        ═════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'tracker' && (
          <div className="space-y-6 animate-fade-in">
            {/* Header & Stats Banner */}
            <div className="p-5 rounded-2xl bg-gradient-to-r from-cyan-950/30 via-slate-900/60 to-cyan-950/20 border border-cyan-500/30 shadow-2xl backdrop-blur-md">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <FileText className="w-5 h-5 text-cyan-400" />
                    <h3 className="text-base font-extrabold text-white">Paper Trading Ledger &amp; Setup Tracker</h3>
                  </div>
                  <p className="text-xs text-slate-400 max-w-2xl">
                    Test your trading discipline without real risk. Track suggestions, execute partial profit booking at Target 1,
                    trail stops to breakeven, and export journal records.
                  </p>
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    onClick={exportPaperTradesCSV}
                    disabled={paperTrades.length === 0}
                    className="px-3 py-1.5 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] disabled:opacity-40 text-white font-semibold text-xs transition cursor-pointer flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Export Journal CSV</span>
                  </button>
                  <button
                    onClick={clearAllPaperTrades}
                    disabled={paperTrades.length === 0}
                    className="px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 disabled:opacity-40 border border-red-500/30 text-red-300 font-semibold text-xs transition cursor-pointer flex items-center gap-1.5"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear Ledger</span>
                  </button>
                </div>
              </div>

              {/* Ledger Aggregate KPIs */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                <div className="p-3 rounded-xl bg-black/40 border border-white/[0.08]">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">Active Capital Deployed</div>
                  <div className="text-base font-extrabold text-white font-mono mt-0.5">
                    ₹{paperStats.totalCapital.toLocaleString('en-IN')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{paperStats.activeCount} open positions</div>
                </div>

                <div className="p-3 rounded-xl bg-black/40 border border-white/[0.08]">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">Realized P&amp;L</div>
                  <div
                    className={`text-base font-extrabold font-mono mt-0.5 ${
                      paperStats.totalRealized >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}
                  >
                    {paperStats.totalRealized >= 0 ? '+' : ''}₹{paperStats.totalRealized.toLocaleString('en-IN')}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">{paperStats.closedCount} completed trades</div>
                </div>

                <div className="p-3 rounded-xl bg-black/40 border border-white/[0.08]">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">Win Rate</div>
                  <div className="text-base font-extrabold text-amber-400 font-mono mt-0.5">
                    {paperStats.winRate}%
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Based on target hits vs SL</div>
                </div>

                <div className="p-3 rounded-xl bg-black/40 border border-white/[0.08]">
                  <div className="text-[10px] text-slate-400 uppercase font-semibold">Total Setups Logged</div>
                  <div className="text-base font-extrabold text-cyan-400 font-mono mt-0.5">
                    {paperTrades.length}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Persisted in browser storage</div>
                </div>
              </div>
            </div>

            {/* Empty State */}
            {paperTrades.length === 0 && (
              <div className="py-16 text-center border border-dashed border-white/[0.1] rounded-2xl p-8 space-y-3">
                <FileText className="w-10 h-10 text-slate-600 mx-auto" />
                <h3 className="text-base font-bold text-white">Your Paper Trading Ledger is Empty</h3>
                <p className="text-xs text-slate-400 max-w-md mx-auto">
                  Start tracking setups by clicking <strong>&quot;Track Setup&quot;</strong> on any suggestion card,
                  or allocate a portfolio from the <strong>Smart Basket Studio</strong>.
                </p>
                <button
                  onClick={() => setActiveTab('radar')}
                  className="mt-2 px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black font-bold text-xs cursor-pointer transition shadow-lg"
                >
                  Explore Scrip Signal Radar
                </button>
              </div>
            )}

            {/* Trades Table */}
            {paperTrades.length > 0 && (
              <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-black/40">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.08] bg-white/[0.02] text-slate-400 font-semibold uppercase text-[10px] tracking-wider">
                      <th className="p-3.5">Scrip &amp; Setup</th>
                      <th className="p-3.5">Entry &amp; Shares</th>
                      <th className="p-3.5">Outlay</th>
                      <th className="p-3.5">Target 1 &amp; SL</th>
                      <th className="p-3.5">Status</th>
                      <th className="p-3.5">Realized P&amp;L</th>
                      <th className="p-3.5 text-right">Actions &amp; Milestones</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.04]">
                    {paperTrades.map((t) => (
                      <tr key={t.id} className="hover:bg-white/[0.02] transition">
                        <td className="p-3.5">
                          <div className="flex items-center gap-2">
                            <span className="font-extrabold text-white font-mono text-sm">{t.display_symbol}</span>
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-white/[0.06] text-slate-300">
                              {t.sector}
                            </span>
                          </div>
                          <div className="text-[11px] text-purple-300 mt-0.5">{t.setup}</div>
                          <div className="text-[10px] text-slate-500 mt-0.5">Logged: {t.entry_date}</div>
                          {/* Visual Trade Lifecycle Stepper */}
                          <div className="flex items-center gap-1 mt-2 text-[9px] font-mono">
                            <span
                              className={`px-1.5 py-0.5 rounded border ${
                                t.status === 'ACTIVE' || t.status === 'T1_HIT' || t.status === 'T2_HIT'
                                  ? 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300 font-bold'
                                  : 'bg-white/[0.03] border-white/[0.06] text-slate-500'
                              }`}
                            >
                              Entry ₹{t.entry_price.toFixed(1)}
                            </span>
                            <span className="text-slate-600">→</span>
                            <span
                              className={`px-1.5 py-0.5 rounded border ${
                                t.status === 'T1_HIT' || t.status === 'T2_HIT'
                                  ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300 font-bold'
                                  : 'bg-white/[0.03] border-white/[0.06] text-slate-500'
                              }`}
                            >
                              T1 Hit (50%)
                            </span>
                            <span className="text-slate-600">→</span>
                            <span
                              className={`px-1.5 py-0.5 rounded border ${
                                t.status === 'T2_HIT'
                                  ? 'bg-purple-500/20 border-purple-500/40 text-purple-300 font-bold'
                                  : 'bg-white/[0.03] border-white/[0.06] text-slate-500'
                              }`}
                            >
                              T2 Runner
                            </span>
                          </div>
                        </td>

                        <td className="p-3.5 font-mono">
                          <div className="text-white font-bold">₹{t.entry_price.toFixed(2)}</div>
                          <div className="text-slate-400 text-[11px] font-semibold">{t.shares} shares</div>
                        </td>

                        <td className="p-3.5 font-mono">
                          <div className="text-slate-200 font-bold">₹{t.outlay.toLocaleString('en-IN')}</div>
                        </td>

                        <td className="p-3.5 font-mono">
                          <div className="text-emerald-400">T1: ₹{t.target_1.toFixed(2)}</div>
                          <div className="text-red-400 text-[11px]">SL: ₹{t.stop_loss.toFixed(2)}</div>
                        </td>

                        <td className="p-3.5">
                          <span
                            className={`text-[10px] px-2.5 py-1 rounded-full font-bold uppercase ${
                              t.status === 'ACTIVE'
                                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 animate-pulse'
                                : t.status === 'T1_HIT'
                                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                                : t.status === 'T2_HIT'
                                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40'
                                : t.status === 'STOPPED_OUT'
                                ? 'bg-red-500/20 text-red-300 border border-red-500/40'
                                : 'bg-slate-500/20 text-slate-300 border border-slate-500/40'
                            }`}
                          >
                            {t.status === 'T1_HIT' ? 'T1 Hit (50% Booked)' : t.status}
                          </span>
                        </td>

                        <td className="p-3.5 font-mono font-bold">
                          <span
                            className={
                              (t.realized_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }
                          >
                            {(t.realized_pnl || 0) >= 0 ? '+' : ''}₹{(t.realized_pnl || 0).toLocaleString('en-IN')}
                          </span>
                        </td>

                        <td className="p-3.5 text-right">
                          <div className="flex items-center justify-end gap-1.5 flex-wrap">
                            {t.status === 'ACTIVE' && (
                              <>
                                <button
                                  onClick={() =>
                                    updateTradeStatus(
                                      t.id,
                                      'T1_HIT',
                                      Math.round(t.shares * 0.5 * (t.target_1 - t.entry_price))
                                    )
                                  }
                                  className="px-2 py-1 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[10px] font-bold cursor-pointer"
                                  title="Book 50% profit at Target 1 and trail stop to breakeven"
                                >
                                  🎯 Book 50% at T1
                                </button>
                                <button
                                  onClick={() =>
                                    updateTradeStatus(
                                      t.id,
                                      'STOPPED_OUT',
                                      -Math.round(t.shares * (t.entry_price - t.stop_loss))
                                    )
                                  }
                                  className="px-2 py-1 rounded bg-red-500/20 hover:bg-red-500/30 text-red-300 text-[10px] font-bold cursor-pointer"
                                  title="Record stop-loss exit"
                                >
                                  🛑 Hit SL
                                </button>
                              </>
                            )}

                            {t.status === 'T1_HIT' && (
                              <>
                                <button
                                  onClick={() =>
                                    updateTradeStatus(
                                      t.id,
                                      'T2_HIT',
                                      Math.round(t.shares * 0.5 * (t.target_2 - t.entry_price))
                                    )
                                  }
                                  className="px-2 py-1 rounded bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 text-[10px] font-bold cursor-pointer"
                                  title="Target 2 reached: Exit remaining 50%"
                                >
                                  🚀 Book Rest at T2
                                </button>
                                <button
                                  onClick={() => updateTradeStatus(t.id, 'CLOSED', 0)}
                                  className="px-2 py-1 rounded bg-white/[0.08] hover:bg-white/[0.14] text-slate-300 text-[10px] font-semibold cursor-pointer"
                                  title="Exit remaining 50% at Breakeven"
                                >
                                  Breakeven Exit
                                </button>
                              </>
                            )}

                            <button
                              onClick={() => removePaperTrade(t.id)}
                              className="p-1 rounded text-slate-500 hover:text-red-400 transition cursor-pointer"
                              title="Delete from ledger"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ═════════════════════════════════════════════════════════════════════
            VIEW 2: SIGNAL RADAR & QUANTITATIVE DOSSIERS
        ═════════════════════════════════════════════════════════════════════ */}
        {activeTab === 'radar' && (
          <div className="space-y-6">
            {/* ── Custom Scrip Instant Evaluator Bar ─────────────────────────── */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-white/[0.04] via-white/[0.02] to-white/[0.04] border border-white/[0.1] backdrop-blur-sm shadow-xl">
              <form onSubmit={handleEvaluateCustom} className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 flex-shrink-0">
                  <Zap className="w-4 h-4 text-emerald-400" />
                  <span>Evaluate Any Custom Scrip:</span>
                </div>
                <div className="relative flex-1">
                  <input
                    type="text"
                    value={customTicker}
                    onChange={(e) => setCustomTicker(e.target.value)}
                    placeholder="Enter any NSE/BSE ticker, e.g. TATASTEEL, SUZLON, NHPC, IOC, IRFC, BEL..."
                    className="w-full pl-3 pr-8 py-2 rounded-lg bg-black/50 border border-white/[0.15] focus:border-emerald-500 focus:outline-none text-xs text-white placeholder-slate-500 font-mono"
                  />
                  {customTicker && (
                    <button
                      type="button"
                      onClick={() => setCustomTicker('')}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 cursor-pointer"
                    >
                      ×
                    </button>
                  )}
                </div>
                <button
                  type="submit"
                  disabled={customEvaluating || !customTicker.trim()}
                  className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 text-black font-bold text-xs transition cursor-pointer flex items-center justify-center gap-1.5 flex-shrink-0"
                >
                  {customEvaluating ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Computing Models...</span>
                    </>
                  ) : (
                    <>
                      <BarChart2 className="w-3.5 h-3.5" />
                      <span>Generate Reasoning</span>
                    </>
                  )}
                </button>
              </form>

              {customError && (
                <div className="mt-3 p-2.5 rounded-lg bg-red-500/10 border border-red-500/25 text-red-400 text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  <span>{customError}</span>
                </div>
              )}

              {customResult && (
                <div className="mt-4 pt-4 border-t border-white/[0.08]">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Instant Custom Scrip Dossier:
                    </span>
                    <button
                      onClick={() => setCustomResult(null)}
                      className="text-xs text-slate-500 hover:text-slate-300 cursor-pointer"
                    >
                      Dismiss
                    </button>
                  </div>
                  <SuggestionCard
                    suggestion={customResult}
                    isExpanded={expandedCards.has(customResult.symbol)}
                    onToggle={() => toggleCard(customResult.symbol)}
                    isWatchlisted={watchlist.includes(customResult.symbol)}
                    onToggleWatchlist={() => toggleWatchlist(customResult.symbol)}
                    isCalcOpen={activeCalcSymbol === customResult.symbol}
                    onToggleCalc={() =>
                      setActiveCalcSymbol(activeCalcSymbol === customResult.symbol ? null : customResult.symbol)
                    }
                    onAddPaperTrade={(s) => addPaperTrade(s)}
                    highlighted
                  />
                </div>
              )}
            </div>

            {/* ── Playbook & Setup Filters ───────────────────────────────────── */}
            <div className="space-y-3">
              {/* Theme Playbook Pills */}
              <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
                <span className="text-xs font-semibold text-slate-400 flex items-center gap-1 pl-1 flex-shrink-0">
                  <Flame className="w-3.5 h-3.5 text-amber-400" />
                  Playbook:
                </span>
                {THEME_PLAYBOOKS.map((p) => {
                  const active = selectedTheme === p.id;
                  return (
                    <button
                      key={p.id}
                      onClick={() => setSelectedTheme(p.id)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer flex-shrink-0 border flex items-center gap-1.5 ${
                        active
                          ? 'bg-amber-500/20 border-amber-500/60 text-amber-300 font-semibold shadow-sm'
                          : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:bg-white/[0.06] hover:text-slate-200'
                      }`}
                    >
                      <span>{p.icon}</span>
                      <span>{p.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Price Range Pills */}
              <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
                <span className="text-xs font-semibold text-slate-400 flex items-center gap-1 pl-1 flex-shrink-0">
                  <Target className="w-3.5 h-3.5 text-cyan-400" />
                  Price:
                </span>
                {PRICE_TIERS.map((tier) => {
                  const active = priceTier === tier.value;
                  return (
                    <button
                      key={tier.label}
                      onClick={() => setPriceTier(tier.value)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer flex-shrink-0 border ${
                        active
                          ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300 font-semibold shadow-sm'
                          : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:bg-white/[0.06] hover:text-slate-200'
                      }`}
                    >
                      {tier.label}
                    </button>
                  );
                })}
              </div>

              {/* Setup Types Pills */}
              <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none">
                <span className="text-xs font-semibold text-slate-400 flex items-center gap-1 pl-1 flex-shrink-0">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-purple-400" />
                  Setup:
                </span>
                {SETUP_FILTERS.map((s) => {
                  const active = setupFilter === s.value;
                  return (
                    <button
                      key={s.value}
                      onClick={() => setSetupFilter(s.value)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer flex-shrink-0 border ${
                        active
                          ? 'bg-purple-500/20 border-purple-500/60 text-purple-300 font-semibold shadow-sm'
                          : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:bg-white/[0.06] hover:text-slate-200'
                      }`}
                    >
                      {s.label}
                    </button>
                  );
                })}
              </div>

              {/* Secondary Controls: Sector, Sort, Expand, Search */}
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2 border-t border-white/[0.06]">
                <div className="flex items-center gap-2 flex-wrap">
                  {data?.available_sectors && (
                    <div className="relative">
                      <select
                        value={selectedSector}
                        onChange={(e) => setSelectedSector(e.target.value)}
                        className="px-3 py-1.5 pr-8 rounded-lg bg-black/60 border border-white/[0.12] text-xs text-slate-300 focus:outline-none focus:border-emerald-500 cursor-pointer appearance-none"
                      >
                        <option value="all">All Sectors ({data.total_scanned})</option>
                        {data.available_sectors.map((sec) => (
                          <option key={sec} value={sec}>
                            {sec}
                          </option>
                        ))}
                      </select>
                      <ChevronDown className="w-3 h-3 text-slate-500 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    </div>
                  )}

                  <div className="relative">
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value)}
                      className="px-3 py-1.5 pr-8 rounded-lg bg-black/60 border border-white/[0.12] text-xs text-slate-300 focus:outline-none focus:border-emerald-500 cursor-pointer appearance-none"
                    >
                      {SORT_OPTIONS.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          Sort: {opt.label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown className="w-3 h-3 text-slate-500 absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                  </div>

                  <div className="flex items-center gap-1 text-xs">
                    <button
                      onClick={expandAll}
                      className="px-2.5 py-1 text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] rounded cursor-pointer transition"
                    >
                      Expand All
                    </button>
                    <span className="text-slate-600">·</span>
                    <button
                      onClick={collapseAll}
                      className="px-2.5 py-1 text-slate-400 hover:text-slate-200 hover:bg-white/[0.05] rounded cursor-pointer transition"
                    >
                      Collapse All
                    </button>
                  </div>
                </div>

                <div className="relative min-w-[200px]">
                  <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Filter by scrip, name, sector..."
                    className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-black/50 border border-white/[0.12] text-xs text-slate-300 focus:outline-none focus:border-emerald-500"
                  />
                </div>
              </div>
            </div>

            {/* ── Status Messages ──────────────────────────────────────────── */}
            {loading && (
              <div className="py-24 text-center">
                <RefreshCw className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-3" />
                <p className="text-sm font-semibold text-white">Scanning Active Equities Under ₹200...</p>
                <p className="text-xs text-slate-500 mt-1">
                  Computing 20/50/200 DMAs, Wilder RSI-14, Relative Volume, and Operator Liquidity Guardrails...
                </p>
              </div>
            )}

            {error && (
              <div className="p-6 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-center max-w-xl mx-auto my-12">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
                <p className="text-sm font-semibold">{error}</p>
                <button
                  onClick={() => fetchSuggestions(true)}
                  className="mt-4 px-4 py-2 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-white text-xs font-semibold cursor-pointer"
                >
                  Retry Connection
                </button>
              </div>
            )}

            {/* ── Empty State ──────────────────────────────────────────────── */}
            {!loading && !error && displayedSuggestions.length === 0 && (
              <div className="py-16 text-center border border-dashed border-white/[0.1] rounded-2xl p-8">
                <Target className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                <h3 className="text-base font-bold text-white">No Scrips Match the Selected Filters</h3>
                <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto">
                  Try adjusting the price ceiling, selecting &quot;All Setups&quot;, or clearing the sector filter.
                </p>
                <button
                  onClick={() => {
                    setPriceTier(200);
                    setSetupFilter('all');
                    setSelectedTheme('all');
                    setSelectedSector('all');
                    setSearchQuery('');
                  }}
                  className="mt-4 px-4 py-2 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] text-xs font-semibold text-white cursor-pointer transition"
                >
                  Reset Filters to Default
                </button>
              </div>
            )}

            {/* ── Suggestions Grid ─────────────────────────────────────────── */}
            {!loading && !error && displayedSuggestions.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center justify-between text-xs text-slate-400 px-1">
                  <span>
                    Showing <strong className="text-white font-mono">{displayedSuggestions.length}</strong> qualified
                    recommendations {priceTier ? `under ₹${priceTier}` : ''}
                  </span>
                  <span>Click any card to inspect full quantitative reasoning dossier &amp; position sizing</span>
                </div>

                {displayedSuggestions.map((item) => (
                  <SuggestionCard
                    key={item.symbol}
                    suggestion={item}
                    isExpanded={expandedCards.has(item.symbol)}
                    onToggle={() => toggleCard(item.symbol)}
                    isWatchlisted={watchlist.includes(item.symbol)}
                    onToggleWatchlist={() => toggleWatchlist(item.symbol)}
                    isCalcOpen={activeCalcSymbol === item.symbol}
                    onToggleCalc={() => setActiveCalcSymbol(activeCalcSymbol === item.symbol ? null : item.symbol)}
                    onAddPaperTrade={(s) => addPaperTrade(s)}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Individual Suggestion Card with Liquidity Guardrails & Position Sizing
───────────────────────────────────────────────────────────────────────────── */
function SuggestionCard({
  suggestion,
  isExpanded,
  onToggle,
  isWatchlisted,
  onToggleWatchlist,
  isCalcOpen,
  onToggleCalc,
  onAddPaperTrade,
  highlighted = false,
}) {
  const { symbol, display_symbol, name, sector, setup_label, setup_icon, conviction_score, conviction_tier } = suggestion;
  const { trade_plan, technicals, reasoning, liquidity_guardrails, execution_playbook } = suggestion;

  const isGreen = suggestion.day_change_pct >= 0;

  // 52-Week Range position
  const low52 = technicals.low_52;
  const high52 = technicals.high_52;
  const currentPrice = technicals.current_price;
  const rangeSpan = Math.max(high52 - low52, 0.01);
  const positionPct = Math.min(100, Math.max(0, ((currentPrice - low52) / rangeSpan) * 100));

  // Interactive Position Sizing State (default ₹1,00,000 portfolio, 1% risk)
  const [userCapital, setUserCapital] = useState(100000);
  const [userRiskPct, setUserRiskPct] = useState(1.0);

  // Dossier Sub-Tabs & Broker Order State
  const [dossierTab, setDossierTab] = useState('reasoning'); // 'reasoning' | 'audit' | 'roadmap' | 'broker'
  const [copiedOrderFmt, setCopiedOrderFmt] = useState('');

  // Dynamic calculations
  const riskBudget = userCapital * (userRiskPct / 100.0);
  const riskPerShare = Math.max(trade_plan.risk_per_share || currentPrice - trade_plan.stop_loss, 0.05);
  const calcShares = Math.max(1, Math.floor(riskBudget / riskPerShare));
  const calcOutlay = calcShares * currentPrice;
  const calcT1Gain = calcShares * (trade_plan.target_1 - currentPrice);
  const calcT2Gain = calcShares * (trade_plan.target_2 - currentPrice);
  const calcMaxLoss = calcShares * riskPerShare;

  return (
    <div
      className={`rounded-2xl transition-all duration-200 border ${
        highlighted
          ? 'bg-[#0f1422] border-emerald-500/50 shadow-xl shadow-emerald-950/20 ring-1 ring-emerald-500/30'
          : isExpanded
          ? 'bg-[#0c101a] border-white/[0.18] shadow-2xl'
          : 'bg-[#0a0d15]/90 hover:bg-[#0d121c] border-white/[0.08] hover:border-white/[0.14]'
      }`}
    >
      {/* ── Main Top Card Summary ────────────────────────────────────────── */}
      <div className="p-4 sm:p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Scrip info & primary tags */}
          <div className="flex items-start gap-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onToggleWatchlist();
              }}
              className={`p-2 rounded-xl transition cursor-pointer flex-shrink-0 ${
                isWatchlisted
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                  : 'bg-white/[0.04] text-slate-500 hover:text-amber-400 hover:bg-white/[0.08] border border-white/[0.06]'
              }`}
              title={isWatchlisted ? 'Remove from Watchlist' : 'Add to Watchlist'}
            >
              <Star className={`w-4 h-4 ${isWatchlisted ? 'fill-amber-400' : ''}`} />
            </button>

            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <Link
                  href={`/?ticker=${encodeURIComponent(symbol)}`}
                  className="text-lg font-extrabold text-white hover:text-emerald-400 transition font-mono tracking-wide"
                  title="Open full interactive dashboard"
                >
                  {display_symbol}
                </Link>
                <span className="text-xs px-2 py-0.5 rounded-md bg-white/[0.06] border border-white/[0.1] text-slate-300 font-medium">
                  {sector}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-md bg-purple-500/15 border border-purple-500/30 text-purple-300 font-semibold flex items-center gap-1">
                  <span>{setup_icon}</span>
                  <span>{setup_label}</span>
                </span>
                {technicals.current_price <= 200 && (
                  <span className="text-[11px] px-1.5 py-0.2 rounded bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-mono">
                    Under ₹200
                  </span>
                )}
                {/* Circuit Headroom Badge */}
                {suggestion.circuit_metrics && (
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-medium border flex items-center gap-1 ${
                      suggestion.circuit_metrics.headroom_pct <= 5.0
                        ? 'bg-amber-500/15 border-amber-500/35 text-amber-300'
                        : 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                    }`}
                    title={`Upper Circuit Limit: ₹${suggestion.circuit_metrics.upper_circuit} (${suggestion.circuit_metrics.circuit_band_pct}% band). Status: ${suggestion.circuit_metrics.circuit_status}`}
                  >
                    <Zap className="w-3 h-3 text-amber-400" />
                    <span>Circuit: +{suggestion.circuit_metrics.headroom_pct}%</span>
                  </span>
                )}
                {/* Delivery Accumulation Badge */}
                {suggestion.delivery_metrics && (
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-medium border ${
                      suggestion.delivery_metrics.delivery_quality === 'Institutional Grade'
                        ? 'bg-cyan-500/10 border-cyan-500/25 text-cyan-300'
                        : 'bg-white/[0.06] border-white/[0.1] text-slate-300'
                    }`}
                    title={suggestion.delivery_metrics.delivery_verdict}
                  >
                    📦 {suggestion.delivery_metrics.estimated_delivery_pct}% Deliv ({suggestion.delivery_metrics.delivery_quality.split(' ')[0]})
                  </span>
                )}
                {/* Liquidity Guardrail Tag */}
                {liquidity_guardrails && (
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-md font-medium border ${
                      liquidity_guardrails.operator_trap_risk === 'MINIMAL'
                        ? 'bg-emerald-500/10 border-emerald-500/25 text-emerald-300'
                        : liquidity_guardrails.operator_trap_risk === 'LOW'
                        ? 'bg-cyan-500/10 border-cyan-500/25 text-cyan-300'
                        : 'bg-amber-500/10 border-amber-500/25 text-amber-300'
                    }`}
                  >
                    🛡️ Turnover ₹{liquidity_guardrails.avg_turnover_cr} Cr/day
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-0.5 max-w-xl truncate">{name}</p>
            </div>
          </div>

          {/* Current Price & Conviction */}
          <div className="flex items-center gap-6 self-start lg:self-center">
            <div className="text-left lg:text-right">
              <div className="text-xl font-black text-white font-mono tracking-tight">
                ₹{technicals.current_price.toFixed(2)}
              </div>
              <div
                className={`text-xs font-semibold flex items-center lg:justify-end gap-1 ${
                  isGreen ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {isGreen ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                <span>
                  {isGreen ? '+' : ''}
                  {suggestion.day_change_pct.toFixed(2)}%
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 pl-3 border-l border-white/[0.08]">
              <div
                className="relative flex items-center justify-center w-14 h-14 rounded-2xl bg-black/60 border shadow-inner backdrop-blur-md"
                style={{ borderColor: `${suggestion.conviction_color}40` }}
              >
                <div className="text-center">
                  <div
                    className="text-lg font-black font-mono leading-none tracking-tight"
                    style={{ color: suggestion.conviction_color }}
                  >
                    {conviction_score}
                  </div>
                  <div
                    className="text-[9px] font-extrabold uppercase tracking-tight mt-1"
                    style={{ color: suggestion.conviction_color }}
                  >
                    {conviction_tier.split(' ')[0]}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Key Execution Blueprint Matrix ─────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 mt-4 pt-3.5 border-t border-white/[0.06]">
          <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Suggested Entry</div>
            <div className="text-xs font-bold text-white font-mono mt-0.5">{trade_plan.entry_display}</div>
            <div className="text-[10px] text-slate-500 mt-0.5">Optimal accumulation band</div>
          </div>

          <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 1 (Conservative)</div>
            <div className="text-xs font-bold text-emerald-400 font-mono mt-0.5">
              ₹{trade_plan.target_1.toFixed(2)}{' '}
              <span className="text-[10px] font-semibold text-emerald-500">(+{trade_plan.target_1_pct}%)</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">2x ATR expansion</div>
          </div>

          <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 2 (Runner)</div>
            <div className="text-xs font-bold text-cyan-400 font-mono mt-0.5">
              ₹{trade_plan.target_2.toFixed(2)}{' '}
              <span className="text-[10px] font-semibold text-cyan-500">(+{trade_plan.target_2_pct}%)</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Multi-week target</div>
          </div>

          <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Stop Loss (Hard SL)</div>
            <div className="text-xs font-bold text-red-400 font-mono mt-0.5">
              ₹{trade_plan.stop_loss.toFixed(2)}{' '}
              <span className="text-[10px] font-semibold text-red-500">({trade_plan.stop_loss_pct}%)</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">Strict invalidation level</div>
          </div>

          <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06] col-span-2 sm:col-span-1">
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Risk : Reward</div>
            <div className="text-xs font-bold text-amber-400 font-mono mt-0.5">
              1 : {trade_plan.risk_reward_ratio}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">{suggestion.horizon}</div>
          </div>
        </div>

        {/* ── Technical Indicator Confluence Strip ─────────────────────────── */}
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2.5 text-xs pt-2.5 border-t border-white/[0.04]">
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Visual RSI-14 Meter */}
            <div className="flex items-center gap-1.5 bg-black/40 px-2.5 py-1 rounded-lg border border-white/[0.06]">
              <span className="text-[10px] text-slate-400 font-semibold">RSI(14):</span>
              <strong
                className={`font-mono text-xs ${
                  technicals.rsi >= 60
                    ? 'text-emerald-400'
                    : technicals.rsi <= 35
                    ? 'text-amber-400'
                    : 'text-slate-200'
                }`}
              >
                {technicals.rsi.toFixed(1)}
              </strong>
              <div className="w-10 h-1.5 bg-slate-800 rounded-full overflow-hidden relative">
                <div
                  className={`h-full rounded-full ${
                    technicals.rsi >= 60 ? 'bg-emerald-400' : technicals.rsi <= 35 ? 'bg-amber-400' : 'bg-cyan-400'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, technicals.rsi))}%` }}
                />
              </div>
            </div>

            {/* Visual RVOL Meter */}
            <div className="flex items-center gap-1.5 bg-black/40 px-2.5 py-1 rounded-lg border border-white/[0.06]">
              <Flame className={`w-3.5 h-3.5 ${technicals.rvol >= 1.25 ? 'text-amber-400 animate-pulse' : 'text-slate-500'}`} />
              <span className="text-[10px] text-slate-400 font-semibold">RVOL:</span>
              <strong className={`font-mono text-xs ${technicals.rvol >= 1.25 ? 'text-emerald-400' : 'text-slate-200'}`}>
                {technicals.rvol.toFixed(2)}x
              </strong>
            </div>

            <span className="text-slate-400 text-[11px] bg-black/30 px-2 py-1 rounded border border-white/[0.04]">
              20 DMA: <strong className="font-mono text-slate-200">₹{technicals.sma20.toFixed(2)}</strong> (
              <span className={technicals.pct_vs_sma20 >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                {technicals.pct_vs_sma20 >= 0 ? '+' : ''}{technicals.pct_vs_sma20.toFixed(1)}%
              </span>)
            </span>

            <span className="text-slate-400 text-[11px] bg-black/30 px-2 py-1 rounded border border-white/[0.04]">
              50 DMA: <strong className="font-mono text-slate-200">₹{technicals.sma50.toFixed(2)}</strong>
            </span>

            <span className="text-slate-400 text-[11px] bg-black/30 px-2 py-1 rounded border border-white/[0.04]">
              200 DMA: <strong className="font-mono text-slate-200">₹{technicals.sma200.toFixed(2)}</strong>
            </span>
          </div>

          {/* 52-Week Range Bar */}
          <div className="flex items-center gap-2 min-w-[180px] bg-black/40 px-2.5 py-1 rounded-lg border border-white/[0.06]">
            <span className="text-[10px] text-slate-500 font-mono">₹{low52.toFixed(1)}</span>
            <div className="flex-1 h-1.5 bg-slate-800 rounded-full relative overflow-hidden">
              <div
                className="absolute top-0 bottom-0 bg-gradient-to-r from-emerald-500 to-cyan-400 rounded-full"
                style={{ width: `${positionPct}%` }}
              />
            </div>
            <span className="text-[10px] text-slate-500 font-mono">₹{high52.toFixed(1)}</span>
            <span className="text-[9px] font-mono text-cyan-300 font-bold">{positionPct.toFixed(0)}%</span>
          </div>
        </div>

        {/* ── Primary Thesis Banner & Action Triggers ───────────────────────── */}
        <div className="mt-3.5 p-3 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-between gap-3 flex-wrap">
          <p className="text-xs text-slate-300 font-medium leading-relaxed flex-1 min-w-[280px]">
            <span className="text-emerald-400 font-bold mr-1.5">⚡ Quantitative Rationale:</span>
            {reasoning.primary_thesis}
          </p>

          <div className="flex items-center gap-2 flex-shrink-0 flex-wrap">
            {/* Quick Track Setup Button */}
            {onAddPaperTrade && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onAddPaperTrade(suggestion);
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/35 text-emerald-300 transition cursor-pointer"
                title="Track this setup in Paper Trading Ledger"
              >
                <FileText className="w-3.5 h-3.5 text-emerald-400" />
                <span>Track Setup</span>
              </button>
            )}

            {/* Position sizing toggle */}
            <button
              onClick={onToggleCalc}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border ${
                isCalcOpen
                  ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                  : 'bg-white/[0.05] border-white/[0.1] text-slate-300 hover:text-white hover:bg-white/[0.09]'
              }`}
            >
              <Calculator className="w-3.5 h-3.5 text-amber-400" />
              <span>{isCalcOpen ? 'Close Sizing' : 'Position Sizing'}</span>
            </button>

            {/* Read reasoning toggle */}
            <button
              onClick={onToggle}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] text-xs font-semibold text-white transition cursor-pointer"
            >
              <span>{isExpanded ? 'Hide Dossier' : 'Read Full Reasoning'}</span>
              {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>

      {/* ── INTERACTIVE POSITION SIZING STUDIO (Inside Card) ────────────────── */}
      {isCalcOpen && (
        <div className="border-t border-amber-500/25 bg-amber-950/10 p-4 sm:p-5 animate-fade-in">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-300 uppercase tracking-wider">
              <Calculator className="w-4 h-4 text-amber-400" />
              <span>Real-World Capital Allocation &amp; Risk Sizing ({display_symbol})</span>
            </div>
            <span className="text-[11px] text-slate-400">Never risk &gt; 1% to 1.5% of total capital on one scrip</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {/* Capital input */}
            <div>
              <label className="text-xs text-slate-400 block mb-1">Your Total Trading Capital:</label>
              <div className="flex items-center gap-1.5 mb-1.5">
                {[50000, 100000, 200000].map((amt) => (
                  <button
                    key={amt}
                    onClick={() => setUserCapital(amt)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono cursor-pointer border ${
                      userCapital === amt
                        ? 'bg-amber-500/30 border-amber-500/60 text-amber-200'
                        : 'bg-white/[0.04] border-white/[0.08] text-slate-400'
                    }`}
                  >
                    ₹{(amt / 1000).toFixed(0)}k
                  </button>
                ))}
              </div>
              <input
                type="number"
                value={userCapital}
                onChange={(e) => setUserCapital(Math.max(5000, Number(e.target.value)))}
                className="w-full px-3 py-1.5 rounded-lg bg-black/60 border border-white/[0.12] text-xs font-mono text-white focus:outline-none focus:border-amber-500"
                step={5000}
              />
            </div>

            {/* Risk tolerance */}
            <div>
              <label className="text-xs text-slate-400 block mb-1">Risk Budget Per Trade:</label>
              <div className="flex items-center gap-1.5 mb-1.5">
                {[0.5, 1.0, 1.5, 2.0].map((pct) => (
                  <button
                    key={pct}
                    onClick={() => setUserRiskPct(pct)}
                    className={`px-2 py-0.5 rounded text-[11px] font-mono cursor-pointer border ${
                      userRiskPct === pct
                        ? 'bg-amber-500/30 border-amber-500/60 text-amber-200'
                        : 'bg-white/[0.04] border-white/[0.08] text-slate-400'
                    }`}
                  >
                    {pct}%
                  </button>
                ))}
              </div>
              <div className="text-xs font-mono text-amber-300 font-bold mt-2">
                Max Allowable Loss: ₹{riskBudget.toFixed(2)} ({userRiskPct}%)
              </div>
            </div>

            {/* Calculated summary */}
            <div className="p-3 rounded-lg bg-black/50 border border-white/[0.08]">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Risk Per Share</div>
              <div className="text-xs font-mono font-bold text-white mt-0.5">
                ₹{riskPerShare.toFixed(2)}{' '}
                <span className="text-[10px] text-slate-500">
                  (Entry ₹{currentPrice.toFixed(2)} - SL ₹{trade_plan.stop_loss.toFixed(2)})
                </span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Risk-to-Reward Ratio: <strong className="text-amber-400 font-mono">1 : {trade_plan.risk_reward_ratio}</strong>
              </div>
            </div>
          </div>

          {/* Real-world results grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-white/[0.06]">
            <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Recommended Quantity</div>
              <div className="text-base font-extrabold text-white font-mono mt-0.5">{calcShares} shares</div>
              <div className="text-[10px] text-slate-500 mt-0.5">
                Outlay: ₹{calcOutlay.toLocaleString('en-IN')} ({((calcOutlay / userCapital) * 100).toFixed(1)}% of capital)
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Max Loss at Stop Loss</div>
              <div className="text-base font-extrabold text-red-400 font-mono mt-0.5">
                -₹{calcMaxLoss.toFixed(2)}
              </div>
              <div className="text-[10px] text-red-500 mt-0.5">
                Exactly {((calcMaxLoss / userCapital) * 100).toFixed(2)}% of your portfolio
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 1 Potential Profit</div>
              <div className="text-base font-extrabold text-emerald-400 font-mono mt-0.5">
                +₹{calcT1Gain.toFixed(2)}
              </div>
              <div className="text-[10px] text-emerald-500 mt-0.5">
                +{((calcT1Gain / userCapital) * 100).toFixed(2)}% portfolio boost
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06]">
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Target 2 Potential Profit</div>
              <div className="text-base font-extrabold text-cyan-400 font-mono mt-0.5">
                +₹{calcT2Gain.toFixed(2)}
              </div>
              <div className="text-[10px] text-cyan-500 mt-0.5">
                +{((calcT2Gain / userCapital) * 100).toFixed(2)}% portfolio boost
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── EXPANDABLE IN-DEPTH REASONING DOSSIER ──────────────────────────── */}
      {isExpanded && (
        <div className="border-t border-white/[0.1] bg-black/40 p-4 sm:p-6 rounded-b-2xl space-y-5 animate-fade-in">
          {/* Header */}
          <div className="flex items-center justify-between pb-2 border-b border-white/[0.06]">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                Full Quantitative Dossier &amp; Trade Management Protocol
              </h4>
            </div>
            <span className="text-[11px] text-slate-500 font-mono">Audited by StockIQ Pro Models</span>
          </div>

          {/* Dossier Navigation Sub-Tabs */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-black/60 border border-white/[0.08] overflow-x-auto scrollbar-none">
            {[
              { id: 'reasoning', label: '1. Pillars & Thesis', icon: Sparkles },
              { id: 'audit', label: '2. Risk & Circuit Audit', icon: ShieldCheck },
              { id: 'roadmap', label: '3. Trailing Roadmap', icon: Target },
              { id: 'broker', label: '4. Instant Broker Order', icon: Briefcase },
            ].map((tab) => {
              const TabIcon = tab.icon;
              const isActive = dossierTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setDossierTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer flex-shrink-0 ${
                    isActive
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-950/40'
                      : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
                  }`}
                >
                  <TabIcon className={`w-3.5 h-3.5 ${isActive ? 'text-emerald-400' : 'text-slate-400'}`} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* TAB 1: 5 QUANTITATIVE PILLARS & THESIS */}
          {dossierTab === 'reasoning' && (
            <div className="space-y-4 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                <div className="p-4 rounded-xl bg-gradient-to-br from-cyan-950/20 to-black/40 border border-cyan-500/20">
                  <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 mb-2">
                    <BarChart2 className="w-4 h-4" />
                    <span>1. Price Action & Moving Average Alignment</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{reasoning.price_action}</p>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-950/20 to-black/40 border border-emerald-500/20">
                  <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 mb-2">
                    <Activity className="w-4 h-4" />
                    <span>2. Volume Footprint & Institutional Absorption</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{reasoning.volume_footprint}</p>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-950/20 to-black/40 border border-purple-500/20">
                  <div className="flex items-center gap-2 text-xs font-bold text-purple-400 mb-2">
                    <Zap className="w-4 h-4" />
                    <span>3. Momentum & Indicator Confluence</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{reasoning.momentum_indicators}</p>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-br from-amber-950/20 to-black/40 border border-amber-500/20">
                  <div className="flex items-center gap-2 text-xs font-bold text-amber-400 mb-2">
                    <Target className="w-4 h-4" />
                    <span>4. Mathematical Target & Stop-Loss Justification</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">{reasoning.levels_rationale}</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-gradient-to-br from-blue-950/20 to-black/40 border border-blue-500/20">
                <div className="flex items-center gap-2 text-xs font-bold text-blue-400 mb-2">
                  <Layers className="w-4 h-4" />
                  <span>5. Fundamental Drivers & Sector Tailwinds</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{reasoning.sector_catalyst}</p>
                {suggestion.description && (
                  <p className="text-[11px] text-slate-400 mt-2 italic border-t border-white/[0.06] pt-2">
                    {suggestion.description}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: RISK, CIRCUIT & DELIVERY AUDIT */}
          {dossierTab === 'audit' && (
            <div className="space-y-4 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {/* Circuit Ceiling & Headroom */}
                {suggestion.circuit_metrics && (
                  <div className="p-4 rounded-xl bg-black/50 border border-amber-500/25">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-amber-400">
                        <Zap className="w-4 h-4" />
                        <span>Exchange Circuit Limit & Headroom</span>
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono font-bold">
                        {suggestion.circuit_metrics.circuit_band_pct}% Band
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Upper Circuit Ceiling:</span>
                        <strong className="text-emerald-400 font-mono">
                          ₹{suggestion.circuit_metrics.upper_circuit.toFixed(2)} (+{suggestion.circuit_metrics.headroom_pct}%)
                        </strong>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Lower Circuit Floor:</span>
                        <strong className="text-red-400 font-mono">
                          ₹{suggestion.circuit_metrics.lower_circuit.toFixed(2)}
                        </strong>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Circuit Status:</span>
                        <strong className="text-slate-200 capitalize font-medium">
                          {suggestion.circuit_metrics.circuit_status.replace('_', ' ')}
                        </strong>
                      </div>
                    </div>
                  </div>
                )}

                {/* Institutional Delivery Quality */}
                {suggestion.delivery_metrics && (
                  <div className="p-4 rounded-xl bg-black/50 border border-cyan-500/25">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-cyan-400">
                        <Briefcase className="w-4 h-4" />
                        <span>Delivery Accumulation Audit</span>
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">
                        {suggestion.delivery_metrics.delivery_quality}
                      </span>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Estimated Delivery:</span>
                        <strong className="text-cyan-300 font-mono text-sm">
                          {suggestion.delivery_metrics.estimated_delivery_pct}%
                        </strong>
                      </div>
                      <p className="text-[11px] text-slate-300 leading-relaxed italic">
                        {suggestion.delivery_metrics.delivery_verdict}
                      </p>
                    </div>
                  </div>
                )}

                {/* Operator Risk & Liquidity Guardrail */}
                {liquidity_guardrails && (
                  <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-white">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" />
                        <span>Liquidity & Operator Trap Audit</span>
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-bold">
                        {liquidity_guardrails.operator_trap_label}
                      </span>
                    </div>

                    <div className="space-y-2 text-xs text-slate-300">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">20-Day Daily Turnover:</span>
                        <strong className="text-white font-mono">₹{liquidity_guardrails.avg_turnover_cr} Cr / day</strong>
                      </div>
                      <div className="text-[11px] text-slate-400">{liquidity_guardrails.circuit_band_desc}</div>
                    </div>
                  </div>
                )}

                {/* Nifty Beta Sensitivity */}
                {suggestion.nifty_beta_sensitivity && (
                  <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-cyan-400">
                        <Activity className="w-4 h-4" />
                        <span>Nifty Beta Sensitivity ({suggestion.nifty_beta_sensitivity.beta}x)</span>
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-white/[0.06] text-slate-300 font-mono">
                        {suggestion.nifty_beta_sensitivity.defensive_score}
                      </span>
                    </div>

                    <div className="space-y-1.5 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">If Nifty Drops -1.0%:</span>
                        <strong className="text-red-400 font-mono">{suggestion.nifty_beta_sensitivity.nifty_drop_1pct}%</strong>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">If Nifty Drops -2.0% (Correction):</span>
                        <strong className="text-red-400 font-mono">{suggestion.nifty_beta_sensitivity.nifty_drop_2pct}%</strong>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">If Nifty Rallies +1.0%:</span>
                        <strong className="text-emerald-400 font-mono">+{suggestion.nifty_beta_sensitivity.nifty_rally_1pct}%</strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Invalidation Rules */}
              <div className="p-4 rounded-xl bg-red-950/20 border border-red-500/25">
                <div className="flex items-center gap-2 text-xs font-bold text-red-400 mb-2">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Downside Risk & Strict Invalidation Rules (Zero Sugarcoating)</span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">{reasoning.invalidation_warning}</p>
              </div>
            </div>
          )}

          {/* TAB 3: 3-STAGE TRAILING ROADMAP */}
          {dossierTab === 'roadmap' && execution_playbook && (
            <div className="space-y-4 animate-fade-in">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
                <div className="p-4 rounded-xl bg-gradient-to-b from-blue-950/30 to-black/60 border border-blue-500/25 relative overflow-hidden">
                  <div className="w-7 h-7 rounded-lg bg-blue-500/20 border border-blue-500/40 text-blue-300 font-bold flex items-center justify-center text-xs mb-3">
                    1
                  </div>
                  <div className="text-xs font-bold text-slate-100">{execution_playbook.stage_1_entry.title}</div>
                  <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                    {execution_playbook.stage_1_entry.description}
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-b from-emerald-950/30 to-black/60 border border-emerald-500/25 relative overflow-hidden">
                  <div className="w-7 h-7 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-bold flex items-center justify-center text-xs mb-3">
                    2
                  </div>
                  <div className="text-xs font-bold text-emerald-300">{execution_playbook.stage_2_target_1.title}</div>
                  <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                    {execution_playbook.stage_2_target_1.description}
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-gradient-to-b from-purple-950/30 to-black/60 border border-purple-500/25 relative overflow-hidden">
                  <div className="w-7 h-7 rounded-lg bg-purple-500/20 border border-purple-500/40 text-purple-300 font-bold flex items-center justify-center text-xs mb-3">
                    3
                  </div>
                  <div className="text-xs font-bold text-purple-300">{execution_playbook.stage_3_target_2.title}</div>
                  <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                    {execution_playbook.stage_3_target_2.description}
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-center gap-3 text-xs text-slate-400">
                <Info className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                <span>
                  <strong>Discipline Reminder:</strong> In low-ticket scrips under ₹200, retail traders often hold losing trades hoping for a rebound. Always honor the stop loss at ₹{trade_plan.stop_loss} without hesitation.
                </span>
              </div>
            </div>
          )}

          {/* TAB 4: INSTANT 1-CLICK BROKER ORDER STUDIO */}
          {dossierTab === 'broker' && (
            <div className="space-y-4 animate-fade-in">
              <div className="p-4 rounded-xl bg-black/60 border border-white/[0.08] space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="text-xs font-bold text-white flex items-center gap-2">
                    <Briefcase className="w-4 h-4 text-emerald-400" />
                    <span>Copy Pre-Formatted Terminal Order for {display_symbol}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Size: {calcShares} Qty | Outlay: ₹{calcOutlay.toFixed(0)}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {/* Zerodha Kite Format */}
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08] flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between text-xs font-bold text-slate-200 mb-1">
                        <span>🪁 Zerodha Kite (Basket Limit)</span>
                        <span className="text-[9px] text-slate-500 font-mono">NSE CNC</span>
                      </div>
                      <div className="p-2 rounded bg-black/80 font-mono text-[11px] text-emerald-400 break-all select-all border border-white/[0.04]">
                        {display_symbol},NSE,BUY,CNC,LIMIT,{trade_plan.entry_max},{calcShares}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const txt = `${display_symbol},NSE,BUY,CNC,LIMIT,${trade_plan.entry_max},${calcShares}`;
                        navigator.clipboard.writeText(txt);
                        setCopiedOrderFmt('zerodha');
                        setTimeout(() => setCopiedOrderFmt(''), 2000);
                      }}
                      className="mt-2 w-full py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 text-emerald-300 font-semibold text-xs transition cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      {copiedOrderFmt === 'zerodha' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy Zerodha CSV</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Angel One Format */}
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08] flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between text-xs font-bold text-slate-200 mb-1">
                        <span>👼 Angel One Bulk Order</span>
                        <span className="text-[9px] text-slate-500 font-mono">DELIVERY</span>
                      </div>
                      <div className="p-2 rounded bg-black/80 font-mono text-[11px] text-cyan-400 break-all select-all border border-white/[0.04]">
                        BUY {display_symbol} QTY:{calcShares} LIMIT:₹{trade_plan.entry_max} SL:₹{trade_plan.stop_loss} T1:₹{trade_plan.target_1}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const txt = `BUY ${display_symbol} QTY:${calcShares} LIMIT:${trade_plan.entry_max} SL:${trade_plan.stop_loss} T1:${trade_plan.target_1}`;
                        navigator.clipboard.writeText(txt);
                        setCopiedOrderFmt('angel');
                        setTimeout(() => setCopiedOrderFmt(''), 2000);
                      }}
                      className="mt-2 w-full py-1.5 rounded-lg bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-300 font-semibold text-xs transition cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      {copiedOrderFmt === 'angel' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy Angel One Order</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Groww / Upstox Format */}
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08] flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between text-xs font-bold text-slate-200 mb-1">
                        <span>🌱 Groww / Upstox Quick Limit</span>
                        <span className="text-[9px] text-slate-500 font-mono">DELIVERY</span>
                      </div>
                      <div className="p-2 rounded bg-black/80 font-mono text-[11px] text-purple-300 break-all select-all border border-white/[0.04]">
                        NSE:{display_symbol} | BUY DELIVERY | LIMIT ₹{trade_plan.entry_max} | QTY: {calcShares} | SL: ₹{trade_plan.stop_loss}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const txt = `NSE:${display_symbol} | BUY DELIVERY | LIMIT ₹${trade_plan.entry_max} | QTY: ${calcShares} | SL: ₹${trade_plan.stop_loss}`;
                        navigator.clipboard.writeText(txt);
                        setCopiedOrderFmt('groww');
                        setTimeout(() => setCopiedOrderFmt(''), 2000);
                      }}
                      className="mt-2 w-full py-1.5 rounded-lg bg-purple-500/15 hover:bg-purple-500/25 border border-purple-500/30 text-purple-300 font-semibold text-xs transition cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      {copiedOrderFmt === 'groww' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-purple-400" />
                          <span>Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5" />
                          <span>Copy Groww Note</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Trade Journal & Social Sharing */}
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.08] flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between text-xs font-bold text-slate-200 mb-1">
                        <span>📱 WhatsApp / Journal Note</span>
                        <span className="text-[9px] text-slate-500 font-mono">LOG</span>
                      </div>
                      <div className="p-2 rounded bg-black/80 font-mono text-[10px] text-slate-300 line-clamp-2 select-all border border-white/[0.04]">
                        🎯 Setup: {display_symbol} ({sector}) | BUY: ₹{trade_plan.entry_min}-₹{trade_plan.entry_max} | SL: ₹{trade_plan.stop_loss} | T1: ₹{trade_plan.target_1} | T2: ₹{trade_plan.target_2}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const txt = `🎯 StockIQ Quantitative Setup: ${display_symbol} (${sector})\nAction: BUY between ₹${trade_plan.entry_min} - ₹${trade_plan.entry_max}\nStop Loss: ₹${trade_plan.stop_loss} | Target 1: ₹${trade_plan.target_1} | Target 2: ₹${trade_plan.target_2}\nConviction: ${conviction_score}/100 | Setup: ${setup_label}\nCalculated Size: ${calcShares} shares (~₹${calcOutlay.toFixed(0)})\nRationale: ${reasoning.primary_thesis}`;
                        navigator.clipboard.writeText(txt);
                        setCopiedOrderFmt('journal');
                        setTimeout(() => setCopiedOrderFmt(''), 2000);
                      }}
                      className="mt-2 w-full py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.1] text-slate-200 font-semibold text-xs transition cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      {copiedOrderFmt === 'journal' ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                          <span>Copied Full Plan!</span>
                        </>
                      ) : (
                        <>
                          <Share2 className="w-3.5 h-3.5" />
                          <span>Copy Trading Plan</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Action links */}
          <div className="flex items-center justify-between pt-3 border-t border-white/[0.06] flex-wrap gap-3">
            <div className="flex items-center gap-2">
              <Link
                href={`/?ticker=${encodeURIComponent(symbol)}`}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-black font-bold text-xs transition cursor-pointer"
              >
                <span>Open Full Analysis</span>
                <ExternalLink className="w-3 h-3" />
              </Link>

              <Link
                href={`/intraday?ticker=${encodeURIComponent(symbol)}`}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/[0.12] text-white font-semibold text-xs transition cursor-pointer"
              >
                <span>Intraday Trading Desk</span>
                <ArrowRight className="w-3 h-3 text-slate-400" />
              </Link>
            </div>

            <button
              onClick={onToggleWatchlist}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer border ${
                isWatchlisted
                  ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                  : 'bg-white/[0.04] border-white/[0.08] text-slate-300 hover:bg-white/[0.08]'
              }`}
            >
              <Star className={`w-3.5 h-3.5 ${isWatchlisted ? 'fill-amber-400 text-amber-400' : ''}`} />
              <span>{isWatchlisted ? 'Watchlisted' : 'Add to Watchlist'}</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
