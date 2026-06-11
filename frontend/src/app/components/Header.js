'use client';

import { useState } from 'react';
import { 
  Search, 
  TrendingUp, 
  Brain, 
  Shield, 
  Menu, 
  X,
  BarChart3,
  Zap,
  Star
} from 'lucide-react';

const Header = ({ onTickerSelect, currentTicker }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

  const searchTickers = async (query) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }
    
    setIsSearching(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/tickers?q=${encodeURIComponent(query)}`);
      const data = await response.json();
      setSearchResults(data.tickers || []);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    searchTickers(query);
  };

  const selectTicker = (ticker) => {
    onTickerSelect(ticker.symbol);
    setSearchQuery('');
    setSearchResults([]);
    setMobileMenuOpen(false);
  };

  const popularStocks = [
    { symbol: 'HDFCBANK.NS', name: 'HDFC Bank' },
    { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
    { symbol: 'TCS.NS', name: 'Tata Consultancy Services' },
    { symbol: 'INFY.NS', name: 'Infosys' }
  ];

  return (
    <>
      <header className="sticky top-0 z-50 border-b border-indigo-500/20 glass-card rounded-none shadow-2xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <div className="flex items-center gap-3">
              <div className="gradient-border pulse-glow">
                <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-2.5 rounded-[14px]">
                  <TrendingUp className="h-6 w-6 text-indigo-400" />
                </div>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-black gradient-text">StockIQ Pro</h1>
                <p className="text-xs text-slate-400">Professional Stock Analysis</p>
              </div>
            </div>

            {/* Search Bar - Desktop */}
            <div className="hidden md:block flex-1 max-w-md mx-8">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-indigo-400" />
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearch}
                  placeholder="Search stocks (e.g. HDFC, Reliance, TCS)"
                  className="block w-full pl-11 pr-4 py-3 border-2 border-slate-700/60 rounded-xl bg-slate-800/80 text-slate-200 placeholder-slate-500 focus-ring focus:border-indigo-500 text-sm font-medium transition-all"
                />
                
                {/* Search Results Dropdown */}
                {(searchResults.length > 0 || isSearching) && (
                  <div className="absolute top-full mt-2 w-full glass-card border-indigo-500/20 shadow-2xl z-50 max-h-80 overflow-y-auto">
                    {isSearching ? (
                      <div className="p-6 text-center text-slate-400">
                        <div className="animate-spin h-6 w-6 border-2 border-indigo-500 border-t-transparent rounded-full mx-auto"></div>
                      </div>
                    ) : (
                      searchResults.map((ticker) => (
                        <button
                          key={ticker.symbol}
                          onClick={() => selectTicker(ticker)}
                          className="w-full px-4 py-3 text-left hover:bg-indigo-600/10 border-b border-slate-700/40 last:border-0 focus:bg-indigo-600/20 focus:outline-none transition-all group"
                        >
                          <div className="flex justify-between items-start">
                            <div>
                              <p className="font-bold text-white text-sm group-hover:text-indigo-300 transition-colors">
                                {ticker.symbol.replace('.NS', '')}
                              </p>
                              <p className="text-xs text-slate-400 truncate max-w-xs">
                                {ticker.name}
                              </p>
                            </div>
                            {ticker.sector && (
                              <span className="text-xs bg-indigo-600/20 text-indigo-300 px-2 py-1 rounded-md border border-indigo-500/30 font-medium">
                                {ticker.sector}
                              </span>
                            )}
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Navigation - Desktop */}
            <nav className="hidden md:flex items-center gap-4">
              <div className="flex items-center gap-2 text-xs text-emerald-400 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 font-medium">
                <Shield className="h-4 w-4" />
                <span>Secure</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-indigo-400 px-3 py-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 font-medium">
                <Brain className="h-4 w-4" />
                <span>AI-Powered</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-purple-400 px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20 font-medium">
                <Zap className="h-4 w-4" />
                <span>Real-time</span>
              </div>
            </nav>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-slate-300 hover:text-white hover:bg-indigo-600/20 rounded-lg transition-colors focus-ring border border-slate-700/40"
            >
              {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>

          {/* Current Stock Display */}
          {currentTicker && (
            <div className="pb-4 border-t border-indigo-500/20 mt-3 pt-4">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-5 w-5 text-indigo-400" />
                <span className="text-sm font-semibold text-slate-300">
                  Analyzing: 
                </span>
                <span className="text-sm font-bold text-white bg-gradient-to-r from-indigo-600/30 to-purple-600/30 border border-indigo-500/40 px-4 py-2 rounded-lg neon-glow">
                  {currentTicker.replace('.NS', '')}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-indigo-500/20 bg-slate-900/98 backdrop-blur-xl">
            <div className="px-4 py-6 space-y-4">
              {/* Mobile Search */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-5 w-5 text-indigo-400" />
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearch}
                  placeholder="Search stocks..."
                  className="block w-full pl-11 pr-4 py-3 border-2 border-slate-700/60 rounded-xl bg-slate-800/80 text-slate-200 placeholder-slate-500 focus-ring text-sm"
                />
              </div>

              {/* Popular Stocks - Mobile */}
              <div>
                <h3 className="text-sm font-bold text-slate-300 mb-3 flex items-center gap-2">
                  <Star className="h-4 w-4 text-yellow-400" />
                  Popular Stocks
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  {popularStocks.map((stock) => (
                    <button
                      key={stock.symbol}
                      onClick={() => selectTicker(stock)}
                      className="text-left p-4 glass-card card-hover border-indigo-500/20 transition-all focus-ring"
                    >
                      <p className="font-bold text-base text-white">
                        {stock.symbol.replace('.NS', '')}
                      </p>
                      <p className="text-xs text-slate-400 truncate mt-1">
                        {stock.name}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Mobile Search Results */}
              {searchResults.length > 0 && (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {searchResults.map((ticker) => (
                    <button
                      key={ticker.symbol}
                      onClick={() => selectTicker(ticker)}
                      className="w-full p-4 text-left glass-card card-hover border-indigo-500/20 transition-all"
                    >
                      <div className="flex justify-between items-start">
                        <div className="min-w-0 flex-1">
                          <p className="font-bold text-white text-sm">
                            {ticker.symbol.replace('.NS', '')}
                          </p>
                          <p className="text-xs text-slate-400 truncate mt-1">
                            {ticker.name}
                          </p>
                        </div>
                        {ticker.sector && (
                          <span className="text-xs bg-indigo-600/20 text-indigo-300 px-2 py-1 rounded-md ml-2 flex-shrink-0 border border-indigo-500/30">
                            {ticker.sector}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </header>
    </>
  );
};

export default Header;