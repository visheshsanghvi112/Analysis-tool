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
      <header className="sticky top-0 z-50 border-b border-slate-800/60 glass-card">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo and Brand */}
            <div className="flex items-center gap-3">
              <div className="gradient-border">
                <div className="bg-slate-900 p-2 rounded-xl">
                  <TrendingUp className="h-6 w-6 text-indigo-400" />
                </div>
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold gradient-text">StockIQ Pro</h1>
                <p className="text-xs text-slate-500">Professional Stock Analysis</p>
              </div>
            </div>

            {/* Search Bar - Desktop */}
            <div className="hidden md:block flex-1 max-w-md mx-8">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-slate-500" />
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearch}
                  placeholder="Search stocks (e.g. HDFC, Reliance, TCS)"
                  className="block w-full pl-10 pr-3 py-2 border border-slate-700 rounded-lg bg-slate-800/60 text-slate-200 placeholder-slate-500 focus-ring focus:border-indigo-500 text-sm"
                />
                
                {/* Search Results Dropdown */}
                {(searchResults.length > 0 || isSearching) && (
                  <div className="absolute top-full mt-1 w-full bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto">
                    {isSearching ? (
                      <div className="p-4 text-center text-slate-500">
                        <div className="animate-spin h-5 w-5 border-2 border-indigo-500 border-t-transparent rounded-full mx-auto"></div>
                      </div>
                    ) : (
                      searchResults.map((ticker) => (
                        <button
                          key={ticker.symbol}
                          onClick={() => selectTicker(ticker)}
                          className="w-full px-4 py-3 text-left hover:bg-slate-700/60 border-b border-slate-700/40 last:border-0 focus:bg-slate-700/60 focus:outline-none transition-colors"
                        >
                          <div className="flex justify-between items-start">
                            <div>
                              <p className="font-semibold text-slate-200 text-sm">
                                {ticker.symbol.replace('.NS', '')}
                              </p>
                              <p className="text-xs text-slate-500 truncate max-w-xs">
                                {ticker.name}
                              </p>
                            </div>
                            {ticker.sector && (
                              <span className="text-xs bg-slate-700 text-slate-400 px-2 py-1 rounded-md">
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
            <nav className="hidden md:flex items-center gap-6">
              <div className="flex items-center gap-1 text-xs text-slate-400">
                <Shield className="h-4 w-4" />
                <span>Secure</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-slate-400">
                <Brain className="h-4 w-4" />
                <span>AI-Powered</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-slate-400">
                <Zap className="h-4 w-4" />
                <span>Real-time</span>
              </div>
            </nav>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors focus-ring"
            >
              {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>

          {/* Current Stock Display */}
          {currentTicker && (
            <div className="pb-3 border-t border-slate-800/40 mt-3 pt-3">
              <div className="flex items-center gap-3">
                <BarChart3 className="h-4 w-4 text-indigo-400" />
                <span className="text-sm font-semibold text-slate-300">
                  Analyzing: 
                </span>
                <span className="text-sm font-bold text-white bg-indigo-600/20 border border-indigo-500/30 px-2 py-1 rounded-md">
                  {currentTicker.replace('.NS', '')}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-slate-800/60 bg-slate-900/95 backdrop-blur-sm">
            <div className="px-4 py-4 space-y-4">
              {/* Mobile Search */}
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-slate-500" />
                </div>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearch}
                  placeholder="Search stocks..."
                  className="block w-full pl-10 pr-3 py-2 border border-slate-700 rounded-lg bg-slate-800/60 text-slate-200 placeholder-slate-500 focus-ring text-sm"
                />
              </div>

              {/* Popular Stocks - Mobile */}
              <div>
                <h3 className="text-sm font-semibold text-slate-400 mb-2 flex items-center gap-2">
                  <Star className="h-4 w-4" />
                  Popular Stocks
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  {popularStocks.map((stock) => (
                    <button
                      key={stock.symbol}
                      onClick={() => selectTicker(stock)}
                      className="text-left p-3 bg-slate-800/60 hover:bg-slate-700/60 rounded-lg border border-slate-700/40 transition-colors focus-ring"
                    >
                      <p className="font-semibold text-sm text-slate-200">
                        {stock.symbol.replace('.NS', '')}
                      </p>
                      <p className="text-xs text-slate-500 truncate">
                        {stock.name}
                      </p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Mobile Search Results */}
              {searchResults.length > 0 && (
                <div className="space-y-1 max-h-60 overflow-y-auto">
                  {searchResults.map((ticker) => (
                    <button
                      key={ticker.symbol}
                      onClick={() => selectTicker(ticker)}
                      className="w-full p-3 text-left bg-slate-800/60 hover:bg-slate-700/60 rounded-lg border border-slate-700/40 transition-colors"
                    >
                      <div className="flex justify-between items-start">
                        <div className="min-w-0 flex-1">
                          <p className="font-semibold text-slate-200 text-sm">
                            {ticker.symbol.replace('.NS', '')}
                          </p>
                          <p className="text-xs text-slate-500 truncate">
                            {ticker.name}
                          </p>
                        </div>
                        {ticker.sector && (
                          <span className="text-xs bg-slate-700 text-slate-400 px-2 py-1 rounded-md ml-2 flex-shrink-0">
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