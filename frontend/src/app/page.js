'use client';

import { useState } from 'react';
import Header from './components/Header';
import LivePrice from './components/LivePrice';
import StockChart from './components/StockChart';
import MLPrediction from './components/MLPrediction';
import AdvancedNews from './components/AdvancedNews';
import PortfolioMetrics from './components/PortfolioMetrics';
import { 
  TrendingUp, 
  Brain, 
  Newspaper, 
  Shield, 
  BarChart3, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Target,
  PieChart,
  Activity
} from 'lucide-react';

const FeatureCard = ({ icon: Icon, title, description, status = 'active' }) => (
  <div className="glass-card p-6 border border-slate-800/60 hover:border-slate-700/80 transition-all duration-200 group">
    <div className="flex items-start justify-between mb-4">
      <div className={`p-3 rounded-xl ${
        status === 'active' ? 'bg-indigo-600/20 text-indigo-400' :
        status === 'loading' ? 'bg-yellow-600/20 text-yellow-400' :
        'bg-slate-600/20 text-slate-500'
      }`}>
        <Icon className="h-6 w-6" />
      </div>
      <div className={`flex items-center gap-1 text-xs ${
        status === 'active' ? 'text-emerald-400' :
        status === 'loading' ? 'text-yellow-400' :
        'text-slate-500'
      }`}>
        {status === 'active' && <CheckCircle className="h-3 w-3" />}
        {status === 'loading' && <Clock className="h-3 w-3 animate-spin" />}
        {status === 'inactive' && <AlertTriangle className="h-3 w-3" />}
        <span className="font-semibold uppercase tracking-wider">
          {status === 'active' ? 'Live' : status === 'loading' ? 'Loading' : 'Offline'}
        </span>
      </div>
    </div>
    <h3 className="text-lg font-bold text-slate-200 mb-2 group-hover:text-white transition-colors">
      {title}
    </h3>
    <p className="text-sm text-slate-500 leading-relaxed">
      {description}
    </p>
  </div>
);

const WelcomeSection = () => (
  <div className="text-center py-12 px-6">
    <div className="max-w-4xl mx-auto">
      <div className="inline-flex items-center gap-3 mb-6">
        <div className="gradient-border p-1">
          <div className="bg-slate-900 p-4 rounded-xl">
            <TrendingUp className="h-12 w-12 text-indigo-400" />
          </div>
        </div>
      </div>
      
      <h1 className="text-4xl md:text-6xl font-black gradient-text mb-6 tracking-tight">
        StockIQ Pro
      </h1>
      
      <p className="text-xl text-slate-400 mb-8 max-w-2xl mx-auto leading-relaxed">
        Professional-grade stock analysis with <span className="text-indigo-400 font-semibold">ML predictions</span>, 
        <span className="text-emerald-400 font-semibold"> real-time intelligence</span>, and 
        <span className="text-purple-400 font-semibold"> institutional analytics</span>
      </p>
      
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        <div className="glass-card p-6 border border-slate-800/60">
          <Brain className="h-8 w-8 text-indigo-400 mb-4 mx-auto" />
          <h3 className="font-bold text-slate-200 mb-2">AI-Powered Predictions</h3>
          <p className="text-sm text-slate-500">Random Forest ML models with 20+ technical indicators</p>
        </div>
        
        <div className="glass-card p-6 border border-slate-800/60">
          <Shield className="h-8 w-8 text-emerald-400 mb-4 mx-auto" />
          <h3 className="font-bold text-slate-200 mb-2">Risk Management</h3>
          <p className="text-sm text-slate-500">VaR, Sharpe ratios, and portfolio-grade analytics</p>
        </div>
        
        <div className="glass-card p-6 border border-slate-800/60">
          <Zap className="h-8 w-8 text-purple-400 mb-4 mx-auto" />
          <h3 className="font-bold text-slate-200 mb-2">Real-Time Intelligence</h3>
          <p className="text-sm text-slate-500">Live prices, news sentiment, and market analysis</p>
        </div>
      </div>
      
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <p className="text-slate-500 text-sm flex items-center gap-2">
          <Target className="h-4 w-4" />
          Search for any NSE/BSE stock above to begin analysis
        </p>
      </div>
    </div>
  </div>
);

const StatsOverview = ({ ticker }) => (
  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
    <FeatureCard
      icon={Activity}
      title="Live Prices"
      description="Real-time quotes with 15-min delay from Yahoo Finance"
      status="active"
    />
    <FeatureCard
      icon={Brain}
      title="ML Predictions"
      description="5-day price forecasts using Random Forest algorithms"
      status={ticker ? "active" : "inactive"}
    />
    <FeatureCard
      icon={Newspaper}
      title="News Intelligence"
      description="AI sentiment analysis from multiple financial sources"
      status={ticker ? "active" : "inactive"}
    />
    <FeatureCard
      icon={PieChart}
      title="Risk Analytics"
      description="Professional portfolio metrics and risk assessments"
      status={ticker ? "active" : "inactive"}
    />
  </div>
);

export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleTickerSelect = (ticker) => {
    setIsLoading(true);
    setSelectedTicker(ticker);
    // Simulate loading delay for better UX
    setTimeout(() => setIsLoading(false), 500);
  };

  return (
    <div className="min-h-screen bg-slate-900">
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!selectedTicker ? (
          <WelcomeSection />
        ) : (
          <div className="space-y-8">
            {/* Loading State */}
            {isLoading && (
              <div className="text-center py-12">
                <div className="animate-spin h-12 w-12 border-4 border-indigo-600 border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-slate-400">Loading analysis for {selectedTicker.replace('.NS', '')}...</p>
              </div>
            )}
            
            {!isLoading && (
              <>
                {/* Stats Overview */}
                <StatsOverview ticker={selectedTicker} />
                
                {/* Main Content Grid */}
                <div className="grid lg:grid-cols-3 gap-8">
                  {/* Left Column - Price & Chart */}
                  <div className="lg:col-span-2 space-y-6">
                    <LivePrice ticker={selectedTicker} />
                    <StockChart ticker={selectedTicker} />
                  </div>
                  
                  {/* Right Column - ML & News */}
                  <div className="space-y-6">
                    <MLPrediction ticker={selectedTicker} />
                    <AdvancedNews ticker={selectedTicker} />
                  </div>
                </div>
                
                {/* Bottom Section - Portfolio Metrics */}
                <div className="mt-8">
                  <PortfolioMetrics ticker={selectedTicker} />
                </div>
              </>
            )}
          </div>
        )}
      </main>
      
      {/* Footer */}
      <footer className="border-t border-slate-800/60 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-indigo-400" />
              <span className="font-semibold text-slate-300">StockIQ Pro</span>
              <span className="text-sm text-slate-500">by Vishesh Sanghvi</span>
            </div>
            
            <div className="flex items-center gap-6 text-sm text-slate-500">
              <span className="flex items-center gap-1">
                <Shield className="h-4 w-4" />
                Secure & Private
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                Real-time Data
              </span>
              <span className="flex items-center gap-1">
                <Brain className="h-4 w-4" />
                AI-Powered
              </span>
            </div>
          </div>
          
          <div className="mt-6 pt-6 border-t border-slate-800/40 text-center text-xs text-slate-600">
            <p>
              Professional stock analysis platform. Data provided by Yahoo Finance (~15 min delay). 
              Not financial advice. Always consult a qualified financial advisor.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}