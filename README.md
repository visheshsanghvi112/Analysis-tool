# 🚀 StockIQ Pro - Professional Stock Analysis Platform

> **Advanced stock analysis with ML predictions, risk metrics, and real-time intelligence for smart investors.**

![StockIQ Pro](https://img.shields.io/badge/StockIQ-Pro-indigo?style=for-the-badge)
![Next.js](https://img.shields.io/badge/Next.js-16.2.7-black?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## 🌟 Features

### 💰 **Real-Time Market Data**
- Live stock prices with 15-minute delay
- NSE/BSE ticker search and selection
- Interactive price charts with technical indicators
- Day range visualization and volume analysis

### 🧠 **AI-Powered Predictions**
- Random Forest ML models with 20+ technical indicators
- 5-day price forecasting with confidence intervals
- Signal generation (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)
- Automated model retraining capabilities

### 📊 **Professional Risk Analytics**
- Value at Risk (VaR) calculations (95%, 99%)
- Sharpe ratio, Sortino ratio, Information ratio
- Maximum drawdown and volatility analysis
- Beta and correlation analysis vs market indices

### 📰 **News Intelligence**
- Multi-source news aggregation from major financial outlets
- AI-powered sentiment analysis with market impact scoring
- Breaking news detection with urgency classification
- Company-specific relevance filtering

### 📈 **Technical Analysis**
- Multi-indicator signal system (RSI, MACD, ADX, Bollinger Bands)
- Support/resistance identification via swing pivot points
- Fibonacci retracement levels
- Relative strength vs Nifty 50 analysis

### 🎯 **Portfolio-Grade Features**
- Black-Scholes options pricing with Greeks
- Expected Shortfall calculations
- Portfolio optimization metrics
- Risk-adjusted performance analysis

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm/yarn
- **Python** 3.9+
- **Git**

### 1. Clone Repository
```bash
git clone https://github.com/your-username/stockiq-pro.git
cd stockiq-pro
```

### 2. Backend Setup
```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start the backend server
python main.py
```
**Backend will run on:** `http://localhost:8000`

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
# Edit .env.local with your API URL

# Start the development server
npm run dev
```
**Frontend will run on:** `http://localhost:3000`

---

## 🛠️ Architecture

### Frontend (Next.js)
```
frontend/
├── src/
│   ├── app/
│   │   ├── components/          # React components
│   │   │   ├── Header.js        # Navigation & search
│   │   │   ├── LivePrice.js     # Real-time quotes
│   │   │   ├── StockChart.js    # Interactive charts
│   │   │   ├── MLPrediction.js  # ML forecasts
│   │   │   ├── AdvancedNews.js  # News sentiment
│   │   │   └── PortfolioMetrics.js # Risk analytics
│   │   ├── globals.css          # Global styles
│   │   ├── layout.js           # App layout
│   │   └── page.js             # Main dashboard
│   └── middleware.js           # Security headers
├── .env.local                  # Environment config
├── package.json               # Dependencies
└── vercel.json               # Deployment config
```

### Backend (FastAPI)
```
backend/
├── api/
│   └── index.py               # API route handler
├── engine.py                  # Technical analysis engine
├── ml_models.py              # ML prediction models
├── news_intelligence.py      # News aggregation & sentiment
├── main.py                   # FastAPI app & endpoints
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
└── vercel.json              # Serverless deployment
```

---

## 🔐 Security Features

### Backend Security
- **Rate limiting**: 30 requests/minute per IP
- **Input validation**: Ticker format validation & sanitization
- **CORS protection**: Restricted origins in production
- **Request timeout**: 10-second timeout for external APIs
- **Error handling**: Secure error messages without sensitive data

### Frontend Security
- **CSP headers**: Content Security Policy for XSS protection
- **Security headers**: X-Frame-Options, X-Content-Type-Options
- **Environment isolation**: Separate configs for dev/production
- **Error boundaries**: Graceful error handling with user feedback
- **Request sanitization**: Input encoding for API calls

---

## 📈 Performance Optimizations

### Frontend
- **Code splitting**: Dynamic imports for heavy components
- **Image optimization**: Next.js automatic image optimization
- **Caching**: Intelligent API response caching
- **Lazy loading**: Components load on demand
- **Mobile-first**: Responsive design with mobile optimizations

### Backend  
- **Async processing**: Non-blocking API calls
- **Connection pooling**: Efficient database connections
- **Data compression**: Gzip compression for responses
- **Concurrent processing**: Multi-threaded news fetching
- **Caching**: Redis caching for frequently accessed data

---

## 🚀 Deployment Guide

### Deploy to Vercel (Recommended)

#### Frontend Deployment
1. **Connect GitHub repository** to Vercel
2. **Set environment variables:**
   ```
   NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
   NEXT_PUBLIC_APP_ENV=production
   ```
3. **Deploy** - Vercel will auto-deploy from your repository

#### Backend Deployment Options

**Option 1: Vercel Serverless (Recommended)**
```bash
cd backend
vercel --prod
```

**Option 2: Railway**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy to Railway
cd backend
railway login
railway init
railway up
```

**Option 3: Render**
```bash
# Connect GitHub repository to Render
# Set Python version to 3.9+
# Use build command: pip install -r requirements.txt
# Use start command: python main.py
```

### Environment Variables

#### Backend (.env)
```bash
ENVIRONMENT=production
API_SECRET_KEY=your-secure-secret-key
ALLOWED_HOSTS=your-frontend-domain.vercel.app
RATE_LIMIT_PER_MINUTE=30
YAHOO_FINANCE_TIMEOUT=10
LOG_LEVEL=INFO
```

#### Frontend (.env.production)
```bash
NEXT_PUBLIC_API_URL=https://your-backend-domain.vercel.app
NEXT_PUBLIC_APP_ENV=production
NEXT_PUBLIC_ANALYTICS_ID=your-analytics-id
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn
```

---

## 💡 Value Proposition

### **For Retail Investors**
- **Professional analytics** at $0 cost vs $24,000/year Bloomberg Terminal
- **Institutional-grade risk metrics** typically unavailable to retail
- **AI predictions** with confidence intervals for informed decisions
- **Multi-source news intelligence** with sentiment analysis

### **For Developers**
- **Modern tech stack** with Next.js 16 + FastAPI
- **Scalable architecture** ready for millions of users
- **Professional codebase** with security best practices
- **Comprehensive documentation** and deployment guides

### **For Financial Advisors**
- **White-label potential** for client-facing tools
- **API access** for integration with existing systems
- **Risk management tools** for portfolio construction
- **Real-time data** for client meetings and reports

---

## 🔧 API Endpoints

### Core Endpoints
```bash
GET  /api/live?ticker=HDFCBANK.NS          # Live price data
GET  /api/analyze?ticker=HDFCBANK.NS       # Full technical analysis
GET  /api/ml-predict?ticker=HDFCBANK.NS    # ML price predictions
GET  /api/advanced-news?ticker=HDFCBANK.NS # News sentiment analysis
GET  /api/portfolio-metrics?ticker=HDFCBANK.NS # Risk metrics
GET  /api/tickers?q=hdfc                   # Stock search
POST /api/retrain-model                    # Model retraining
GET  /api/compare?tickers=HDFC,ICICI       # Peer comparison
```

### Response Examples
```json
// Live Price Response
{
  "ticker": "HDFCBANK.NS",
  "price": 1645.50,
  "change": 12.30,
  "changePct": 0.75,
  "dayHigh": 1650.00,
  "dayLow": 1630.00,
  "volume": 2547832,
  "timestamp": "15:30:00"
}

// ML Prediction Response
{
  "ticker": "HDFCBANK.NS", 
  "prediction": {
    "target_price": 1680.25,
    "confidence": 78.5,
    "signal": "BUY",
    "signal_strength": 7.2,
    "forecast_horizon": "5_days"
  }
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards
- **Frontend**: ESLint + Prettier for Next.js/React
- **Backend**: Black + isort for Python formatting
- **Commits**: Conventional commits format
- **Testing**: Jest for frontend, pytest for backend

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Vishesh Sanghvi**
- Email: [your-email@example.com](mailto:your-email@example.com)
- LinkedIn: [linkedin.com/in/vishesh-sanghvi](https://linkedin.com/in/vishesh-sanghvi)
- GitHub: [@vishesh-sanghvi](https://github.com/vishesh-sanghvi)

---

## 🙏 Acknowledgments

- **Yahoo Finance** for market data API
- **Next.js team** for the amazing React framework  
- **FastAPI** for the high-performance Python API framework
- **Tailwind CSS** for the utility-first styling system
- **Lucide React** for beautiful icons
- **Recharts** for interactive data visualization

---

## 🆘 Support

Having issues? We're here to help!

- **Documentation**: Check our [Wiki](https://github.com/your-repo/wiki)
- **Issues**: [Create an issue](https://github.com/your-repo/issues/new)
- **Discussions**: [Join discussions](https://github.com/your-repo/discussions)
- **Email**: [support@stockiq-pro.com](mailto:support@stockiq-pro.com)

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ by [Vishesh Sanghvi](https://github.com/vishesh-sanghvi)

</div>