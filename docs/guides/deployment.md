# 🚀 Deployment Guide - StockIQ Pro

## ✅ What's Been Improved

### 🔒 Security Enhancements

#### Backend Security
✅ **Rate Limiting** - 30 requests/minute per IP using slowapi  
✅ **CORS Protection** - Restricted origins, configurable per environment  
✅ **Input Validation** - Regex validation for ticker symbols  
✅ **Request Sanitization** - Proper input encoding and validation  
✅ **Security Headers** - X-Frame-Options, Content-Type protection  
✅ **Environment Isolation** - Separate configs for dev/prod  

#### Frontend Security
✅ **CSP Headers** - Content Security Policy implementation  
✅ **Security Middleware** - Next.js middleware for headers  
✅ **Error Boundaries** - Graceful error handling  
✅ **Request Encoding** - Proper URL encoding for API calls  
✅ **Environment Variables** - Secure config management  

### 🎨 UI/UX Improvements

#### Modern Design System
✅ **Glass Morphism** - Modern glass-card effects  
✅ **Gradient Accents** - Professional color gradients  
✅ **Dark Theme** - Optimized dark mode design  
✅ **Custom Scrollbars** - Styled scrollbars for consistency  
✅ **Loading States** - Shimmer effects and skeletons  
✅ **Micro-interactions** - Smooth hover and transition effects  

#### Enhanced Components
✅ **Professional Header** - Sticky header with search and branding  
✅ **Live Price Card** - Real-time updates with visual indicators  
✅ **Welcome Section** - Engaging landing page  
✅ **Feature Cards** - Status indicators for each feature  
✅ **Responsive Footer** - Professional footer with features  

#### Mobile Optimization
✅ **Mobile-First Design** - Optimized for small screens  
✅ **Touch-Friendly** - Larger tap targets  
✅ **Responsive Grid** - Adapts to all screen sizes  
✅ **Mobile Menu** - Collapsible navigation  

### ⚡ Performance Optimizations

✅ **Code Splitting** - Dynamic imports for heavy components  
✅ **Request Cancellation** - AbortController for API calls  
✅ **Lazy Loading** - On-demand component loading  
✅ **Image Optimization** - Next.js automatic optimization  
✅ **Caching Strategy** - Intelligent data caching  

---

## 📦 Deployment Steps

### Option 1: Vercel (Recommended) ⭐

#### Frontend Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Production-ready deployment"
git push origin main
```

2. **Connect to Vercel**
- Go to [vercel.com](https://vercel.com)
- Click "Import Project"
- Select your GitHub repository
- Configure:
  - Framework: Next.js
  - Root Directory: `frontend`
  - Build Command: `npm run build`
  - Output Directory: `.next`

3. **Environment Variables** (in Vercel Dashboard)
```
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
NEXT_PUBLIC_APP_ENV=production
NEXT_PUBLIC_RATE_LIMIT_ENABLED=true
```

4. **Deploy** - Click Deploy!

#### Backend Deployment

**Method A: Vercel Serverless**
```bash
cd backend
npm install -g vercel
vercel login
vercel --prod
```

Set environment variables in Vercel:
```
ENVIRONMENT=production
ALLOWED_HOSTS=your-frontend.vercel.app
RATE_LIMIT_PER_MINUTE=30
```

**Method B: Railway (Better for Python)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
cd backend
railway login
railway init
railway up
```

---

### Option 2: Railway (Full Stack)

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Deploy Backend**
```bash
cd backend
railway login
railway init --name stockiq-backend
railway up
```

3. **Deploy Frontend**
```bash
cd frontend
railway init --name stockiq-frontend
railway up
```

4. **Link Services**
- Get backend URL from Railway dashboard
- Set `NEXT_PUBLIC_API_URL` in frontend

---

### Option 3: Render

#### Backend Deployment
1. Connect GitHub repo to Render
2. Create new "Web Service"
3. Configure:
   - Environment: Python 3.9+
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Root Directory: `backend`

4. Environment Variables:
```
ENVIRONMENT=production
ALLOWED_HOSTS=your-frontend-domain.com
RATE_LIMIT_PER_MINUTE=30
PYTHON_VERSION=3.9.0
```

#### Frontend Deployment
1. Create new "Static Site"
2. Configure:
   - Build Command: `npm run build`
   - Publish Directory: `.next`
   - Root Directory: `frontend`

---

## 🔧 Configuration Files

### Backend Configuration

**backend/.env** (Create this file)
```bash
ENVIRONMENT=production
API_SECRET_KEY=your-super-secret-key-change-this
ALLOWED_HOSTS=your-frontend-domain.vercel.app,localhost
RATE_LIMIT_PER_MINUTE=30
RATE_LIMIT_PER_HOUR=1000
YAHOO_FINANCE_TIMEOUT=10
NEWS_FETCH_TIMEOUT=15
LOG_LEVEL=INFO
```

### Frontend Configuration

**frontend/.env.production** (Already created)
```bash
NEXT_PUBLIC_API_URL=https://your-backend-url.vercel.app
NEXT_PUBLIC_APP_ENV=production
NEXT_PUBLIC_RATE_LIMIT_ENABLED=true
NEXT_PUBLIC_SECURE_HEADERS=true
```

---

## 🧪 Testing Before Deployment

### Backend Tests
```bash
cd backend

# Test health endpoint
curl http://localhost:8000/health

# Test live price endpoint
curl http://localhost:8000/api/live?ticker=HDFCBANK.NS

# Test rate limiting (run multiple times quickly)
for i in {1..35}; do curl http://localhost:8000/api/live?ticker=HDFCBANK.NS; done
```

### Frontend Tests
```bash
cd frontend

# Build for production
npm run build

# Test production build locally
npm start

# Check for build errors
npm run lint
```

---

## 📊 Post-Deployment Checklist

### Functionality Tests
- [ ] Home page loads correctly
- [ ] Stock search works
- [ ] Live price updates
- [ ] Charts render properly
- [ ] ML predictions load
- [ ] News sentiment displays
- [ ] Portfolio metrics calculate
- [ ] Mobile responsive
- [ ] Error handling works

### Security Tests
- [ ] CORS headers set correctly
- [ ] Rate limiting works
- [ ] CSP headers present
- [ ] No sensitive data in errors
- [ ] HTTPS enabled
- [ ] Environment variables secure

### Performance Tests
- [ ] Page load < 3 seconds
- [ ] API responses < 2 seconds
- [ ] No console errors
- [ ] Images optimized
- [ ] Lighthouse score > 80

---

## 🎯 Domain Setup (Optional)

### Custom Domain with Vercel

1. **Add Domain in Vercel Dashboard**
   - Project Settings → Domains
   - Add your domain (e.g., stockiq.pro)

2. **Configure DNS**
   - Add CNAME record: `www` → `cname.vercel-dns.com`
   - Add A record: `@` → Vercel IP

3. **SSL Certificate**
   - Vercel automatically provisions SSL

---

## 🚨 Troubleshooting

### Common Issues

**Issue: Backend CORS errors**
```python
# Solution: Update allowed_origins in backend/main.py
allowed_origins = [
    "https://your-actual-domain.vercel.app",
    "http://localhost:3000",
]
```

**Issue: Environment variables not loading**
```bash
# Solution: Redeploy after setting variables
vercel env pull
vercel --prod
```

**Issue: Rate limit errors**
```python
# Solution: Adjust rate limit in backend/main.py
limiter = Limiter(key_func=get_remote_address, default_limits=["50/minute"])
```

**Issue: Build fails on Vercel**
```bash
# Solution: Check Node.js version
# Add vercel.json in frontend with:
{
  "build": {
    "env": {
      "NODE_VERSION": "18"
    }
  }
}
```

---

## 📈 Monitoring & Analytics

### Recommended Tools

1. **Vercel Analytics** (Built-in)
   - Automatic performance monitoring
   - Core Web Vitals tracking

2. **Sentry** (Error Tracking)
```bash
npm install @sentry/nextjs
# Configure NEXT_PUBLIC_SENTRY_DSN
```

3. **LogRocket** (Session Replay)
```bash
npm install logrocket
# Add to layout.js
```

---

## 💰 Cost Estimation

### Free Tier Limits

**Vercel (Frontend)**
- ✅ Unlimited deployments
- ✅ 100GB bandwidth/month
- ✅ Custom domains
- ✅ SSL certificates
- ⚠️ Serverless functions: 100GB-hours

**Railway (Backend)**
- ✅ $5 free credit/month
- ✅ ~500 hours free runtime
- ✅ Shared resources

**Render (Alternative)**
- ✅ 750 hours/month free
- ⚠️ Spins down after inactivity

### Recommended Production Setup
- **Frontend**: Vercel Pro ($20/month) - Better performance
- **Backend**: Railway Pro ($5-20/month) - Always-on
- **Total**: ~$25-40/month for production-grade hosting

---

## 📚 Additional Resources

- [Next.js Deployment Docs](https://nextjs.org/docs/deployment)
- [Vercel Platform Docs](https://vercel.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Railway Deployment Docs](https://docs.railway.app/)

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**:
   - Vercel: Dashboard → Project → Deployments → View Logs
   - Railway: Dashboard → Project → Deployments → Logs

2. **Verify environment variables**:
   - Ensure all required variables are set
   - Check for typos in variable names

3. **Test locally first**:
   - Always test production build locally before deploying

4. **Contact support**:
   - Create an issue on GitHub
   - Check documentation
   - Community forums

---

<div align="center">

**🎉 Your app is now production-ready!**

Made with ❤️ by Vishesh Sanghvi

</div>