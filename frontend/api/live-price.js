// Vercel Serverless Function for Live Prices
import yahooFinance from 'yahoo-finance2';

export default async function handler(req, res) {
  const { ticker } = req.query;
  
  if (!ticker) {
    return res.status(400).json({ error: 'Ticker required' });
  }

  try {
    const quote = await yahooFinance.quoteSummary(ticker, {
      modules: ['price', 'summaryDetail']
    });
    
    const price = quote.price;
    const summary = quote.summaryDetail;
    
    const result = {
      ticker: ticker.toUpperCase(),
      price: price.regularMarketPrice,
      change: price.regularMarketChange,
      changePct: price.regularMarketChangePercent * 100,
      dayHigh: price.regularMarketDayHigh,
      dayLow: price.regularMarketDayLow,
      volume: price.regularMarketVolume,
      marketCap: summary.marketCap,
      prevClose: price.regularMarketPreviousClose,
      timestamp: new Date().toISOString()
    };
    
    res.status(200).json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}