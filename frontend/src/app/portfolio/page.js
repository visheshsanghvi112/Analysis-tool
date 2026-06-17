import PortfolioTracker from '../components/PortfolioTracker';

export const metadata = {
  title: 'Portfolio Tracker — StockIQ Pro',
  description: 'Track your Indian stock portfolio with live P&L, covariance-matrix risk analytics, allocation breakdown, and correlation heatmap. Built for NSE and BSE investors.',
};

export default function PortfolioPage() {
  return <PortfolioTracker />;
}
