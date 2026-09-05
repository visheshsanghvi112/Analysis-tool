import IntradayTerminal from '../components/IntradayTerminal';

export const metadata = {
  title: 'Intraday Trading Desk & Microstructure Terminal — StockIQ Pro',
  description: 'Institutional high-frequency trading desk with session-anchored VWAP, standard deviation volatility bands, Volume Profile (VPVR with POC), Camarilla & Floor pivots, Opening Range Breakout (ORB 15m), and Cumulative Volume Delta.',
};

export default function IntradayPage() {
  return <IntradayTerminal />;
}
