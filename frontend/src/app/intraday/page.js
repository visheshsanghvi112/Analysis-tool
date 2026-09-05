import { Suspense } from 'react';
import IntradayTerminal from '../components/IntradayTerminal';

export const metadata = {
  title: 'Intraday Trading Desk & Microstructure Terminal — StockIQ Pro',
  description: 'Institutional high-frequency trading desk with session-anchored VWAP, standard deviation volatility bands, Volume Profile (VPVR with POC), Camarilla & Floor pivots, Opening Range Breakout (ORB 15m), and Cumulative Volume Delta.',
};

export default function IntradayPage() {
  return (
    <Suspense fallback={
      <div className="w-full min-h-screen bg-slate-950 flex items-center justify-center text-cyan-400 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
          <span>Loading Intraday Quantitative Desk...</span>
        </div>
      </div>
    }>
      <IntradayTerminal />
    </Suspense>
  );
}
