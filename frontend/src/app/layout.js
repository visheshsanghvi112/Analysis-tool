import { Inter } from "next/font/google";
import "./globals.css";
import ErrorBoundary from "./components/ErrorBoundary";

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata = {
  title: "StockIQ Pro – Professional Stock Analysis",
  description: "Advanced NSE/BSE stock analysis with ML predictions, risk metrics, and real-time news intelligence.",
  keywords: "stock analysis, ML predictions, NSE, BSE, technical analysis, portfolio analytics",
  authors: [{ name: "Vishesh Sanghvi" }],
  robots: "index, follow",
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0f172a',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} h-full dark`}>
      <body className="min-h-full bg-slate-900 text-slate-100 antialiased" suppressHydrationWarning>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
