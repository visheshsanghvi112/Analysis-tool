import { Inter, Poppins } from "next/font/google";
import "./globals.css";
import ErrorBoundary from "./components/ErrorBoundary";

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const poppins = Poppins({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-poppins',
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
  themeColor: '#000000',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${poppins.variable} h-full dark`} suppressHydrationWarning>
      <body className="min-h-full bg-black text-slate-100 antialiased" suppressHydrationWarning>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
