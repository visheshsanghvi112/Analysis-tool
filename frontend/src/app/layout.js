import { Geist, Geist_Mono, Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";
import ErrorBoundary from "./components/ErrorBoundary";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
});

const spaceGrotesk = Space_Grotesk({ 
  subsets: ['latin'],
  variable: '--font-space-grotesk',
});

export const metadata = {
  title: "StockIQ Pro - Professional Stock Analysis Platform",
  description: "Advanced stock analysis with ML predictions, risk metrics, and real-time intelligence for smart investors. NSE/BSE stocks with institutional-grade analytics.",
  keywords: "stock analysis, ML predictions, portfolio management, risk analytics, Indian stocks, NSE, BSE, technical analysis",
  author: "Vishesh Sanghvi",
  robots: "index, follow",
  viewport: "width=device-width, initial-scale=1",
  themeColor: "#0f172a",
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${inter.variable} ${spaceGrotesk.variable} h-full antialiased dark`}
    >
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      </head>
      <body className="min-h-full flex flex-col bg-slate-900 text-slate-100 font-inter" suppressHydrationWarning>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  );
}
