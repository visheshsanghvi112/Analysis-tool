/**
 * Centralized API configuration for StockIQ Pro.
 * Dynamically resolves backend URL:
 * - Uses NEXT_PUBLIC_API_URL if set in environment.
 * - When running in browser on localhost/127.0.0.1/LAN, routes requests to http://localhost:8000.
 * - In production environments, falls back to the deployed production backend.
 */

export const getApiBaseUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.startsWith('192.168.') ||
      hostname.startsWith('10.') ||
      hostname.endsWith('.local')
    ) {
      return 'http://localhost:8000';
    }
  }
  return 'https://stock-analysis-backend-seven.vercel.app';
};

export const API_BASE_URL = getApiBaseUrl();
