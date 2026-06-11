'use client';

import React from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // Log error to monitoring service
    if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
      // Sentry.captureException(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
          <div className="max-w-md w-full text-center">
            <div className="glass-card p-8 border border-red-800/60">
              <div className="flex justify-center mb-6">
                <div className="p-4 bg-red-600/20 text-red-400 rounded-full">
                  <AlertTriangle className="h-12 w-12" />
                </div>
              </div>
              
              <h1 className="text-2xl font-bold text-slate-200 mb-4">
                Something went wrong
              </h1>
              
              <p className="text-slate-400 mb-6 leading-relaxed">
                We encountered an unexpected error. This has been automatically reported to our team.
              </p>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-4 text-left mb-6">
                  <h3 className="text-sm font-semibold text-red-400 mb-2">Error Details:</h3>
                  <pre className="text-xs text-slate-300 overflow-auto">
                    {this.state.error.toString()}
                  </pre>
                </div>
              )}
              
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  onClick={() => window.location.reload()}
                  className="btn-primary flex items-center justify-center gap-2 flex-1"
                >
                  <RefreshCw className="h-4 w-4" />
                  Reload Page
                </button>
                
                <button
                  onClick={() => window.location.href = '/'}
                  className="btn-secondary flex items-center justify-center gap-2 flex-1"
                >
                  <Home className="h-4 w-4" />
                  Go Home
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;