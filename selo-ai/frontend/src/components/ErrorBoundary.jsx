import React from 'react';
import { logger } from '../utils/logger';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console and potentially to monitoring service
    logger.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI
      return (
        <div className="min-h-screen bg-dark-bg flex items-center justify-center p-4">
          <div className="bg-darker-bg border border-red-500/50 rounded-lg p-6 max-w-md w-full">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center">
                <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <h2 className="text-red-500 text-lg font-semibold">Something went wrong</h2>
            </div>
            
            <p className="text-gray-300 mb-4">
              SELO encountered an unexpected error. The development team has been notified.
            </p>
            
            <div className="space-y-3">
              <button
                onClick={() => window.location.reload()}
                className="w-full px-4 py-2 bg-[var(--color-bg-elev-1)] hover:bg-[var(--color-bg-elev-2)] text-[var(--color-accent)] border border-[var(--color-accent)]/50 rounded-lg font-medium transition-colors"
              >
                Reload Page
              </button>
              
              {process.env.NODE_ENV === 'development' && (
                <details className="mt-4">
                  <summary className="text-gray-400 text-sm cursor-pointer hover:text-gray-300">
                    Show Error Details (Development)
                  </summary>
                  <div className="mt-2 p-3 bg-gray-900 rounded text-xs text-gray-300 font-mono overflow-auto max-h-40">
                    <div className="text-red-400 font-semibold mb-2">Error:</div>
                    <div className="mb-3">{this.state.error && this.state.error.toString()}</div>
                    <div className="text-red-400 font-semibold mb-2">Stack Trace:</div>
                    <div className="whitespace-pre-wrap">{this.state.errorInfo.componentStack}</div>
                  </div>
                </details>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
