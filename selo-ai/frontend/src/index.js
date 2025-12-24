import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';

// Global error handlers for better error visibility
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled Promise Rejection:', event.reason);
  console.error('Promise:', event.promise);
  
  // Log to backend if available
  try {
    const errorData = {
      type: 'unhandledRejection',
      message: event.reason?.message || String(event.reason),
      stack: event.reason?.stack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
    };
    
    // Send to backend error logging endpoint (non-blocking)
    fetch('/api/client-errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorData),
    }).catch(() => {}); // Silently fail if backend unavailable
  } catch (e) {
    // Error logging failed - just log to console
  }
  
  // Prevent default browser error handling
  event.preventDefault();
});

window.addEventListener('error', (event) => {
  console.error('Uncaught Error:', event.error || event.message);
  
  // Log to backend if available
  try {
    const errorData = {
      type: 'uncaughtError',
      message: event.message || String(event.error),
      stack: event.error?.stack,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
    };
    
    fetch('/api/client-errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorData),
    }).catch(() => {});
  } catch (e) {
    // Error logging failed - just log to console
  }
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
