/**
 * Production-safe logging utility for SELO AI Frontend.
 * 
 * Provides conditional logging that can be disabled in production builds
 * while maintaining debug capabilities during development.
 */

const isDevelopment = process.env.NODE_ENV === 'development';
const isDebugEnabled = isDevelopment || localStorage.getItem('debug') === 'true';

class Logger {
  constructor(context = '') {
    this.context = context;
  }

  log(message, ...args) {
    if (isDebugEnabled) {
      const prefix = this.context ? `[${this.context}]` : '';
      console.log(`${prefix} ${message}`, ...args);
    }
  }

  warn(message, ...args) {
    if (isDebugEnabled) {
      const prefix = this.context ? `[${this.context}]` : '';
      console.warn(`${prefix} ${message}`, ...args);
    }
  }

  error(message, ...args) {
    // Always log errors, even in production
    const prefix = this.context ? `[${this.context}]` : '';
    console.error(`${prefix} ${message}`, ...args);
  }

  debug(message, ...args) {
    if (isDebugEnabled) {
      const prefix = this.context ? `[${this.context}]` : '';
      console.debug(`${prefix} ${message}`, ...args);
    }
  }

  info(message, ...args) {
    if (isDebugEnabled) {
      const prefix = this.context ? `[${this.context}]` : '';
      console.info(`${prefix} ${message}`, ...args);
    }
  }
}

// Create logger instances for different components
export const createLogger = (context) => new Logger(context);

// Default logger
export const logger = new Logger();

// Component-specific loggers
export const chatLogger = new Logger('Chat');
export const reflectionLogger = new Logger('Reflection');
export const personaLogger = new Logger('Persona');
export const socketLogger = new Logger('Socket');

export default logger;
