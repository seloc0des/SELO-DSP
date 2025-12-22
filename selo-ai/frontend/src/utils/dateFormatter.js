/**
 * Centralized date formatting utilities
 * Provides consistent date/time display across the application
 */

import { formatDistanceToNow, parseISO, format, isValid } from 'date-fns';
import { logger } from './logger';

const coerceDate = (value) => {
  if (!value) return null;
  if (value instanceof Date) {
    const copy = new Date(value.getTime());
    return isNaN(copy.getTime()) ? null : copy;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const hasTimezone = /[zZ]$|[+-]\d{2}:?\d{2}$/.test(trimmed);
    const isoLike = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?$/.test(trimmed);
    const normalized = !hasTimezone && isoLike ? `${trimmed}Z` : trimmed;
    const parsed = new Date(normalized);
    if (!isNaN(parsed.getTime())) return parsed;
  }
  try {
    const fallback = new Date(value);
    return isNaN(fallback.getTime()) ? null : fallback;
  } catch (error) {
    return null;
  }
};

/**
 * Format a date string as a relative time (e.g., "3 hours ago")
 * @param {string|Date} dateString - Date to format
 * @param {object} options - Options for formatDistanceToNow
 * @returns {string} Formatted relative time string
 */
export const formatRelativeTime = (dateString, options = { addSuffix: true }) => {
  if (!dateString) return 'Unknown time';
  try {
    const date = coerceDate(dateString);
    if (!date) return 'Unknown time';
    return formatDistanceToNow(date, options);
  } catch (error) {
    logger.error('Error formatting relative time:', error);
    return 'Unknown time';
  }
};

/**
 * Format a date string as an absolute date/time
 * @param {string|Date} dateString - Date to format
 * @param {object} options - Options for toLocaleString
 * @returns {string} Formatted date/time string
 */
export const formatAbsoluteTime = (dateString, options = {}) => {
  if (!dateString) return 'Unknown time';
  
  const defaultOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...options
  };
  
  try {
    const date = coerceDate(dateString);
    if (!date) return 'Unknown time';
    return date.toLocaleString('en-US', defaultOptions);
  } catch (error) {
    logger.error('Error formatting absolute time:', error);
    return 'Unknown time';
  }
};

/**
 * Check if a date string is valid
 * @param {string|Date} dateString - Date to validate
 * @returns {boolean} True if valid date
 */
export const isValidDate = (dateString) => {
  if (!dateString) return false;
  const date = coerceDate(dateString);
  return Boolean(date);
};

/**
 * Format timestamp for reflection cards
 * Combines relative and absolute time for better UX
 * @param {string|Date} dateString - Date to format
 * @returns {string} Formatted timestamp
 */
export const formatReflectionTimestamp = (dateString) => {
  if (!dateString) return 'Unknown time';
  
  try {
    const date = coerceDate(dateString);
    if (!date) return 'Unknown time';
    
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    // If less than 24 hours ago, show relative time
    if (diffInHours < 24) {
      return formatDistanceToNow(date, { addSuffix: true });
    }
    
    // Otherwise show absolute date
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (error) {
    logger.error('Error formatting reflection timestamp:', error);
    return 'Unknown time';
  }
};
