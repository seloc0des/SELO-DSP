/**
 * Format trait name for display by replacing underscores with spaces
 * and capitalizing each word.
 * 
 * @param {string} name - Raw trait name (e.g., "genuine_care")
 * @returns {string} - Formatted name (e.g., "Genuine Care")
 */
export const formatTraitName = (name) => {
  if (!name || typeof name !== 'string') return '';
  
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

/**
 * Format trait name for display (lowercase with spaces)
 * 
 * @param {string} name - Raw trait name (e.g., "genuine_care")
 * @returns {string} - Formatted name (e.g., "genuine care")
 */
export const formatTraitNameLowercase = (name) => {
  if (!name || typeof name !== 'string') return '';
  
  return name.replace(/_/g, ' ');
};
