import React from 'react';
import { formatTraitName } from '../../utils/formatTraitName';

/**
 * PersonaTraitSliders (read-only)
 * Visualizes persona trait weights as horizontal bars.
 * @param {Array} traits - List of trait objects with { name, value, category, description }
 */
const PersonaTraitSliders = ({ traits = [] }) => {
  if (!traits.length) return (
    <div className="text-gray-500 text-xs italic">No traits defined</div>
  );

  // Sort by category then name for grouping
  const sortedTraits = [...traits].sort((a, b) => {
    if (a.category !== b.category) return a.category.localeCompare(b.category);
    return a.name.localeCompare(b.name);
  });

  return (
    <div className="space-y-3 mt-2">
      {sortedTraits.map((trait, idx) => (
        <div key={trait.name + trait.category}>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-semibold text-[var(--color-accent)]">{formatTraitName(trait.name)}</span>
            <span className="text-xs text-gray-400">{(trait.value * 100).toFixed(0)}%</span>
          </div>
          <div className="w-full h-3 bg-gray-800 rounded">
            <div
              className="h-3 rounded bg-[var(--color-accent)] transition-all"
              style={{ width: `${Math.round(trait.value * 100)}%` }}
              title={trait.description || ''}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default PersonaTraitSliders;
