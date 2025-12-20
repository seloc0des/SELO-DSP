import React from 'react';

/**
 * ReflectionTrigger component for initiating new AI reflections
 * This displays a card with information about the reflection type and allows triggering a generation
 */
const ReflectionTrigger = ({ reflectionType, isGenerating, onGenerate }) => {
  // Configuration for different reflection types
  const reflectionConfig = {
    daily: {
      title: "Daily Reflection",
      description: "Generate a daily reflection based on recent memories and experiences",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
      )
    },
    weekly: {
      title: "Weekly Reflection",
      description: "Generate a comprehensive weekly review to identify patterns and growth opportunities",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      )
    },
    emotional: {
      title: "Emotional Analysis",
      description: "Analyze current emotional state and identify patterns in emotional responses",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    }
  };

  const config = reflectionConfig[reflectionType] || reflectionConfig.daily;

  return (
    <div className="border border-[var(--color-border)] bg-[var(--color-bg-elev-1)] rounded-lg overflow-hidden">
      <div className="p-5">
        <div className="flex items-start space-x-4">
          <div className="text-[var(--color-accent)]">
            {config.icon}
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-medium text-[var(--color-accent)]">
              {config.title}
            </h3>
            <p className="text-sm text-gray-400 mt-1">
              {config.description}
            </p>
          </div>
        </div>
        
        <div className="mt-4">
          <button
            onClick={onGenerate}
            disabled={isGenerating}
            className={`w-full py-3 px-4 rounded-lg flex items-center justify-center transition-colors border ${
              isGenerating
                ? 'bg-[var(--color-bg-elev-2)] text-[var(--color-text-muted)] border-[var(--color-border)] cursor-not-allowed'
                : 'bg-[var(--color-accent)] text-black border-transparent hover:opacity-90'
            }`}
          >
            {isGenerating ? (
              <>
                <span className="animate-pulse mr-2">⏳</span>
                <span>Generating reflection...</span>
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Generate New Reflection
              </>
            )}
          </button>
        </div>
      </div>
      
      <div className="px-5 py-3 bg-[var(--color-bg-elev-2)] border-t border-[var(--color-border)]">
        <p className="text-xs text-[var(--color-text-muted)]">
          Reflections are generated based on persona memories, emotional state, and learned attributes. 
          They help your AI evolve its understanding and identity.
        </p>
      </div>
    </div>
  );
};

export default ReflectionTrigger;
