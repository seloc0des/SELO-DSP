import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';
import ReflectionView from './components/Reflection/ReflectionView';
import PersonaView from './components/Persona/PersonaView';
import AgentStateView from './components/AgentState/AgentStateView';
import ErrorBoundary from './components/ErrorBoundary';
import NavSidebar from './components/Layout/NavSidebar';
import DiagnosticsBanner from './components/System/DiagnosticsBanner';
// License components removed - open source release
import { getOrCreateSessionId } from './services/sessionService';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [userId, setUserId] = useState('');

  // Generate/retrieve a stable session/user ID via centralized service
  useEffect(() => {
    const id = getOrCreateSessionId();
    setUserId(id);
  }, []);

  // License event handlers removed - open source release

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-[var(--color-bg)] text-[var(--color-text-primary)] flex">
        {/* Sidebar */}
        <NavSidebar activeTab={activeTab} onChange={setActiveTab} />

        {/* Main content area */}
        <div className="flex-1 flex flex-col">
          {/* Minimal header */}
          <header className="h-16 border-b border-[var(--color-border)] flex items-center justify-between px-4">
            <div className="flex flex-col leading-tight">
              <div className="text-base font-semibold text-[var(--color-text-primary)]">SELO DSP</div>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-xs text-[var(--color-text-muted)]">{new Date().getFullYear()}</div>
            </div>
          </header>

          {/* Mobile tab switcher (sidebar hidden on mobile) */}
          <div className="md:hidden border-b border-[var(--color-border)] px-3 py-2">
            <div className="grid grid-cols-4 gap-2">
              <button
                className={`text-sm px-2 py-2 rounded border transition-colors ${activeTab==='chat' ? 'border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-bg-elev-2)]' : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elev-2)]'}`}
                onClick={() => setActiveTab('chat')}
                aria-current={activeTab==='chat' ? 'page' : undefined}
              >Chat</button>
              <button
                className={`text-sm px-2 py-2 rounded border transition-colors ${activeTab==='reflection' ? 'border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-bg-elev-2)]' : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elev-2)]'}`}
                onClick={() => setActiveTab('reflection')}
                aria-current={activeTab==='reflection' ? 'page' : undefined}
              >Reflect</button>
              <button
                className={`text-sm px-2 py-2 rounded border transition-colors ${activeTab==='persona' ? 'border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-bg-elev-2)]' : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elev-2)]'}`}
                onClick={() => setActiveTab('persona')}
                aria-current={activeTab==='persona' ? 'page' : undefined}
              >Persona</button>
              <button
                className={`text-sm px-2 py-2 rounded border transition-colors ${activeTab==='agent-state' ? 'border-[var(--color-accent)] text-[var(--color-accent)] bg-[var(--color-bg-elev-2)]' : 'border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elev-2)]'}`}
                onClick={() => setActiveTab('agent-state')}
                aria-current={activeTab==='agent-state' ? 'page' : undefined}
              >State</button>
            </div>
          </div>

          {/* Diagnostics banner (GPU/CPU status) */}
          <DiagnosticsBanner />

          {/* Content */}
          <main className="flex-1 p-4 overflow-y-auto">
            <div className="max-w-6xl mx-auto">
              {/* Individual ErrorBoundaries for each tab to prevent one view from crashing others */}
              {activeTab === 'chat' ? (
                <ErrorBoundary>
                  <Chat userId={userId} />
                </ErrorBoundary>
              ) : activeTab === 'reflection' ? (
                <ErrorBoundary>
                  <ReflectionView userId={userId} />
                </ErrorBoundary>
              ) : activeTab === 'persona' ? (
                <ErrorBoundary>
                  <PersonaView userId={userId} />
                </ErrorBoundary>
              ) : activeTab === 'agent-state' ? (
                <ErrorBoundary>
                  <AgentStateView userId={userId} />
                </ErrorBoundary>
              ) : null}
            </div>
          </main>
        </div>
      </div>
      
      {/* License modal removed - open source release */}
    </ErrorBoundary>
  );
}

export default App;
