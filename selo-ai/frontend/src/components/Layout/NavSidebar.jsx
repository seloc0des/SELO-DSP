import React from 'react';

const NavItem = ({ id, label, icon, active, onClick, disabled = false }) => (
  <button
    onClick={() => { if (!disabled) onClick(id); }}
    className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors
      ${disabled
        ? 'text-[var(--color-text-muted)] cursor-not-allowed'
        : active
          ? 'bg-[var(--color-bg-elev-2)] text-[var(--color-accent)]'
          : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-elev-2)]'}`}
    aria-current={active ? 'page' : undefined}
    aria-disabled={disabled || undefined}
    disabled={disabled}
  >
    <span aria-hidden>{icon}</span>
    <span className="truncate">{label}</span>
  </button>
);

const NavChildItem = ({ label }) => (
  <button
    type="button"
    className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm text-[var(--color-text-muted)] cursor-not-allowed pl-9"
    aria-disabled
    disabled
  >
    <span aria-hidden className="w-2 h-2 rounded-full bg-[var(--color-border)]" />
    <span className="truncate">{label}</span>
  </button>
);

const NavSidebar = ({ activeTab, onChange }) => {
  return (
    <aside className="hidden md:block md:w-64 shrink-0 border-r border-[var(--color-border)] bg-[var(--color-bg-elev-1)]">
      <div className="p-4">
        <div className="mb-4">
          <h1 className="text-xl font-bold tracking-wide text-[var(--color-accent)]">SELO DSP</h1>
        </div>
        <nav className="space-y-1" aria-label="Primary">
          <NavItem id="chat" label="Chat" active={activeTab==='chat'} onClick={onChange} icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M21 12c0 3.866-3.582 7-8 7-1.17 0-2.279-.197-3.287-.555L3 20l1.693-4.063C4.246 14.84 4 13.945 4 13c0-3.866 3.582-7 8-7s9 3.134 9 7z" stroke="currentColor" strokeWidth="1.5"/></svg>
          } />
          <NavItem id="reflection" label="Reflections" active={activeTab==='reflection'} onClick={onChange} icon={
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 3v18M3 12h18" stroke="currentColor" strokeWidth="1.5"/></svg>
          } />
          <NavItem id="persona" label="Persona" active={activeTab==='persona'} onClick={onChange} icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 12a5 5 0 100-10 5 5 0 000 10zM3 21a9 9 0 1118 0v0H3z" stroke="currentColor" strokeWidth="1.5"/></svg>
          } />
          <NavItem id="agent-state" label="Agent State" active={activeTab==='agent-state'} onClick={onChange} icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" stroke="currentColor" strokeWidth="1.5"/></svg>
          } />
          <NavItem id="plugins" label="Plugins (coming soon)" disabled onClick={onChange} icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M7 3h4v4h2V3h4v4h3v4h-3v2h3v4h-3v4h-4v-4h-2v4H7v-4H4v-4h3v-2H4V7h3V3z" stroke="currentColor" strokeWidth="1.5" fill="none"/>
            </svg>
          } />
          <div className="mt-1 space-y-1" aria-label="Upcoming plugins">
            {['Google', 'Spotify', 'Unreal Engine', 'Unity Engine', 'Gauntlet PVP'].map((name) => (
              <NavChildItem key={name} label={name} />
            ))}
          </div>
        </nav>
      </div>
    </aside>
  );
};

export default NavSidebar;
