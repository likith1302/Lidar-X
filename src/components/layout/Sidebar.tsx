import React from 'react';
import {
  LayoutDashboard,
  Compass,
  Network,
  Gauge,
  Film,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  X,
} from 'lucide-react';
import { useAppState } from '../../context/AppStateContext';
import { NavigationTab } from '../../types/navigation';
import { clsx } from 'clsx';

interface NavItem {
  id: NavigationTab;
  label: string;
  icon: React.ElementType;
  tag?: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'mapping-console', label: 'Mapping Console', icon: Compass },
  { id: 'architecture', label: 'Architecture', icon: Network },
  { id: 'performance-lab', label: 'Performance Lab', icon: Gauge },
  { id: 'scenes-replay', label: 'Scenes / Replay', icon: Film },
  { id: 'diagnostics', label: 'Diagnostics', icon: ShieldCheck },
];

export const Sidebar: React.FC = () => {
  const {
    currentTab,
    setCurrentTab,
    isSidebarCollapsed,
    setIsSidebarCollapsed,
    isMobileDrawerOpen,
    setIsMobileDrawerOpen,
  } = useAppState();

  const handleNavClick = (tabId: NavigationTab) => {
    setCurrentTab(tabId);
    setIsMobileDrawerOpen(false);
  };

  const navContent = (
    <div className="flex flex-col h-full bg-dark-900 border-r border-white/10 select-none">
      {/* Brand Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10 h-16">
        <div
          onClick={() => handleNavClick('overview')}
          className="flex items-center gap-3 cursor-pointer overflow-hidden"
        >
          <div className="w-8 h-8 rounded-lg overflow-hidden border border-cyan-500/30 flex items-center justify-center shadow-glow-cyan shrink-0 bg-dark-950">
            <img src="/favicon.png" alt="LiDAR-X Logo" className="w-full h-full object-cover" />
          </div>
          {!isSidebarCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-bold tracking-tight text-white text-base leading-tight font-sans truncate">
                LiDAR-X
              </span>
              <span className="text-[10px] font-mono text-gray-400 tracking-wider uppercase truncate">
                3D Perception & Mapping
              </span>
            </div>
          )}
        </div>

        {/* Mobile close button */}
        <button
          onClick={() => setIsMobileDrawerOpen(false)}
          className="md:hidden p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/5"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Navigation List */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              title={isSidebarCollapsed ? item.label : undefined}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group relative',
                isActive
                  ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-inner-dark'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-white/5 border border-transparent'
              )}
            >
              <Icon
                className={clsx(
                  'w-5 h-5 shrink-0 transition-colors',
                  isActive ? 'text-cyan-400' : 'text-gray-400 group-hover:text-gray-200'
                )}
              />
              {!isSidebarCollapsed && (
                <span className="truncate font-sans text-left">{item.label}</span>
              )}

              {/* Active Indicator bar */}
              {isActive && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-cyan-400 rounded-r-full shadow-[0_0_8px_#00E5FF]" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Collapse Toggle for Desktop */}
      <div className="hidden md:flex items-center justify-between px-4 py-2 border-t border-white/5 bg-dark-950/40">
        {!isSidebarCollapsed && (
          <span className="text-[11px] font-mono text-gray-500">Collapse View</span>
        )}
        <button
          onClick={() => setIsSidebarCollapsed((prev: boolean) => !prev)}
          className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors ml-auto"
          title={isSidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {isSidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside
        className={clsx(
          'hidden md:block h-screen sticky top-0 transition-all duration-300 z-30 shrink-0',
          isSidebarCollapsed ? 'w-20' : 'w-64'
        )}
      >
        {navContent}
      </aside>

      {/* Mobile Backdrop & Drawer */}
      {isMobileDrawerOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-black/80 backdrop-blur-sm"
            onClick={() => setIsMobileDrawerOpen(false)}
          />
          <div className="relative w-72 max-w-[80vw] h-full shadow-2xl z-10">
            {navContent}
          </div>
        </div>
      )}
    </>
  );
};
