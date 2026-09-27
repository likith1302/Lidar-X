import React, { useState } from 'react';
import { AppStateProvider, useAppState } from './context/AppStateContext';
import { Sidebar } from './components/layout/Sidebar';
import { DemoView } from './components/views/DemoView';
import { OverviewView } from './components/views/OverviewView';
import { MappingConsoleView } from './components/views/MappingConsoleView';
import { ArchitectureView } from './components/views/ArchitectureView';
import { PerformanceLabView } from './components/views/PerformanceLabView';
import { DiagnosticsView } from './components/views/DiagnosticsView';
import { BackendConnectionModal } from './components/common/BackendConnectionModal';
import { Menu } from 'lucide-react';

const MainContent: React.FC = () => {
  const { currentTab, setIsMobileDrawerOpen } = useAppState();

  // Show the backend connection modal on initial load when entering the demo
  const [isConnectionModalOpen, setIsConnectionModalOpen] = useState<boolean>(() => {
    // Only display modal if on demo view on initial load
    return currentTab === 'demo';
  });

  const renderCurrentView = () => {
    switch (currentTab) {
      case 'demo':
        return <DemoView />;
      case 'overview':
        return <OverviewView />;
      case 'mapping-console':
        return <MappingConsoleView />;
      case 'architecture':
        return <ArchitectureView />;
      case 'performance-lab':
        return <PerformanceLabView />;
      case 'scenes-replay':
        return <MappingConsoleView initialMode="sequence-replay" autoStartDemo={true} />;
      case 'diagnostics':
        return <DiagnosticsView />;
      default:
        return <DemoView />;
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-dark-950 text-gray-100 bg-tech-grid bg-tech-radial font-sans">
      {/* Backend Connecting & Initializing Modal */}
      <BackendConnectionModal
        isOpen={isConnectionModalOpen}
        onReady={() => setIsConnectionModalOpen(false)}
        onDismiss={() => setIsConnectionModalOpen(false)}
      />

      {/* Sidebar Navigation (Collapsible desktop + mobile drawer) */}
      <Sidebar />

      {/* Main App Container */}
      <div className="flex-1 flex flex-col h-full min-w-0 overflow-hidden relative">
        {/* Mobile drawer toggle */}
        <button
          onClick={() => setIsMobileDrawerOpen(true)}
          className="md:hidden fixed top-3 left-3 z-40 p-2 rounded-xl bg-dark-900/90 border border-white/10 text-gray-400 hover:text-white shadow-lg cursor-pointer"
          title="Open Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Scrollable View Area - optimized for full viewport demo viewing */}
        <main
          className={`flex-1 overflow-y-auto ${
            currentTab === 'demo' ? 'p-2 md:p-4' : 'px-4 md:px-8 py-6 md:py-8'
          }`}
        >
          {renderCurrentView()}
        </main>
      </div>
    </div>
  );
};

export default function App() {
  return (
    <AppStateProvider>
      <MainContent />
    </AppStateProvider>
  );
}
