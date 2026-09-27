import React, { useState, useEffect } from 'react';
import { useAppState } from '../../context/AppStateContext';
import { Loader2, Server, CheckCircle2, RefreshCw, X, Play } from 'lucide-react';
import { Badge } from './Badge';

export const BackendLoadingModal: React.FC = () => {
  const {
    isBackendConnected,
    backendElapsedSeconds,
    hasDismissedBackendLoading,
    setHasDismissedBackendLoading,
    retryBackendConnection,
  } = useAppState();

  const [hasClosed, setHasClosed] = useState(false);
  const [showSuccessToast, setShowSuccessToast] = useState(false);

  useEffect(() => {
    if (isBackendConnected) {
      // Backend connected! Show brief 600ms success indication then automatically close
      setShowSuccessToast(true);
      const timer = setTimeout(() => {
        setHasClosed(true);
        setHasDismissedBackendLoading(true);
      }, 700);
      return () => clearTimeout(timer);
    }
  }, [isBackendConnected, setHasDismissedBackendLoading]);

  // If already connected and closed, or dismissed, do not render
  if (hasClosed || hasDismissedBackendLoading) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md transition-all duration-300">
      <div className="relative w-full max-w-lg bg-dark-900 border border-cyan-500/40 rounded-3xl p-6 md:p-8 shadow-2xl shadow-cyan-500/20 text-center space-y-6 overflow-hidden">
        {/* Glowing top line */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-pulse" />

        {/* Close / Dismiss button */}
        <button
          onClick={() => {
            setHasClosed(true);
            setHasDismissedBackendLoading(true);
          }}
          className="absolute top-4 right-4 p-2 rounded-xl text-gray-400 hover:text-white hover:bg-white/5 transition-colors cursor-pointer"
          title="Dismiss dialog"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Center Graphic */}
        <div className="flex justify-center pt-2">
          {showSuccessToast ? (
            <div className="w-16 h-16 rounded-2xl bg-emerald-500/20 border border-emerald-500/50 flex items-center justify-center shadow-[0_0_25px_rgba(16,185,129,0.4)] animate-bounce">
              <CheckCircle2 className="w-8 h-8 text-emerald-400" />
            </div>
          ) : (
            <div className="relative w-16 h-16 rounded-2xl bg-cyan-500/10 border border-cyan-500/40 flex items-center justify-center shadow-glow-cyan">
              <Server className="w-8 h-8 text-cyan-400" />
              <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-cyan-400 animate-ping" />
              <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-cyan-500 border-2 border-dark-900" />
            </div>
          )}
        </div>

        {/* Main Headings */}
        <div className="space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Badge variant={showSuccessToast ? 'emerald' : 'cyan'} className="text-[10px] tracking-wider uppercase font-mono">
              {showSuccessToast ? 'CONNECTION ESTABLISHED' : 'BACKEND INITIALIZING'}
            </Badge>
          </div>
          <h2 className="text-xl md:text-2xl font-bold text-white tracking-tight">
            {showSuccessToast
              ? 'Backend Connected!'
              : 'Waiting for Backend to Connect...'}
          </h2>
          <p className="text-xs md:text-sm text-gray-300 font-sans leading-relaxed max-w-md mx-auto">
            {showSuccessToast ? (
              <span className="text-emerald-300 font-medium">
                Neural perception engine connected! Starting Sequence 00 video replay...
              </span>
            ) : (
              <span>
                Please wait while the LiDAR-X backend server is loading. Deployed cloud servers (e.g. Render / Koyeb) take 20–50s on standby cold-start.
              </span>
            )}
          </p>
        </div>

        {/* Status card */}
        {!showSuccessToast && (
          <div className="bg-dark-950/80 rounded-2xl border border-white/10 p-4 space-y-3 text-left">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-gray-400 flex items-center gap-2">
                <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                Connecting to FastAPI server...
              </span>
              <span className="text-cyan-400 font-bold">
                {backendElapsedSeconds}s elapsed
              </span>
            </div>

            {/* Pulsing visual bar */}
            <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-400 animate-pulse w-full" />
            </div>

            <div className="text-[11px] font-mono text-gray-400 flex items-center gap-2">
              <Play className="w-3 h-3 text-cyan-400 shrink-0" />
              <span><strong>Sequence 00</strong> will auto-play immediately upon connection.</span>
            </div>
          </div>
        )}

        {/* Footer Actions */}
        {!showSuccessToast && (
          <div className="flex items-center justify-center gap-3 pt-1">
            <button
              onClick={() => retryBackendConnection()}
              className="px-4 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-medium flex items-center gap-2 transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Ping Now</span>
            </button>
            <button
              onClick={() => {
                setHasClosed(true);
                setHasDismissedBackendLoading(true);
              }}
              className="px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white border border-white/10 text-xs font-mono transition-colors cursor-pointer"
            >
              <span>Explore UI in Standby</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
