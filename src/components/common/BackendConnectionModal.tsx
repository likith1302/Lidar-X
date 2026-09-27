import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Radio,
  Cpu,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import { terrainService } from '../../services/terrainService';
import { useAppState } from '../../context/AppStateContext';

export type BackendConnectionState = 'CONNECTING' | 'BACKEND_READY' | 'BACKEND_ERROR';

interface BackendConnectionModalProps {
  isOpen: boolean;
  onReady: () => void;
  onDismiss?: () => void;
}

export const BackendConnectionModal: React.FC<BackendConnectionModalProps> = ({
  isOpen,
  onReady,
  onDismiss,
}) => {
  const { setIsBackendConnected } = useAppState();
  const [state, setState] = useState<BackendConnectionState>('CONNECTING');
  const [attemptCount, setAttemptCount] = useState<number>(1);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('Initializing LiDAR-X perception backend...');
  const [subMessage, setSubMessage] = useState<string>('Please wait while the perception engine connects...');
  const [isRetrying, setIsRetrying] = useState<boolean>(false);

  const maxAttempts = 16; // ~40 seconds total
  const isMountedRef = useRef<boolean>(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const checkConnection = useCallback(async (currentAttempt: number) => {
    if (!isMountedRef.current) return;
    setState('CONNECTING');
    setIsRetrying(false);

    try {
      setStatusMessage('Initializing LiDAR-X perception backend...');
      setSubMessage(`Connecting to perception engine & 2.5D foveated grid... (Attempt ${currentAttempt}/${maxAttempts})`);

      const res = await terrainService.checkHealth();

      if (!isMountedRef.current) return;

      if (res.success && res.data && res.data.status === 'healthy') {
        setState('BACKEND_READY');
        setStatusMessage('Backend connected — loading demo');
        setSubMessage('Fast-FRNet neural models and 2.5D grid engine verified online.');
        setIsBackendConnected(true);

        // Allow judge to see confirmed state briefly before smooth dismissal
        timerRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            onReady();
          }
        }, 750);
        return;
      }
      throw new Error(res.message || 'Health check returned non-healthy state');
    } catch {
      if (!isMountedRef.current) return;

      if (currentAttempt < maxAttempts) {
        setAttemptCount(currentAttempt + 1);
        // Wait 2.4s before next retry (sensible interval for cloud wake-up)
        timerRef.current = setTimeout(() => {
          if (isMountedRef.current) {
            checkConnection(currentAttempt + 1);
          }
        }, 2400);
      } else {
        setState('BACKEND_ERROR');
        setStatusMessage('Backend connection failed');
        setSubMessage('Perception backend did not respond within the timeout window.');
      }
    }
  }, [maxAttempts, onReady, setIsBackendConnected]);

  useEffect(() => {
    isMountedRef.current = true;
    if (isOpen) {
      setElapsedSeconds(0);
      setAttemptCount(1);
      checkConnection(1);
    }
    return () => {
      isMountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isOpen, checkConnection]);

  // Elapsed seconds timer
  useEffect(() => {
    if (!isOpen || state === 'BACKEND_READY') return;
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isOpen, state]);

  const handleManualRetry = () => {
    setIsRetrying(true);
    setAttemptCount(1);
    setElapsedSeconds(0);
    if (timerRef.current) clearTimeout(timerRef.current);
    checkConnection(1);
  };

  const handleContinueOffline = () => {
    if (onDismiss) {
      onDismiss();
    } else {
      onReady();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md transition-opacity duration-300 select-none animate-fadeIn">
      {/* Instrumentation Card */}
      <div className="relative w-full max-w-lg rounded-2xl bg-[#0d121a] border border-[#f97316]/40 shadow-2xl p-6 sm:p-8 text-gray-100 overflow-hidden font-sans">
        {/* Subtle orange accent top bar */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-orange-500 via-amber-400 to-cyan-500" />

        {/* Technical Header */}
        <div className="flex items-center justify-between border-b border-white/10 pb-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-orange-500/15 border border-orange-500/30 flex items-center justify-center text-orange-400 shadow-sm">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight text-white text-base">LiDAR-X</span>
                <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-orange-500/10 border border-orange-500/30 text-orange-400 font-semibold tracking-wider">
                  SIH Demo
                </span>
              </div>
              <p className="text-[11px] font-mono text-gray-400 tracking-wider uppercase">
                Autonomous Spatial Perception
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-dark-950 border border-white/10 text-[11px] font-mono text-gray-400">
            <Radio className="w-3 h-3 text-orange-400 animate-pulse" />
            <span>{elapsedSeconds}s</span>
          </div>
        </div>

        {/* Center Graphic & Status */}
        <div className="py-4 flex flex-col items-center text-center space-y-5">
          {/* Animated Graphic Indicator */}
          {state === 'CONNECTING' && (
            <div className="relative flex items-center justify-center w-20 h-20">
              {/* Outer pulsing ring */}
              <div className="absolute inset-0 rounded-full border-2 border-orange-500/20 animate-ping opacity-35" />
              {/* Spinning technical ring */}
              <div className="w-16 h-16 rounded-full border-2 border-transparent border-t-orange-500 border-r-amber-400 animate-spin" />
              {/* Center pulsing radar dot */}
              <div className="absolute w-6 h-6 rounded-full bg-orange-500/20 border border-orange-500/60 flex items-center justify-center">
                <div className="w-2.5 h-2.5 rounded-full bg-orange-400 animate-pulse" />
              </div>
            </div>
          )}

          {state === 'BACKEND_READY' && (
            <div className="w-16 h-16 rounded-full bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 animate-scaleUp">
              <CheckCircle2 className="w-8 h-8 stroke-[2.5]" />
            </div>
          )}

          {state === 'BACKEND_ERROR' && (
            <div className="w-16 h-16 rounded-full bg-red-500/15 border border-red-500/40 flex items-center justify-center text-red-400">
              <AlertTriangle className="w-8 h-8" />
            </div>
          )}

          {/* Status Label & Details */}
          <div className="space-y-1.5 max-w-sm">
            <h3 className="text-lg font-bold tracking-tight text-white">
              {statusMessage}
            </h3>
            <p className="text-xs sm:text-sm text-gray-400 leading-relaxed font-sans">
              {subMessage}
            </p>
          </div>

          {/* Status Badge Line */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-dark-950/80 border border-white/10 text-xs font-mono">
            {state === 'CONNECTING' && (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-amber-300">CONNECTING... (Attempt {attemptCount}/{maxAttempts})</span>
              </>
            )}
            {state === 'BACKEND_READY' && (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span className="text-emerald-300 font-semibold">BACKEND_READY — LAUNCHING</span>
              </>
            )}
            {state === 'BACKEND_ERROR' && (
              <>
                <span className="w-2 h-2 rounded-full bg-red-400" />
                <span className="text-red-300 font-semibold">BACKEND_ERROR</span>
              </>
            )}
          </div>
        </div>

        {/* Telemetry info footer */}
        <div className="mt-4 pt-4 border-t border-white/10 flex flex-col gap-3">
          <div className="flex items-center justify-between text-[11px] font-mono text-gray-400">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-orange-400" />
              <span>Inference Engine: Fast-FRNet</span>
            </span>
            <span>Target: /health</span>
          </div>

          {/* Action buttons on Error */}
          {state === 'BACKEND_ERROR' && (
            <div className="flex flex-col sm:flex-row items-center gap-2.5 pt-2">
              <button
                type="button"
                onClick={handleManualRetry}
                disabled={isRetrying}
                className="w-full sm:flex-1 py-2.5 px-4 rounded-xl bg-orange-500 hover:bg-orange-400 text-black font-bold text-xs uppercase tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isRetrying ? 'animate-spin' : ''}`} />
                <span>Retry Connection</span>
              </button>

              <button
                type="button"
                onClick={handleContinueOffline}
                className="w-full sm:w-auto py-2.5 px-4 rounded-xl bg-dark-900 hover:bg-dark-850 text-gray-300 hover:text-white border border-white/10 text-xs font-mono transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
                title="Proceed to view the demonstration video"
              >
                <span>Continue to Demo</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
