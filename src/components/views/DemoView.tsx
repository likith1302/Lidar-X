import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Play,
  Pause,
  Volume2,
  VolumeX,
  Maximize,
  Minimize,
  RotateCcw,
  Sparkles,
  Layers,
  Activity,
  Cpu,
  Radio,
  ExternalLink,
  Info,
  CheckCircle2,
} from 'lucide-react';
import { useAppState } from '../../context/AppStateContext';
import { Badge } from '../common/Badge';
import { MappingConsoleView } from './MappingConsoleView';

export const DemoView: React.FC = () => {
  const { isBackendConnected, setCurrentTab, setIsSidebarCollapsed } = useAppState();

  // Mode: 'video' (primary judge demonstration) vs 'live-console' (interactive 3D WebGL)
  const [activeMode, setActiveMode] = useState<'video' | 'live-console'>('video');

  // Video State
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const [volume, setVolume] = useState<number>(1);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [autoplayBlocked, setAutoplayBlocked] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);
  const [showControls, setShowControls] = useState<boolean>(true);
  const hideControlsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-collapse sidebar on mount to maximize demo viewing area
  useEffect(() => {
    setIsSidebarCollapsed(true);
  }, [setIsSidebarCollapsed]);

  // Attempt autoplay as soon as video element is ready
  const attemptAutoplay = useCallback(async () => {
    const video = videoRef.current;
    if (!video) return;

    try {
      video.muted = true;
      setIsMuted(true);
      const playPromise = video.play();
      if (playPromise !== undefined) {
        await playPromise;
        setIsPlaying(true);
        setAutoplayBlocked(false);
      }
    } catch (err) {
      console.warn('Autoplay prevented by browser security policy:', err);
      setIsPlaying(false);
      setAutoplayBlocked(true);
    }
  }, []);

  useEffect(() => {
    if (activeMode === 'video') {
      attemptAutoplay();
    }
  }, [activeMode, attemptAutoplay]);

  const handlePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play().then(() => {
        setIsPlaying(true);
        setAutoplayBlocked(false);
      }).catch((e) => {
        console.warn('Playback error:', e);
      });
    } else {
      video.pause();
      setIsPlaying(false);
    }
  };

  const handleToggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isMuted) {
      video.muted = false;
      video.volume = volume || 0.8;
      setIsMuted(false);
    } else {
      video.muted = true;
      setIsMuted(true);
    }
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVol = parseFloat(e.target.value);
    setVolume(newVol);
    if (videoRef.current) {
      videoRef.current.volume = newVol;
      if (newVol > 0 && isMuted) {
        videoRef.current.muted = false;
        setIsMuted(false);
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackSpeed(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  const handleToggleFullscreen = () => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch(err => {
        console.warn('Fullscreen error:', err);
      });
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false);
      }).catch(err => {
        console.warn('Exit fullscreen error:', err);
      });
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      attemptAutoplay();
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleMouseMove = () => {
    setShowControls(true);
    if (hideControlsTimerRef.current) {
      clearTimeout(hideControlsTimerRef.current);
    }
    hideControlsTimerRef.current = setTimeout(() => {
      if (isPlaying) {
        setShowControls(false);
      }
    }, 3000);
  };

  return (
    <div className="flex flex-col h-full w-full max-w-[1800px] mx-auto space-y-3 font-sans pb-4">
      {/* 1. COMPACT TOP TELEMETRY RIBBON */}
      <header className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl bg-dark-900/90 border border-white/10 shadow-lg">
        {/* Brand Title */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-500 animate-pulse" />
            <h1 className="text-base sm:text-lg font-extrabold text-white tracking-tight flex items-center gap-2">
              <span>LiDAR-X</span>
              <span className="text-gray-400 font-normal hidden sm:inline">|</span>
              <span className="text-xs font-mono text-gray-300 font-medium hidden sm:inline uppercase tracking-wide">
                Autonomous 2.5D Perception Engine
              </span>
            </h1>
          </div>
          <Badge variant="cyan" className="text-[10px] tracking-wider py-0.5">
            SIH DEMO EXPERIENCE
          </Badge>
        </div>

        {/* Action Controls & Mode Switcher */}
        <div className="flex items-center gap-2.5">
          {/* Mode Switcher */}
          <div className="flex items-center p-1 rounded-lg bg-dark-950 border border-white/10 text-xs font-mono">
            <button
              onClick={() => setActiveMode('video')}
              className={`px-3 py-1.5 rounded-md transition-all font-medium flex items-center gap-1.5 cursor-pointer ${
                activeMode === 'video'
                  ? 'bg-orange-500 text-black shadow-glow-orange font-bold'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              <span>Demo Video</span>
            </button>
            <button
              onClick={() => setActiveMode('live-console')}
              className={`px-3 py-1.5 rounded-md transition-all font-medium flex items-center gap-1.5 cursor-pointer ${
                activeMode === 'live-console'
                  ? 'bg-cyan-500 text-black shadow-glow-cyan font-bold'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Interactive 3D Stream</span>
            </button>
          </div>

          {/* Backend Status Pill */}
          <div
            className={`hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-mono ${
              isBackendConnected
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
            }`}
          >
            <Radio className="w-3 h-3 animate-pulse" />
            <span>{isBackendConnected ? 'PERCEPTION ENGINE ONLINE' : 'STANDBY'}</span>
          </div>
        </div>
      </header>

      {/* 2. MAIN DEMO VISUALIZATION AREA (MAXIMIZED VIEWPORT SPACE) */}
      <main className="flex-1 flex flex-col min-h-0 relative">
        {activeMode === 'video' ? (
          <div
            ref={containerRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => isPlaying && setShowControls(false)}
            className="relative flex-1 w-full rounded-2xl overflow-hidden bg-black border border-white/15 shadow-2xl flex items-center justify-center group min-h-[460px] md:min-h-[580px]"
          >
            {/* HTML5 Video Element */}
            <video
              ref={videoRef}
              src="/demo-video.mp4"
              playsInline
              muted
              autoPlay
              loop
              preload="auto"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onClick={handlePlayPause}
              className="w-full h-full object-contain cursor-pointer max-h-[82vh]"
            >
              <source src="/Lidar 2.5d.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>

            {/* AUTOPLAY BLOCKED OVERLAY FALLBACK */}
            {autoplayBlocked && (
              <div
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.play().then(() => {
                      setIsPlaying(true);
                      setAutoplayBlocked(false);
                    });
                  }
                }}
                className="absolute inset-0 z-30 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center cursor-pointer transition-all hover:bg-black/50"
              >
                <div className="flex flex-col items-center space-y-4 p-6 rounded-2xl bg-dark-900/90 border border-orange-500/40 shadow-glow-orange max-w-sm text-center">
                  <div className="w-16 h-16 rounded-full bg-orange-500/20 border border-orange-500/60 flex items-center justify-center text-orange-400 animate-pulse">
                    <Play className="w-8 h-8 ml-1 fill-orange-400 stroke-none" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white tracking-tight">Click to Start Demo</h3>
                    <p className="text-xs text-gray-400 mt-1 font-mono">
                      Browser autoplay policy paused initial stream. Click anywhere to play.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="px-5 py-2.5 rounded-xl bg-orange-500 hover:bg-orange-400 text-black font-bold text-xs uppercase tracking-wider transition-all shadow-md"
                  >
                    Start Demonstration
                  </button>
                </div>
              </div>
            )}

            {/* UNMUTE AUDIO FLOATING BADGE (Helps judges enable sound if video starts muted) */}
            {isMuted && isPlaying && !autoplayBlocked && (
              <button
                type="button"
                onClick={handleToggleMute}
                className="absolute top-4 right-4 z-20 flex items-center gap-2 px-3 py-2 rounded-xl bg-dark-900/90 hover:bg-orange-500 hover:text-black border border-orange-500/30 text-orange-300 text-xs font-mono font-medium shadow-lg backdrop-blur-md transition-all cursor-pointer group"
                title="Click to unmute video audio"
              >
                <VolumeX className="w-4 h-4 group-hover:scale-110 transition-transform" />
                <span>Audio Muted (Click to Unmute)</span>
              </button>
            )}

            {/* BOTTOM HUD VIDEO CONTROLS */}
            <div
              className={`absolute bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-black/95 via-black/70 to-transparent p-4 transition-opacity duration-300 ${
                showControls || !isPlaying ? 'opacity-100' : 'opacity-0 pointer-events-none'
              }`}
            >
              {/* Timeline Scrubber */}
              <div className="flex items-center gap-3 mb-2.5">
                <input
                  type="range"
                  min={0}
                  max={duration || 100}
                  step={0.1}
                  value={currentTime}
                  onChange={handleSeek}
                  className="w-full h-1.5 bg-gray-700/80 rounded-lg appearance-none cursor-pointer accent-orange-500 hover:accent-orange-400 transition-all"
                />
              </div>

              {/* Main Controls Row */}
              <div className="flex items-center justify-between text-gray-200">
                {/* Left Controls: Play, Time, Volume */}
                <div className="flex items-center gap-3 sm:gap-4">
                  {/* Play/Pause */}
                  <button
                    onClick={handlePlayPause}
                    className="w-9 h-9 flex items-center justify-center rounded-lg bg-orange-500 hover:bg-orange-400 text-black font-bold transition-all shadow-md cursor-pointer"
                    title={isPlaying ? 'Pause' : 'Play'}
                  >
                    {isPlaying ? <Pause className="w-4 h-4 fill-black" /> : <Play className="w-4 h-4 fill-black ml-0.5" />}
                  </button>

                  {/* Restart */}
                  <button
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = 0;
                        videoRef.current.play();
                        setIsPlaying(true);
                      }
                    }}
                    className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                    title="Restart Demo"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>

                  {/* Volume Toggle & Slider */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={handleToggleMute}
                      className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                      title={isMuted ? 'Unmute' : 'Mute'}
                    >
                      {isMuted || volume === 0 ? (
                        <VolumeX className="w-4 h-4 text-orange-400" />
                      ) : (
                        <Volume2 className="w-4 h-4" />
                      )}
                    </button>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={isMuted ? 0 : volume}
                      onChange={handleVolumeChange}
                      className="w-16 sm:w-20 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-orange-500 hidden sm:block"
                    />
                  </div>

                  {/* Time Display */}
                  <span className="text-xs font-mono text-gray-400">
                    <span className="text-white font-medium">{formatTime(currentTime)}</span>
                    <span className="mx-1">/</span>
                    <span>{formatTime(duration)}</span>
                  </span>
                </div>

                {/* Right Controls: Speed, Fullscreen, Console Switch */}
                <div className="flex items-center gap-2">
                  {/* Speed Selector */}
                  <div className="flex items-center gap-1 bg-dark-950/80 rounded-lg p-0.5 border border-white/10 text-[11px] font-mono">
                    {[1, 1.25, 1.5].map((spd) => (
                      <button
                        key={spd}
                        onClick={() => handleSpeedChange(spd)}
                        className={`px-2 py-0.5 rounded cursor-pointer ${
                          playbackSpeed === spd
                            ? 'bg-orange-500 text-black font-bold'
                            : 'text-gray-400 hover:text-white'
                        }`}
                      >
                        {spd}x
                      </button>
                    ))}
                  </div>

                  {/* Fullscreen Button */}
                  <button
                    onClick={handleToggleFullscreen}
                    className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                    title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
                  >
                    {isFullscreen ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* LIVE INTERACTIVE 3D WEBGL MAPPING CONSOLE MODE */
          <div className="flex-1 w-full h-full min-h-0">
            <MappingConsoleView initialMode="sequence-replay" autoStartDemo={true} />
          </div>
        )}
      </main>

      {/* 3. SLIM TECHNICAL HIGHLIGHTS FOOTER */}
      <footer className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 select-none font-mono">
        <div className="px-3 py-2 rounded-xl bg-dark-900/80 border border-white/5 flex items-center justify-between">
          <span className="text-[11px] text-gray-400">Model:</span>
          <span className="text-xs font-semibold text-orange-400">Fast-FRNet (Dual-Path)</span>
        </div>
        <div className="px-3 py-2 rounded-xl bg-dark-900/80 border border-white/5 flex items-center justify-between">
          <span className="text-[11px] text-gray-400">Map Engine:</span>
          <span className="text-xs font-semibold text-cyan-400">2.5D Adaptive Foveated</span>
        </div>
        <div className="px-3 py-2 rounded-xl bg-dark-900/80 border border-white/5 flex items-center justify-between">
          <span className="text-[11px] text-gray-400">Sensors:</span>
          <span className="text-xs font-semibold text-purple-400">SemanticKITTI + RELLIS-3D</span>
        </div>
        <div className="px-3 py-2 rounded-xl bg-dark-900/80 border border-white/5 flex items-center justify-between">
          <span className="text-[11px] text-gray-400">Target Latency:</span>
          <span className="text-xs font-semibold text-emerald-400">&lt; 50 ms / frame</span>
        </div>
      </footer>
    </div>
  );
};
