/**
 * LiDAR Sequence Replay and Live Stream Client Service
 * Coordinates with backend for ZIP sequence ingestion, playback lifecycle control,
 * and continuous WebSocket frame streaming with backpressure.
 *
 * The WebSocket helper here implements an automatic reconnect loop with
 * exponential backoff.  The previous version connected once and silently
 * dropped the link on any close, which forced the user to re-upload on a
 * transient network blip.  Reconnects now re-emit the playback state so the
 * UI can resync.
 */

import { apiClient, ApiResponse } from './api';
import {
  ReplayUploadResponse,
  ReplaySessionStatus,
  ReplayFrameStreamPayload,
  ReplaySeekRequest,
  ReplayStartRequest,
} from '../types/replay';

export interface ReplayPrecomputeStatus {
  session_id: string;
  sequence_name: string;
  processed_frames: number;
  total_frames: number;
  percent_complete: number;
  is_complete: boolean;
  is_running: boolean;
  elapsed_seconds: number;
  eta_seconds: number;
  current_stage?: string;
  failed_count?: number;
  error_message?: string | null;
}

export interface ReplayWebSocketHandle {
  send: (msg: any) => void;
  close: () => void;
  /** True when the underlying socket is currently connected. */
  isConnected: () => boolean;
}

export class ReplayService {
  /**
   * Upload a SemanticKITTI sequence ZIP package (containing velodyne/*.bin and predictions/*.label)
   */
  public async uploadSequence(
    file: File,
    sessionId?: string,
    playbackMode?: 'offline_precomputed_replay' | 'live_processing'
  ): Promise<ApiResponse<ReplayUploadResponse>> {
    const queryParams: Record<string, string> = {};
    if (sessionId) queryParams['session_id'] = sessionId;
    if (playbackMode) queryParams['playback_mode'] = playbackMode;

    return apiClient.uploadFile<ReplayUploadResponse>(
      '/replay/upload-sequence',
      file,
      'file',
      queryParams
    );
  }

  /**
   * Get playback status of an active replay session.  No silent mock: if the
   * backend is unreachable the request throws (so the UI knows it's offline).
   */
  public async getStatus(sessionId: string): Promise<ApiResponse<ReplaySessionStatus>> {
    return apiClient.get<ReplaySessionStatus>(
      `/replay/${encodeURIComponent(sessionId)}/status`
    );
  }

  public async startReplay(
    sessionId: string,
    fps?: number
  ): Promise<ApiResponse<ReplaySessionStatus>> {
    return apiClient.post<ReplaySessionStatus>(
      `/replay/${encodeURIComponent(sessionId)}/start`,
      { fps: fps ?? 10.0 }
    );
  }

  public async pauseReplay(sessionId: string): Promise<ApiResponse<ReplaySessionStatus>> {
    return apiClient.post<ReplaySessionStatus>(
      `/replay/${encodeURIComponent(sessionId)}/pause`,
      {}
    );
  }

  public async stopReplay(sessionId: string): Promise<ApiResponse<ReplaySessionStatus>> {
    return apiClient.post<ReplaySessionStatus>(
      `/replay/${encodeURIComponent(sessionId)}/stop`,
      {}
    );
  }

  public async seekReplay(
    sessionId: string,
    frameIndex: number
  ): Promise<ApiResponse<ReplaySessionStatus>> {
    return apiClient.post<ReplaySessionStatus>(
      `/replay/${encodeURIComponent(sessionId)}/seek`,
      { frame_index: frameIndex }
    );
  }

  public async getNextFrame(sessionId: string): Promise<ApiResponse<ReplayFrameStreamPayload>> {
    return apiClient.get<ReplayFrameStreamPayload>(
      `/replay/${encodeURIComponent(sessionId)}/next-frame`
    );
  }

  /**
   * Look up a single cell from the session-global fused map.  Used to keep a
   * selected cell inspectable across replay frames.
   */
  public async getGlobalCell(sessionId: string, cellKey: string): Promise<ApiResponse<any>> {
    return apiClient.get<any>(
      `/replay/${encodeURIComponent(sessionId)}/global-cells/${encodeURIComponent(cellKey)}`
    );
  }

  /**
   * Start background precomputation of all sequence frames for 60 FPS playback
   */
  public async triggerPrecompute(
    sessionId: string
  ): Promise<ApiResponse<ReplayPrecomputeStatus>> {
    return apiClient.post<ReplayPrecomputeStatus>(
      `/replay/${encodeURIComponent(sessionId)}/precompute`,
      {}
    );
  }

  /**
   * Ingest sequence ZIP archive directly from local file path (zero HTTP upload overhead)
   */
  public async ingestArchive(
    archivePath: string,
    sessionId?: string,
    playbackMode: 'offline_precomputed_replay' | 'live_processing' = 'offline_precomputed_replay'
  ): Promise<ApiResponse<ReplayUploadResponse>> {
    return apiClient.post<ReplayUploadResponse>('/replay/ingest-archive', {
      archive_path: archivePath,
      session_id: sessionId,
      playback_mode: playbackMode,
    });
  }

  /**
   * Cancel in-progress sequence precomputation
   */
  public async cancelPrecompute(
    sessionId: string
  ): Promise<ApiResponse<{ session_id: string; cancelled: boolean }>> {
    return apiClient.post<{ session_id: string; cancelled: boolean }>(
      `/replay/${encodeURIComponent(sessionId)}/cancel-precompute`,
      {}
    );
  }

  /**
   * Get precomputation progress status
   */
  public async getPrecomputeStatus(
    sessionId: string
  ): Promise<ApiResponse<ReplayPrecomputeStatus>> {
    return apiClient.get<ReplayPrecomputeStatus>(
      `/replay/${encodeURIComponent(sessionId)}/precompute-status`
    );
  }

  /**
   * Connect to WebSocket live feed stream with automatic message parsing, control
   * helpers, AND automatic reconnect with exponential backoff.
   *
   * Behaviour:
   *   - On open, calls ``onConnected`` and starts sending frames to ``onFrame``.
   *   - On close (other than explicit close()), waits a backoff interval
   *     (1s, 2s, 4s, ... capped at 15s) and tries to reconnect.
   *   - On each (re)connect, the server emits a ``replay_status`` event which
   *     is forwarded to ``onStatus`` with ``resync: true``.  Callers can use
   *     this to re-hydrate their state without waiting for the next frame.
   */
  public connectWebSocketStream(
    sessionId: string,
    onFrame: (frame: ReplayFrameStreamPayload) => void,
    onStatus?: (status: ReplaySessionStatus, info: { resync: boolean }) => void,
    onError?: (err: any) => void,
    onConnectionChange?: (connected: boolean) => void
  ): ReplayWebSocketHandle {
    const baseUrl = apiClient.getBaseUrl();
    const wsProto = baseUrl.startsWith('https') ? 'wss:' : 'ws:';
    const hostAndPath = baseUrl.replace(/^https?:\/\//, '');
    const wsUrl = `${wsProto}//${hostAndPath}/replay/${encodeURIComponent(sessionId)}/stream`;

    let ws: WebSocket | null = null;
    let isClosedExplicitly = false;
    let isConnected = false;
    let attempt = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const notify = (conn: boolean) => {
      isConnected = conn;
      try { onConnectionChange?.(conn); } catch { /* ignore */ }
    };

    const open = () => {
      if (isClosedExplicitly) return;
      try {
        ws = new WebSocket(wsUrl);
      } catch (err) {
        if (onError) onError(err);
        scheduleReconnect();
        return;
      }

      ws.onopen = () => {
        attempt = 0;
        notify(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.event === 'status' && data.status && onStatus) {
            onStatus(data.status, { resync: false });
          } else if (data.event === 'replay_status' && data.status && onStatus) {
            onStatus(data.status, { resync: !!data.resync });
          } else if (data.session_id && data.points_sample) {
            onFrame(data as ReplayFrameStreamPayload);
          }
        } catch (e) {
          console.warn('Error parsing replay WebSocket frame:', e);
        }
      };

      ws.onerror = (err) => {
        if (onError && !isClosedExplicitly) {
          try { onError(err); } catch { /* ignore */ }
        }
      };

      ws.onclose = () => {
        notify(false);
        if (!isClosedExplicitly) {
          scheduleReconnect();
        }
      };
    };

    const scheduleReconnect = () => {
      if (isClosedExplicitly) return;
      attempt += 1;
      const backoffMs = Math.min(15000, 1000 * 2 ** Math.min(attempt - 1, 4));
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(open, backoffMs);
    };

    open();

    return {
      send: (msg: any) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
        }
      },
      close: () => {
        isClosedExplicitly = true;
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        if (ws) {
          try { ws.close(); } catch { /* ignore */ }
          ws = null;
        }
        notify(false);
      },
      isConnected: () => isConnected,
    };
  }

  /**
   * List all available replay sessions discovered by the backend.
   */
  public async listSessions(): Promise<ApiResponse<any[]>> {
    return apiClient.get<any[]>('/replay/sessions');
  }

  /**
   * Delete a replay session and clean up all cached data.
   */
  public async deleteSession(sessionId: string): Promise<ApiResponse<any>> {
    return apiClient.delete<any>(`/replay/${encodeURIComponent(sessionId)}`);
  }
}

export const replayService = new ReplayService();
