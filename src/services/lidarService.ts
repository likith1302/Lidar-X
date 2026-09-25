/**
 * LiDAR Data Ingestion Service
 * Handles point cloud frame retrieval, binary upload, and sensor metadata.
 */

import { apiClient, ApiResponse } from './api';
import { LidarFrame, FrameMetadata } from '../types/lidar';
import { FrameUploadResponse } from '../types/terrain';
import { mockLidarFrame, mockFrameMetadata } from '../mocks/mockLidarData';

export class LidarService {
  /**
   * Upload a SemanticKITTI .bin LiDAR frame to backend
   */
  public async uploadFrameBin(
    file: File,
    frameId?: string
  ): Promise<ApiResponse<FrameUploadResponse>> {
    const queryParams: Record<string, string> = {};
    if (frameId) {
      queryParams['frame_id'] = frameId;
    }
    return apiClient.uploadFile<FrameUploadResponse>(
      '/frames/upload',
      file,
      'file',
      queryParams
    );
  }

  /**
   * Fetch LiDAR frame (from live backend or mock point cloud)
   */
  public async getFrame(frameId?: string): Promise<ApiResponse<LidarFrame>> {
    const targetId = frameId || 'latest';

    return apiClient.request<LidarFrame>(
      `/frames/${encodeURIComponent(targetId)}?sample_points=6000`,
      {},
      () => ({
        ...mockLidarFrame,
        frameId: targetId,
      })
    ).then((res) => {
      // If live backend returned FrameDetailsResponse shape, adapt it to LidarFrame
      if (!res.isMock && (res.data as any).sample_points) {
        const details = res.data as any;
        const adaptedPoints = (details.sample_points || []).map((p: any) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          intensity: p.intensity ?? 0.5,
          semanticClass: 'unknown',
        }));

        return {
          ...res,
          data: {
            frameId: details.frame_id,
            sequenceId: 'Uploaded Sequence',
            points: adaptedPoints,
            metadata: {
              scanSource: details.metadata?.scan_source || 'Live Backend Ingestion',
              timestamp: details.created_at || details.metadata?.timestamp || new Date().toISOString(),
              calibration: 'Sensor Coordinate Frame',
              labels: 'Ready for Geometric Terrain & AI Analysis',
              vehiclePose: 'Sensor Origin Body Frame [0.0, 0.0, 0.0]',
              sensorConfiguration: details.metadata?.format || 'SemanticKITTI float32',
              beamCount: `${details.point_count} pts total`,
              horizontalFov: '360 deg Continuous Sweep',
              rangeCapability: `${Math.round(details.bounds?.max_x || 80)}m Radial Radius`,
            },
          },
        };
      }
      return res;
    });
  }

  /**
   * Fetch sensor calibration and frame metadata
   */
  public async getFrameMetadata(frameId?: string): Promise<ApiResponse<FrameMetadata>> {
    return apiClient.request<FrameMetadata>(
      `/frames/${frameId || 'latest'}`,
      {},
      () => mockFrameMetadata
    );
  }

  /**
   * List all stored frame IDs from backend
   */
  public async listFrames(): Promise<ApiResponse<string[]>> {
    return apiClient.get<string[]>(
      '/frames/',
      () => [
        'FRAME_0001 (Mock)',
        'FRAME_0002 (Mock)',
        'FRAME_0003 (Mock)',
      ]
    );
  }

  /**
   * Available synthetic sequences
   */
  public getAvailableSequences(): string[] {
    return [
      'SEQ_CAMPUS_AUTONOMOUS_01 (Mock)',
      'SEQ_URBAN_INTERSECTION_02 (Mock)',
      'SEQ_HIGHWAY_CORRIDOR_03 (Mock)',
    ];
  }
}

export const lidarService = new LidarService();
