/**
 * Base API Service Layer
 * Provides typed HTTP communication with configurable mock mode and live backend connectivity.
 */

export interface ApiResponse<T> {
  success: boolean;
  data: T;
  isMock: boolean;
  message?: string;
  timestamp: string;
}

class ApiClient {
  private baseUrl: string;
  private mockMode: boolean;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';
    
    // Explicit VITE_USE_MOCKS takes priority; default to REAL backend mode (false)
    if (import.meta.env.VITE_USE_MOCKS !== undefined) {
      this.mockMode = import.meta.env.VITE_USE_MOCKS === 'true';
    } else if (import.meta.env.VITE_ENABLE_MOCK_FALLBACK !== undefined) {
      this.mockMode = import.meta.env.VITE_ENABLE_MOCK_FALLBACK === 'true';
    } else {
      // Real backend data mode by default
      this.mockMode = false;
    }
  }

  public getBaseUrl(): string {
    return this.baseUrl;
  }

  public isMockMode(): boolean {
    return this.mockMode;
  }

  public setMockMode(enabled: boolean): void {
    this.mockMode = enabled;
  }

  public buildUrl(endpoint: string): string {
    if (endpoint.startsWith('http://') || endpoint.startsWith('https://')) {
      return endpoint;
    }
    const cleanBase = this.baseUrl.replace(/\/+$/, '');
    let cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    if (cleanBase.endsWith('/api/v1') && cleanEndpoint.startsWith('/api/v1/')) {
      cleanEndpoint = cleanEndpoint.replace(/^\/api\/v1/, '');
    }
    return `${cleanBase}${cleanEndpoint}`;
  }

  /**
   * Generic request wrapper.
   * If mockMode is false, attempts live fetch and throws on disconnection.
   * If mockMode is true, attempts live fetch first; if unreachable, falls back to mockDataFactory.
   */
  public async request<T>(
    endpoint: string,
    options?: RequestInit,
    mockDataFactory?: () => T
  ): Promise<ApiResponse<T>> {
    const url = this.buildUrl(endpoint);

    // 1. If mock mode is explicitly disabled, attempt live backend only
    if (!this.mockMode) {
      try {
        const response = await fetch(url, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...(options?.headers || {}),
          },
        });

        if (!response.ok) {
          const errorJson = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorJson.detail || `HTTP Error ${response.status}: ${response.statusText}`);
        }

        const data: T = await response.json();
        return {
          success: true,
          data,
          isMock: false,
          timestamp: new Date().toISOString(),
        };
      } catch (err: any) {
        throw new Error(
          `[Disconnected] Backend service unreachable at ${url}. ` +
          `Ensure FastAPI server is running on ${this.baseUrl} or enable mock mode. Error: ${err?.message || err}`
        );
      }
    }

    // 2. In mock mode, attempt live fetch first if network is reachable, else fall back gracefully
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(options?.headers || {}),
        },
      });

      if (response.ok) {
        const data: T = await response.json();
        return {
          success: true,
          data,
          isMock: false,
          timestamp: new Date().toISOString(),
        };
      }
    } catch {
      // Live backend unreachable - proceed to mock data fallback
    }

    if (mockDataFactory) {
      return {
        success: true,
        data: mockDataFactory(),
        isMock: true,
        message: 'Operating in frontend mock mode. Live backend integration ready.',
        timestamp: new Date().toISOString(),
      };
    }

    throw new Error(`Endpoint ${endpoint} unreachable and no mock fallback provided.`);
  }

  /**
   * Multipart file upload wrapper for binary LiDAR frames (.bin)
   */
  public async uploadFile<T>(
    endpoint: string,
    file: File,
    fieldName: string = 'file',
    queryParams?: Record<string, string>
  ): Promise<ApiResponse<T>> {
    let url = this.buildUrl(endpoint);
    if (queryParams) {
      const qs = new URLSearchParams(queryParams).toString();
      url += `?${qs}`;
    }

    const formData = new FormData();
    formData.append(fieldName, file);

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorJson.detail || `Upload failed: ${response.status} ${response.statusText}`);
      }

      const data: T = await response.json();
      return {
        success: true,
        data,
        isMock: false,
        timestamp: new Date().toISOString(),
      };
    } catch (err: any) {
      throw new Error(
        `Failed to upload file to ${url}. Ensure backend is running. Error: ${err?.message || err}`
      );
    }
  }

  /**
   * Helper for POST JSON requests
   */
  public async post<T>(endpoint: string, body: any, mockDataFactory?: () => T): Promise<ApiResponse<T>> {
    return this.request<T>(
      endpoint,
      {
        method: 'POST',
        body: JSON.stringify(body),
      },
      mockDataFactory
    );
  }

  /**
   * Helper for GET requests
   */
  public async get<T>(endpoint: string, mockDataFactory?: () => T): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'GET' }, mockDataFactory);
  }

  /**
   * Helper for DELETE requests
   */
  public async delete<T>(endpoint: string, mockDataFactory?: () => T): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, { method: 'DELETE' }, mockDataFactory);
  }
}

export const apiClient = new ApiClient();
