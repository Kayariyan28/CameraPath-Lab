import type {
  AnalysisResult, EnvironmentInfo, HealthInfo, Job, JobEvent, OutputsListing,
  ShotMotionPayload, ShotTrajectory, SolveSettings,
} from './types'

const BASE = '/api'

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    // FastAPI returns {detail: ...}; surface that rather than a bare status.
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) {
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  health: () => request<HealthInfo>('/system/health'),
  environment: () => request<EnvironmentInfo>('/system/environment'),
  capabilities: () => request<Record<string, unknown>>('/system/capabilities'),

  createJob: () => request<Job>('/jobs', { method: 'POST' }),
  listJobs: () => request<unknown[]>('/jobs'),
  getJob: (id: string) => request<Job>(`/jobs/${id}`),
  deleteJob: (id: string) => request<{ deleted: boolean }>(`/jobs/${id}`, { method: 'DELETE' }),

  /** Upload with real progress — fetch cannot report request progress, so XHR. */
  uploadVideo: (
    id: string,
    file: File,
    onProgress?: (fraction: number) => void,
  ): Promise<Job> =>
    new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/jobs/${id}/video`)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as Job)
          } catch (err) {
            reject(new ApiError(`malformed response: ${String(err)}`, xhr.status))
          }
        } else {
          let detail = `${xhr.status} ${xhr.statusText}`
          try {
            const body = JSON.parse(xhr.responseText)
            if (body?.detail) detail = body.detail
          } catch { /* non-JSON */ }
          reject(new ApiError(detail, xhr.status))
        }
      }
      xhr.onerror = () => reject(new ApiError('network error during upload', 0))
      xhr.onabort = () => reject(new ApiError('upload cancelled', 0))
      xhr.send(form)
    }),

  analyze: (id: string, settings?: Partial<SolveSettings>) =>
    request<{ job_id: string; started: boolean }>(`/jobs/${id}/analyze`, {
      method: 'POST',
      body: JSON.stringify(settings ?? {}),
    }),

  /** Geometry, fusion, validation and exports — plus the proxy render unless
   *  `render` is false. */
  solve: (id: string, settings?: Partial<SolveSettings>, render = true) =>
    request<{ job_id: string; started: boolean }>(`/jobs/${id}/solve?render=${render}`, {
      method: 'POST',
      body: JSON.stringify(settings ?? {}),
    }),

  /** Re-render only; the backend reads the exported trajectory, so no geometry
   *  is recomputed. Only output-side settings are applied. */
  render: (id: string, settings?: Partial<SolveSettings>) =>
    request<{ job_id: string; started: boolean }>(`/jobs/${id}/render`, {
      method: 'POST',
      body: JSON.stringify(settings ?? {}),
    }),

  trajectory: (id: string) =>
    request<{ job_id: string; shots: ShotTrajectory[] }>(`/jobs/${id}/trajectory`),
  outputs: (id: string) => request<OutputsListing>(`/jobs/${id}/outputs`),
  outputUrl: (id: string, name: string) => `${BASE}/jobs/${id}/outputs/${encodeURIComponent(name)}`,

  cancel: (id: string) =>
    request<{ cancelling: boolean }>(`/jobs/${id}/cancel`, { method: 'POST' }),

  analysis: (id: string) => request<AnalysisResult>(`/jobs/${id}/analysis`),
  motion: (id: string, shotId: number) =>
    request<ShotMotionPayload>(`/jobs/${id}/motion/${shotId}`),
  log: (id: string) => request<{ lines: string[] }>(`/jobs/${id}/log`),

  frameUrl: (id: string, frameIndex: number, width = 640) =>
    `${BASE}/jobs/${id}/frame/${frameIndex}?width=${width}`,
  sourceUrl: (id: string) => `${BASE}/jobs/${id}/source`,
}

/**
 * Subscribe to a job's event stream.
 *
 * EventSource reconnects on its own, and the backend replays its backlog on
 * connect, so a dropped connection self-heals without losing events.
 */
export function subscribeToJob(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onError?: (err: Event) => void,
): () => void {
  const source = new EventSource(`${BASE}/jobs/${jobId}/events`)
  source.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as JobEvent)
    } catch {
      /* ignore malformed frame */
    }
  }
  source.onerror = (err) => onError?.(err)
  return () => source.close()
}
