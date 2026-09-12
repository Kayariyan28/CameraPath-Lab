import { useCallback, useEffect, useRef, useState } from 'react'
import { api, subscribeToJob } from '../api/client'
import type { Job, JobEvent, ShotMotionPayload, SolveSettings } from '../api/types'

export interface JobController {
  job: Job | null
  events: JobEvent[]
  progress: number
  stage: string
  busy: boolean
  uploadFraction: number | null
  error: string | null
  motion: Record<number, ShotMotionPayload>

  upload: (file: File) => Promise<void>
  analyze: (settings?: Partial<SolveSettings>) => Promise<void>
  cancel: () => Promise<void>
  reset: () => void
  loadMotion: (shotId: number) => Promise<void>
}

const TERMINAL: ReadonlySet<string> = new Set([
  'analyzed', 'solved', 'complete', 'failed', 'cancelled',
])

export function useJob(): JobController {
  const [job, setJob] = useState<Job | null>(null)
  const [events, setEvents] = useState<JobEvent[]>([])
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploadFraction, setUploadFraction] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [motion, setMotion] = useState<Record<number, ShotMotionPayload>>({})

  const unsubscribe = useRef<(() => void) | null>(null)
  const jobId = job?.id ?? null

  // Re-fetch the job record whenever the backend says its state changed. The
  // event stream carries progress; the job record carries results.
  const refresh = useCallback(async (id: string) => {
    try {
      setJob(await api.getJob(id))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  useEffect(() => {
    if (!jobId) return
    unsubscribe.current?.()
    unsubscribe.current = subscribeToJob(jobId, (event) => {
      setEvents((prev) => (prev.length > 600 ? [...prev.slice(-600), event] : [...prev, event]))

      if (typeof event.progress === 'number') setProgress(event.progress)
      if (event.stage) setStage(event.stage)

      if (event.kind === 'error') setError(event.message)

      if (event.kind === 'state') {
        const next = String(event.data?.state ?? '')
        void refresh(jobId)
        if (TERMINAL.has(next)) setBusy(false)
      }
      if (event.kind === 'done') {
        setBusy(false)
        void refresh(jobId)
      }
    })
    return () => {
      unsubscribe.current?.()
      unsubscribe.current = null
    }
  }, [jobId, refresh])

  const upload = useCallback(async (file: File) => {
    setError(null)
    setEvents([])
    setProgress(0)
    setStage('')
    setMotion({})
    setUploadFraction(0)
    try {
      // A fresh job per upload keeps each source video's artefacts isolated.
      const created = await api.createJob()
      setJob(created)
      const withVideo = await api.uploadVideo(created.id, file, setUploadFraction)
      setJob(withVideo)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setUploadFraction(null)
    }
  }, [])

  const analyze = useCallback(async (settings?: Partial<SolveSettings>) => {
    if (!jobId) return
    setError(null)
    setBusy(true)
    setProgress(0)
    setMotion({})
    try {
      await api.analyze(jobId, settings)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }, [jobId])

  const cancel = useCallback(async () => {
    if (!jobId) return
    try {
      await api.cancel(jobId)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [jobId])

  const loadMotion = useCallback(async (shotId: number) => {
    if (!jobId) return
    setMotion((prev) => {
      if (prev[shotId]) return prev
      void api.motion(jobId, shotId)
        .then((payload) => setMotion((p) => ({ ...p, [shotId]: payload })))
        .catch(() => { /* absent motion data is not an error worth surfacing */ })
      return prev
    })
  }, [jobId])

  const reset = useCallback(() => {
    unsubscribe.current?.()
    unsubscribe.current = null
    setJob(null); setEvents([]); setProgress(0); setStage('')
    setBusy(false); setError(null); setMotion({}); setUploadFraction(null)
  }, [])

  return {
    job, events, progress, stage, busy, uploadFraction, error, motion,
    upload, analyze, cancel, reset, loadMotion,
  }
}
