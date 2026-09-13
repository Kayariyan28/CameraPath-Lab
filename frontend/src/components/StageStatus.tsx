import type { Job, JobEvent } from '../api/types'
import { stateTone, titleCase } from '../state/format'

/** Exactly the stage strings the backend emits (spec §2.6). */
const STAGES = [
  'Preparing frames',
  'Tracking features',
  'Estimating 2D motion',
  'Estimating camera geometry',
  'Optimizing camera poses',
  'Recovering lens motion',
  'Validating trajectory',
  'Creating Blender scene',
  'Rendering MP4',
]

/** How many leading stages a settled job state has completed. */
const COMPLETED_BY_STATE: Record<string, number> = {
  analyzed: 3,
  solved: 7,
  complete: STAGES.length,
}

/** Stages a running state has certainly finished, before the live stage says more. */
const FLOOR_BY_STATE: Record<string, number> = {
  analyzing: 0,
  solving: 3,
  rendering: 7,
}

interface Props {
  job: Job | null
  stage: string
  progress: number
  busy: boolean
  events: JobEvent[]
}

export function StageStatus({ job, stage, progress, busy, events }: Props) {
  if (!job) return null
  const latest = [...events].reverse().find((e) => e.kind === 'log' || e.kind === 'progress')
  const settled = COMPLETED_BY_STATE[job.state]
  const live = STAGES.indexOf(stage)
  const completed = settled ?? Math.max(FLOOR_BY_STATE[job.state] ?? 0, live)
  const full = settled !== undefined && job.state !== 'failed'

  return (
    <div className="section">
      <div className="section-title">
        Pipeline
        <span className={`badge ${stateTone(job.state)}`}>{titleCase(job.state)}</span>
        <span className="spacer" />
        {busy && <span className="tiny mono dim">{Math.round(progress * 100)}%</span>}
      </div>

      <div className={`bar ${job.state === 'failed' ? 'bad' : full ? 'ok' : ''}`}>
        <i style={{ width: `${Math.round((full ? 1 : progress) * 100)}%` }} />
      </div>

      <div className="stack tight" style={{ marginTop: 9 }}>
        {STAGES.map((s, i) => {
          const active = busy && s === stage
          const done = !active && i < completed
          return (
            <div key={s} className="row tiny">
              <span style={{
                width: 12, textAlign: 'center',
                color: active ? 'var(--accent)' : done ? 'var(--ok)' : 'var(--fg-3)',
              }}>
                {active ? '▸' : done ? '✓' : '·'}
              </span>
              <span style={{ color: active ? 'var(--fg-0)' : done ? 'var(--fg-1)' : 'var(--fg-3)' }}>
                {s}
              </span>
            </div>
          )
        })}
      </div>

      {busy && latest?.message && (
        <div className="tiny dim mono" style={{ marginTop: 8 }}>{latest.message}</div>
      )}

      {job.error && (
        <div className="note bad" style={{ marginTop: 9 }}>
          <strong>Failed:</strong> {job.error}
        </div>
      )}
    </div>
  )
}
