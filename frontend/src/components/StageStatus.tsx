import type { Job, JobEvent } from '../api/types'
import { stateTone, titleCase } from '../state/format'

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

/** Stages implemented so far. Shown differently from pending ones so the UI
 *  never implies a later phase already ran. */
const IMPLEMENTED = new Set(STAGES.slice(0, 3))

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
  const reachedIndex = STAGES.indexOf(stage)

  return (
    <div className="section">
      <div className="section-title">
        Pipeline
        <span className={`badge ${stateTone(job.state)}`}>{titleCase(job.state)}</span>
        <span className="spacer" />
        {busy && <span className="tiny mono dim">{Math.round(progress * 100)}%</span>}
      </div>

      <div className={`bar ${job.state === 'failed' ? 'bad' : job.state === 'analyzed' ? 'ok' : ''}`}>
        <i style={{ width: `${Math.round((job.state === 'analyzed' ? 1 : progress) * 100)}%` }} />
      </div>

      <div className="stack tight" style={{ marginTop: 9 }}>
        {STAGES.map((s, i) => {
          const done = reachedIndex > i || job.state === 'analyzed' && IMPLEMENTED.has(s)
          const active = busy && s === stage
          const implemented = IMPLEMENTED.has(s)
          return (
            <div key={s} className="row tiny" style={{ opacity: implemented ? 1 : 0.42 }}>
              <span style={{
                width: 12, textAlign: 'center',
                color: active ? 'var(--accent)' : done ? 'var(--ok)' : 'var(--fg-3)',
              }}>
                {active ? '▸' : done ? '✓' : '·'}
              </span>
              <span style={{ color: active ? 'var(--fg-0)' : done ? 'var(--fg-1)' : 'var(--fg-3)' }}>
                {s}
              </span>
              <span className="spacer" />
              {!implemented && <span className="tiny dimmer mono">phase 2+</span>}
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
