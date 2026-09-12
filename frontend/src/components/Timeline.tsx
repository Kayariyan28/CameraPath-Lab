import type { Shot } from '../api/types'
import { timecode } from '../state/format'

interface Props {
  duration: number
  shots: Shot[]
  cursorTime: number | null
  fps: number
  onScrub: (time: number) => void
}

const SHOT_TINTS = ['#4d9fff', '#3ecf8e', '#e5a13a', '#8b7fe8', '#e5564b', '#46c8d8']

export function Timeline({ duration, shots, cursorTime, fps, onScrub }: Props) {
  const span = Math.max(duration, 1e-6)

  const handle = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
    onScrub(frac * span)
  }

  return (
    <div>
      <div
        className="timeline"
        onMouseDown={handle}
        onMouseMove={(e) => { if (e.buttons === 1) handle(e) }}
      >
        <div className="timeline-track">
          {shots.map((s, i) => (
            <div
              key={s.id}
              className="timeline-shot"
              title={`Shot ${s.id + 1}: ${s.start_time.toFixed(2)}–${s.end_time.toFixed(2)}s`}
              style={{
                left: `${(s.start_time / span) * 100}%`,
                width: `${((s.end_time - s.start_time) / span) * 100}%`,
                background: SHOT_TINTS[i % SHOT_TINTS.length],
                opacity: 0.55,
              }}
            />
          ))}
        </div>
        {cursorTime !== null && (
          <div className="timeline-head" style={{ left: `${(cursorTime / span) * 100}%` }} />
        )}
      </div>
      <div className="row tiny dimmer mono" style={{ padding: '0 12px 6px' }}>
        <span>00:00.00</span>
        <span className="spacer" />
        {cursorTime !== null && (
          <span style={{ color: 'var(--accent)' }}>
            {timecode(cursorTime, fps)} · frame {Math.round(cursorTime * fps)}
          </span>
        )}
        <span className="spacer" />
        <span>{timecode(duration, fps)}</span>
      </div>
    </div>
  )
}
