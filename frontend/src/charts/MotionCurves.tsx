import { useMemo, useState } from 'react'
import type { MotionFrame, Shot } from '../api/types'
import { LineChart, type Series } from './LineChart'
import { px } from '../state/format'

interface Props {
  frames: MotionFrame[]
  shots: Shot[]
  cursorTime: number | null
  onScrub: (time: number) => void
}

type Tab = 'translation' | 'rotation' | 'zoom' | 'quality'

const TABS: { id: Tab; label: string }[] = [
  { id: 'translation', label: 'Translation' },
  { id: 'rotation', label: 'Rotation' },
  { id: 'zoom', label: 'Zoom / Radial' },
  { id: 'quality', label: 'Solve quality' },
]

/**
 * Per-frame motion curves.
 *
 * These plot the *image-space* signal, which is what has actually been measured
 * at this stage of the pipeline. The headings say so explicitly: showing a
 * "speed" curve here would imply a physical camera velocity that no geometric
 * solve has yet produced.
 */
export function MotionCurves({ frames, shots, cursorTime, onScrub }: Props) {
  const [tab, setTab] = useState<Tab>('translation')

  const times = useMemo(() => frames.map((f) => f.timestamp), [frames])
  const cutTimes = useMemo(() => shots.slice(1).map((s) => s.start_time), [shots])

  const series = useMemo<{ list: Series[]; symmetric: boolean; domain?: [number, number]; unit: string; note: string }>(() => {
    switch (tab) {
      case 'translation':
        return {
          list: [
            { values: frames.map((f) => f.dx_pixels), color: '#4d9fff', label: 'dx' },
            { values: frames.map((f) => f.dy_pixels), color: '#3ecf8e', label: 'dy' },
          ],
          symmetric: true,
          unit: '',
          note: 'Image-space displacement per frame, at analysis resolution. Positive dx = content moved right.',
        }
      case 'rotation':
        return {
          list: [{ values: frames.map((f) => f.rotation_deg), color: '#8b7fe8', label: 'roll' }],
          symmetric: true,
          unit: '°',
          note: 'In-plane image rotation per frame. Caused by camera roll, or by yaw/pitch on a tilted camera.',
        }
      case 'zoom':
        return {
          list: [
            { values: frames.map((f) => f.radial_flow), color: '#e5a13a', label: 'radial' },
            { values: frames.map((f) => (f.scale - 1) * 100), color: '#4d9fff', label: 'scale %' },
          ],
          symmetric: true,
          unit: '',
          note: 'Outward flow about the frame centre. A dolly-in and a zoom-in both produce this; separating them needs parallax.',
        }
      case 'quality':
        return {
          list: [
            { values: frames.map((f) => f.inlier_ratio), color: '#3ecf8e', label: 'inliers', fill: true },
            { values: frames.map((f) => f.confidence), color: '#4d9fff', label: 'confidence' },
            { values: frames.map((f) => f.dynamic_area_fraction), color: '#e5564b', label: 'dynamic' },
          ],
          symmetric: false,
          domain: [0, 1],
          unit: '',
          note: 'Fraction of tracks agreeing with the dominant model, per-transition confidence, and the share of frame area judged to be moving content.',
        }
    }
  }, [tab, frames])

  const atCursor = useMemo(() => {
    if (cursorTime === null || !frames.length) return null
    let best = frames[0]!
    let bestD = Infinity
    for (const f of frames) {
      const d = Math.abs(f.timestamp - cursorTime)
      if (d < bestD) { bestD = d; best = f }
    }
    return best
  }, [cursorTime, frames])

  if (!frames.length) {
    return (
      <div className="empty">
        <div>No motion data yet</div>
        <div className="tiny dimmer">Run analysis to compute per-frame motion curves</div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1 }}>
      <div className="row" style={{ padding: '6px 12px 0' }}>
        <div className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="spacer" />
        {atCursor && (
          <div className="tiny mono dim">
            f{atCursor.frame_index} · {atCursor.timestamp.toFixed(3)}s ·
            {' '}dx {px(atCursor.dx_pixels)} · dy {px(atCursor.dy_pixels)} ·
            {' '}{atCursor.rotation_deg.toFixed(2)}° ·
            {' '}inliers {(atCursor.inlier_ratio * 100).toFixed(0)}% ·
            {' '}{atCursor.model_used}
          </div>
        )}
      </div>

      <div className="chart-wrap" style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
        <div className="chart-title">
          {series.list.map((s) => (
            <span key={s.label} style={{ color: s.color }}>■ {s.label}</span>
          ))}
        </div>
        <LineChart
          series={series.list}
          times={times}
          height={116}
          symmetric={series.symmetric}
          domain={series.domain}
          cursorTime={cursorTime}
          cutTimes={cutTimes}
          unit={series.unit}
          onScrub={onScrub}
        />
        <div className="tiny dimmer" style={{ marginTop: 6 }}>{series.note}</div>
        {cutTimes.length > 0 && (
          <div className="tiny dimmer" style={{ marginTop: 3 }}>
            <span style={{ color: 'var(--bad)' }}>┆</span> hard cut — each shot is solved
            in its own coordinate system
          </div>
        )}
      </div>
    </div>
  )
}
