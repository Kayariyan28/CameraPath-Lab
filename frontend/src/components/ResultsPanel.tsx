import { useEffect, useRef } from 'react'
import { api } from '../api/client'
import type { ConfidenceReport, Job, OutputsListing, ShotTrajectory } from '../api/types'

interface Props {
  job: Job
  trajectory: ShotTrajectory
  outputs: OutputsListing | null
  cursorTime: number | null
}

const LEVEL_TONE: Record<string, string> = { high: 'ok', medium: 'warn', low: 'bad' }

/** The deliverables and the honest account of how far to trust them (spec §22-23, §33). */
export function ResultsPanel({ job, trajectory, outputs, cursorTime }: Props) {
  const proxy = outputs?.files.find((f) => f.name === 'motion_proxy.mp4')
  const player = useRef<HTMLVideoElement>(null)

  // Keep the proxy frame-locked to the reference timeline: the proxy starts at the
  // shot's first source frame, so its clock is offset by the shot start.
  useEffect(() => {
    const el = player.current
    if (!el || cursorTime === null || !el.paused) return
    const start = trajectory.poses[0]?.timestamp ?? 0
    const t = Math.max(0, cursorTime - start)
    if (Number.isFinite(t) && Math.abs(el.currentTime - t) > 0.02) el.currentTime = t
  }, [cursorTime, trajectory])

  return (
    <div className="stack" style={{ gap: 0 }}>
      <div className="section">
        <div className="section-title">
          Motion proxy
          <span className="spacer" />
          {proxy && <span className="badge mute">{(proxy.size_bytes / 1e6).toFixed(1)} MB</span>}
        </div>
        {proxy ? (
          <video ref={player} className="video-el" src={api.outputUrl(job.id, proxy.name)}
            controls muted playsInline loop key={proxy.size_bytes} />
        ) : (
          <div className="note tiny">
            {job.state === 'rendering' ? 'Rendering…' : 'No proxy rendered yet — exports below are still usable.'}
          </div>
        )}
        <div className="tiny dimmer" style={{ marginTop: 6 }}>
          Neutral geometry filmed by the recovered camera. No source pixels are included.
        </div>
      </div>

      <ConfidenceCard report={trajectory.confidence} />

      <div className="section">
        <div className="section-title">Camera move</div>
        <div className="tiny" style={{ lineHeight: 1.55 }}>{trajectory.summary}</div>
        {trajectory.classified_moves.length > 0 && (
          <div className="row" style={{ flexWrap: 'wrap', gap: 5, marginTop: 7 }}>
            {trajectory.classified_moves.map((m, i) => (
              <span key={i} className="badge mute" title={m.description}>
                {m.label.replace(/_/g, ' ')} {Math.round(m.strength * 100)}%
              </span>
            ))}
          </div>
        )}
        <dl className="kv" style={{ marginTop: 9 }}>
          <dt>solved by</dt><dd>{trajectory.pipeline_mode_used.replace(/_/g, ' ')}</dd>
          <dt>scale</dt><dd>{trajectory.scale_mode === 'metric' ? `metric (m)` : 'normalized — not metres'}</dd>
          <dt>path length</dt><dd>{trajectory.confidence.translation_observable ? `${trajectory.total_path_length.toFixed(2)} ${trajectory.scale_units === 'm' ? 'm' : 'u'}` : 'not observable'}</dd>
          <dt>total rotation</dt><dd>{trajectory.total_rotation_deg.toFixed(1)}°</dd>
          <dt>frames · fps</dt><dd>{trajectory.frame_count} · {trajectory.fps.toFixed(3)}</dd>
          <dt>fidelity</dt><dd>{trajectory.motion_fidelity}</dd>
        </dl>
      </div>

      <div className="section">
        <div className="section-title">Solvers</div>
        <div className="stack tight">
          {trajectory.solver_decisions.map((d, i) => (
            <div key={i} className="tiny">
              <span className={`badge ${d.selected ? 'ok' : d.succeeded ? 'mute' : 'bad'}`}>
                {d.solver}{d.selected ? ' · selected' : d.succeeded ? '' : ' · failed'}
              </span>{' '}
              <span className="dim mono">{d.duration_seconds.toFixed(1)}s</span>
              <div className="dimmer" style={{ marginTop: 2 }}>{d.message}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-title">Export</div>
        {outputs && outputs.files.length > 0 ? (
          <div className="stack tight">
            {outputs.files.map((f) => (
              <a key={f.name} className="output-link" href={api.outputUrl(job.id, f.name)} download={f.name}>
                <span className="mono">{f.name}</span>
                <span className="spacer" />
                <span className="dimmer tiny">{formatBytes(f.size_bytes)}</span>
              </a>
            ))}
          </div>
        ) : <div className="tiny dimmer">No exports yet</div>}
      </div>
    </div>
  )
}

function ConfidenceCard({ report }: { report: ConfidenceReport }) {
  const bar = (label: string, value: number, note?: string) => (
    <div style={{ marginBottom: 6 }}>
      <div className="row tiny"><span className="dim">{label}</span><span className="spacer" /><span className="mono">{value.toFixed(2)}</span></div>
      <div className="bar"><i style={{ width: `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%` }} /></div>
      {note && <div className="tiny dimmer">{note}</div>}
    </div>
  )
  return (
    <div className="section">
      <div className="section-title">
        Confidence
        <span className="spacer" />
        <span className={`badge ${LEVEL_TONE[report.level] ?? 'mute'}`}>{report.level.toUpperCase()} {report.score.toFixed(2)}</span>
      </div>
      <div className={`note ${LEVEL_TONE[report.level] ?? ''} tiny`} style={{ marginBottom: 9 }}>{report.headline}</div>
      {bar('rotation', report.rotation_confidence)}
      {bar('translation', report.translation_confidence, report.translation_observable ? undefined : 'not observable in this shot')}
      {bar('lens / zoom', report.zoom_confidence)}
      <ul className="reasons">
        {report.reasons.map((r, i) => <li key={i}>{r}</li>)}
      </ul>
      <dl className="kv" style={{ marginTop: 6 }}>
        <dt>registered</dt><dd>{(report.registered_frame_ratio * 100).toFixed(0)}%</dd>
        <dt>reprojection</dt><dd>{report.mean_reprojection_error === null ? '—' : `${report.mean_reprojection_error.toFixed(2)} px`}</dd>
        <dt>parallax</dt><dd>{report.baseline_parallax_score.toFixed(2)}</dd>
        <dt>temporal consistency</dt><dd>{report.temporal_consistency.toFixed(2)}</dd>
        <dt>solver agreement</dt><dd>{report.solver_agreement === null ? 'single solver' : report.solver_agreement.toFixed(2)}</dd>
      </dl>
    </div>
  )
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(1)} MB`
}
