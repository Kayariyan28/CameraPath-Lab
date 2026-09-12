import type { AnalysisResult, ShotAnalysis } from '../api/types'
import { complexityTone, degrees, percent, px, scoreTone, titleCase } from '../state/format'

interface Props {
  analysis: AnalysisResult
  selectedShot: number
  onSelectShot: (id: number) => void
}

export function AnalysisPanel({ analysis, selectedShot, onSelectShot }: Props) {
  const current = analysis.shot_analyses.find((s) => s.shot.id === selectedShot)
    ?? analysis.shot_analyses[0]

  return (
    <div className="stack" style={{ gap: 0 }}>
      <ShotList
        analyses={analysis.shot_analyses}
        cutCount={analysis.cut_candidates.filter((c) => c.accepted).length}
        rejectedCount={analysis.cut_candidates.filter((c) => !c.accepted).length}
        selected={selectedShot}
        onSelect={onSelectShot}
      />
      {current && <ShotDetail analysis={current} />}
      {analysis.warnings.length > 0 && <Warnings warnings={analysis.warnings} />}
      <CutDiagnostics analysis={analysis} />
    </div>
  )
}

function ShotList({ analyses, cutCount, rejectedCount, selected, onSelect }: {
  analyses: ShotAnalysis[]
  cutCount: number
  rejectedCount: number
  selected: number
  onSelect: (id: number) => void
}) {
  return (
    <div className="section">
      <div className="section-title">
        Shots
        <span className="badge mute">{analyses.length}</span>
        <span className="spacer" />
        {cutCount > 0 && <span className="tiny dimmer">{cutCount} cut{cutCount > 1 ? 's' : ''}</span>}
      </div>
      <div className="stack tight">
        {analyses.map((a) => (
          <div
            key={a.shot.id}
            className={`shot ${a.shot.id === selected ? 'selected' : ''}`}
            onClick={() => onSelect(a.shot.id)}
          >
            <div className="shot-head">
              <span className="shot-name">Shot {a.shot.id + 1}</span>
              <span className={`badge ${complexityTone(a.complexity)}`}>{a.complexity}</span>
              <span className="spacer" />
              <span className="tiny dimmer mono">
                {a.shot.start_time.toFixed(2)}–{a.shot.end_time.toFixed(2)}s
              </span>
            </div>
            <div className="tiny dim mono">
              f{a.shot.start_frame}–{a.shot.end_frame} · {a.shot.end_frame - a.shot.start_frame + 1} frames
            </div>
          </div>
        ))}
      </div>
      {rejectedCount > 0 && (
        <div className="tiny dimmer" style={{ marginTop: 8 }}>
          {rejectedCount} appearance spike{rejectedCount > 1 ? 's' : ''} examined and
          rejected as camera motion rather than a cut.
        </div>
      )}
    </div>
  )
}

function ShotDetail({ analysis }: { analysis: ShotAnalysis }) {
  const s = analysis.signature
  const translationObservable = s.parallax_score >= 0.15

  return (
    <>
      <div className="section">
        <div className="section-title">Measured image motion</div>
        <div className="metric-grid">
          <Metric
            label="Mean flow"
            value={px(s.mean_flow_magnitude)}
            note={`peak ${px(s.peak_flow_magnitude, 0)}`}
          />
          <Metric
            label="Model agreement"
            value={percent(s.mean_inlier_ratio)}
            tone={scoreTone(s.mean_inlier_ratio, 0.6, 0.35)}
            note="tracks fitting one model"
          />
          <Metric
            label="Net displacement"
            value={px(Math.hypot(s.net_dx_pixels, s.net_dy_pixels), 0)}
            note={`dx ${s.net_dx_pixels.toFixed(0)} · dy ${s.net_dy_pixels.toFixed(0)}`}
          />
          <Metric
            label="Image rotation"
            value={degrees(s.total_image_rotation_deg, 1)}
            note="cumulative in-plane"
          />
          <Metric
            label="Scale change"
            value={`${s.net_scale_change.toFixed(3)}×`}
            note={s.net_scale_change > 1.01 ? 'content expanding'
              : s.net_scale_change < 0.99 ? 'content contracting'
              : 'no net scale change'}
          />
          <Metric
            label="Jitter"
            value={s.jitter_score.toFixed(2)}
            tone={s.jitter_score > 0.25 ? 'warn' : 'mute'}
            note={s.jitter_score > 0.25 ? 'handheld character' : 'smooth'}
          />
        </div>
      </div>

      <div className="section">
        <div className="section-title">
          Observability
          <span className={`badge ${translationObservable ? 'ok' : 'warn'}`}>
            {translationObservable ? 'translation observable' : 'rotation / zoom only'}
          </span>
        </div>
        <div className="metric-grid">
          <Metric
            label="Parallax"
            value={s.parallax_score.toFixed(2)}
            tone={scoreTone(s.parallax_score, 0.3, 0.15)}
            note="depth-dependent flow"
          />
          <Metric
            label="Homography fit"
            value={percent(s.homography_dominance)}
            tone={s.homography_dominance > 0.85 ? 'warn' : 'mute'}
            note="planar / distant"
          />
          <Metric
            label="Texture"
            value={s.texture_score.toFixed(2)}
            tone={scoreTone(s.texture_score, 0.4, 0.2)}
            note="trackable detail"
          />
          <Metric
            label="Sharpness"
            value={s.blur_score.toFixed(2)}
            tone={scoreTone(s.blur_score, 0.4, 0.25)}
            note="1 = crisp"
          />
        </div>

        {!translationObservable && (
          <div className="note warn" style={{ marginTop: 9 }}>
            A single homography explains {percent(s.homography_dominance)} of this shot's
            motion. That means the scene is planar, far away, or the camera only rotated
            and zoomed — physical translation is <strong>not measurable</strong> here.
            Reporting a dolly would be a fabrication, so screen-space matching is used
            instead.
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">
          Pipeline routing
          <span className="badge info">{titleCase(analysis.recommended_mode)}</span>
        </div>
        <div className="note info">{analysis.recommendation_reason}</div>
        {analysis.complexity_reasons.length > 0 && (
          <ul className="tiny dim" style={{ margin: '9px 0 0', paddingLeft: 16 }}>
            {analysis.complexity_reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
        )}
      </div>
    </>
  )
}

function Warnings({ warnings }: { warnings: string[] }) {
  return (
    <div className="section">
      <div className="section-title">
        Notes <span className="badge warn">{warnings.length}</span>
      </div>
      <div className="stack tight">
        {warnings.map((w) => <div key={w} className="note warn">{w}</div>)}
      </div>
    </div>
  )
}

function CutDiagnostics({ analysis }: { analysis: AnalysisResult }) {
  if (!analysis.cut_candidates.length) return null
  return (
    <div className="section">
      <div className="section-title">Cut decisions</div>
      <div className="stack tight">
        {analysis.cut_candidates.map((c) => (
          <div key={c.frame_index} className="note" style={{
            borderLeftColor: c.accepted ? 'var(--bad)' : 'var(--line-strong)',
          }}>
            <div className="row tiny mono" style={{ marginBottom: 2 }}>
              <span className={`badge ${c.accepted ? 'bad' : 'mute'}`}>
                {c.accepted ? 'cut' : 'kept'}
              </span>
              <span className="dimmer">frame {c.frame_index} · {c.time_seconds.toFixed(3)}s</span>
            </div>
            <div className="tiny dim">{c.reason}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

function Metric({ label, value, note, tone }: {
  label: string; value: string; note?: string; tone?: string
}) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={tone && tone !== 'mute' ? { color: `var(--${tone})` } : undefined}>
        {value}
      </div>
      {note && <div className="metric-note">{note}</div>}
    </div>
  )
}
