import { useState } from 'react'
import type { MotionFidelity, PipelineMode, ProxyStyle, ScaleMode, SolveSettings } from '../api/types'

interface Props {
  settings: Partial<SolveSettings>
  onChange: (patch: Partial<SolveSettings>) => void
  disabled?: boolean
  canRender: boolean
  canReconstruct3D: boolean
  vggtAvailable: boolean
}

const MODES: { id: PipelineMode; label: string; hint: string }[] = [
  { id: 'auto', label: 'Auto', hint: 'Picks the strongest pipeline each shot can actually support, and says why.' },
  { id: 'physical_3d', label: 'Physical 3D', hint: 'Force geometric reconstruction. Will report low confidence rather than invent translation where parallax is absent.' },
  { id: 'perceptual_match', label: 'Perceptual match', hint: 'Match screen-space motion instead of reconstructing a physical path. Right for pure pan, zoom, or distant scenery.' },
  { id: 'fast', label: 'Fast', hint: 'Lower analysis resolution and sparser keyframes. For previewing a move.' },
  { id: 'high_accuracy', label: 'High accuracy', hint: 'Higher resolution and denser keyframes. Slower.' },
]

const FIDELITY: { id: MotionFidelity; label: string; hint: string }[] = [
  { id: 'exact', label: 'Exact', hint: 'Preserve intentional micro-movement and handheld jitter. Default — this is reconstruction, not beautification.' },
  { id: 'clean', label: 'Clean', hint: 'Remove obvious solver noise only. Real camera shake survives.' },
  { id: 'smooth', label: 'Smooth', hint: 'Deliberate cinematic smoothing. This changes the move — handheld becomes gimbal-like.' },
]

export function Controls({
  settings, onChange, disabled, canRender, canReconstruct3D, vggtAvailable,
}: Props) {
  const [advanced, setAdvanced] = useState(false)
  const mode = settings.mode ?? 'auto'
  const fidelity = settings.motion_fidelity ?? 'exact'
  const scale = settings.scale_mode ?? 'normalized'

  const modeHint = MODES.find((m) => m.id === mode)?.hint
  const fidelityHint = FIDELITY.find((f) => f.id === fidelity)?.hint

  return (
    <>
      <div className="section">
        <div className="section-title">
          Accuracy
          <span className="spacer" />
          <button className="btn sm" onClick={() => setAdvanced((v) => !v)}>
            {advanced ? 'basic' : 'advanced'}
          </button>
        </div>
        <div className="stack tight">
          <label className="field">
            <span>Pipeline</span>
            <select
              value={mode}
              disabled={disabled}
              onChange={(e) => onChange({ mode: e.target.value as PipelineMode })}
            >
              {MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </label>
          {modeHint && <div className="tiny dimmer">{modeHint}</div>}

          <label className="field">
            <span>Motion fidelity</span>
            <select
              value={fidelity}
              disabled={disabled}
              onChange={(e) => onChange({ motion_fidelity: e.target.value as MotionFidelity })}
            >
              {FIDELITY.map((f) => <option key={f.id} value={f.id}>{f.label}</option>)}
            </select>
          </label>
          {fidelityHint && (
            <div className={`tiny ${fidelity === 'smooth' ? 'note warn' : 'dimmer'}`}>
              {fidelityHint}
            </div>
          )}
        </div>
      </div>

      {advanced && (
        <>
          <div className="section">
            <div className="section-title">Solve</div>
            <div className="stack tight">
              <label className="field">
                <span>Max analysis resolution</span>
                <select
                  value={settings.max_analysis_resolution ?? 0}
                  disabled={disabled}
                  onChange={(e) => onChange({
                    max_analysis_resolution: Number(e.target.value) || null,
                  })}
                >
                  <option value={0}>Auto (from available memory)</option>
                  <option value={720}>720 px long edge</option>
                  <option value={1080}>1080 px</option>
                  <option value={1440}>1440 px</option>
                  <option value={2160}>2160 px</option>
                </select>
              </label>

              <label className="field">
                <span>
                  Keyframe density
                  <em className="mono dim">{(settings.keyframe_density ?? 1).toFixed(2)}×</em>
                </span>
                <input
                  type="range" min={0.25} max={4} step={0.25}
                  value={settings.keyframe_density ?? 1}
                  disabled={disabled}
                  onChange={(e) => onChange({ keyframe_density: Number(e.target.value) })}
                />
              </label>

              <label className="field">
                <span>
                  Dynamic rejection
                  <em className="mono dim">{(settings.dynamic_rejection_strength ?? 0.5).toFixed(2)}</em>
                </span>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={settings.dynamic_rejection_strength ?? 0.5}
                  disabled={disabled}
                  onChange={(e) => onChange({ dynamic_rejection_strength: Number(e.target.value) })}
                />
              </label>
              <div className="tiny dimmer">
                How aggressively features that disagree with the dominant scene model are
                excluded. Raise it when a moving subject fills the frame.
              </div>

              <label className="field">
                <span>
                  Confidence threshold
                  <em className="mono dim">{(settings.confidence_threshold ?? 0.35).toFixed(2)}</em>
                </span>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={settings.confidence_threshold ?? 0.35}
                  disabled={disabled}
                  onChange={(e) => onChange({ confidence_threshold: Number(e.target.value) })}
                />
              </label>
              <div className="tiny dimmer">
                Below this, Auto falls further down the solver ladder.
              </div>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Lens &amp; scale</div>
            <div className="stack tight">
              <label className="field">
                <span>Lens mode</span>
                <select
                  value={settings.lens_mode ?? 'auto'}
                  disabled={disabled}
                  onChange={(e) => onChange({ lens_mode: e.target.value as 'auto' | 'fixed' | 'variable' })}
                >
                  <option value="auto">Auto-detect zoom</option>
                  <option value="fixed">Fixed focal length</option>
                  <option value="variable">Variable (expect zoom)</option>
                </select>
              </label>

              <label className="field">
                <span>FOV override</span>
                <input
                  type="number" min={5} max={175} step={1}
                  placeholder="estimate from footage"
                  value={settings.fov_override_degrees ?? ''}
                  disabled={disabled}
                  onChange={(e) => onChange({
                    fov_override_degrees: e.target.value ? Number(e.target.value) : null,
                  })}
                />
              </label>

              <label className="field">
                <span>Scale</span>
                <select
                  value={scale}
                  disabled={disabled}
                  onChange={(e) => onChange({ scale_mode: e.target.value as ScaleMode })}
                >
                  <option value="normalized">Normalized (default)</option>
                  <option value="metric">Metric (needs calibration)</option>
                </select>
              </label>
              <div className={`tiny ${scale === 'metric' ? 'note warn' : 'dimmer'}`}>
                {scale === 'metric'
                  ? 'Metric output requires a real-world reference. Without one, translation cannot honestly be labelled in metres.'
                  : 'Monocular video has no absolute scale. Translation ratios and all timing are preserved; units are Blender units, not metres.'}
              </div>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Solvers</div>
            <div className="stack tight">
              <label className="row tiny">
                <input
                  type="checkbox"
                  checked={settings.enable_colmap ?? true}
                  disabled={disabled || !canReconstruct3D}
                  onChange={(e) => onChange({ enable_colmap: e.target.checked })}
                />
                <span>COLMAP geometric reconstruction</span>
              </label>
              {!canReconstruct3D && (
                <div className="tiny dimmer">pycolmap unavailable — OpenCV solver will be used</div>
              )}
              <label className="row tiny">
                <input
                  type="checkbox"
                  checked={settings.enable_vggt ?? false}
                  disabled={disabled || !vggtAvailable}
                  onChange={(e) => onChange({ enable_vggt: e.target.checked })}
                />
                <span>VGGT learned geometry</span>
              </label>
              {!vggtAvailable && (
                <div className="tiny dimmer">
                  Requires PyTorch. Install with
                  {' '}<code className="mono">./scripts/bootstrap_macos.sh --with-torch</code>
                </div>
              )}
            </div>
          </div>

          <div className="section">
            <div className="section-title">Output</div>
            <div className="stack tight">
              <label className="field">
                <span>Proxy scene</span>
                <select
                  value={settings.proxy_style ?? 'motion_cage'}
                  disabled={disabled}
                  onChange={(e) => onChange({ proxy_style: e.target.value as ProxyStyle })}
                >
                  <option value="motion_cage">Motion cage (recommended)</option>
                  <option value="depth_poles">Depth poles</option>
                  <option value="ground_grid">Ground grid</option>
                  <option value="minimal">Minimal</option>
                </select>
              </label>
              <label className="row tiny">
                <input
                  type="checkbox"
                  checked={settings.render_trajectory_preview ?? true}
                  disabled={disabled}
                  onChange={(e) => onChange({ render_trajectory_preview: e.target.checked })}
                />
                <span>Also render third-person trajectory preview</span>
              </label>
              {!canRender && (
                <div className="note warn">
                  Blender was not found, so MP4 rendering is unavailable. Analysis and
                  trajectory export still work.
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  )
}
