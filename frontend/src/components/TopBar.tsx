import { useEffect, useState } from 'react'
import { api } from '../api/client'
import type { EnvironmentInfo, HealthInfo } from '../api/types'

export function TopBar({ right }: { right?: React.ReactNode }) {
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const [env, setEnv] = useState<EnvironmentInfo | null>(null)
  const [open, setOpen] = useState(false)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    let live = true
    const poll = async () => {
      try {
        const [h, e] = await Promise.all([api.health(), api.environment()])
        if (!live) return
        setHealth(h); setEnv(e); setOffline(false)
      } catch {
        if (live) setOffline(true)
      }
    }
    void poll()
    const id = setInterval(poll, 20_000)
    return () => { live = false; clearInterval(id) }
  }, [])

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <span>CameraPath Lab</span>
          <span className="brand-sub">camera motion reconstruction</span>
        </div>
        <div className="spacer" />
        {right}
        {offline ? (
          <span className="badge bad">backend offline</span>
        ) : health ? (
          <button className="btn sm" onClick={() => setOpen((v) => !v)}>
            <Cap ok={health.can_ingest} label="ingest" />
            <Cap ok={health.can_reconstruct_3d} label="3D" />
            <Cap ok={health.can_render} label="render" />
            {env && <span className="tiny dimmer mono">{env.chip}</span>}
          </button>
        ) : (
          <span className="badge mute">connecting…</span>
        )}
      </header>
      {open && env && <EnvironmentSheet env={env} health={health} onClose={() => setOpen(false)} />}
    </>
  )
}

function Cap({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="tiny mono" style={{ color: ok ? 'var(--ok)' : 'var(--warn)' }}>
      {ok ? '●' : '○'}{label}
    </span>
  )
}

function EnvironmentSheet({ env, health, onClose }: {
  env: EnvironmentInfo; health: HealthInfo | null; onClose: () => void
}) {
  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        background: 'rgba(0,0,0,.55)', display: 'flex',
        alignItems: 'flex-start', justifyContent: 'flex-end', padding: 12,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 460, maxHeight: '90vh', overflowY: 'auto',
          background: 'var(--bg-1)', border: '1px solid var(--line-strong)',
          borderRadius: 'var(--r-lg)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pane-head" style={{ borderRadius: 'var(--r-lg) var(--r-lg) 0 0' }}>
          Machine &amp; toolchain
          <button className="btn sm" onClick={onClose}>close</button>
        </div>

        <div className="section">
          <div className="section-title">Host</div>
          <dl className="kv">
            <dt>Chip</dt><dd>{env.chip}</dd>
            <dt>Apple Silicon</dt><dd>{env.is_apple_silicon ? 'yes' : 'no'}</dd>
            <dt>Unified memory</dt>
            <dd>{env.total_memory_gb.toFixed(0)} GB ({env.available_memory_gb.toFixed(1)} free)</dd>
            <dt>Cores</dt>
            <dd>{env.cpu_cores_total} ({env.cpu_cores_performance}P/{env.cpu_cores_efficiency}E)</dd>
            <dt>Disk free</dt><dd>{env.disk_free_gb.toFixed(1)} GB</dd>
            <dt>macOS</dt><dd>{env.macos_version}</dd>
          </dl>
        </div>

        <div className="section">
          <div className="section-title">Tools</div>
          <div className="stack tight">
            {Object.entries(env.tools).map(([name, t]) => (
              <div key={name} className="row tiny">
                <span className={`badge ${t.available ? 'ok' : 'warn'}`}>{name}</span>
                <span className="dim mono" style={{
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {t.available ? t.version : t.detail}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="section">
          <div className="section-title">Python packages</div>
          <div className="row wrap" style={{ gap: 5 }}>
            {Object.entries(env.packages).map(([name, p]) => (
              <span key={name} className={`badge ${p.available ? 'ok' : 'mute'}`}>
                {name} {p.available ? p.version : '—'}
              </span>
            ))}
          </div>
        </div>

        <div className="section">
          <div className="section-title">Acceleration</div>
          <div className={`note ${env.mps_available ? 'info' : ''}`}>
            <strong>MPS:</strong> {env.mps_detail}
          </div>
        </div>

        <div className="section">
          <div className="section-title">Derived resource policy</div>
          <div className="tiny dimmer" style={{ marginBottom: 7 }}>
            {env.resource_policy.reason}
          </div>
          <dl className="kv">
            <dt>Analysis resolution</dt>
            <dd>{env.resource_policy.analysis_long_edge} px</dd>
            <dt>Geometry resolution</dt>
            <dd>{env.resource_policy.geometry_long_edge} px</dd>
            <dt>Decode chunk</dt>
            <dd>{env.resource_policy.decode_chunk_frames} frames</dd>
            <dt>Tracked features</dt>
            <dd>{env.resource_policy.max_tracked_features}</dd>
            <dt>VGGT window</dt>
            <dd>{env.resource_policy.vggt_window_frames} / stride {env.resource_policy.vggt_window_stride}</dd>
            <dt>Worker threads</dt>
            <dd>{env.resource_policy.worker_threads}</dd>
          </dl>
        </div>

        {(health?.warnings.length ?? 0) > 0 && (
          <div className="section">
            <div className="section-title">Notes</div>
            <div className="stack tight">
              {health!.warnings.map((w) => <div key={w} className="note warn">{w}</div>)}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
