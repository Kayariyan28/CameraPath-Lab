import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api/client'
import type { HealthInfo, SolveSettings } from './api/types'
import { KinematicsCurves } from './charts/KinematicsCurves'
import { MotionCurves } from './charts/MotionCurves'
import { AnalysisPanel } from './components/AnalysisPanel'
import { Controls } from './components/Controls'
import { Dropzone, FileSummary } from './components/Dropzone'
import { LogPanel } from './components/LogPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { StageStatus } from './components/StageStatus'
import { Timeline } from './components/Timeline'
import { TopBar } from './components/TopBar'
import { VideoInfoPanel } from './components/VideoInfoPanel'
import { useJob } from './state/useJob'
import { MotionPathView } from './trajectory/MotionPathView'
import { TrajectoryView } from './trajectory/TrajectoryView'

type BottomTab = 'kinematics' | 'curves' | 'log'
type RightTab = 'results' | 'analysis'

export function App() {
  const job = useJob()
  const [settings, setSettings] = useState<Partial<SolveSettings>>({
    mode: 'auto',
    motion_fidelity: 'exact',
    scale_mode: 'normalized',
    proxy_style: 'motion_cage',
  })
  const [selectedShot, setSelectedShot] = useState(0)
  const [cursorTime, setCursorTime] = useState<number | null>(null)
  const [bottomTab, setBottomTab] = useState<BottomTab>('curves')
  const [rightTab, setRightTab] = useState<RightTab>('analysis')
  const [health, setHealth] = useState<HealthInfo | null>(null)
  const video = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    const linked = new URLSearchParams(window.location.search).get('job')
    if (linked) void job.open(linked)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Keep the URL pointing at the current job so a reload reopens it.
  useEffect(() => {
    const url = new URL(window.location.href)
    if (job.job?.id) url.searchParams.set('job', job.job.id)
    else url.searchParams.delete('job')
    window.history.replaceState(null, '', url)
  }, [job.job?.id])

  const analysis = job.job?.analysis ?? null
  const info = job.job?.video ?? null
  const shots = analysis?.shots ?? []
  const fps = info?.fps_average || 30

  // Pull motion curves for whichever shot is selected.
  useEffect(() => {
    if (analysis) void job.loadMotion(selectedShot)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis, selectedShot])

  useEffect(() => { setSelectedShot(0) }, [job.job?.id])

  const trajectory = job.job?.trajectories?.find((t) => t.shot_id === selectedShot) ?? null
  const hasTrajectory = (job.job?.trajectories?.length ?? 0) > 0

  // When a solve lands, show what it produced.
  useEffect(() => {
    if (hasTrajectory) { setRightTab('results'); setBottomTab('kinematics') }
  }, [hasTrajectory, job.job?.id])

  const motion = job.motion[selectedShot]
  const frames = motion?.motion_frames ?? []

  const scrub = (time: number) => {
    setCursorTime(time)
    if (video.current && Number.isFinite(time)) {
      video.current.currentTime = Math.max(0, time)
    }
  }

  const selectShot = (id: number) => {
    setSelectedShot(id)
    const shot = shots.find((s) => s.id === id)
    if (shot) scrub(shot.start_time)
  }

  const canAnalyze = Boolean(job.job?.video) && !job.busy
  const canGenerate = Boolean(analysis) && !job.busy
  const aspect = info ? info.width / Math.max(info.height, 1) : 16 / 9
  const analysisSize = useMemo(
    () => motion?.analysis_size ?? analysis?.analysis_resolution ?? [1080, 608],
    [motion, analysis],
  )

  return (
    <div className="app">
      <TopBar
        right={
          job.job && (
            <>
              {job.busy && (
                <button className="btn sm danger" onClick={() => void job.cancel()}>
                  cancel
                </button>
              )}
              <button className="btn sm" onClick={job.reset}>new</button>
            </>
          )
        }
      />

      <div className="main">
        {/* ------------------------------------------------ LEFT: reference */}
        <section className="pane">
          <div className="pane-head">
            Reference
            {info && <span className="badge mute">{info.width}×{info.height}</span>}
          </div>
          <div className="pane-body">
            {!job.job?.video ? (
              <Dropzone
                onFile={(f) => void job.upload(f)}
                uploadFraction={job.uploadFraction}
              />
            ) : (
              <>
                <div className="section">
                  <video
                    ref={video}
                    className="video-el"
                    src={api.sourceUrl(job.job.id)}
                    controls
                    muted
                    playsInline
                    onTimeUpdate={(e) => setCursorTime(e.currentTarget.currentTime)}
                  />
                  <div style={{ marginTop: 7 }}>
                    <FileSummary name={info!.filename} size={info!.size_bytes} />
                  </div>
                </div>

                <div className="section">
                  <div className="section-title">Container</div>
                  <VideoInfoPanel info={info!} />
                </div>

                <Controls
                  settings={settings}
                  onChange={(patch) => setSettings((s) => ({ ...s, ...patch }))}
                  disabled={job.busy}
                  canRender={health?.can_render ?? false}
                  canReconstruct3D={health?.can_reconstruct_3d ?? false}
                  vggtAvailable={health?.can_use_vggt ?? false}
                />

                <div className="section">
                  {analysis ? (
                    <>
                      <button
                        className="btn primary cta cta-generate"
                        disabled={!canGenerate}
                        onClick={() => void job.solve(settings)}
                      >
                        {job.busy ? 'Working…' : hasTrajectory ? 'REGENERATE MOTION REFERENCE' : 'GENERATE MOTION REFERENCE'}
                      </button>
                      <div className="row" style={{ marginTop: 7, gap: 6 }}>
                        <button className="btn sm" disabled={!canAnalyze} onClick={() => void job.analyze(settings)}>
                          Re-analyse
                        </button>
                        {hasTrajectory && (
                          <button className="btn sm" disabled={job.busy || !(health?.can_render ?? true)}
                            title="Re-render with the current proxy style and output size; geometry is not recomputed"
                            onClick={() => void job.render(settings)}>
                            Re-render proxy only
                          </button>
                        )}
                      </div>
                      <div className="tiny dimmer" style={{ marginTop: 7 }}>
                        Solves the camera, exports the trajectory and renders the neutral motion
                        proxy MP4. Timing follows the source exactly.
                      </div>
                    </>
                  ) : (
                    <button
                      className="btn primary cta"
                      disabled={!canAnalyze}
                      onClick={() => void job.analyze(settings)}
                    >
                      {job.busy ? 'Analysing…' : 'Analyse video'}
                    </button>
                  )}
                </div>
              </>
            )}

            {job.error && (
              <div className="note bad" style={{ margin: 12 }}>{job.error}</div>
            )}
          </div>
        </section>

        {/* --------------------------------------------- CENTRE: 3D viewport */}
        <section className="pane">
          <div className="pane-head">
            {trajectory ? 'Camera trajectory' : 'Motion path'}
            <span className="spacer" />
            {trajectory ? (
              <span className={`badge ${trajectory.pipeline_mode_used === 'physical_3d' ? 'ok' : 'warn'}`}>
                {trajectory.pipeline_mode_used === 'physical_3d' ? 'recovered 3D camera path' : 'perceptual match · screen-space equivalent'}
              </span>
            ) : frames.length > 0 && (
              <span className="badge warn">image-space · not yet a 3D camera path</span>
            )}
          </div>
          <div className="pane-body" style={{ overflow: 'hidden' }}>
            {trajectory ? (
              <TrajectoryView trajectory={trajectory} cursorTime={cursorTime} aspect={aspect} />
            ) : frames.length > 0 ? (
              <MotionPathView
                frames={frames}
                cursorTime={cursorTime}
                analysisSize={analysisSize}
              />
            ) : (
              <div className="empty">
                <div style={{ fontSize: 22, opacity: 0.35 }}>◈</div>
                <div>{job.job?.video ? 'Run analysis to see the motion path' : 'No video loaded'}</div>
                <div className="tiny dimmer" style={{ maxWidth: 380 }}>
                  After analysis this shows the measured image-space motion; after
                  Generate it shows the recovered camera trajectory with keyframe frustums.
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ------------------------------------------- RIGHT: camera analysis */}
        <section className="pane">
          <div className="pane-head">
            {hasTrajectory ? (
              <div className="tabs">
                <button className={`tab ${rightTab === 'results' ? 'active' : ''}`} onClick={() => setRightTab('results')}>Results</button>
                <button className={`tab ${rightTab === 'analysis' ? 'active' : ''}`} onClick={() => setRightTab('analysis')}>Analysis</button>
              </div>
            ) : 'Camera analysis'}
            <span className="spacer" />
            {analysis && rightTab === 'analysis' && (
              <span className="badge mute">
                {analysis.analysis_resolution[0]}×{analysis.analysis_resolution[1]} analysis
              </span>
            )}
          </div>
          <div className="pane-body">
            <StageStatus
              job={job.job}
              stage={job.stage}
              progress={job.progress}
              busy={job.busy}
              events={job.events}
            />
            {hasTrajectory && rightTab === 'results' && job.job && trajectory ? (
              <ResultsPanel job={job.job} trajectory={trajectory} outputs={job.outputs} cursorTime={cursorTime} />
            ) : analysis ? (
              <AnalysisPanel
                analysis={analysis}
                selectedShot={selectedShot}
                onSelectShot={selectShot}
              />
            ) : (
              !job.busy && (
                <div className="empty tiny">
                  {job.job?.video
                    ? 'Analysis results will appear here'
                    : 'Load a reference video to begin'}
                </div>
              )
            )}
          </div>
        </section>
      </div>

      {/* ------------------------------------------------------ BOTTOM: time */}
      <div className="bottom">
        <div className="pane-head">
          <div className="tabs">
            {hasTrajectory && (
              <button
                className={`tab ${bottomTab === 'kinematics' ? 'active' : ''}`}
                onClick={() => setBottomTab('kinematics')}
              >
                Camera kinematics
              </button>
            )}
            <button
              className={`tab ${bottomTab === 'curves' ? 'active' : ''}`}
              onClick={() => setBottomTab('curves')}
            >
              Image motion
            </button>
            <button
              className={`tab ${bottomTab === 'log' ? 'active' : ''}`}
              onClick={() => setBottomTab('log')}
            >
              Pipeline log
            </button>
          </div>
          <span className="spacer" />
          {shots.length > 1 && (
            <div className="row tiny">
              <span className="dimmer">shot</span>
              {shots.map((s) => (
                <button
                  key={s.id}
                  className={`tab ${s.id === selectedShot ? 'active' : ''}`}
                  onClick={() => selectShot(s.id)}
                >
                  {s.id + 1}
                </button>
              ))}
            </div>
          )}
        </div>

        {info && (
          <Timeline
            duration={info.duration_seconds}
            shots={shots}
            cursorTime={cursorTime}
            fps={fps}
            onScrub={scrub}
          />
        )}

        <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex' }}>
          {bottomTab === 'kinematics' && trajectory ? (
            <KinematicsCurves trajectory={trajectory} cursorTime={cursorTime} onScrub={scrub} />
          ) : bottomTab === 'curves' || bottomTab === 'kinematics' ? (
            <MotionCurves
              frames={frames}
              shots={shots}
              cursorTime={cursorTime}
              onScrub={scrub}
            />
          ) : (
            <div style={{ flex: 1, overflowY: 'auto' }}>
              <LogPanel events={job.events} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
