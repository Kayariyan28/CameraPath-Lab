import { useMemo } from 'react'
import type { ShotTrajectory } from '../api/types'
import { LineChart } from './LineChart'

interface Props {
  trajectory: ShotTrajectory
  cursorTime: number | null
  onScrub: (time: number) => void
}

/**
 * Camera kinematics of the recovered trajectory (spec §14): speed, yaw / pitch /
 * roll rate and FOV against the source's own timestamps.
 *
 * Speed is in the trajectory's scale units per second — normalized unless a metric
 * calibration was supplied (I5) — and is not drawn at all when translation was not
 * observable, since a zero line would read as "the camera stood still" rather than
 * "this could not be measured".
 */
export function KinematicsCurves({ trajectory, cursorTime, onScrub }: Props) {
  const k = trajectory.kinematics
  const times = useMemo(() => k.map((x) => x.timestamp), [k])
  const lensTimes = useMemo(() => trajectory.poses.map((p) => p.timestamp), [trajectory.poses])
  const translates = trajectory.confidence.translation_observable
  const unit = trajectory.scale_units === 'm' ? 'm/s' : 'u/s'

  // Magnitudes are plotted from zero. An auto-scaled axis stretched a 0.35% speed
  // flutter on a constant-speed orbit to full chart height, where it read as
  // acceleration that is not in the source.
  const speedDomain = useMemo<[number, number]>(
    () => [0, Math.max(1e-6, ...k.map((x) => x.speed)) * 1.15], [k],
  )
  const fovDomain = useMemo<[number, number]>(() => {
    const f = trajectory.poses.map((p) => p.fov_horizontal)
    const lo = Math.min(...f), hi = Math.max(...f)
    const pad = Math.max(5, (hi - lo) * 0.15)
    return [Math.max(0, lo - pad), hi + pad]
  }, [trajectory.poses])

  if (!k.length) return <div className="empty tiny">No kinematics for this shot</div>

  const row = (title: string, note: string, chart: JSX.Element) => (
    <div style={{ marginBottom: 10 }}>
      <div className="chart-title">{title}<span className="spacer" /><span className="dimmer">{note}</span></div>
      {chart}
    </div>
  )

  return (
    <div className="chart-wrap" style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
      {translates
        ? row(
            `Speed (${unit})`,
            trajectory.scale_mode === 'metric' ? 'calibrated metres' : 'normalized units — relative speed only',
            <LineChart series={[{ values: k.map((x) => x.speed), color: '#4d9fff', label: 'speed', fill: true }]}
              times={times} height={64} domain={speedDomain} cursorTime={cursorTime} onScrub={onScrub} unit="" />,
          )
        : <div className="note warn tiny" style={{ marginBottom: 10 }}>
            Speed not shown: translation is not observable in this shot, so there is no measured
            travel to plot. Rotation and lens curves below are measured.
          </div>}
      {row('Yaw rate (°/s)', 'about the camera’s up axis',
        <LineChart series={[{ values: k.map((x) => x.angular_velocity[0]), color: '#e5a13a', label: 'yaw' }]}
          times={times} height={56} symmetric cursorTime={cursorTime} onScrub={onScrub} unit="°" />)}
      {row('Pitch rate (°/s)', 'about the camera’s right axis',
        <LineChart series={[{ values: k.map((x) => x.angular_velocity[1]), color: '#3ecf8e', label: 'pitch' }]}
          times={times} height={56} symmetric cursorTime={cursorTime} onScrub={onScrub} unit="°" />)}
      {row('Roll rate (°/s)', 'about the viewing axis',
        <LineChart series={[{ values: k.map((x) => x.angular_velocity[2]), color: '#8b7fe8', label: 'roll' }]}
          times={times} height={56} symmetric cursorTime={cursorTime} onScrub={onScrub} unit="°" />)}
      {row('Horizontal FOV (°)', trajectory.confidence.zoom_confidence < 0.3 ? 'prior — focal not measured in this shot' : 'recovered lens curve',
        <LineChart series={[{ values: trajectory.poses.map((p) => p.fov_horizontal), color: '#e8eaee', label: 'fov' }]}
          times={lensTimes} height={56} domain={fovDomain} cursorTime={cursorTime} onScrub={onScrub} unit="°" />)}
    </div>
  )
}
