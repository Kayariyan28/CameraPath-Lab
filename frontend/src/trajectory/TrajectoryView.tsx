import { useCallback, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { CameraPose, Kinematics, ShotTrajectory } from '../api/types'
import { Viewport, clearGroup, type SceneHandle } from './Viewport'

interface Props {
  trajectory: ShotTrajectory
  cursorTime: number | null
  /** Source frame aspect (width / height), for frustum shape. */
  aspect: number
}

/** Confidence bands matching validation/confidence.py (HIGH >= 0.72, MEDIUM >= 0.45). */
function confidenceColor(c: number): THREE.Color {
  if (c >= 0.72) return new THREE.Color(0x3ecf8e)
  if (c >= 0.45) return new THREE.Color(0xe5a13a)
  return new THREE.Color(0xe5564b)
}

/** CameraPath quaternion [w,x,y,z] -> three.js (x,y,z,w). Both worlds are
 *  right-handed and Z-up, so no axis change is needed — only the order. */
function toThreeQuat(q: CameraPose['quaternion']): THREE.Quaternion {
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]).normalize()
}

/** The camera's own axes in world space. CameraPath cameras look along local
 *  +Y with +Z up and +X right. */
function cameraAxes(pose: CameraPose) {
  const q = toThreeQuat(pose.quaternion)
  return {
    forward: new THREE.Vector3(0, 1, 0).applyQuaternion(q),
    up: new THREE.Vector3(0, 0, 1).applyQuaternion(q),
    right: new THREE.Vector3(1, 0, 0).applyQuaternion(q),
  }
}

function frustumPoints(pose: CameraPose, depth: number, aspect: number): THREE.Vector3[] {
  const { forward, up, right } = cameraAxes(pose)
  const apex = new THREE.Vector3(...pose.position)
  const halfW = depth * Math.tan((pose.fov_horizontal * Math.PI) / 360)
  const halfH = halfW / Math.max(aspect, 1e-3)
  const centre = apex.clone().addScaledVector(forward, depth)
  const c = [
    centre.clone().addScaledVector(right, -halfW).addScaledVector(up, halfH),
    centre.clone().addScaledVector(right, halfW).addScaledVector(up, halfH),
    centre.clone().addScaledVector(right, halfW).addScaledVector(up, -halfH),
    centre.clone().addScaledVector(right, -halfW).addScaledVector(up, -halfH),
  ]
  // Apex-to-corner edges, the image rectangle, and a short "up" tick on the top
  // edge so roll is readable at a glance.
  const topMid = c[0]!.clone().lerp(c[1]!, 0.5)
  return [
    apex, c[0]!, apex, c[1]!, apex, c[2]!, apex, c[3]!,
    c[0]!, c[1]!, c[1]!, c[2]!, c[2]!, c[3]!, c[3]!, c[0]!,
    topMid, topMid.clone().addScaledVector(up, halfH * 0.45),
  ]
}

function nearestIndex(poses: CameraPose[], time: number): number {
  let lo = 0
  let hi = poses.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (poses[mid]!.timestamp < time) lo = mid
    else hi = mid
  }
  return Math.abs(poses[lo]!.timestamp - time) <= Math.abs(poses[hi]!.timestamp - time) ? lo : hi
}

/**
 * The recovered optical-camera trajectory in 3D.
 *
 * Positions are in the trajectory's scale mode — normalized units unless the user
 * calibrated a metric scale (I5) — and the HUD says which. When translation was
 * not observable, every position is the same point by design; the look-direction
 * trail is what shows the move then, and the overlay says so rather than letting
 * a single dot read as "the camera did not move".
 */
export function TrajectoryView({ trajectory, cursorTime, aspect }: Props) {
  const handle = useRef<SceneHandle | null>(null)
  const poses = trajectory.poses
  const kin = trajectory.kinematics

  const extent = useMemo(() => {
    if (!poses.length) return 1
    const box = new THREE.Box3()
    for (const p of poses) box.expandByPoint(new THREE.Vector3(...p.position))
    return box.getSize(new THREE.Vector3()).length()
  }, [poses])
  const translates = extent > 1e-6 && trajectory.confidence.translation_observable
  // Visual scale for frustums and the look trail: a fraction of the path, or a
  // fixed size when there is no path to measure against.
  const unit = translates ? Math.max(extent * 0.045, 1e-3) : 1.0

  const build = useCallback(() => {
    const h = handle.current
    if (!h) return
    clearGroup(h.content)
    const cur = h.scene.getObjectByName('current-pose')
    if (cur) h.scene.remove(cur)
    if (!poses.length) return

    // --- path, coloured per segment by confidence -------------------------
    if (translates) {
      const seg: number[] = []
      const col: number[] = []
      for (let i = 0; i < poses.length - 1; i += 1) {
        const a = poses[i]!
        const b = poses[i + 1]!
        seg.push(...a.position, ...b.position)
        const ca = confidenceColor(a.confidence)
        const cb = confidenceColor(b.confidence)
        col.push(ca.r, ca.g, ca.b, cb.r, cb.g, cb.b)
      }
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3))
      g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
      h.content.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })))
    }

    // --- look-direction trail -----------------------------------------------
    // Where the camera pointed over time, drawn `trail` units ahead of it. This
    // is the only visible cue for a pure pan/tilt, and it makes orbits legible.
    const trail = unit * 2.2
    const look = poses.map((p) => new THREE.Vector3(...p.position).addScaledVector(cameraAxes(p).forward, trail))
    h.content.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(look),
      new THREE.LineBasicMaterial({ color: 0x8b7fe8, transparent: true, opacity: 0.55 }),
    ))

    // --- keyframe frustums (geometric anchors), thinned for clarity -----------
    const anchors = poses.filter((p) => p.is_anchor)
    const shown = anchors.length > 40
      ? anchors.filter((_, i) => i % Math.ceil(anchors.length / 40) === 0)
      : anchors
    const fr: number[] = []
    for (const p of shown) for (const v of frustumPoints(p, unit, aspect)) fr.push(v.x, v.y, v.z)
    if (fr.length) {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(fr, 3))
      h.content.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({
        color: 0x6e7686, transparent: true, opacity: 0.6,
      })))
    }

    // --- start / end -------------------------------------------------------
    const dot = (p: CameraPose, color: number) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(unit * 0.22, 18, 12),
        new THREE.MeshBasicMaterial({ color }),
      )
      m.position.set(...p.position)
      return m
    }
    h.content.add(dot(poses[0]!, 0x3ecf8e))
    if (translates) h.content.add(dot(poses[poses.length - 1]!, 0xe5564b))

    h.frameContent()
  }, [poses, translates, unit, aspect])

  useEffect(() => { build() }, [build])

  // --- current pose, following the timeline ---------------------------------
  const index = cursorTime === null || !poses.length ? 0 : nearestIndex(poses, cursorTime)
  useEffect(() => {
    const h = handle.current
    if (!h || !poses.length) return
    const old = h.scene.getObjectByName('current-pose')
    if (old) {
      h.scene.remove(old)
      old.traverse((o) => {
        const m = o as THREE.Mesh
        m.geometry?.dispose?.()
        const mat = m.material
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose())
        else mat?.dispose?.()
      })
    }
    const pose = poses[index]!
    const group = new THREE.Group()
    group.name = 'current-pose'
    const pts = frustumPoints(pose, unit * 1.6, aspect)
    group.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x4d9fff }),
    ))
    const { forward } = cameraAxes(pose)
    group.add(new THREE.ArrowHelper(forward, new THREE.Vector3(...pose.position), unit * 3.2, 0x4d9fff, unit * 0.5, unit * 0.28))
    h.scene.add(group)
  }, [index, poses, unit, aspect])

  const pose = poses[index]
  const k: Kinematics | undefined = kin[index]
  const units = trajectory.scale_units === 'm' ? 'm' : 'u'

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Viewport onReady={(h) => { handle.current = h; build() }} />

      {pose && (
        <div className="hud hud-tl">
          <div><span className="dimmer">frame</span> {pose.frame_index} <span className="dimmer">t</span> {pose.timestamp.toFixed(3)}s</div>
          {k && translates && <div><span className="dimmer">speed</span> {k.speed.toFixed(2)} {units}/s <span className="dimmer">({Math.round(k.speed_normalized * 100)}%)</span></div>}
          {k && <div><span className="dimmer">yaw/pitch/roll</span> {k.angular_velocity.map((v) => v.toFixed(1)).join(' / ')} °/s</div>}
          <div><span className="dimmer">FOV</span> {pose.fov_horizontal.toFixed(1)}°</div>
          <div><span className="dimmer">source</span> {pose.solver_source}{pose.is_anchor ? ' · anchor' : ''} <span className="dimmer">conf</span> {pose.confidence.toFixed(2)}</div>
        </div>
      )}

      {!translates && (
        <div className="hud hud-tr note warn" style={{ maxWidth: 300 }}>
          Translation not observable in this shot — the camera is shown at one point and
          only its orientation (purple look trail) and lens are recovered.
        </div>
      )}

      <div className="hud hud-bl">
        <div>
          <span style={{ color: '#3ecf8e' }}>●</span> start{translates && <>{'  '}<span style={{ color: '#e5564b' }}>●</span> end</>}
          {'  '}<span style={{ color: '#4d9fff' }}>▲</span> current
          {'  '}<span style={{ color: '#8b7fe8' }}>—</span> look direction
          {'  '}<span style={{ color: '#6e7686' }}>◇</span> keyframes
          {translates && <>{'  '}path: <span style={{ color: '#3ecf8e' }}>high</span>/<span style={{ color: '#e5a13a' }}>med</span>/<span style={{ color: '#e5564b' }}>low</span> confidence</>}
        </div>
        <div className="dimmer">
          {trajectory.scale_mode === 'metric' ? 'metres (calibrated)' : 'normalized units — not metres'} ·
          optical camera pose, not a drone body · drag orbit · scroll zoom · right-drag pan
        </div>
      </div>
    </div>
  )
}
