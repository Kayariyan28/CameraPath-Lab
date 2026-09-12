import { useCallback, useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { MotionFrame } from '../api/types'
import { Viewport, clearGroup, type SceneHandle } from './Viewport'

interface Props {
  frames: MotionFrame[]
  cursorTime: number | null
  analysisSize: number[]
}

/**
 * Phase-1 viewport content: the cumulative *image-space* motion path.
 *
 * This is deliberately NOT presented as a camera trajectory. Nothing here has
 * been through a geometric solve, so there is no 3D position to show — what is
 * drawn is the accumulated pan/tilt of the frame contents, laid out on a plane,
 * with rotation and scale change encoded along it. Drawing this as a 3D camera
 * path would be exactly the confusion between optical flow and physical
 * translation that invariant I6 forbids.
 *
 * Phase 2 replaces this with real recovered poses and frustums in the same
 * viewport.
 */
export function MotionPathView({ frames, cursorTime, analysisSize }: Props) {
  const handle = useRef<SceneHandle | null>(null)

  const build = useCallback(() => {
    const h = handle.current
    if (!h) return
    clearGroup(h.content)
    if (!frames.length) return

    const longEdge = Math.max(analysisSize[0] ?? 1080, analysisSize[1] ?? 608, 1)
    // Map pixels to viewport units so a full-frame pan spans ~10 units,
    // keeping the plot readable regardless of analysis resolution.
    const k = 10 / longEdge

    let x = 0
    let y = 0
    let roll = 0
    const points: THREE.Vector3[] = [new THREE.Vector3(0, 0, 0)]
    const colors: number[] = []
    const speeds: number[] = []

    for (const f of frames) {
      x += f.dx_pixels * k
      y -= f.dy_pixels * k // screen +y is down; world +y is away
      roll += f.rotation_deg
      // Height encodes cumulative scale change: rising = content expanding
      // (dolly-in or zoom-in, not separable without parallax).
      const z = (Math.log(Math.max(f.scale, 1e-3)) * 40) * k * longEdge * 0.01
      points.push(new THREE.Vector3(x, y, points[points.length - 1]!.z + z))
      speeds.push(Math.hypot(f.dx_pixels, f.dy_pixels) / Math.max(f.dt, 1e-6))
    }

    const maxSpeed = Math.max(...speeds, 1e-6)
    const cold = new THREE.Color(0x2b6fb5)
    const hot = new THREE.Color(0xffb63a)
    for (const s of speeds) {
      const c = cold.clone().lerp(hot, Math.min(1, s / maxSpeed))
      colors.push(c.r, c.g, c.b, c.r, c.g, c.b)
    }

    // Line segments so each step can carry its own speed colour.
    const segPositions: number[] = []
    for (let i = 0; i < points.length - 1; i += 1) {
      const a = points[i]!
      const b = points[i + 1]!
      segPositions.push(a.x, a.y, a.z, b.x, b.y, b.z)
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.Float32BufferAttribute(segPositions, 3))
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    h.content.add(new THREE.LineSegments(
      geom,
      new THREE.LineBasicMaterial({ vertexColors: true }),
    ))

    // Endpoints.
    const marker = (p: THREE.Vector3, color: number, size: number) => {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(size, 18, 12),
        new THREE.MeshBasicMaterial({ color }),
      )
      m.position.copy(p)
      return m
    }
    h.content.add(marker(points[0]!, 0x3ecf8e, 0.16))
    h.content.add(marker(points[points.length - 1]!, 0xe5564b, 0.16))

    // Roll indicator ticks every ~10 frames, so in-plane rotation is visible.
    const tickMat = new THREE.LineBasicMaterial({ color: 0x8b7fe8, transparent: true, opacity: 0.7 })
    let acc = 0
    for (let i = 0; i < frames.length; i += 1) {
      acc += frames[i]!.rotation_deg
      if (i % 10 !== 0) continue
      const p = points[i + 1]!
      const rad = (acc * Math.PI) / 180
      const len = 0.45
      const g = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p.x - Math.cos(rad) * len, p.y - Math.sin(rad) * len, p.z),
        new THREE.Vector3(p.x + Math.cos(rad) * len, p.y + Math.sin(rad) * len, p.z),
      ])
      h.content.add(new THREE.Line(g, tickMat))
    }

    h.frameContent()
  }, [frames, analysisSize])

  useEffect(() => { build() }, [build])

  // Cursor marker follows the scrubbed time.
  useEffect(() => {
    const h = handle.current
    if (!h) return
    const existing = h.scene.getObjectByName('cursor-marker')
    if (existing) {
      h.scene.remove(existing)
      ;(existing as THREE.Mesh).geometry?.dispose()
    }
    if (cursorTime === null || !frames.length) return

    let idx = 0
    let best = Infinity
    for (let i = 0; i < frames.length; i += 1) {
      const d = Math.abs(frames[i]!.timestamp - cursorTime)
      if (d < best) { best = d; idx = i }
    }
    const line = h.content.children.find((c) => c instanceof THREE.LineSegments) as
      THREE.LineSegments | undefined
    if (!line) return
    const pos = line.geometry.getAttribute('position')
    const at = Math.min(idx * 2, pos.count - 1)
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.2, 18, 12),
      new THREE.MeshBasicMaterial({ color: 0x4d9fff }),
    )
    mesh.name = 'cursor-marker'
    mesh.position.set(pos.getX(at), pos.getY(at), pos.getZ(at))
    h.scene.add(mesh)
  }, [cursorTime, frames])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <Viewport onReady={(h) => { handle.current = h; build() }} />
      <div style={{
        position: 'absolute', left: 12, bottom: 10, pointerEvents: 'none',
        fontSize: 10.5, lineHeight: 1.6, color: 'var(--fg-2)',
        fontFamily: 'var(--mono)', textShadow: '0 1px 3px #000',
      }}>
        <div><span style={{ color: '#3ecf8e' }}>●</span> start
          {'  '}<span style={{ color: '#e5564b' }}>●</span> end
          {'  '}<span style={{ color: '#8b7fe8' }}>—</span> roll
          {'  '}<span style={{ color: '#2b6fb5' }}>—</span>→<span style={{ color: '#ffb63a' }}>—</span> speed</div>
        <div style={{ color: 'var(--fg-3)' }}>drag orbit · scroll zoom · right-drag pan</div>
      </div>
    </div>
  )
}
