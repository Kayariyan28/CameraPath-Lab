import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/**
 * Three.js viewport for trajectory inspection.
 *
 * Written against three.js directly rather than react-three-fiber: this is a
 * single imperative scene whose contents are rebuilt from a pose array, React
 * reconciliation buys nothing here, and it sidesteps the
 * three/fiber/drei version matrix (drei already reported a three-version
 * incompatibility at install time).
 */

export interface SceneHandle {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  controls: OrbitControls
  /** Group that callers populate; cleared on every rebuild. */
  content: THREE.Group
  frameContent: () => void
}

interface Props {
  onReady: (handle: SceneHandle) => void
  className?: string
}

export function Viewport({ onReady, className }: Props) {
  const mount = useRef<HTMLDivElement>(null)
  const handleRef = useRef<SceneHandle | null>(null)

  useEffect(() => {
    const host = mount.current
    if (!host) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0b0d)

    const camera = new THREE.PerspectiveCamera(46, 1, 0.05, 4000)
    camera.position.set(16, -22, 13)
    camera.up.set(0, 0, 1) // Z-up, matching the CameraPath world convention

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    } catch {
      host.innerHTML =
        '<div class="empty">WebGL is unavailable in this browser.</div>'
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    host.appendChild(renderer.domElement)
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.09
    controls.target.set(0, 0, 0)

    // --- static reference furniture -------------------------------------
    const grid = new THREE.GridHelper(120, 60, 0x2a2f3a, 0x1c2027)
    // GridHelper is XZ-planar; rotate it into the XY plane for a Z-up world.
    grid.rotation.x = Math.PI / 2
    scene.add(grid)

    const axes = new THREE.AxesHelper(3)
    scene.add(axes)

    scene.add(new THREE.AmbientLight(0xffffff, 0.75))
    const key = new THREE.DirectionalLight(0xffffff, 0.7)
    key.position.set(8, -12, 18)
    scene.add(key)

    const content = new THREE.Group()
    scene.add(content)

    const frameContent = () => {
      const box = new THREE.Box3().setFromObject(content)
      if (box.isEmpty()) {
        camera.position.set(16, -22, 13)
        controls.target.set(0, 0, 0)
        controls.update()
        return
      }
      const size = box.getSize(new THREE.Vector3())
      const centre = box.getCenter(new THREE.Vector3())
      const radius = Math.max(size.length() * 0.5, 1)
      const dist = radius / Math.tan((camera.fov * Math.PI) / 360) * 1.5
      const dir = new THREE.Vector3(0.6, -0.85, 0.45).normalize()
      camera.position.copy(centre).addScaledVector(dir, dist)
      controls.target.copy(centre)
      camera.near = Math.max(dist / 800, 0.01)
      camera.far = dist * 12
      camera.updateProjectionMatrix()
      controls.update()
    }

    const handle: SceneHandle = { scene, camera, renderer, controls, content, frameContent }
    handleRef.current = handle

    const resize = () => {
      const w = host.clientWidth || 1
      const h = host.clientHeight || 1
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(host)

    let raf = 0
    const tick = () => {
      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)

    onReady(handle)

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
      controls.dispose()
      // Dispose every GPU resource: a viewport that is mounted and unmounted
      // repeatedly across jobs otherwise leaks buffers until the context dies.
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh
        mesh.geometry?.dispose?.()
        const mat = mesh.material
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
        else mat?.dispose?.()
      })
      renderer.dispose()
      if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement)
      handleRef.current = null
    }
    // onReady is intentionally excluded: the scene is built once per mount and
    // re-running this effect would tear down and rebuild the WebGL context.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <div ref={mount} className={className} style={{ width: '100%', height: '100%' }} />
}

/** Remove and dispose everything inside a group, leaving the group attached. */
export function clearGroup(group: THREE.Group): void {
  for (let i = group.children.length - 1; i >= 0; i -= 1) {
    const child = group.children[i]!
    group.remove(child)
    child.traverse((obj) => {
      const mesh = obj as THREE.Mesh
      mesh.geometry?.dispose?.()
      const mat = mesh.material
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
      else mat?.dispose?.()
    })
  }
}
