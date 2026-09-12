import { useMemo } from 'react'

export interface Series {
  values: number[]
  color: string
  label: string
  /** Draw as a filled area rather than a stroke. */
  fill?: boolean
}

interface Props {
  series: Series[]
  times: number[]
  height?: number
  /** Force a symmetric domain about zero — right for signed rates, where the
   *  sign is the meaning (pan left vs right) and an asymmetric axis hides it. */
  symmetric?: boolean
  /** Fixed domain, e.g. [0,1] for ratios, so small noise does not fill the plot. */
  domain?: [number, number]
  cursorTime?: number | null
  cutTimes?: number[]
  unit?: string
  onScrub?: (time: number) => void
}

const PAD_L = 38
const PAD_R = 6
const PAD_T = 6
const PAD_B = 14

/** Hand-rolled SVG rather than a charting library: these plots need exact
 *  control of the zero line, shared time axis and cut markers, and a library
 *  would be more code to configure than to draw. */
export function LineChart({
  series, times, height = 64, symmetric = false, domain,
  cursorTime = null, cutTimes = [], unit = '', onScrub,
}: Props) {
  const W = 1000 // viewBox units; the SVG scales to its container width

  const { lo, hi } = useMemo(() => {
    if (domain) return { lo: domain[0], hi: domain[1] }
    let min = Infinity
    let max = -Infinity
    for (const s of series) {
      for (const v of s.values) {
        if (!Number.isFinite(v)) continue
        if (v < min) min = v
        if (v > max) max = v
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { lo: 0, hi: 1 }
    if (symmetric) {
      const m = Math.max(Math.abs(min), Math.abs(max)) || 1
      return { lo: -m * 1.12, hi: m * 1.12 }
    }
    if (max - min < 1e-9) return { lo: min - 0.5, hi: max + 0.5 }
    const pad = (max - min) * 0.1
    return { lo: min - pad, hi: max + pad }
  }, [series, symmetric, domain])

  const t0 = times.length ? (times[0] ?? 0) : 0
  const t1 = times.length ? (times[times.length - 1] ?? 1) : 1
  const span = Math.max(t1 - t0, 1e-6)

  const xOf = (t: number) => PAD_L + ((t - t0) / span) * (W - PAD_L - PAD_R)
  const yOf = (v: number) =>
    PAD_T + (1 - (v - lo) / Math.max(hi - lo, 1e-9)) * (height - PAD_T - PAD_B)

  const paths = useMemo(() => series.map((s) => {
    const pts: string[] = []
    for (let i = 0; i < s.values.length; i += 1) {
      const v = s.values[i]
      const t = times[i]
      if (v === undefined || t === undefined || !Number.isFinite(v)) continue
      pts.push(`${pts.length === 0 ? 'M' : 'L'}${xOf(t).toFixed(2)},${yOf(v).toFixed(2)}`)
    }
    const d = pts.join(' ')
    const area = d && s.fill
      ? `${d} L${xOf(t1).toFixed(2)},${yOf(Math.max(lo, 0)).toFixed(2)} ` +
        `L${xOf(t0).toFixed(2)},${yOf(Math.max(lo, 0)).toFixed(2)} Z`
      : ''
    return { ...s, d, area }
  }), [series, times, lo, hi, height]) // eslint-disable-line react-hooks/exhaustive-deps

  const zeroVisible = lo < 0 && hi > 0

  const handle = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onScrub) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    const vx = frac * W
    const inner = (vx - PAD_L) / Math.max(W - PAD_L - PAD_R, 1)
    onScrub(t0 + Math.min(1, Math.max(0, inner)) * span)
  }

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${height}`}
      preserveAspectRatio="none"
      style={{ height, cursor: onScrub ? 'crosshair' : 'default' }}
      onMouseDown={handle}
      onMouseMove={(e) => { if (e.buttons === 1) handle(e) }}
    >
      <line className="axis" x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={height - PAD_B} />
      <line className="grid" x1={PAD_L} y1={height - PAD_B} x2={W - PAD_R} y2={height - PAD_B} />

      <text x={PAD_L - 5} y={PAD_T + 7} textAnchor="end">
        {formatTick(hi)}{unit}
      </text>
      <text x={PAD_L - 5} y={height - PAD_B} textAnchor="end">
        {formatTick(lo)}{unit}
      </text>

      {zeroVisible && (
        <line className="zero" x1={PAD_L} y1={yOf(0)} x2={W - PAD_R} y2={yOf(0)} />
      )}

      {cutTimes.map((t) => (
        <line key={`cut-${t}`} className="cutline" x1={xOf(t)} y1={PAD_T} x2={xOf(t)} y2={height - PAD_B} />
      ))}

      {paths.map((p) => (
        <g key={p.label}>
          {p.area && <path d={p.area} fill={p.color} opacity={0.16} />}
          {p.d && (
            <path
              d={p.d} fill="none" stroke={p.color}
              strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </g>
      ))}

      {cursorTime !== null && Number.isFinite(cursorTime) && (
        <line className="cursor" x1={xOf(cursorTime)} y1={PAD_T} x2={xOf(cursorTime)} y2={height - PAD_B} />
      )}
    </svg>
  )
}

function formatTick(v: number): string {
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1000) return v.toExponential(0)
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  if (a >= 1) return v.toFixed(2)
  return v.toFixed(3)
}
