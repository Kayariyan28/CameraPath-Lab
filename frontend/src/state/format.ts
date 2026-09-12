/** Display helpers. Every one of these exists so the UI never implies more
 *  precision than the underlying measurement carries. */

export function seconds(value: number, places = 3): string {
  if (!Number.isFinite(value)) return '—'
  return `${value.toFixed(places)}s`
}

export function timecode(value: number, fps = 30): string {
  if (!Number.isFinite(value) || value < 0) return '00:00.00'
  const total = Math.max(0, value)
  const m = Math.floor(total / 60)
  const s = Math.floor(total % 60)
  const f = Math.floor((total % 1) * fps)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(f).padStart(2, '0')}`
}

export function bytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = value
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1 }
  return `${v < 10 && i > 0 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}

export function percent(value: number, places = 0): string {
  if (!Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(places)}%`
}

export function fps(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '—'
  // 23.976 and 29.97 must not be rounded to 24 and 30 — the difference is the
  // whole reason timing is read from container timestamps.
  const rounded = Math.round(value)
  return Math.abs(value - rounded) < 0.001 ? String(rounded) : value.toFixed(3)
}

export function px(value: number, places = 1): string {
  if (!Number.isFinite(value)) return '—'
  return `${value.toFixed(places)} px`
}

export function degrees(value: number, places = 2): string {
  if (!Number.isFinite(value)) return '—'
  return `${value.toFixed(places)}°`
}

export function titleCase(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export type Severity = 'ok' | 'warn' | 'bad' | 'info' | 'mute'

export function complexityTone(level: string): Severity {
  switch (level) {
    case 'trivial': case 'low': return 'ok'
    case 'moderate': return 'info'
    case 'high': return 'warn'
    case 'extreme': return 'bad'
    default: return 'mute'
  }
}

export function stateTone(state: string): Severity {
  switch (state) {
    case 'complete': case 'analyzed': case 'solved': return 'ok'
    case 'failed': return 'bad'
    case 'cancelled': return 'warn'
    case 'analyzing': case 'solving': case 'rendering': return 'info'
    default: return 'mute'
  }
}

/** A score where higher is better. */
export function scoreTone(value: number, warnBelow = 0.5, badBelow = 0.25): Severity {
  if (value < badBelow) return 'bad'
  if (value < warnBelow) return 'warn'
  return 'ok'
}
