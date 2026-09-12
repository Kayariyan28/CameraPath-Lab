import { useEffect, useRef } from 'react'
import type { JobEvent } from '../api/types'

export function LogPanel({ events }: { events: JobEvent[] }) {
  const end = useRef<HTMLDivElement>(null)
  useEffect(() => { end.current?.scrollIntoView({ block: 'end' }) }, [events.length])

  const shown = events.filter(
    (e) => e.kind !== 'progress' && e.kind !== 'synced',
  )

  if (!shown.length) {
    return <div className="empty tiny">Pipeline log will appear here</div>
  }

  return (
    <div className="log">
      {shown.map((e, i) => (
        <div key={`${e.timestamp}-${i}`} className={`l-${e.kind}`}>
          <span className="dimmer">
            {new Date(e.timestamp * 1000).toLocaleTimeString([], {
              hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
            })}{' '}
          </span>
          <span className="dimmer">{e.kind.padEnd(7)} </span>
          {e.shot_id !== undefined && e.shot_id !== null && (
            <span className="dimmer">shot{String(e.shot_id).padStart(2, '0')} </span>
          )}
          {e.message}
        </div>
      ))}
      <div ref={end} />
    </div>
  )
}
