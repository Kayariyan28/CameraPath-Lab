import { useCallback, useRef, useState } from 'react'
import { bytes } from '../state/format'

interface Props {
  onFile: (file: File) => void
  uploadFraction: number | null
  disabled?: boolean
}

const ACCEPT = '.mp4,.mov,.m4v,.mkv,.avi,.webm,.mpg,.mpeg,.mts,.m2ts'

export function Dropzone({ onFile, uploadFraction, disabled }: Props) {
  const [active, setActive] = useState(false)
  const [rejected, setRejected] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const accept = useCallback((files: FileList | null) => {
    setRejected(null)
    const file = files?.[0]
    if (!file) return
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPT.split(',').includes(ext)) {
      setRejected(`${ext || 'that file type'} is not a supported video container`)
      return
    }
    onFile(file)
  }, [onFile])

  if (uploadFraction !== null) {
    return (
      <div className="dropzone" style={{ cursor: 'default' }}>
        <div className="dropzone-title">Uploading…</div>
        <div className="bar" style={{ margin: '10px 0 6px' }}>
          <i style={{ width: `${Math.round(uploadFraction * 100)}%` }} />
        </div>
        <div className="tiny dim mono">{Math.round(uploadFraction * 100)}%</div>
      </div>
    )
  }

  return (
    <>
      <div
        className={`dropzone ${active ? 'active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setActive(true) }}
        onDragLeave={() => setActive(false)}
        onDrop={(e) => {
          e.preventDefault()
          setActive(false)
          if (!disabled) accept(e.dataTransfer.files)
        }}
        onClick={() => !disabled && input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') input.current?.click() }}
      >
        <div className="dropzone-title">Drop a reference video</div>
        <div className="tiny dim">or click to choose a file</div>
        <div className="tiny dimmer" style={{ marginTop: 9 }}>
          MP4 · MOV · MKV · AVI · WebM · MTS
        </div>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => accept(e.target.files)}
        />
      </div>
      {rejected && (
        <div className="note bad" style={{ margin: '0 12px 12px' }}>{rejected}</div>
      )}
    </>
  )
}

export function FileSummary({ name, size }: { name: string; size: number }) {
  return (
    <div className="row tiny">
      <span className="mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {name}
      </span>
      <span className="spacer" />
      <span className="dimmer mono">{bytes(size)}</span>
    </div>
  )
}
