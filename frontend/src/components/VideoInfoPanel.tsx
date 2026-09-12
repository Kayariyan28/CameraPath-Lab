import type { VideoInfo } from '../api/types'
import { bytes, fps, seconds } from '../state/format'

/**
 * Container facts. Every row distinguishes measured from declared, because the
 * difference decides whether timing can be trusted (invariant I2). A stream
 * that declares 30 fps but plays at a variable rate is common, and silently
 * treating it as CFR is how a reconstruction ends up with the right shape and
 * the wrong speed.
 */
export function VideoInfoPanel({ info }: { info: VideoInfo }) {
  const authoritative = info.timing_source === 'container_pts'
  const vfr = info.frame_rate_mode === 'variable'

  return (
    <div className="stack tight">
      <dl className="kv">
        <dt>Resolution</dt>
        <dd>
          {info.width}×{info.height}
          {info.rotation_degrees !== 0 && (
            <span className="dimmer"> (rot {info.rotation_degrees}°)</span>
          )}
        </dd>

        <dt>Duration</dt>
        <dd>{seconds(info.duration_seconds)}</dd>

        <dt>Frames</dt>
        <dd>
          {info.frame_count.toLocaleString()}
          {!info.frame_count_is_exact && <span className="dimmer"> est.</span>}
        </dd>

        <dt>Frame rate</dt>
        <dd>
          {fps(info.fps_average)}
          {Math.abs(info.fps_average - info.fps_nominal) > 0.01 && (
            <span className="dimmer"> (declared {fps(info.fps_nominal)})</span>
          )}
        </dd>

        <dt>Rate mode</dt>
        <dd>
          <span className={`badge ${vfr ? 'warn' : info.frame_rate_mode === 'constant' ? 'ok' : 'mute'}`}>
            {info.frame_rate_mode}
          </span>
        </dd>

        <dt>Codec</dt>
        <dd>{info.codec_name}{info.profile ? ` · ${info.profile}` : ''}</dd>

        <dt>Pixel format</dt>
        <dd>{info.pix_fmt ?? '—'}</dd>

        <dt>Bitrate</dt>
        <dd>{info.bit_rate ? `${(info.bit_rate / 1e6).toFixed(1)} Mb/s` : '—'}</dd>

        <dt>File size</dt>
        <dd>{bytes(info.size_bytes)}</dd>

        <dt>Timing</dt>
        <dd>
          <span className={`badge ${authoritative ? 'ok' : 'warn'}`}>
            {authoritative ? 'container PTS' : 'synthesised'}
          </span>
        </dd>
      </dl>

      {!authoritative && (
        <div className="note warn">
          This container gave no usable presentation timestamps, so frame times were
          synthesised from the declared frame rate. If the source is genuinely
          variable-rate, output timing may drift from the original.
        </div>
      )}
      {vfr && authoritative && (
        <div className="note info">
          Variable frame rate ({(info.fps_jitter * 100).toFixed(1)}% interval jitter).
          Real per-frame timestamps are being used, so original timing is preserved.
        </div>
      )}
    </div>
  )
}
