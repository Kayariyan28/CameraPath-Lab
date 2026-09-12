// Mirrors backend/app/models/schemas/*.py. Kept hand-written rather than
// generated so the field comments that carry the honesty guarantees (what is
// measured vs assumed, what units apply) travel with the types.

export type TimingSource = 'container_pts' | 'nominal_fps'
export type FrameRateMode = 'constant' | 'variable' | 'unknown'

export interface VideoInfo {
  path: string
  filename: string
  size_bytes: number
  duration_seconds: number
  frame_count: number
  /** False when the count was estimated from duration x fps rather than counted. */
  frame_count_is_exact: boolean
  width: number
  height: number
  sample_aspect_ratio: string | null
  display_aspect_ratio: string | null
  rotation_degrees: number
  fps_nominal: number
  fps_average: number
  frame_rate_mode: FrameRateMode
  fps_jitter: number
  codec_name: string
  codec_long_name: string | null
  pix_fmt: string | null
  bit_rate: number | null
  profile: string | null
  time_base: string | null
  /** Whether frame times came from container PTS or were synthesised. */
  timing_source: TimingSource
  has_audio: boolean
  metadata_tags: Record<string, string>
}

export type ShotComplexity = 'trivial' | 'low' | 'moderate' | 'high' | 'extreme'

export interface Shot {
  id: number
  start_frame: number
  end_frame: number
  start_time: number
  end_time: number
  confidence: number
}

export interface CutCandidate {
  frame_index: number
  time_seconds: number
  histogram_score: number
  structural_score: number
  match_collapse_score: number
  flow_coherence_score: number
  combined_score: number
  accepted: boolean
  reason: string
}

export type MotionModel = 'none' | 'translation' | 'euclidean' | 'affine' | 'homography'

/** Image-space motion for one frame transition. NOT physical camera translation. */
export interface MotionFrame {
  frame_index: number
  timestamp: number
  /** Real elapsed seconds from measured timestamps, not 1/fps. */
  dt: number
  dx_pixels: number
  dy_pixels: number
  rotation_deg: number
  scale: number
  model_used: MotionModel
  affine: number[] | null
  homography: number[] | null
  flow_magnitude: number
  flow_magnitude_p90: number
  median_flow: number[]
  /** Outward flow about the centre. Caused by dolly OR zoom — not separable alone. */
  radial_flow: number
  flow_divergence: number
  flow_curl: number
  tracks_in: number
  tracks_survived: number
  inlier_ratio: number
  confidence: number
  background_track_count: number
  rejected_track_count: number
  dynamic_area_fraction: number
}

export interface MotionSignature {
  frame_count: number
  duration: number
  mean_flow_magnitude: number
  peak_flow_magnitude: number
  mean_inlier_ratio: number
  total_image_rotation_deg: number
  net_dx_pixels: number
  net_dy_pixels: number
  cumulative_path_pixels: number
  mean_radial_flow: number
  net_scale_change: number
  /** Evidence of depth-dependent flow, i.e. observable translation. */
  parallax_score: number
  homography_dominance: number
  rotation_dominance: number
  texture_score: number
  blur_score: number
  jitter_score: number
}

export type PipelineMode =
  | 'auto' | 'physical_3d' | 'perceptual_match' | 'fast' | 'high_accuracy'
export type MotionFidelity = 'exact' | 'clean' | 'smooth'
export type ScaleMode = 'normalized' | 'metric'
export type ProxyStyle = 'motion_cage' | 'depth_poles' | 'ground_grid' | 'minimal'
export type LensMode = 'auto' | 'fixed' | 'variable'

export interface ShotAnalysis {
  shot: Shot
  signature: MotionSignature
  complexity: ShotComplexity
  complexity_reasons: string[]
  recommended_mode: PipelineMode
  recommendation_reason: string
}

export interface AnalysisResult {
  video: VideoInfo
  shots: Shot[]
  cut_candidates: CutCandidate[]
  shot_analyses: ShotAnalysis[]
  analysis_resolution: number[]
  warnings: string[]
}

export interface SolveSettings {
  mode: PipelineMode
  motion_fidelity: MotionFidelity
  scale_mode: ScaleMode
  scale_calibration: unknown | null
  max_analysis_resolution: number | null
  keyframe_density: number
  dynamic_rejection_strength: number
  confidence_threshold: number
  lens_mode: LensMode
  fov_override_degrees: number | null
  enable_vggt: boolean
  enable_colmap: boolean
  proxy_style: ProxyStyle
  output_width: number
  output_height: number
  match_source_aspect: boolean
  output_fps: number | null
  render_trajectory_preview: boolean
}

export type JobState =
  | 'created' | 'uploaded' | 'analyzing' | 'analyzed'
  | 'solving' | 'solved' | 'rendering' | 'complete' | 'failed' | 'cancelled'

export interface StageProgress {
  stage: string
  progress: number
  message: string
  shot_id: number | null
  started_at: string | null
  completed_at: string | null
}

export interface JobOutputs {
  motion_proxy_mp4: string | null
  trajectory_preview_mp4: string | null
  trajectory_json: string | null
  trajectory_csv: string | null
  analysis_json: string | null
  camera_scene_blend: string | null
}

export interface Job {
  id: string
  state: JobState
  created_at: string
  updated_at: string
  video: VideoInfo | null
  settings: SolveSettings
  analysis: AnalysisResult | null
  trajectories: unknown[]
  outputs: JobOutputs
  current_stage: StageProgress | null
  stage_history: StageProgress[]
  error: string | null
  error_detail: string | null
  log: string[]
}

export interface ShotMotionPayload {
  shot_id: number
  analysis_size: number[]
  texture: number
  blur: number
  persistent_track_count: number
  mean_track_age: number
  dynamic_region_fraction: number
  signature: MotionSignature
  motion_frames: MotionFrame[]
}

export interface JobEvent {
  job_id: string
  kind: 'stage' | 'progress' | 'log' | 'solver' | 'warning' | 'error'
       | 'state' | 'done' | 'synced'
  timestamp: number
  stage?: string
  progress?: number
  message: string
  shot_id?: number
  data?: Record<string, unknown>
}

export interface ToolInfo {
  name: string
  available: boolean
  path: string | null
  version: string | null
  detail: string
}

export interface ResourcePolicy {
  analysis_long_edge: number
  geometry_long_edge: number
  decode_chunk_frames: number
  max_tracked_features: number
  colmap_max_image_size: number
  colmap_max_num_features: number
  vggt_window_frames: number
  vggt_window_stride: number
  worker_threads: number
  blender_threads: number
  reason: string
}

export interface EnvironmentInfo {
  platform: string
  machine: string
  is_apple_silicon: boolean
  chip: string
  macos_version: string
  total_memory_gb: number
  available_memory_gb: number
  cpu_cores_total: number
  cpu_cores_performance: number
  cpu_cores_efficiency: number
  disk_free_gb: number
  mps_available: boolean
  mps_detail: string
  tools: Record<string, ToolInfo>
  packages: Record<string, ToolInfo>
  warnings: string[]
  resource_policy: ResourcePolicy
}

export interface HealthInfo {
  status: 'ok' | 'degraded'
  can_ingest: boolean
  can_reconstruct_3d: boolean
  can_render: boolean
  can_use_vggt: boolean
  warnings: string[]
}
