export type Status = 'queued' | 'running' | 'awaiting_selection' | 'done' | 'failed'
export type StageState = 'pending' | 'running' | 'awaiting' | 'done' | 'failed'
export type Engine = 'manim' | 'hyperframes'

export interface Retrieved { resource_id: string; score: number; resource_type?: string; text?: string }
export interface GoldResource {
  resource_id: string; resource_type?: string; summary?: string; description?: string; text?: string
  concepts?: string[]; image_path?: string
}
export interface Stage1 {
  mode: string; query?: string; query_id?: string
  retrieved: Retrieved[]; pool_size?: number; awaiting_selection?: boolean
  selection: {
    shortlist: string[]; selected: string[]; n?: number; B?: number; parse_path?: string; suggested_n?: number
    fell_back?: boolean; repaired?: boolean; problem?: string; llm_sec?: number
  }
  gold_set?: { gold_sets?: { resources_used?: GoldResource[] }[] }
  seconds?: number
}
export interface Segment {
  segment_id?: string | number; order: number; role: string; duration_sec: number
  resource_ids?: string[]; narration?: string; on_screen?: string; concepts?: string[]
}
export interface StoryboardInner {
  storyboard_id?: string; query?: string; intent?: string; total_duration_sec?: number
  role_template?: string | string[]; segments?: Segment[]
}
export interface Stage2 { mode: string; storyboard?: { storyboard?: StoryboardInner } & StoryboardInner; seconds?: number }
export interface Stage3 {
  engine?: string; ok?: boolean; video?: string | null; status?: string | null
  timing_sec?: Record<string, number>; seconds?: number
}
export interface Job {
  id: string; spec?: Record<string, unknown>; status: Status; stage: number
  stages: Record<string, StageState>; log: string[]; error?: string | null
  video_url: string | null; stage1: Stage1 | null; stage2: Stage2 | null; stage3: Stage3 | null
  progress?: Progress
}
export interface Progress { pct: number; label: string; eta_sec: number }
export interface JobSummary {
  id: string; query: string; engine: string; status: Status; stages: Record<string, StageState>
  premade?: boolean; query_id?: string | null; selection?: string; stage2?: string
  stage1_label?: string | null; stage2_label?: string | null
}
export interface BenchQuery { query_id: string; query: string; T: number; n_gold: number }
export interface Health {
  generator?: { url?: string; up: boolean; sleeping?: boolean }
  verifier?: { url?: string; up: boolean; sleeping?: boolean }
  stage1_bge?: boolean
  engines?: Record<string, { label: string; eta_min: string | number | number[] }>
  queue?: number; busy?: boolean
}
