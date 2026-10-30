import type { BenchQuery, Health, Job, JobSummary } from './types'

// '/' in dev, '/VideoModule/' when built for the mmir VM (vite build --base /VideoModule/): every API path is relative to it
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '')
export const apiPath = (p: string) => (p.startsWith('/api') ? BASE + p : p)

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init)
  if (!r.ok) {
    let detail = ''
    try { detail = JSON.stringify((await r.json()).detail ?? '') } catch { /* ignore */ }
    throw new Error(`HTTP ${r.status} ${r.statusText} ${detail}`.trim())
  }
  return (await r.json()) as T
}

export interface JobSpec {
  query: string; engine: 'manim' | 'hyperframes'; n: number | null
  duration_sec: number; max_rounds: number; stop_after: 1 | 2 | 3; selection: Stage1Method
  stage2: Stage2Method; query_id: string | null
}
export type Stage1Method = 'auto' | 'manual' | 'lw' | 'e2e' | 'optiset' | 'gold'
export type Stage2Method = 'set2story' | 'freeform' | 'reference'
export const createJob = (spec: JobSpec) =>
  req<{ id: string; queue_position: number }>(apiPath('/api/jobs'), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(spec),
  })
export const submitSelection = (id: string, resource_ids: string[]) =>
  req<{ id: string; selected: string[] }>(apiPath(`/api/jobs/${encodeURIComponent(id)}/selection`), {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resource_ids }),
  })
export const getJob = (id: string) => req<Job>(apiPath(`/api/jobs/${encodeURIComponent(id)}`))
export const listJobs = () => req<JobSummary[]>(apiPath('/api/jobs'))
export const listBenchmark = () => req<BenchQuery[]>(apiPath('/api/benchmark'))
export const getHealth = () => req<Health>(apiPath('/api/health'))
export const figureUrl = (id: string) => apiPath(`/api/figure/${encodeURIComponent(id)}`)
export const isFig = (id: string) => id.startsWith('FIG_')
