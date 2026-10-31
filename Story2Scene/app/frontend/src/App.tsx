import { useCallback, useEffect, useRef, useState } from 'react'
import { createJob, getHealth, getJob, listBenchmark, listJobs } from './api'
import type { Stage1Method, Stage2Method } from './api'
import type { BenchQuery, Health, Job, JobSummary, StageState } from './types'
import { Panel, ProgressBar, StatusBadge } from './ui'
import Premade from './Premade'
import Study, { StudyResults } from './Study'
import Stage1View from './Stage1'
import Stage2View from './Stage2'
import Stage3View from './Stage3'
import CompareTab, { S1L, S2L } from './Compare'

const LS = 'videoexplain.lastJob'
const lsGet = () => { try { return localStorage.getItem(LS) } catch { return null } }
const lsSet = (v: string) => { try { localStorage.setItem(LS, v) } catch { /* ignore */ } }
const lsClear = () => { try { localStorage.removeItem(LS) } catch { /* ignore */ } }
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e))
// tabs by URL hash, so the study can be shared as .../VideoModule/#study
const hashTab = () => (location.hash === '#study' ? 'study' : location.hash === '#study-results' ? 'results' : location.hash === '#compare' ? 'compare' : 'make')
const etaText = (e?: { eta_min: unknown }) => (e ? (Array.isArray(e.eta_min) ? e.eta_min.join('-') : String(e.eta_min)) : '?')

export default function App() {
  const [tab, setTab] = useState(hashTab())
  useEffect(() => { const f = () => setTab(hashTab()); window.addEventListener('hashchange', f); return () => window.removeEventListener('hashchange', f) }, [])
  const tabCls = (on: boolean) => `rounded-t-lg border border-b-0 px-4 py-2 text-sm font-medium ${on ? 'border-slate-300 bg-white' : 'border-transparent text-slate-600 hover:text-slate-900'}`
  return (
    <div className="mx-auto max-w-[1600px] p-3">
      <nav className="flex gap-1 border-b border-slate-300">
        <a href="#" className={tabCls(tab === 'make')}>🎬 Make a video</a>
        <a href="#compare" className={tabCls(tab === 'compare')}>⚖️ Compare</a>
        <a href="#study" className={tabCls(tab === 'study')}>🗳️ User study</a>
      </nav>
      <div className="pt-3">{tab === 'study' ? <Study /> : tab === 'results' ? <StudyResults /> : tab === 'compare' ? <CompareTab lastJob={lsGet()} /> : <Maker />}</div>
    </div>
  )
}

function Maker() {
  const [query, setQuery] = useState('')
  const [engine, setEngine] = useState<'manim' | 'hyperframes'>('manim')
  const [duration, setDuration] = useState(180)
  const [n, setN] = useState('')
  const [selection, setSelection] = useState<Stage1Method>('auto')
  const [stage2, setStage2] = useState<Stage2Method>('set2story')
  const [bench, setBench] = useState<BenchQuery[]>([])
  const [queryId, setQueryId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(lsGet())
  const timer = useRef<number | null>(null)

  const refreshSide = useCallback(() => {
    getHealth().then(setHealth).catch(() => setHealth(null))
    listJobs().then(setJobs).catch(() => {})
  }, [])

  useEffect(() => { refreshSide(); const t = window.setInterval(refreshSide, 10000); return () => clearInterval(t) }, [refreshSide])
  useEffect(() => { listBenchmark().then(setBench).catch(() => {}) }, [])
  // the benchmark gold set needs a benchmark query, the reference storyboard needs the gold set
  useEffect(() => { if (!queryId && selection === 'gold') setSelection('auto') }, [queryId, selection])
  useEffect(() => { if (selection !== 'gold' && stage2 === 'reference') setStage2('set2story') }, [selection, stage2])

  // poll the selected job
  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    const tick = async () => {
      try {
        const j = await getJob(jobId)
        if (cancelled) return
        setJob(j); setErr(null)
        // keep polling while awaiting a manual selection, so the job resumes in view once it is submitted
        if (j.status === 'queued' || j.status === 'running' || j.status === 'awaiting_selection') timer.current = window.setTimeout(tick, 2000)
        else refreshSide()
      } catch (e) {
        if (cancelled) return
        // the remembered job is gone (archived or deleted): forget it instead of showing an error
        if (errMsg(e).startsWith('HTTP 404')) { lsClear(); setJobId(null); setJob(null); setErr(null); return }
        setErr(`Job ${jobId}: ${errMsg(e)}`)
        timer.current = window.setTimeout(tick, 4000) // keep retrying on network errors
      }
    }
    tick()
    return () => { cancelled = true; if (timer.current) clearTimeout(timer.current) }
  }, [jobId, refreshSide])

  const active = job != null && (job.status === 'queued' || job.status === 'running' || job.status === 'awaiting_selection')

  async function submit() {
    setErr(null); setSubmitting(true)
    try {
      const nn = n.trim() === '' ? null : Number(n)
      if (nn !== null && (!Number.isInteger(nn) || nn < 1)) throw new Error('n must be a positive integer or blank')
      const r = await createJob({ query: query.trim(), engine, n: nn, duration_sec: duration, max_rounds: 3, stop_after: 3, selection, stage2, query_id: queryId })
      lsSet(r.id); setJob(null); setJobId(r.id); refreshSide()
    } catch (e) { setErr('Submit failed: ' + errMsg(e)) } finally { setSubmitting(false) }
  }
  function open(id: string) { lsSet(id); setJob(null); setJobId(id) }
  function pickBench(id: string) {
    const b = bench.find((x) => x.query_id === id)
    setQueryId(b ? b.query_id : null)
    if (b) { setQuery(b.query); if ([120, 180, 240].includes(b.T)) setDuration(b.T) }
  }
  const premade = jobs.filter((j) => j.premade)
  const recent = jobs.filter((j) => !j.premade)

  const st = (k: string): StageState => job?.stages?.[k] ?? 'pending'
  const eng = health?.engines
  const dot = (up?: boolean, sl?: boolean) => (up ? (sl ? 'sleeping' : 'up') : 'down')

  return (
    <div className="space-y-3">
      <div className="space-y-2 rounded-lg border border-slate-300 bg-white p-3">
        <label className="block text-sm">Benchmark query <span className="text-slate-500">(optional: needed for the benchmark gold set and reference storyboard)</span>
          <select value={queryId ?? ''} onChange={(e) => pickBench(e.target.value)} className="mt-1 block w-full min-w-0 rounded border p-1">
            <option value="">— none: type your own question below —</option>
            {bench.map((b) => <option key={b.query_id} value={b.query_id}>{b.query_id} · {b.query.length > 110 ? b.query.slice(0, 110) + '…' : b.query}</option>)}
          </select>
        </label>
        <textarea value={query} onChange={(e) => { setQuery(e.target.value); if (queryId && e.target.value !== bench.find((b) => b.query_id === queryId)?.query) setQueryId(null) }} rows={2} placeholder="Ask a question to explain, e.g. 'Why do we have seasons?'" className="w-full rounded border border-slate-300 p-2" />
        <div className="flex flex-wrap items-start gap-x-6 gap-y-2 text-sm">
          <fieldset className="space-y-1">
            <legend className="font-medium">Engine</legend>
            <label className="block"><input type="radio" name="engine" checked={engine === 'manim'} onChange={() => setEngine('manim')} /> Manim <span className="text-slate-500">(~{etaText(eng?.manim)} min)</span></label>
            <label className="block"><input type="radio" name="engine" checked={engine === 'hyperframes'} onChange={() => setEngine('hyperframes')} /> HyperFrames</label>
            <p className="max-w-xs text-xs text-orange-700">HyperFrames: more polished, but slower (~{etaText(eng?.hyperframes)} min)</p>
          </fieldset>
          <fieldset className="space-y-1">
            <legend className="font-medium">Stage 1 resources</legend>
            <label className="block"><input type="radio" name="selection" checked={selection === 'auto'} onChange={() => setSelection('auto')} /> SetCoRe-LLM <span className="text-slate-500">(ours: setCore + Llama-70B pick the set)</span></label>
            <label className="block"><input type="radio" name="selection" checked={selection === 'manual'} onChange={() => setSelection('manual')} /> Manual <span className="text-slate-500">(you tick the resources)</span></label>
            <label className="block"><input type="radio" name="selection" checked={selection === 'lw'} onChange={() => setSelection('lw')} /> SetCoRe-LW <span className="text-slate-500">(ours, lightweight: concept coverage, CPU, ~30 s)</span></label>
            <label className="block"><input type="radio" name="selection" checked={selection === 'e2e'} onChange={() => setSelection('e2e')} /> SetCoRe-E2E <span className="text-slate-500">(ours, learned set predictor, CPU, ~30 s)</span></label>
            <label className="block"><input type="radio" name="selection" checked={selection === 'optiset'} onChange={() => setSelection('optiset')} /> OptiSet <span className="text-slate-500">(baseline: Llama-70B, ~5-10 min)</span></label>
            <label className={`block ${queryId ? '' : 'text-slate-400'}`}><input type="radio" name="selection" disabled={!queryId} checked={selection === 'gold'} onChange={() => setSelection('gold')} /> Benchmark gold set <span className="text-slate-500">({queryId ? 'annotated set of this query' : 'pick a benchmark query'})</span></label>
          </fieldset>
          <fieldset className="space-y-1">
            <legend className="font-medium">Stage 2 storyboard</legend>
            <label className="block"><input type="radio" name="stage2" checked={stage2 === 'set2story'} onChange={() => setStage2('set2story')} /> Set2Story <span className="text-slate-500">(ours)</span></label>
            <label className="block"><input type="radio" name="stage2" checked={stage2 === 'freeform'} onChange={() => setStage2('freeform')} /> FreeForm <span className="text-slate-500">(baseline: one gemma-12B call)</span></label>
            <label className={`block ${selection === 'gold' ? '' : 'text-slate-400'}`}><input type="radio" name="stage2" disabled={selection !== 'gold'} checked={stage2 === 'reference'} onChange={() => setStage2('reference')} /> Benchmark reference <span className="text-slate-500">({selection === 'gold' ? 'its own timing' : 'needs the gold set'})</span></label>
          </fieldset>
          <label>Target duration<br />
            <select value={duration} disabled={stage2 === 'reference'} onChange={(e) => setDuration(Number(e.target.value))} className="rounded border p-1 disabled:text-slate-400">
              {[120, 180, 240].map((d) => <option key={d} value={d}>{d} s</option>)}
            </select>
          </label>
          <label>Gold-set size n<br />
            <input value={n} disabled={selection === 'optiset' || selection === 'gold'} onChange={(e) => setN(e.target.value)} placeholder="auto" inputMode="numeric" className="w-20 rounded border p-1 disabled:bg-slate-100" />
          </label>
          <button onClick={submit} disabled={(active && job?.status !== 'awaiting_selection') || submitting || !query.trim()} className="self-end rounded bg-blue-600 px-4 py-2 text-white disabled:bg-slate-400">
            {active && job?.status !== 'awaiting_selection' ? (job?.status === 'queued' ? 'Queued…' : 'Running…') : 'Run pipeline'}
          </button>
        </div>
        <div className="flex flex-wrap gap-3 border-t pt-2 text-xs text-slate-600">
          {health ? (<>
            <span>generator: <b>{dot(health.generator?.up, health.generator?.sleeping)}</b></span>
            <span>verifier: <b>{dot(health.verifier?.up, health.verifier?.sleeping)}</b></span>
            <span>{health.stage1_bge ? 'BGE retrieval ok' : <b className="text-orange-700">dummy retrieval (no BGE)</b>}</span>
            <span>queue: {health.queue ?? 0}{health.busy ? ' (busy)' : ''}</span>
          </>) : <span className="text-red-700">backend unreachable</span>}
        </div>
      </div>

      {err && <div role="alert" className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-800">{err}</div>}
      {job?.error && <div role="alert" className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-800">Job error: {job.error}</div>}
      {job && (
        <div className="space-y-2 rounded-lg border border-slate-300 bg-white p-3">
          <p className="text-sm">Job <span className="font-mono">{job.id}</span> <StatusBadge s={job.status} /> {String(job.spec?.query ?? '')}</p>
          <p className="text-xs text-slate-600">Stage 1: <b>{String(job.spec?.stage1_label ?? S1L[String(job.spec?.selection ?? 'auto')] ?? job.spec?.selection)}</b> · Stage 2: <b>{String(job.spec?.stage2_label ?? S2L[String(job.spec?.stage2 ?? 'set2story')] ?? job.spec?.stage2)}</b> · {String(job.spec?.engine ?? '')}</p>
          {!String(job.id).startsWith('premade-') && <ProgressBar p={job.progress} status={job.status} />}
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <Panel title="Stage 1: Retrieval -> gold set" state={st('1')} mode={job?.stage1?.mode}>
          <Stage1View s={job?.stage1 ?? null} jobId={job?.id} awaiting={job?.status === 'awaiting_selection'} />
        </Panel>
        <Panel title="Stage 2: Storyboard" state={st('2')} mode={job?.stage2?.mode}>
          <Stage2View s={job?.stage2 ?? null} />
        </Panel>
        <Panel title="Stage 3: Video" state={st('3')} mode={job?.stage3?.engine}>
          {job ? <Stage3View job={job} /> : <p className="text-sm text-slate-500">No data yet.</p>}
        </Panel>
      </div>

      {job?.status === 'done' && jobs.filter((j) => j.status === 'done' && (job.spec?.query_id ? j.query_id === job.spec.query_id : j.query === job.spec?.query)).length > 1 && (
        <p className="text-sm"><a href="#compare" className="text-blue-700 underline">⚖️ Compare this question's configurations side by side</a></p>
      )}

      <details open className="rounded-lg border border-slate-300 bg-white p-3 text-sm">
        <summary className="cursor-pointer font-medium">Premade results ({premade.length} videos)</summary>
        <Premade jobs={premade} current={jobId} open={open} />
      </details>

      <details className="rounded-lg border border-slate-300 bg-white p-3 text-sm">
        <summary className="cursor-pointer font-medium">Recent jobs ({recent.length})</summary>
        <ul className="mt-2 space-y-1">
          {recent.map((j) => (
            <li key={j.id}>
              <button onClick={() => open(j.id)} className={`text-left underline ${j.id === jobId ? 'font-bold' : ''}`}>{j.query || j.id}</button>{' '}
              <span className="text-xs text-slate-500">{j.engine} · {S1L[j.selection ?? 'auto'] ?? j.selection} + {S2L[j.stage2 ?? 'set2story'] ?? j.stage2}</span> <StatusBadge s={j.status} />
            </li>
          ))}
        </ul>
      </details>
    </div>
  )
}
