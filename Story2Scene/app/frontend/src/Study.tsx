import { useEffect, useRef, useState } from 'react'
import { apiPath } from './api'

// User study E2 (tap-only): two unlabeled videos + the explanation set they were made from -> three questions
// (correctness, source faithfulness, coherence), each Video 1 / Video 2 / Same / Can't judge -> reveal.
interface Res { id: string; kind: 'text' | 'figure'; source: string; text: string; summary: string; caption: string }
interface Pair {
  k: number; query: string; answered: boolean; same_set: boolean; sets: { a: Res[]; b?: Res[] }
  a?: string; b?: string; a_ours?: boolean; b_ours?: boolean; comparison?: string; choices?: Record<string, string>
}
interface Session { sid: string; profile?: { role: string; level: string }; pairs: Pair[]; finished: boolean; n_extra: number }

const LS = 'videoexplain.study'
const lsGet = () => { try { return localStorage.getItem(LS) } catch { return null } }
const lsSet = (v: string | null) => { try { if (v) localStorage.setItem(LS, v); else localStorage.removeItem(LS) } catch { /* ignore */ } }
async function call<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(apiPath(path), body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) { let d = ''; try { d = (await r.json()).detail } catch { /* ignore */ } throw new Error(d || `HTTP ${r.status}`) }
  return (await r.json()) as T
}

// the three questions; examples are on topics that are not in the study, so they do not hint at any answer
const QUESTIONS = [
  {
    key: 'correct', icon: '✅', title: 'Correctness', ask: 'Which video has fewer errors in the concepts it explains?',
    look: 'an incorrect definition, a wrong claim, a wrong example (or a wrong formula, unit or label)',
    example: 'A video says “plants take in oxygen to make their food”. That is wrong: photosynthesis uses carbon dioxide.',
  },
  {
    key: 'faithful', icon: '📚', title: 'Source faithfulness', ask: 'Which video better keeps the meaning of the material it was made from?',
    look: 'a claim that is not supported by the material, a diagram that is not in the material, a change to what the material says',
    example: 'The material says “water boils at 100 °C at sea level”, but the video says “water boils at 100 °C everywhere, even on mountains”. Even a true fact counts as a problem here if the material does not say it.',
  },
  {
    key: 'coherent', icon: '🧩', title: 'Coherence', ask: 'Which video explains the topic in a more coherent way?',
    look: 'a term that is used but not explained (when it should have been), repetitions, narration that does not follow a logical path',
    example: 'A video uses the term “latent heat” without saying what it is, repeats the same sentence twice, or shows the result before the steps that lead to it.',
  },
]
const CHOICES = [
  { v: 'a', t: '🟦 Video 1', on: 'border-blue-600 bg-blue-600 text-white' },
  { v: 'b', t: '🟧 Video 2', on: 'border-orange-500 bg-orange-500 text-white' },
  { v: 'same', t: '🤝 Same', on: 'border-green-600 bg-green-600 text-white' },
  { v: 'cant', t: '🤷 Can’t judge', on: 'border-slate-600 bg-slate-600 text-white' },
]
const CHOICE_TEXT: Record<string, string> = { a: 'Video 1', b: 'Video 2', same: 'Same', cant: 'Can’t judge' }
const LEVELS: Record<string, string[]> = {
  student: ['Class 6-8', 'Class 9-10', 'Class 11-12', 'Undergraduate', 'Postgraduate'],
  teacher: ['School teacher', 'College faculty', 'Tutor / coach'],
  other: ['Parent', 'Researcher', 'Other'],
}
const NICE = (j: string) => (j.includes('lw__') ? 'SetCoRe (our evidence picker)' : j.includes('optiset') ? 'OptiSet (an LLM baseline)'
  : j.endsWith('__ff') ? 'FreeForm (a one-shot LLM storyboard)' : 'Set2Story (our storyboard planner)')
const btn = 'rounded-xl border-2 px-4 py-3 text-base font-medium transition active:scale-95'

function Dots({ s, at }: { s: Session; at: number }) {
  return (
    <div className="flex items-center justify-center gap-1.5" aria-label={`pair ${at + 1} of ${s.pairs.length}`}>
      {s.pairs.map((p, i) => (
        <span key={i} className={`h-3 rounded-full transition-all ${p.answered ? 'w-3 bg-green-500' : i === at ? 'w-8 bg-blue-600' : 'w-3 bg-slate-300'}`} />
      ))}
      <span className="ml-1 text-xs text-slate-500">{s.pairs.filter((p) => p.answered).length}/{s.pairs.length}</span>
    </div>
  )
}

function QuestionGuide({ q, open }: { q: typeof QUESTIONS[number]; open?: boolean }) {
  // meaning + what to look for + an example, folded behind a tap on the pair page
  const [show, setShow] = useState(!!open)
  return (
    <div className="text-left text-sm">
      {!open && <button onClick={() => setShow(!show)} className="text-blue-700 underline">{show ? 'Hide' : 'What does this mean?'}</button>}
      {show && (
        <div className="mt-1 space-y-1 rounded-lg bg-slate-50 p-2 text-slate-700">
          <p><b>Look for:</b> {q.look}.</p>
          <p><b>Example:</b> <i>{q.example}</i></p>
        </div>
      )}
    </div>
  )
}

function Intro({ onStart }: { onStart: (role: string, level: string) => void }) {
  const [consent, setConsent] = useState(false)
  const [role, setRole] = useState<string | null>(null)
  return (
    <div className="mx-auto max-w-2xl space-y-5 text-center">
      <div className="text-5xl">🎬📚</div>
      <h1 className="text-2xl font-bold">Which explainer video is better?</h1>
      <p className="text-slate-700">You will see <b>4 pairs</b> of short videos (1-3 min each) that answer the same question, together with the
        <b> textbook material</b> each video was made from. For every pair you answer <b>3 questions</b> by tapping. <b>No typing</b>, about <b>20 minutes</b>, completely anonymous.</p>
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div className="rounded-xl bg-blue-50 p-3">👀<br />Watch both videos</div>
        <div className="rounded-xl bg-amber-50 p-3">📚<br />Check the material</div>
        <div className="rounded-xl bg-green-50 p-3">👆<br />Answer 3 questions</div>
      </div>
      <div className="space-y-2 text-left">
        <p className="text-center font-medium">The 3 questions</p>
        {QUESTIONS.map((q) => (
          <div key={q.key} className="rounded-xl border border-slate-200 p-3">
            <p className="font-semibold">{q.icon} {q.title}: <span className="font-normal">{q.ask}</span></p>
            <QuestionGuide q={q} open />
          </div>
        ))}
        <p className="text-center text-sm text-slate-600">Each question has 4 answers: <b>Video 1</b>, <b>Video 2</b>, <b>Same</b> (equally good or equally bad) or <b>Can’t judge</b> (you cannot tell, for example the topic is new to you).</p>
      </div>
      <p className="text-sm text-slate-600">At the end we reveal which method made each video you watched.</p>
      <button onClick={() => setConsent(!consent)} className={`${btn} w-full ${consent ? 'border-green-600 bg-green-50' : 'border-slate-300'}`}>
        {consent ? '☑' : '☐'} I agree to take part. My taps are stored anonymously for research (no name, no e-mail).
      </button>
      {consent && (
        <div className="space-y-2">
          <p className="font-medium">I am a…</p>
          <div className="grid grid-cols-3 gap-2">
            {[['student', '🎒 Student'], ['teacher', '🧑‍🏫 Teacher'], ['other', '🙋 Other']].map(([k, t]) => (
              <button key={k} onClick={() => setRole(k)} className={`${btn} ${role === k ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300'}`}>{t}</button>
            ))}
          </div>
        </div>
      )}
      {consent && role && (
        <div className="space-y-2">
          <p className="font-medium">Tap the one that fits best to start</p>
          <div className="flex flex-wrap justify-center gap-2">
            {LEVELS[role].map((l) => <button key={l} onClick={() => onStart(role, l)} className={`${btn} border-blue-300 hover:bg-blue-50`}>{l} ➜</button>)}
          </div>
        </div>
      )}
    </div>
  )
}

function Player({ src, label, onWatch }: { src: string; label: string; onWatch: (f: number) => void }) {
  // fraction of the video actually played (seconds of playback, so speeding up counts but seeking ahead does not)
  const played = useRef(0)
  const last = useRef<number | null>(null)
  const [f, setF] = useState(0)
  const upd = (v: HTMLVideoElement) => {
    const t = v.currentTime
    if (last.current !== null && t > last.current && t - last.current <= 1.5 * Math.max(1, v.playbackRate)) played.current += t - last.current
    last.current = t
    const x = v.duration ? Math.min(1, played.current / v.duration) : 0
    if (x - f >= 0.01 || (x >= 0.5 && f < 0.5)) { setF(x); onWatch(x) }
  }
  return (
    <div className="min-w-0 space-y-1">
      <div className="flex items-center justify-between text-sm font-bold"><span>{label}</span><span className="text-xs font-normal text-slate-500">{f >= 0.5 ? '✔ watched' : `${Math.round(f * 100)}% watched`}</span></div>
      <video src={src} controls playsInline preload="metadata" className="aspect-video w-full rounded-xl bg-black"
        onTimeUpdate={(e) => upd(e.currentTarget)} onSeeking={(e) => { last.current = e.currentTarget.currentTime }} />
      <div className="h-1.5 rounded-full bg-slate-200"><div className="h-full rounded-full bg-blue-500 transition-all" style={{ width: `${Math.min(100, (f / 0.5) * 100)}%` }} /></div>
    </div>
  )
}

// numbered labels (Text 1, Figure 1, ...) of one explanation set, in its order
const labels = (set: Res[]) => { let t = 0, f = 0; return set.map((r) => (r.kind === 'figure' ? `Figure ${++f}` : `Text ${++t}`)) }

interface Open { set: Res[]; i: number; title: string }

function Reader({ o, onClose }: { o: Open; onClose: () => void }) {
  // one resource in full, with prev / next inside its set
  const [i, setI] = useState(o.i)
  const r = o.set[i], lab = labels(o.set)
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-40 flex items-end justify-center bg-black/50 sm:items-center" onClick={onClose}>
      <div role="dialog" aria-label={`${lab[i]} of ${o.title}`} onClick={(e) => e.stopPropagation()}
        className="flex max-h-[88vh] w-full max-w-3xl flex-col rounded-t-2xl bg-white sm:rounded-2xl">
        <div className="flex items-center gap-2 border-b border-slate-200 p-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs text-slate-500">📚 {o.title}</p>
            <p className="font-semibold">{r.kind === 'figure' ? '🖼️' : '📄'} {lab[i]} <span className="text-xs font-normal text-slate-500">· {r.source}</span></p>
          </div>
          <button onClick={onClose} className="rounded-full border border-slate-300 px-3 py-1 text-lg" aria-label="Close">✕</button>
        </div>
        <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-3 text-sm">
          {r.kind === 'figure' && <img src={apiPath(`/api/figure/${r.id}`)} alt={r.caption || r.summary} className="max-h-[50vh] w-full rounded bg-white object-contain" />}
          {r.caption && <p className="text-xs font-medium">{r.caption}</p>}
          <p className="whitespace-pre-line break-words text-slate-800">{r.text}</p>
        </div>
        {o.set.length > 1 && (
          <div className="flex items-center justify-between gap-2 border-t border-slate-200 p-2">
            <button disabled={i === 0} onClick={() => setI(i - 1)} className={`${btn} py-2 disabled:opacity-30`}>◀ {i > 0 ? lab[i - 1] : ''}</button>
            <span className="text-xs text-slate-500">{i + 1} / {o.set.length}</span>
            <button disabled={i === o.set.length - 1} onClick={() => setI(i + 1)} className={`${btn} py-2 disabled:opacity-30`}>{i < o.set.length - 1 ? lab[i + 1] : ''} ▶</button>
          </div>
        )}
      </div>
    </div>
  )
}

function Material({ title, set, tone, open }: { title: string; set: Res[]; tone: string; open: (o: Open) => void }) {
  // compact, like the Stage 1 panel: thumbnail + one-line summary per resource; tap for the full text
  const lab = labels(set)
  return (
    <div className={`space-y-1 rounded-xl p-2 ${tone}`}>
      <p className="text-sm font-semibold">📚 {title} <span className="font-normal text-slate-500">· tap to read</span></p>
      <ul className="space-y-1">
        {set.map((r, i) => (
          <li key={r.id}>
            <button onClick={() => open({ set, i, title })} className="flex w-full items-center gap-2 rounded-lg border border-slate-200 bg-white p-1.5 text-left hover:border-blue-400 active:scale-[0.99]">
              {r.kind === 'figure'
                ? <img src={apiPath(`/api/figure/${r.id}`)} alt="" loading="lazy" className="h-12 w-12 shrink-0 rounded border bg-white object-contain" />
                : <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded border bg-slate-50 text-xl">📄</span>}
              <span className="min-w-0 flex-1">
                <span className="block text-xs font-semibold">{lab[i]} <span className="font-normal text-slate-500">· {r.source}</span></span>
                <span className="line-clamp-2 block break-words text-xs text-slate-700">{r.summary}</span>
              </span>
              <span className="shrink-0 text-slate-400">▸</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Chips({ who, set, title, tone, open }: { who: string; set: Res[]; title: string; tone: string; open: (o: Open) => void }) {
  // the material a video used, as tappable chips next to a question
  const lab = labels(set)
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <span className="font-medium">{who} used:</span>
      {set.map((r, i) => (
        <button key={r.id} onClick={() => open({ set, i, title })} className={`rounded-full border px-2 py-1 ${tone}`}>{r.kind === 'figure' ? '🖼️' : '📄'} {lab[i]}</button>
      ))}
    </div>
  )
}

function PairView({ s, k, onDone }: { s: Session; k: number; onDone: (s: Session) => void }) {
  const p = s.pairs[k]
  const [watch, setWatch] = useState({ a: 0, b: 0 })
  const [ch, setCh] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [reading, setReading] = useState<Open | null>(null)
  const t0 = useRef(Date.now())
  const T = { a: p.same_set ? 'Both videos were made from this material' : 'Video 1 was made from this material', b: 'Video 2 was made from this material' }
  const unlocked = watch.a >= 0.5 && watch.b >= 0.5
  const ready = unlocked && QUESTIONS.every((q) => ch[q.key])
  const vid = (side: string) => apiPath(`/api/study/${s.sid}/video/${k}/${side}`)
  async function submit() {
    setBusy(true); setErr(null)
    try { onDone(await call<Session>(`/api/study/${s.sid}/answer/${k}`, { choices: ch, watch, seconds: (Date.now() - t0.current) / 1000 })) } catch (e) { setErr(String(e instanceof Error ? e.message : e)); setBusy(false) }
  }
  return (
    <div className="space-y-4">
      <Dots s={s} at={k} />
      <div className="rounded-xl bg-indigo-50 p-3 text-center"><span className="text-xs uppercase tracking-wide text-indigo-700">The question</span><p className="text-lg font-semibold">“{p.query}”</p></div>
      <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <Player src={vid('a')} label="🟦 Video 1" onWatch={(f) => setWatch((w) => ({ ...w, a: f }))} />
          {!p.same_set && <Material title={T.a} set={p.sets.a} tone="bg-blue-50" open={setReading} />}
        </div>
        <div className="min-w-0 space-y-2">
          <Player src={vid('b')} label="🟧 Video 2" onWatch={(f) => setWatch((w) => ({ ...w, b: f }))} />
          {!p.same_set && <Material title={T.b} set={p.sets.b ?? []} tone="bg-orange-50" open={setReading} />}
        </div>
      </div>
      {p.same_set && <Material title={T.a} set={p.sets.a} tone="bg-slate-100" open={setReading} />}
      {reading && <Reader o={reading} onClose={() => setReading(null)} />}
      {!unlocked ? (
        <p className="rounded-xl bg-slate-100 p-4 text-center text-slate-700">🔒 Watch at least half of <b>both</b> videos to unlock the questions (you can speed them up with ⋮ → playback speed). Meanwhile, tap the material to read what each video was made from.</p>
      ) : (
        <div className="space-y-3">
          {QUESTIONS.map((q, i) => (
            <div key={q.key} className="space-y-2 rounded-xl border-2 border-slate-200 p-3">
              <p className="font-semibold"><span className="text-slate-400">{i + 1}/3</span> {q.icon} {q.title}: <span className="font-normal">{q.ask}</span></p>
              <QuestionGuide q={q} />
              {q.key === 'faithful' && (
                <div className="space-y-1 rounded-lg bg-slate-50 p-2">
                  {p.same_set
                    ? <Chips who="Both videos" set={p.sets.a} title={T.a} tone="border-slate-400 bg-white" open={setReading} />
                    : <>
                      <Chips who="🟦 Video 1" set={p.sets.a} title={T.a} tone="border-blue-400 bg-blue-50" open={setReading} />
                      <Chips who="🟧 Video 2" set={p.sets.b ?? []} title={T.b} tone="border-orange-400 bg-orange-50" open={setReading} />
                    </>}
                </div>
              )}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {CHOICES.map((c) => (
                  <button key={c.v} onClick={() => setCh((o) => ({ ...o, [q.key]: c.v }))} aria-pressed={ch[q.key] === c.v}
                    className={`${btn} px-2 ${ch[q.key] === c.v ? c.on : 'border-slate-300 hover:bg-slate-50'}`}>{c.t}</button>
                ))}
              </div>
            </div>
          ))}
          {err && <p role="alert" className="text-center text-red-700">{err}</p>}
          <button onClick={submit} disabled={!ready || busy} className={`${btn} w-full border-blue-700 bg-blue-600 text-white disabled:border-slate-300 disabled:bg-slate-300 ${ready && !busy ? 'animate-pulse' : ''}`}>
            {busy ? 'Saving…' : ready ? 'Next ➜' : `Answer all 3 questions (${QUESTIONS.filter((q) => ch[q.key]).length}/3)`}
          </button>
        </div>
      )}
    </div>
  )
}

function Finish({ s, onMore, onReset }: { s: Session; onMore: () => void; onReset: () => void }) {
  const answered = s.pairs.filter((p) => p.answered)
  return (
    <div className="mx-auto max-w-2xl space-y-4 text-center">
      <div className="text-6xl motion-safe:animate-bounce">🎉</div>
      <h1 className="text-2xl font-bold">Thank you! Your answers are saved.</h1>
      <p className="text-slate-700">Here is the reveal: who made each video you watched?</p>
      <ul className="space-y-2 text-left">
        {answered.map((p) => (
          <li key={p.k} className="rounded-xl border border-slate-200 p-3 text-sm">
            <p className="font-medium">“{p.query}”</p>
            <p className="text-xs text-slate-500">{p.comparison}</p>
            <p>🟦 Video 1: <b>{NICE(p.a ?? '')}</b></p>
            <p>🟧 Video 2: <b>{NICE(p.b ?? '')}</b></p>
            <p className="text-xs text-slate-600">Your answers: {QUESTIONS.map((q) => `${q.title} → ${CHOICE_TEXT[p.choices?.[q.key] ?? ''] ?? '-'}`).join(' · ')}</p>
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap justify-center gap-2">
        <button onClick={onMore} className={`${btn} border-blue-600 bg-blue-600 text-white`}>➕ One more pair</button>
        <a href="#" className={`${btn} border-slate-300`}>🎬 Try making a video</a>
        <button onClick={onReset} className={`${btn} border-slate-300 text-sm`}>🔄 New participant on this device</button>
      </div>
    </div>
  )
}

interface QStat { ours: number; baseline: number; same: number; cant: number; ours_win_rate: number | null; sign_test_p: number | null }
interface Summary {
  sessions: number; finished: number; roles: Record<string, number>
  comparisons: Record<string, { title: string; ours: string; baseline: string; n: number; questions: Record<string, QStat> }>
}
export function StudyResults() {
  const [s, setS] = useState<Summary | null>(null)
  useEffect(() => { call<Summary>('/api/study/summary/all').then(setS).catch(() => {}) }, [])
  if (!s) return <p>Loading…</p>
  return (
    <div className="space-y-3 text-sm">
      <h1 className="text-xl font-bold">User study results (E2)</h1>
      <p>{s.sessions} participants ({s.finished} finished) · {Object.entries(s.roles).map(([k, v]) => `${k}: ${v}`).join(', ') || 'none yet'}</p>
      {Object.values(s.comparisons).map((c) => (
        <div key={c.title} className="overflow-x-auto rounded border border-slate-300 bg-white p-3">
          <h2 className="font-semibold">{c.title} <span className="font-normal text-slate-500">({c.n} pairs answered)</span></h2>
          <table className="mt-2 text-xs">
            <thead><tr>{['question', c.ours, c.baseline, 'same', 'can’t judge', 'win rate', 'sign test p'].map((h) => <th key={h} className="px-2 text-left">{h}</th>)}</tr></thead>
            <tbody>{QUESTIONS.map((q) => { const d = c.questions[q.key]; return (
              <tr key={q.key}><td className="px-2">{q.title}</td>{[d.ours, d.baseline, d.same, d.cant, d.ours_win_rate ?? '-', d.sign_test_p?.toFixed(3) ?? '-'].map((v, i) => <td key={i} className="px-2 tabular-nums">{v}</td>)}</tr>) })}</tbody>
          </table>
        </div>
      ))}
      <p className="text-xs text-slate-500">Win rate = wins of our method among the pairs where one video was picked (same / can’t judge left out); exact two-sided sign test.</p>
    </div>
  )
}

export default function Study() {
  const [s, setS] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [cheer, setCheer] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    const sid = lsGet()
    if (!sid) { setLoading(false); return }
    call<Session>(`/api/study/${sid}`).then(setS).catch(() => lsSet(null)).finally(() => setLoading(false))
  }, [])
  const keep = (n: Session) => { setS(n); lsSet(n.sid); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  async function start(role: string, level: string) {
    setErr(null)
    try { keep(await call<Session>('/api/study/start', { role, level, consent: true })) } catch (e) { setErr(String(e instanceof Error ? e.message : e)) }
  }
  function pairDone(n: Session) { setCheer(true); window.setTimeout(() => setCheer(false), 1400); keep(n) }
  async function more() {
    if (!s) return
    try { keep(await call<Session>(`/api/study/${s.sid}/more`, {})) } catch (e) { setErr(String(e instanceof Error ? e.message : e)) }
  }
  if (loading) return <p className="text-center">Loading…</p>
  const next = s?.pairs.findIndex((p) => !p.answered) ?? -1
  return (
    <div className="relative mx-auto max-w-6xl rounded-2xl border border-slate-300 bg-white p-4 sm:p-6">
      {cheer && (
        <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-white/60">
          <div className="text-center"><div className="text-7xl motion-safe:animate-bounce">🎉</div><p className="text-2xl font-bold">Nice! Saved.</p></div>
        </div>
      )}
      {err && <p role="alert" className="mb-3 text-center text-red-700">{err}</p>}
      {!s ? <Intro onStart={start} />
        : next >= 0 ? <PairView key={`${s.sid}-${next}`} s={s} k={next} onDone={pairDone} />
          : <Finish s={s} onMore={more} onReset={() => { lsSet(null); setS(null) }} />}
    </div>
  )
}
