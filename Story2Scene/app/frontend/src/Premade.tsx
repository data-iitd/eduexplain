import type { JobSummary } from './types'

// Premade results: the demo evaluation videos, one card per benchmark query, one button per Stage 1 + Stage 2
// configuration (SetCoRe-LLM + Set2Story is the full system)
const ORDER = ['SetCoRe-LLM', 'Benchmark gold set', 'OptiSet', 'SetCoRe-LW', 'SetCoRe-E2E']
const ORDER2 = ['Set2Story', 'FreeForm', 'Benchmark reference']
const rank = (j: JobSummary) => ORDER.indexOf(j.stage1_label ?? '') * 10 + ORDER2.indexOf(j.stage2_label ?? '')

export default function Premade({ jobs, current, open }: { jobs: JobSummary[]; current: string | null; open: (id: string) => void }) {
  const byQuery = new Map<string, JobSummary[]>()
  for (const j of jobs) {
    const k = j.query_id ?? j.query
    byQuery.set(k, [...(byQuery.get(k) ?? []), j])
  }
  if (byQuery.size === 0) return <p className="mt-2 text-slate-500">No premade results.</p>
  return (
    <div className="mt-2 space-y-2">
      <p className="text-xs text-slate-600">
        The same benchmark query through different Stage 1 and Stage 2 methods, each rendered by the Manim video module.
        <b> SetCoRe-LLM + Set2Story</b> is our full pipeline; the others swap in a baseline or the benchmark's own gold set / reference storyboard.
      </p>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
        {[...byQuery.entries()].map(([qid, list]) => (
          <div key={qid} className="min-w-0 rounded border border-slate-200 p-2">
            <p className="mb-1.5 break-words font-medium">{list[0].query} <span className="font-mono text-xs text-slate-400">{qid}</span></p>
            <div className="flex flex-wrap gap-1">
              {[...list].sort((a, b) => rank(a) - rank(b)).map((j) => {
                const ours = j.stage1_label === 'SetCoRe-LLM' && j.stage2_label === 'Set2Story'
                return (
                  <button key={j.id} onClick={() => open(j.id)}
                    className={`rounded border px-2 py-1 text-left text-xs ${j.id === current ? 'border-blue-600 bg-blue-600 text-white' : ours ? 'border-green-500 bg-green-50 hover:bg-green-100' : 'border-slate-300 hover:bg-slate-100'}`}>
                    {j.stage1_label} + {j.stage2_label}{ours ? ' ★' : ''}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
