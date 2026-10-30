import type { Stage2 } from './types'
import { isFig } from './api'

export default function Stage2View({ s }: { s: Stage2 | null }) {
  if (!s) return <p className="text-sm text-slate-500">No data yet.</p>
  const sb = s.storyboard?.storyboard ?? s.storyboard
  if (!sb) return <p className="text-sm text-slate-500">Empty storyboard.</p>
  const rt = Array.isArray(sb.role_template) ? sb.role_template.join(' → ') : sb.role_template
  return (
    <div className="space-y-2 text-sm">
      <div className="rounded bg-slate-100 p-2">
        <p><b>Intent:</b> {sb.intent}</p>
        <p><b>Roles:</b> {rt}</p>
        <p><b>Total:</b> {sb.total_duration_sec} s &middot; {sb.segments?.length ?? 0} segments</p>
      </div>
      <ol className="space-y-2">
        {(sb.segments ?? []).map((g) => (
          <li key={`${g.segment_id}-${g.order}`} className="rounded border border-slate-300 p-2">
            <div className="mb-1 flex flex-wrap items-center gap-1 text-xs">
              <span className="font-bold">#{g.order}</span><span className="font-medium">{g.role}</span>
              <span className="text-slate-500">{g.duration_sec}s</span>
              {(g.resource_ids ?? []).map((id) => (
                <span key={id} className={`rounded px-1 font-mono ${isFig(id) ? 'bg-purple-200 text-purple-900 ring-1 ring-purple-400' : 'bg-slate-200'}`}>{id}</span>
              ))}
            </div>
            <p className="break-words"><b>Narration:</b> {g.narration}</p>
            <p className="break-words text-slate-600"><b>On screen:</b> {g.on_screen}</p>
          </li>
        ))}
      </ol>
      {s.seconds != null && <p className="text-xs text-slate-500">{s.seconds.toFixed(1)} s</p>}
    </div>
  )
}
