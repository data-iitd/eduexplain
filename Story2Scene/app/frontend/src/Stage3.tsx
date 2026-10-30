import type { Job } from './types'
import { apiPath } from './api'

export default function Stage3View({ job }: { job: Job }) {
  const s = job.stage3
  const video = job.video_url ? apiPath(job.video_url) : null
  const running = job.stages['3'] === 'running'
  const tail = (job.log ?? []).slice(-30)
  return (
    <div className="space-y-2 text-sm">
      <p><b>Engine:</b> {s?.engine ?? String(job.spec?.engine ?? '')} {s?.status && <>&middot; <b>status:</b> {s.status}</>} {s?.ok === false && <span className="text-red-700">(not ok)</span>}</p>
      {video && (
        <div className="space-y-1">
          <video key={video} src={video} controls className="w-full rounded bg-black" />
          <a href={video} download className="text-blue-700 underline">Download video</a>
        </div>
      )}
      {s?.timing_sec && (
        <table className="text-xs"><tbody>
          {Object.entries(s.timing_sec).map(([k, v]) => <tr key={k}><td className="pr-3">{k}</td><td className="tabular-nums">{typeof v === 'number' ? v.toFixed(1) : String(v)} s</td></tr>)}
        </tbody></table>
      )}
      <details open={running || !job.video_url}>
        <summary className="cursor-pointer text-xs font-medium">Log (last {tail.length})</summary>
        <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded bg-slate-900 p-2 text-xs text-slate-100">{tail.join('\n') || '(empty)'}</pre>
      </details>
    </div>
  )
}
