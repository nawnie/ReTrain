/** Small pure helpers for the Live pane: numbers, time, smoothing, and reading the model's JSON answers. */

import type { GpuSample, LiveRun } from '../types'

export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`
}

export const pct = (v: number | undefined | null) => (v === undefined || v === null ? '—' : `${Math.round(v)}%`)

/** Exponential moving average: shows the trend through step-to-step noise. `alpha` near 1 follows the raw line closely. */
export function ema(values: number[], alpha = 0.3): number[] {
  const out: number[] = []
  let prev: number | null = null
  for (const v of values) {
    prev = prev === null ? v : alpha * v + (1 - alpha) * prev
    out.push(prev)
  }
  return out
}

/** Examples per second between consecutive reports (needs the run's examples-per-epoch). */
export function speedSeries(run: LiveRun): [number, number][] {
  const out: [number, number][] = []
  for (let i = 1; i < run.loss.length; i++) {
    const dt = run.loss[i][4] - run.loss[i - 1][4]
    const de = run.loss[i][0] - run.loss[i - 1][0]
    if (dt > 0.5 && de > 0 && run.trainExamples) out.push([run.loss[i][0], (de * run.trainExamples) / dt])
  }
  return out
}

/** Downsample an array of samples to about `n` points for sparklines. */
export function thin<T>(items: T[], n: number): T[] {
  if (items.length <= n) return items
  const step = items.length / n
  return Array.from({ length: n }, (_, i) => items[Math.floor(i * step)])
}

export function latest(samples: GpuSample[]): GpuSample | null {
  return samples.length ? samples[samples.length - 1] : null
}

/* -- reading the model's answers --------------------------------------------- */

export interface ParsedAnswer {
  ok: boolean
  think: string
  say: string
  actions: { buttons: string[]; frames: number }[]
  raw: string
}

/** Pull the JSON object out of a model reply (prose around it is tolerated). */
export function parseAnswer(text: string): ParsedAnswer {
  const raw = text ?? ''
  const start = raw.indexOf('{')
  const end = raw.lastIndexOf('}')
  if (start >= 0 && end > start) {
    try {
      const d = JSON.parse(raw.slice(start, end + 1)) as Record<string, unknown>
      const actions = Array.isArray(d.actions)
        ? (d.actions as unknown[]).flatMap((a) => {
            const o = a as { buttons?: unknown; frames?: unknown }
            return Array.isArray(o.buttons) ? [{ buttons: o.buttons.map(String), frames: Number(o.frames) || 0 }] : []
          })
        : []
      return { ok: true, think: String(d.think ?? ''), say: String(d.say ?? ''), actions, raw }
    } catch {
      /* fall through to the unparsed result */
    }
  }
  return { ok: false, think: '', say: '', actions: [], raw }
}
