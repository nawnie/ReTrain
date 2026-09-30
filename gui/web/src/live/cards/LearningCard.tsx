/** "Is it actually learning?" - the per-epoch learning checks as a trend, a per-skill heatmap, and a plain-words verdict. */

import { useMemo } from 'react'
import type { EChartsCoreOption } from 'echarts/core'
import type { LiveCheck, LiveRun } from '../../types'
import { Panel } from '../../ui'
import { EChart } from '../EChart'
import { pct } from '../format'
import { baseOption, usePalette } from '../theme'

const METRICS: { key: 'valid_json' | 'actions_match' | 'first_button_match' | 'exact'; label: string; hint: string }[] = [
  { key: 'valid_json', label: 'Valid reply', hint: 'the answer is well-formed JSON with real buttons' },
  { key: 'actions_match', label: 'Right action', hint: 'the same buttons as the reference answer' },
  { key: 'first_button_match', label: 'First button', hint: 'at least the first press is right' },
  { key: 'exact', label: 'Exact copy', hint: 'the whole reply is word-for-word identical' },
]

/** One plain sentence about progress, so nobody has to interpret a chart. */
function verdict(checks: LiveCheck[]): { tone: 'ok' | 'warn' | 'risk' | 'neutral'; text: string } {
  if (checks.length === 0) return { tone: 'neutral', text: 'No learning check has run yet. The first one is the untrained baseline.' }
  const base = checks[0]
  const last = checks[checks.length - 1]
  if (checks.length === 1) return { tone: 'neutral', text: `Baseline recorded: the untrained model picks the right action ${pct(base.actions_match)} of the time. Progress shows after the first epoch.` }
  const gain = (last.actions_match ?? 0) - (base.actions_match ?? 0)
  if (gain >= 20) return { tone: 'ok', text: `Learning: right-action accuracy rose from ${pct(base.actions_match)} to ${pct(last.actions_match)} (+${Math.round(gain)} points) on questions it never trained on.` }
  if (gain > 3) return { tone: 'warn', text: `Improving slowly: right-action accuracy ${pct(base.actions_match)} → ${pct(last.actions_match)}.` }
  return { tone: 'risk', text: `Not learning yet: right-action accuracy is ${pct(last.actions_match)} (baseline ${pct(base.actions_match)}).` }
}

export function LearningCard({ run }: { run: LiveRun }) {
  const p = usePalette()
  const checks = run.checks
  const v = verdict(checks)
  const last = checks[checks.length - 1]
  const first = checks[0]

  const trend = useMemo<EChartsCoreOption>(() => {
    const colours = [p.inkMuted, p.ok, p.accent, p.warn]
    return {
      ...baseOption(p),
      grid: { left: 40, right: 40, top: 28, bottom: 26 },
      legend: { top: 0, textStyle: { color: p.inkMuted }, itemWidth: 14, itemHeight: 8 },
      tooltip: { ...baseOption(p).tooltip, trigger: 'axis', valueFormatter: (x: unknown) => `${Math.round(Number(x))}%` },
      xAxis: { type: 'category', data: checks.map((c) => c.tag), axisLabel: { color: p.inkFaint }, axisLine: { lineStyle: { color: p.line } }, boundaryGap: false },
      yAxis: { type: 'value', min: 0, max: 100, axisLabel: { color: p.inkFaint, formatter: '{value}%' }, splitLine: { lineStyle: { color: p.line, type: 'dashed' } } },
      series: METRICS.map((m, i) => ({
        name: m.label, type: 'line', data: checks.map((c) => c[m.key] ?? null), symbolSize: 8, lineStyle: { width: m.key === 'actions_match' ? 4 : 2, color: colours[i] },
        itemStyle: { color: colours[i] }, endLabel: { show: m.key === 'actions_match', formatter: '{c}%', color: colours[i] },
      })),
    }
  }, [checks, p])

  const heat = useMemo<EChartsCoreOption | null>(() => {
    const cats = Array.from(new Set(checks.flatMap((c) => Object.keys(c.by_category ?? {})))).sort()
    if (!cats.length) return null
    const data: [number, number, number][] = []
    checks.forEach((c, x) => cats.forEach((cat, y) => data.push([x, y, c.by_category?.[cat]?.actions_match ?? 0])))
    return {
      ...baseOption(p),
      grid: { left: 88, right: 12, top: 6, bottom: 24 },
      tooltip: { ...baseOption(p).tooltip, formatter: (q: { value: number[] }) => `<b>${cats[q.value[1]].replace(/_/g, ' ')}</b><br/>${checks[q.value[0]].tag}: ${Math.round(q.value[2])}% right action` },
      xAxis: { type: 'category', data: checks.map((c) => c.tag), axisLabel: { color: p.inkFaint }, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: false } },
      yAxis: { type: 'category', data: cats, axisLabel: { color: p.inkMuted, formatter: (s: string) => s.replace(/_/g, ' ') }, axisLine: { show: false }, axisTick: { show: false } },
      visualMap: { min: 0, max: 100, show: false, inRange: { color: [p.risk, p.warn, p.ok] } },
      series: [{ type: 'heatmap', data, label: { show: true, color: '#fff', fontSize: 11, formatter: (q: { value: number[] }) => `${Math.round(q.value[2])}` }, itemStyle: { borderColor: p.surface, borderWidth: 2, borderRadius: 3 } }],
    }
  }, [checks, p])

  return (
    <Panel title="Is it actually learning?" note="After each epoch the trainer asks the model to answer questions it never trained on, and scores the answers.">
      <p className="verdict" data-tone={v.tone}>{v.text}</p>

      {checks.length ? (
        <>
          <div className="score-row">
            {METRICS.map((m) => {
              const now = last?.[m.key]
              const delta = first && last && first !== last && now !== undefined && first[m.key] !== undefined ? now - (first[m.key] as number) : null
              return (
                <div key={m.key} className="score" title={m.hint}>
                  <span>{m.label}</span>
                  <b>{pct(now)}</b>
                  {delta !== null ? <small data-up={delta >= 0}>{delta >= 0 ? '▲' : '▼'} {Math.abs(Math.round(delta))} pts</small> : <small>baseline</small>}
                </div>
              )
            })}
          </div>
          <EChart option={trend} height={170} label="Learning check scores across baseline and each epoch" />
          {heat ? (
            <>
              <h3 className="sub">Right-action score by skill</h3>
              <EChart option={heat} height={Math.max(90, 24 * new Set(checks.flatMap((c) => Object.keys(c.by_category ?? {}))).size + 34)} label="Right-action score for each skill at each check" />
            </>
          ) : null}
        </>
      ) : null}
    </Panel>
  )
}
