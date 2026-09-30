/** Loss over time: raw readings, a smoothed trend, held-out points, and (below, zoom-linked) gradient size and learning rate. */

import { useMemo, useState } from 'react'
import type { EChartsCoreOption } from 'echarts/core'
import type { LiveRun } from '../../types'
import { Panel } from '../../ui'
import { EChart } from '../EChart'
import { ema } from '../format'
import { baseOption, usePalette, type Palette } from '../theme'

const fmt = (v: number) => (v >= 1 ? v.toFixed(2) : v >= 0.1 ? v.toFixed(3) : v.toFixed(4))

function axis(p: Palette, name: string, extra: Record<string, unknown> = {}) {
  return {
    type: 'value', name, nameTextStyle: { color: p.inkFaint }, axisLine: { lineStyle: { color: p.line } },
    axisLabel: { color: p.inkFaint, fontSize: 11 }, splitLine: { lineStyle: { color: p.line, type: 'dashed' } }, ...extra,
  }
}

export function LossCard({ run }: { run: LiveRun }) {
  const p = usePalette()
  const [log, setLog] = useState(true)
  const [showRaw, setShowRaw] = useState(true)

  const heldAll = [...run.checks.filter((c) => c.eval_loss !== null && c.eval_loss !== undefined).map((c) => c.eval_loss as number), ...run.evals.map((e) => e[1])]
  const heldLatest = heldAll.length ? heldAll[heldAll.length - 1] : null

  const loss = useMemo<EChartsCoreOption>(() => {
    const pts = run.loss
    const total = Math.max(run.totalEpochs, ...pts.map((q) => q[0]), 0.01)
    const trend = ema(pts.map((q) => q[1]), 0.35)
    const held: [number, number][] = [
      ...run.checks.filter((c) => c.eval_loss !== null && c.eval_loss !== undefined && c.epoch !== undefined).map((c) => [c.epoch as number, c.eval_loss as number] as [number, number]),
      ...run.evals.map((e) => [e[0], e[1]] as [number, number]),
    ]
    const best = pts.length ? pts.reduce((a, b) => (b[1] < a[1] ? b : a)) : null
    const all = [...pts.map((q) => q[1]), ...held.map((h) => h[1])].filter((v) => v > 0)
    const nice = (v: number) => Number(v.toPrecision(1))          // round the axis ends to one significant digit so labels read cleanly
    const lo = nice(Math.min(...all) / 1.6)
    const hi = nice(Math.max(...all) * 1.6)
    const boundaries = Array.from({ length: Math.max(0, Math.floor(total) - 0) }, (_, i) => ({ xAxis: i + 1 })).filter((b) => b.xAxis < total)
    return {
      ...baseOption(p),
      grid: { left: 54, right: 16, top: 30, bottom: 46 },
      legend: { top: 0, right: 8, textStyle: { color: p.inkMuted }, itemWidth: 14, itemHeight: 8, data: ['trend', 'held-out'] },
      xAxis: axis(p, 'epoch', { min: 0, max: total, nameLocation: 'end' }),
      yAxis: axis(p, 'loss', { type: log ? 'log' : 'value', min: log ? lo : 0, max: log ? hi : undefined, scale: log, splitNumber: 4, axisLabel: { color: p.inkFaint, fontSize: 11, formatter: (v: number) => fmt(v) } }),
      dataZoom: [{ type: 'inside', xAxisIndex: 0 }, { type: 'slider', xAxisIndex: 0, height: 14, bottom: 6, borderColor: p.line, fillerColor: p.sunken, handleSize: 12 }],
      tooltip: {
        ...baseOption(p).tooltip, trigger: 'axis',
        formatter: (params: unknown) => {
          const rows = params as { seriesName: string; value: number[]; marker: string }[]
          if (!rows.length) return ''
          const e = rows[0].value[0]
          const q = run.loss.reduce((a, b) => (Math.abs(b[0] - e) < Math.abs(a[0] - e) ? b : a), run.loss[0])
          return `<b>epoch ${e.toFixed(2)}</b><br/>` + rows.map((r) => `${r.marker} ${r.seriesName}: <b>${fmt(r.value[1])}</b>`).join('<br/>') +
            (q ? `<br/><span style="opacity:.7">grad-norm ${q[2]?.toFixed(2) ?? '—'} · lr ${q[3]?.toExponential(1) ?? '—'}</span>` : '')
        },
      },
      series: [
        { name: 'raw', type: 'line', data: pts.map((q) => [q[0], q[1]]), showSymbol: false, lineStyle: { width: 1, color: p.accent, opacity: showRaw ? 0.35 : 0 }, z: 1, silent: !showRaw },
        {
          name: 'trend', type: 'line', smooth: true, showSymbol: false, data: pts.map((q, i) => [q[0], trend[i]]), lineStyle: { width: 3, color: p.accent }, z: 3,
          areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${p.accent}55` }, { offset: 1, color: `${p.accent}00` }] } },
          markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: p.inkFaint, type: 'dashed', opacity: 0.6 }, data: boundaries },
          markPoint: best ? { symbol: 'pin', symbolSize: 26, itemStyle: { color: p.ok }, label: { show: false }, data: [{ coord: [best[0], best[1]] }] } : undefined,
        },
        { name: 'held-out', type: 'scatter', data: held, symbolSize: 13, itemStyle: { color: p.warn, borderColor: p.surface, borderWidth: 2 }, z: 5 },
      ],
    }
  }, [run.loss, run.checks, run.evals, run.totalEpochs, p, log, showRaw])

  const grad = useMemo<EChartsCoreOption>(() => {
    const pts = run.loss
    const total = Math.max(run.totalEpochs, ...pts.map((q) => q[0]), 0.01)
    return {
      ...baseOption(p),
      grid: { left: 54, right: 54, top: 22, bottom: 4 },
      legend: { top: 0, right: 8, textStyle: { color: p.inkMuted }, itemWidth: 14, itemHeight: 8 },
      xAxis: axis(p, '', { min: 0, max: total, axisLabel: { show: false }, splitLine: { show: false } }),
      yAxis: [axis(p, '', { scale: true, splitNumber: 2 }), axis(p, '', { splitLine: { show: false }, splitNumber: 2, axisLabel: { color: p.inkFaint, fontSize: 11, formatter: (v: number) => v.toExponential(0) } })],
      dataZoom: [{ type: 'inside', xAxisIndex: 0 }],
      tooltip: { ...baseOption(p).tooltip, trigger: 'axis' },
      series: [
        { name: 'gradient size', type: 'line', data: pts.filter((q) => q[2] !== null).map((q) => [q[0], q[2]]), showSymbol: false, lineStyle: { width: 2, color: p.inkMuted } },
        { name: 'learning rate', type: 'line', yAxisIndex: 1, data: pts.filter((q) => q[3] !== null).map((q) => [q[0], q[3]]), showSymbol: false, lineStyle: { width: 2, color: p.ok, type: 'dashed' } },
      ],
    }
  }, [run.loss, run.totalEpochs, p])

  return (
    <Panel
      title="Loss"
      note="Down and to the right is learning. Amber dots are the held-out questions."
      action={
        <div className="chart-tools">
          <label><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log scale</label>
          <label><input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} /> raw readings</label>
        </div>
      }
    >
      {run.loss.length ? (
        <>
          <EChart key={log ? 'log' : 'lin'} option={loss} height={250} group="loss" label={`Loss from ${fmt(run.loss[0][1])} to ${fmt(run.loss[run.loss.length - 1][1])}`} />
          <EChart option={grad} height={96} group="loss" label="Gradient size and learning rate over the run" />
          <dl className="loss-facts">
            <div><dt>Started at</dt><dd>{fmt(run.loss[0][1])}</dd></div>
            <div><dt>Now</dt><dd>{fmt(run.loss[run.loss.length - 1][1])}</dd></div>
            <div><dt>Best</dt><dd>{run.lossBest === null ? '—' : fmt(run.lossBest)}</dd></div>
            <div><dt>Held-out (latest)</dt><dd>{heldLatest === null ? 'after epoch 1' : fmt(heldLatest)}</dd></div>
          </dl>
          {run.checks.length ? (
            <div className="live-table-wrap">
              <h3 className="sub">Epoch by epoch</h3>
              <table className="live-table">
                <thead>
                  <tr><th scope="col">Check</th><th scope="col">Training loss (that epoch)</th><th scope="col">Held-out loss</th><th scope="col">Right action</th><th scope="col">First button</th></tr>
                </thead>
                <tbody>
                  {run.checks.map((c, i) => {
                    const e = c.epoch ?? 0
                    const inEpoch = run.loss.filter((q) => q[0] > e - 1 + 1e-9 && q[0] <= e + 1e-9).map((q) => q[1])
                    const avg = inEpoch.length ? inEpoch.reduce((a, b) => a + b, 0) / inEpoch.length : null
                    return (
                      <tr key={`${c.tag}-${i}`}>
                        <th scope="row">{c.tag}</th>
                        <td>{avg === null ? '—' : fmt(avg)}</td>
                        <td>{c.eval_loss === null || c.eval_loss === undefined ? '—' : fmt(c.eval_loss)}</td>
                        <td>{c.actions_match === undefined ? '—' : `${Math.round(c.actions_match)}%`}</td>
                        <td>{c.first_button_match === undefined ? '—' : `${Math.round(c.first_button_match)}%`}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
        </>
      ) : (
        <p className="muted">The first loss reading appears after the first few training steps.</p>
      )}
    </Panel>
  )
}
