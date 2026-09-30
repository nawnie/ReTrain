/** Which layers are learning, and when: a layer-by-time heatmap, and the running total per layer split into attention and feed-forward. */

import { useMemo, useState } from 'react'
import type { EChartsCoreOption } from 'echarts/core'
import type { LiveLayers } from '../../types'
import { Panel } from '../../ui'
import { EChart } from '../EChart'
import { baseOption, usePalette } from '../theme'

export function LayersCard({ layers }: { layers: LiveLayers | null }) {
  const p = usePalette()
  const [metric, setMetric] = useState<'grad' | 'upd'>('grad')

  const heat = useMemo<EChartsCoreOption | null>(() => {
    const m = layers?.matrix
    if (!m || !m.steps.length) return null
    const grid = metric === 'grad' ? m.grad : m.upd
    const data: [number, number, number][] = []
    grid.forEach((row, x) => row.forEach((v, y) => data.push([x, y, v])))
    const max = Math.max(...data.map((d) => d[2]), 1e-9)
    return {
      ...baseOption(p),
      grid: { left: 40, right: 12, top: 8, bottom: 52 },
      tooltip: {
        ...baseOption(p).tooltip,
        formatter: (q: { value: number[] }) => `<b>layer ${m.layers[q.value[1]]}</b><br/>epoch ${(m.epochs[q.value[0]] ?? 0).toFixed(2)} · step ${m.steps[q.value[0]]}<br/>${metric === 'grad' ? 'gradient' : 'adapter size'} ${q.value[2].toFixed(3)}`,
      },
      xAxis: { type: 'category', data: m.epochs.map((e) => (e ?? 0).toFixed(1)), axisLabel: { color: p.inkFaint, interval: Math.max(0, Math.floor(m.steps.length / 8) - 1) }, axisLine: { show: false }, axisTick: { show: false }, name: 'epoch', nameLocation: 'middle', nameGap: 26, nameTextStyle: { color: p.inkFaint } },
      yAxis: { type: 'category', data: m.layers.map(String), axisLabel: { color: p.inkFaint, interval: 5 }, axisLine: { show: false }, axisTick: { show: false }, name: 'layer', nameTextStyle: { color: p.inkFaint } },
      visualMap: { min: 0, max, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 90, text: ['more', 'less'], textStyle: { color: p.inkFaint }, inRange: { color: [p.ramp0, p.ramp1, p.ramp2] }, calculable: false },
      series: [{ type: 'heatmap', data, progressive: 0, emphasis: { itemStyle: { borderColor: p.ink, borderWidth: 1 } } }],
    }
  }, [layers, metric, p])

  const bars = useMemo<EChartsCoreOption | null>(() => {
    const t = layers?.totals
    if (!t?.length) return null
    const globalLayers = new Set(t.length === 48 ? t.filter((x) => x.i % 6 === 5).map((x) => x.i) : [])
    const sum = t.reduce((a, x) => a + x.attn + x.mlp, 0) || 1
    return {
      ...baseOption(p),
      grid: { left: 40, right: 12, top: 26, bottom: 24 },
      legend: { top: 0, right: 4, textStyle: { color: p.inkMuted }, itemWidth: 12, itemHeight: 8 },
      tooltip: { ...baseOption(p).tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: unknown) => `${((Number(v) / sum) * 100).toFixed(1)}% of all learning` },
      xAxis: { type: 'category', data: t.map((x) => String(x.i)), axisLabel: { color: p.inkFaint, interval: 3 }, axisLine: { lineStyle: { color: p.line } } },
      yAxis: { type: 'value', axisLabel: { show: false }, splitLine: { lineStyle: { color: p.line, type: 'dashed' } } },
      series: [
        { name: 'attention', type: 'bar', stack: 's', itemStyle: { color: p.ramp1 }, data: t.map((x) => ({ value: x.attn, itemStyle: { color: p.ramp1, borderColor: globalLayers.has(x.i) ? p.ink : undefined, borderWidth: globalLayers.has(x.i) ? 1.5 : 0 } })) },
        { name: 'feed-forward', type: 'bar', stack: 's', itemStyle: { color: p.ramp2 }, data: t.map((x) => ({ value: x.mlp, itemStyle: { color: p.ramp2 } })) },
      ],
    }
  }, [layers, p])

  const facts = useMemo(() => {
    const t = layers?.totals
    if (!t?.length) return null
    const sum = t.reduce((a, x) => a + x.attn + x.mlp, 0) || 1
    const third = Math.ceil(t.length / 3)
    const share = (from: number, to: number) => Math.round((t.slice(from, to).reduce((a, x) => a + x.attn + x.mlp, 0) / sum) * 100)
    const top = [...t].sort((a, b) => b.attn + b.mlp - (a.attn + a.mlp)).slice(0, 5).map((x) => x.i)
    const attn = Math.round((t.reduce((a, x) => a + x.attn, 0) / sum) * 100)
    return { early: share(0, third), middle: share(third, 2 * third), late: share(2 * third, t.length), top, attn, third }
  }, [layers])

  return (
    <Panel
      title="Which layers are learning"
      note="Bigger gradient = the training signal is pushing that layer harder. Adapter size = how far its adapter has moved so far."
      action={
        <div className="segmented" role="group" aria-label="Metric">
          <button type="button" data-on={metric === 'grad'} onClick={() => setMetric('grad')}>Gradient</button>
          <button type="button" data-on={metric === 'upd'} onClick={() => setMetric('upd')}>Adapter size</button>
        </div>
      }
    >
      {!layers || !layers.available || !heat ? (
        <p className="muted">
          {layers?.reason ?? 'Reading…'} Runs started with the ReTrain Gemma 4 lane record this for every layer.
        </p>
      ) : (
        <div className="layers-grid">
          <div>
            <EChart option={heat} height={250} label="Heatmap of learning signal by layer and time" />
          </div>
          <div>
            {bars ? <EChart option={bars} height={170} label="Total learning per layer, attention and feed-forward" /> : null}
            {facts ? (
              <ul className="facts">
                <li>Early layers (0–{facts.third - 1}): <b>{facts.early}%</b></li>
                <li>Middle layers: <b>{facts.middle}%</b></li>
                <li>Late layers: <b>{facts.late}%</b></li>
                <li>Attention vs feed-forward: <b>{facts.attn}%</b> / {100 - facts.attn}%</li>
                <li>Most active: <b>{facts.top.join(', ')}</b></li>
              </ul>
            ) : null}
          </div>
        </div>
      )}
    </Panel>
  )
}
