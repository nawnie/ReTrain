/** The machine while training: GPU memory and load over the last minutes, and how fast examples are being processed. */

import { useMemo } from 'react'
import type { EChartsCoreOption } from 'echarts/core'
import type { GpuSample, LiveRun } from '../../types'
import { Panel } from '../../ui'
import { EChart } from '../EChart'
import { latest, speedSeries } from '../format'
import { baseOption, usePalette } from '../theme'

export function SystemCard({ run, gpu }: { run: LiveRun; gpu: GpuSample[] }) {
  const p = usePalette()
  const now = latest(gpu)

  const gpuOption = useMemo<EChartsCoreOption>(() => {
    const total = now?.totalGb ?? 16
    return {
      ...baseOption(p),
      grid: { left: 40, right: 44, top: 26, bottom: 22 },
      legend: { top: 0, right: 4, textStyle: { color: p.inkMuted }, itemWidth: 14, itemHeight: 8 },
      tooltip: { ...baseOption(p).tooltip, trigger: 'axis' },
      xAxis: { type: 'time', axisLabel: { color: p.inkFaint, hideOverlap: true }, axisLine: { lineStyle: { color: p.line } }, splitLine: { show: false } },
      yAxis: [
        { type: 'value', min: 0, max: Math.ceil(total), axisLabel: { color: p.inkFaint, formatter: '{value} GB' }, splitLine: { lineStyle: { color: p.line, type: 'dashed' } } },
        { type: 'value', min: 0, max: 100, axisLabel: { color: p.inkFaint, formatter: '{value}%' }, splitLine: { show: false } },
      ],
      series: [
        {
          name: 'memory used', type: 'line', showSymbol: false, data: gpu.map((g) => [g.t * 1000, g.usedGb]), lineStyle: { width: 2, color: p.accent }, z: 3,
          areaStyle: { color: `${p.accent}30` },
          markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: p.risk, type: 'dashed' }, data: [{ yAxis: total }] },
        },
        { name: 'GPU busy', type: 'line', yAxisIndex: 1, showSymbol: false, data: gpu.map((g) => [g.t * 1000, g.util]), lineStyle: { width: 1.5, color: p.ok } },
      ],
    }
  }, [gpu, now, p])

  const speed = useMemo<EChartsCoreOption>(() => {
    const s = speedSeries(run)
    return {
      ...baseOption(p),
      grid: { left: 40, right: 12, top: 10, bottom: 22 },
      tooltip: { ...baseOption(p).tooltip, trigger: 'axis', valueFormatter: (v: unknown) => `${Number(v).toFixed(2)} examples/s` },
      xAxis: { type: 'value', min: 0, axisLabel: { color: p.inkFaint }, axisLine: { lineStyle: { color: p.line } }, splitLine: { show: false } },
      yAxis: { type: 'value', min: 0, splitNumber: 2, axisLabel: { color: p.inkFaint }, splitLine: { lineStyle: { color: p.line, type: 'dashed' } } },
      series: [{ name: 'speed', type: 'line', smooth: true, showSymbol: false, data: s, lineStyle: { width: 2, color: p.ok }, areaStyle: { color: `${p.ok}25` } }],
    }
  }, [run, p])

  return (
    <Panel title="The machine" note="Read from nvidia-smi every 2 seconds. Windows keeps a few GB of the card for the desktop itself.">
      <div className="machine-row">
        <div className="machine-stat"><span>Memory</span><b>{now ? `${now.usedGb.toFixed(1)} / ${now.totalGb.toFixed(0)} GB` : '—'}</b></div>
        <div className="machine-stat"><span>Busy</span><b>{now ? `${Math.round(now.util)}%` : '—'}</b></div>
        <div className="machine-stat"><span>Power</span><b>{now ? `${Math.round(now.powerW)} W` : '—'}</b></div>
        <div className="machine-stat"><span>Temp</span><b>{now ? `${Math.round(now.tempC)}°C` : '—'}</b></div>
      </div>
      {gpu.length > 1 ? <EChart option={gpuOption} height={150} label="GPU memory and load over recent minutes" /> : <p className="muted">Waiting for the first GPU reading…</p>}
      <h3 className="sub">Speed (examples per second)</h3>
      {speedSeries(run).length > 1 ? <EChart option={speed} height={100} label="Training speed over the run" /> : <p className="muted">Needs two loss readings.</p>}
    </Panel>
  )
}
