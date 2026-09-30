/**
 * A thin React wrapper around Apache ECharts.
 *
 * ECharts is registered piece by piece (only the chart types and components the
 * Live pane uses), so the bundle does not carry the whole library. The wrapper
 * owns the chart instance's life: created once, resized whenever its box
 * changes size, disposed on unmount. New data arrives as a new `option`, which
 * is merged in, so a zoom the operator set survives the 2-second refresh.
 *
 * Charts sharing a `group` string zoom together (used for loss and gradient).
 */

import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'

echarts.use([
  BarChart, HeatmapChart, LineChart, ScatterChart,
  DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent, TooltipComponent, VisualMapComponent,
  CanvasRenderer,
])

export function EChart(props: {
  option: EChartsCoreOption
  height: number
  /** fill the height the parent gives (never smaller than `height`) instead of staying a fixed size */
  grow?: boolean
  group?: string
  label: string
  /** Called with the chart instance once it exists, so a caller can wire extra behaviour. */
  onReady?: (chart: echarts.ECharts) => void
}) {
  const { option, height, grow, group, label, onReady } = props
  const box = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  // Create once; keep the size in step with the container.
  useEffect(() => {
    if (!box.current) return undefined
    const instance = echarts.init(box.current, undefined, { renderer: 'canvas' })
    chart.current = instance
    if (group) {
      instance.group = group
      echarts.connect(group)
    }
    onReady?.(instance)
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(box.current)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
    // The instance is created once per mount; option changes are applied below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Apply new data. Merging (not replacing) keeps the operator's zoom.
  useEffect(() => {
    chart.current?.setOption(option, { notMerge: false, lazyUpdate: true })
  }, [option])

  return <div ref={box} className="echart" data-grow={grow ? 'true' : undefined} style={grow ? { minHeight: height } : { height }} role="img" aria-label={label} />
}
