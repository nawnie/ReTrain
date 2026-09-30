/**
 * Data for the Live pane: which runs exist, the chosen run's numbers, its layer and answer data, and GPU history.
 *
 * Polling rules: an active run is re-read every 2 seconds, a finished one every 10. Answer samples are only
 * re-fetched when a new learning check appears (they change once per epoch, not every poll). Every request is
 * cancelled when the pane unmounts or the choice changes, so a late reply can never overwrite newer state.
 * "Pause" freezes the display on the current numbers without stopping the run.
 */

import { useEffect, useState } from 'react'
import * as api from '../api'
import type { GpuSample, LiveLayers, LiveRun, LiveRunSummary, LiveSamples } from '../types'

const FAST_MS = 2000
const SLOW_MS = 10000

export interface LiveData {
  runs: LiveRunSummary[]
  selected: string
  select: (id: string) => void
  run: LiveRun | null
  layers: LiveLayers | null
  samples: LiveSamples | null
  gpu: GpuSample[]
  problem: string | null
  paused: boolean
  setPaused: (value: boolean) => void
}

export function useLiveData(): LiveData {
  const [runs, setRuns] = useState<LiveRunSummary[]>([])
  const [selected, setSelected] = useState('')
  const [run, setRun] = useState<LiveRun | null>(null)
  const [layers, setLayers] = useState<LiveLayers | null>(null)
  const [samples, setSamples] = useState<LiveSamples | null>(null)
  const [gpu, setGpu] = useState<GpuSample[]>([])
  const [problem, setProblem] = useState<string | null>(null)
  const [paused, setPaused] = useState(false)

  // The list of watchable runs; the newest real (non-demo) run is chosen first.
  useEffect(() => {
    const controller = new AbortController()
    const load = () =>
      api
        .fetchLiveRuns(controller.signal)
        .then((value) => {
          setRuns(value.runs)
          setSelected((current) => current || (value.runs.find((r) => !r.demo) ?? value.runs[0])?.id || '')
        })
        .catch((error: unknown) => {
          if (!controller.signal.aborted) setProblem(api.readableError(error))
        })
    void load()
    const timer = window.setInterval(load, SLOW_MS)
    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [])

  // GPU history, independent of which run is chosen.
  useEffect(() => {
    if (paused) return undefined
    const controller = new AbortController()
    const load = () =>
      api
        .fetchLiveSystem(controller.signal)
        .then((value) => setGpu(value.samples))
        .catch(() => undefined)
    void load()
    const timer = window.setInterval(load, FAST_MS)
    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [paused])

  // The chosen run and its per-layer data.
  const active = run === null || run.status === 'loading' || run.status === 'training'
  useEffect(() => {
    if (!selected || paused) return undefined
    const controller = new AbortController()
    const load = () =>
      Promise.all([api.fetchLiveRun(selected, controller.signal), api.fetchLiveLayers(selected, controller.signal)])
        .then(([nextRun, nextLayers]) => {
          if ('status' in nextRun && nextRun.status === 'blocked') return
          setRun(nextRun as LiveRun)
          setLayers(nextLayers)
          setProblem(null)
        })
        .catch((error: unknown) => {
          if (!controller.signal.aborted) setProblem(api.readableError(error))
        })
    void load()
    const timer = window.setInterval(load, active ? FAST_MS : SLOW_MS)
    return () => {
      window.clearInterval(timer)
      controller.abort()
    }
  }, [selected, active, paused])

  // Answer samples change when a learning check finishes, so they are keyed on how many checks exist.
  const checkCount = run?.checks.length ?? 0
  useEffect(() => {
    if (!selected) return undefined
    const controller = new AbortController()
    api
      .fetchLiveSamples(selected, controller.signal)
      .then(setSamples)
      .catch(() => undefined)
    return () => controller.abort()
  }, [selected, checkCount])

  const select = (id: string) => {
    setRun(null)
    setLayers(null)
    setSamples(null)
    setSelected(id)
  }

  return { runs, selected, select, run, layers, samples, gpu, problem, paused, setPaused }
}
