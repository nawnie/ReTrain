/**
 * The Layers pane: what the model is made of, and which part of it to train.
 *
 * The question this answers is "where should the adapter go". A transformer is
 * not uniform -- early layers, late layers, attention projections and the MLP
 * block all differ in how much parameter mass they hold and how much of that
 * mass is actually carrying signal. Training the whole stack when a trailing
 * slice would do is the most common way a local run costs more VRAM and more
 * hours than it needed to.
 *
 * Encoding
 * --------
 * Layers run left to right in forward-pass order; module roles are rows, in the
 * same position for every layer, so a row can be read straight across the model
 * -- "does mlp.gate get denser deeper in?" is a glance rather than a
 * calculation.
 *
 * The chosen metric drives two channels at once: how far a block extrudes
 * toward the viewer, and its colour on the sequential ramp. That redundancy is
 * deliberate. Depth alone is hard to judge at an angle, colour alone is lost to
 * a reader with a colour vision deficiency, and the table below carries the
 * same numbers as text for anyone the picture does not serve.
 *
 * The 3D view is a view. Every selection it can make is also makeable from the
 * table, which is the keyboard and screen-reader path, so a machine without
 * WebGL loses the picture and keeps the function.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { LayerBlock, LayerMap, LocalModel, ModelLayer } from '../types'
import { Chip, Empty, Panel } from '../ui'
import { createLayerScene, type LayerInstance, type LayerScene } from '../gl/layerScene'

/* -- Metrics ---------------------------------------------------------------
   Each answers a different sense of the word "dense". They are kept separate
   rather than blended into one score, because a block that is large and quiet
   and a block that is small and loud are different training targets, and a
   single number would hide which one you are looking at.
   -------------------------------------------------------------------------- */

type MetricId = 'params' | 'bytes' | 'rms' | 'active'

interface Metric {
  id: MetricId
  label: string
  hint: string
  /** True when the value is sampled from the weights rather than exact. */
  sampled: boolean
  /** Orders of magnitude apart, so these read better on a square-root scale. */
  compress: boolean
  value: (block: LayerBlock) => number | null
  format: (value: number) => string
}

const METRICS: Metric[] = [
  {
    id: 'params',
    label: 'Parameters',
    hint: 'How much of the model lives here. Exact, from the tensor shapes.',
    sampled: false,
    compress: true,
    value: (block) => block.params,
    format: (value) =>
      value >= 1e9 ? `${(value / 1e9).toFixed(2)}B` : value >= 1e6 ? `${(value / 1e6).toFixed(1)}M` : `${(value / 1e3).toFixed(1)}K`,
  },
  {
    id: 'bytes',
    label: 'Size on disk',
    hint: 'Storage footprint. Exact, from the tensor byte ranges.',
    sampled: false,
    compress: true,
    value: (block) => block.bytes,
    format: (value) =>
      value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(1)} MB`,
  },
  {
    id: 'rms',
    label: 'Weight magnitude',
    hint: 'Root-mean-square weight value. Sampled. Large weights move the activations further.',
    sampled: true,
    compress: false,
    value: (block) => block.stats?.rms ?? null,
    format: (value) => value.toFixed(4),
  },
  {
    id: 'active',
    label: 'Active weights',
    hint: 'Share of weights above 1% of the block peak. Sampled. Low means much of the block is near zero.',
    sampled: true,
    compress: false,
    value: (block) => block.stats?.active ?? null,
    format: (value) => `${(value * 100).toFixed(1)}%`,
  },
]

/* -- Colour ----------------------------------------------------------------
   The ramp is read from the theme tokens rather than written here, so the 3D
   scene and the DOM around it are coloured from one source and both themes
   work without a second palette.
   -------------------------------------------------------------------------- */

type Rgb = [number, number, number]

function parseColor(value: string): Rgb {
  const text = value.trim()
  const hex = text.match(/^#([0-9a-f]{6})$/i)
  if (hex) {
    const n = parseInt(hex[1], 16)
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
  }
  const rgb = text.match(/rgba?\(([^)]+)\)/i)
  if (rgb) {
    const parts = rgb[1].split(/[,/\s]+/).filter(Boolean).map(Number)
    return [(parts[0] || 0) / 255, (parts[1] || 0) / 255, (parts[2] || 0) / 255]
  }
  return [0.5, 0.5, 0.5]
}

/** Three-stop interpolation. Two stops turn the middle of a wide range to mud. */
function rampColor(stops: [Rgb, Rgb, Rgb], t: number): Rgb {
  const clamped = Math.max(0, Math.min(1, t))
  const [low, mid, high] = clamped < 0.5 ? [stops[0], stops[1], clamped * 2] as const
    : [stops[1], stops[2], (clamped - 0.5) * 2] as const
  const k = high as number
  return [
    (low as Rgb)[0] + ((mid as Rgb)[0] - (low as Rgb)[0]) * k,
    (low as Rgb)[1] + ((mid as Rgb)[1] - (low as Rgb)[1]) * k,
    (low as Rgb)[2] + ((mid as Rgb)[2] - (low as Rgb)[2]) * k,
  ]
}

function readThemeColors() {
  const styles = getComputedStyle(document.documentElement)
  const token = (name: string) => parseColor(styles.getPropertyValue(name))
  return {
    ramp: [token('--ramp-0'), token('--ramp-1'), token('--ramp-2')] as [Rgb, Rgb, Rgb],
    background: token('--surface-sunken'),
    highlight: token('--accent'),
  }
}

/* -- Scene layout ------------------------------------------------------------
   Layers run left to right along X in forward-pass order, module roles stack up
   the Y axis, and the chosen metric extrudes toward the viewer along Z.

   Layers were on the vertical axis first. That put a 28-tall, 8-wide model into
   a canvas nearly four times wider than it is high, so the camera had to pull
   back far enough to fit the height and the picture ended up marooned in the
   middle of a mostly empty frame. Running the stack along the long axis uses
   the space, and reading a pipeline left to right is the more natural direction
   anyway. It also puts the role labels on a fixed left edge where they can be
   read as row headings.
   -------------------------------------------------------------------------- */

const LAYER_PITCH = 1.7
const ROLE_PITCH = 2.3
const BLOCK_WIDTH = 1.25
const BLOCK_HEIGHT = 1.7
const MIN_DEPTH = 0.15
const MAX_DEPTH = 7.0

/** Where a label should sit in world space, for the HTML overlay to project. */
interface SceneLabel {
  key: string
  text: string
  x: number
  y: number
  z: number
  kind: 'role' | 'layer'
}

interface SceneBuild {
  instances: LayerInstance[]
  /** Scene id -> what it represents, for hover and click. */
  index: Map<string, { layer: number; block: LayerBlock; value: number | null }>
  roles: string[]
  labels: SceneLabel[]
  maxValue: number
}

function buildScene(map: LayerMap, metric: Metric, colors: ReturnType<typeof readThemeColors>): SceneBuild {
  // Role order comes from the server, which already emits blocks in
  // forward-pass order, so the rows read as the path through a layer.
  const roles: string[] = []
  for (const layer of map.layers) {
    for (const block of layer.blocks) {
      if (!roles.includes(block.kind)) roles.push(block.kind)
    }
  }

  const values: number[] = []
  for (const layer of map.layers) {
    for (const block of layer.blocks) {
      const value = metric.value(block)
      if (value !== null && Number.isFinite(value)) values.push(value)
    }
  }
  const maxValue = values.length ? Math.max(...values) : 1

  const normalise = (value: number) => {
    if (maxValue <= 0) return 0
    const ratio = Math.max(0, value) / maxValue
    // Parameter counts span orders of magnitude between a norm and an MLP
    // projection; a square root keeps the small end visible instead of
    // flattening everything that is not the largest block.
    return metric.compress ? Math.sqrt(ratio) : ratio
  }

  const instances: LayerInstance[] = []
  const index = new Map<string, { layer: number; block: LayerBlock; value: number | null }>()
  const centreX = ((map.layer_count - 1) / 2) * LAYER_PITCH

  for (const layer of map.layers) {
    for (const block of layer.blocks) {
      const row = roles.indexOf(block.kind)
      const raw = metric.value(block)
      const t = raw === null ? 0 : normalise(raw)
      const depth = MIN_DEPTH + t * (MAX_DEPTH - MIN_DEPTH)
      const [r, g, b] = rampColor(colors.ramp, t)
      const id = `L${layer.index}:${block.kind}`
      instances.push({
        id,
        x: layer.index * LAYER_PITCH - centreX,
        y: row * ROLE_PITCH,
        // Extrude forward from a common back plane rather than about the
        // centre, so every block shares a baseline and depths are comparable.
        z: depth / 2,
        sx: BLOCK_WIDTH,
        sy: BLOCK_HEIGHT,
        sz: depth,
        r,
        g,
        b,
        a: 1,
      })
      index.set(id, { layer: layer.index, block, value: raw })
    }
  }

  // Row headings sit just off the left end of the stack; layer ticks sit below
  // it. Both are anchored in world space so they follow the model as it turns.
  const labels: SceneLabel[] = roles.map((role, row) => ({
    key: `role:${role}`,
    text: role,
    x: -centreX - LAYER_PITCH * 1.4,
    y: row * ROLE_PITCH,
    z: 0,
    kind: 'role',
  }))
  // Every layer would be a wall of numbers; a tick every four is enough to
  // locate yourself along the stack.
  const tickEvery = map.layer_count > 40 ? 8 : 4
  for (const layer of map.layers) {
    if (layer.index % tickEvery !== 0 && layer.index !== map.layer_count - 1) continue
    labels.push({
      key: `layer:${layer.index}`,
      text: String(layer.index),
      x: layer.index * LAYER_PITCH - centreX,
      y: -ROLE_PITCH,
      z: 0,
      kind: 'layer',
    })
  }

  return { instances, index, roles, labels, maxValue }
}

/* -- Selection helpers ------------------------------------------------------ */

/**
 * The backend expresses partial training as "the last N layers", so a selection
 * it can act on has to be a contiguous run ending at the final layer. Anything
 * else is a valid thing to look at and not a valid thing to train, and the UI
 * says which it is rather than silently rounding the selection to something
 * the server will accept.
 */
function trailingRunLength(selected: Set<number>, layerCount: number): number | null {
  if (selected.size === 0) return null
  let n = 0
  for (let index = layerCount - 1; index >= 0; index -= 1) {
    if (selected.has(index)) n += 1
    else break
  }
  return n === selected.size ? n : null
}

/* -- Component -------------------------------------------------------------- */

export function Layers(props: {
  models: LocalModel[]
  folder: string
  map: LayerMap | null
  loading: boolean
  notice: string | null
  sample: boolean
  onFolder: (folder: string) => void
  onSample: (sample: boolean) => void
  onApplyScope: (lastNLayers: number) => void
}) {
  const { models, folder, map, loading, notice, sample, onFolder, onSample, onApplyScope } = props

  const [metricId, setMetricId] = useState<MetricId>('params')
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [focused, setFocused] = useState<number | null>(null)
  const [glReady, setGlReady] = useState<boolean | null>(null)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const sceneRef = useRef<LayerScene | null>(null)
  const buildRef = useRef<SceneBuild | null>(null)
  // Label elements are positioned by writing transforms directly rather than
  // through React state: they move on every frame of a drag, and re-rendering
  // the tree at that rate would make the orbit stutter for no benefit.
  const labelRefs = useRef(new Map<string, HTMLSpanElement>())
  const [hover, setHover] = useState<
    { x: number; y: number; layer: number; role: string; text: string } | null
  >(null)
  // The label set is state, not a ref: React has to render the elements before
  // syncLabels has anything to position. Their positions are then written
  // directly to the DOM, so this only changes when the model or metric does.
  const [labels, setLabels] = useState<SceneLabel[]>([])

  const metric = METRICS.find((item) => item.id === metricId) ?? METRICS[0]

  // Held in a ref so the pointer handlers can format a value with the current
  // metric without the whole listener set being torn down and rebound whenever
  // the metric changes.
  const metricRef = useRef(metric)
  metricRef.current = metric

  /** Re-project every world-anchored label onto the canvas. */
  const syncLabels = useCallback(() => {
    const scene = sceneRef.current
    const build = buildRef.current
    if (!scene || !build) return
    for (const label of build.labels) {
      const element = labelRefs.current.get(label.key)
      if (!element) continue
      const point = scene.project(label.x, label.y, label.z)
      if (!point) {
        element.style.visibility = 'hidden'
        continue
      }
      element.style.visibility = 'visible'
      element.style.transform = `translate(${point.x}px, ${point.y}px) translate(-50%, -50%)`
    }
  }, [])

  /* -- Renderer lifecycle ------------------------------------------------- */

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const scene = createLayerScene(canvas)
    sceneRef.current = scene
    setGlReady(scene !== null)
    if (!scene) return undefined

    scene.resize()
    const observer = new ResizeObserver(() => {
      scene.resize()
      syncLabels()
    })
    observer.observe(canvas)
    return () => {
      observer.disconnect()
      scene.dispose()
      sceneRef.current = null
    }
  }, [syncLabels])

  // Rebuild whenever the model, the metric, or the theme changes. The theme is
  // watched because the ramp lives in CSS: a theme switch has to re-colour the
  // scene, and nothing else would tell it to.
  const rebuild = useCallback(() => {
    const scene = sceneRef.current
    if (!scene || !map) return
    const build = buildScene(map, metric, readThemeColors())
    buildRef.current = build
    const colors = readThemeColors()
    scene.setTheme({ background: colors.background, highlight: colors.highlight })
    scene.setInstances(build.instances)
    setLabels(build.labels)
  }, [map, metric])

  useEffect(() => {
    rebuild()
  }, [rebuild])

  // Runs after the label elements exist in the DOM, which is why positioning is
  // a separate effect from building them.
  useEffect(() => {
    syncLabels()
  }, [labels, syncLabels])

  useEffect(() => {
    const observer = new MutationObserver(() => rebuild())
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onScheme = () => rebuild()
    query.addEventListener('change', onScheme)
    return () => {
      observer.disconnect()
      query.removeEventListener('change', onScheme)
    }
  }, [rebuild])

  // Highlight follows the selection. An empty selection highlights everything,
  // which reads as "nothing chosen yet" rather than as "everything dimmed".
  useEffect(() => {
    const scene = sceneRef.current
    const build = buildRef.current
    if (!scene || !build) return
    if (selected.size === 0) {
      scene.setHighlight(new Set())
      return
    }
    const ids = new Set<string>()
    for (const [id, entry] of build.index) {
      if (selected.has(entry.layer)) ids.add(id)
    }
    scene.setHighlight(ids)
  }, [selected, map, metric])

  /* -- Pointer orbit ------------------------------------------------------- */

  useEffect(() => {
    const canvas = canvasRef.current
    const scene = sceneRef.current
    if (!canvas || !scene) return undefined

    let dragging = false
    let moved = 0
    let lastX = 0
    let lastY = 0

    const onPointerDown = (event: PointerEvent) => {
      dragging = true
      moved = 0
      lastX = event.clientX
      lastY = event.clientY
      canvas.setPointerCapture(event.pointerId)
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return
      const dx = event.clientX - lastX
      const dy = event.clientY - lastY
      lastX = event.clientX
      lastY = event.clientY
      moved += Math.abs(dx) + Math.abs(dy)
      const orbit = scene.getOrbit()
      scene.setOrbit({
        azimuth: orbit.azimuth - dx * 0.008,
        elevation: orbit.elevation + dy * 0.006,
      })
      syncLabels()
    }

    // Hover readout. Picking is a ray test against a few hundred boxes, cheap
    // enough to run per move, but it is still coalesced into an animation frame
    // so a fast mouse cannot queue more work than the display can show.
    let hoverFrame = 0
    const onHoverMove = (event: PointerEvent) => {
      if (dragging) return
      if (hoverFrame) return
      hoverFrame = requestAnimationFrame(() => {
        hoverFrame = 0
        const rect = canvas.getBoundingClientRect()
        const localX = event.clientX - rect.left
        const localY = event.clientY - rect.top
        const id = scene.pick(localX, localY)
        const entry = id ? buildRef.current?.index.get(id) : null
        if (!entry) {
          setHover(null)
          return
        }
        setHover({
          x: localX,
          y: localY,
          layer: entry.layer,
          role: entry.block.kind,
          text: entry.value === null ? 'not sampled' : metricRef.current.format(entry.value),
        })
      })
    }

    const onLeave = () => setHover(null)

    const onPointerUp = (event: PointerEvent) => {
      if (!dragging) return
      dragging = false
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
      // A drag that barely moved is a click. Anything further was the operator
      // rotating the view, and must not also change the selection.
      if (moved > 6) return
      const rect = canvas.getBoundingClientRect()
      const id = scene.pick(event.clientX - rect.left, event.clientY - rect.top)
      const entry = id ? buildRef.current?.index.get(id) : null
      if (!entry) return
      setFocused(entry.layer)
      setSelected((current) => {
        const next = new Set(current)
        if (event.shiftKey && focused !== null) {
          const [from, to] = focused < entry.layer ? [focused, entry.layer] : [entry.layer, focused]
          for (let index = from; index <= to; index += 1) next.add(index)
        } else if (next.has(entry.layer)) {
          next.delete(entry.layer)
        } else {
          next.add(entry.layer)
        }
        return next
      })
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const orbit = scene.getOrbit()
      scene.setOrbit({ distance: orbit.distance * (event.deltaY > 0 ? 1.12 : 0.89) })
      syncLabels()
    }

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointermove', onHoverMove)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointercancel', onPointerUp)
    canvas.addEventListener('pointerleave', onLeave)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointermove', onHoverMove)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointercancel', onPointerUp)
      canvas.removeEventListener('pointerleave', onLeave)
      canvas.removeEventListener('wheel', onWheel)
      if (hoverFrame) cancelAnimationFrame(hoverFrame)
    }
  }, [glReady, focused, syncLabels])

  /* -- Derived ------------------------------------------------------------- */

  const trailing = map ? trailingRunLength(selected, map.layer_count) : null

  const selectedTotals = useMemo(() => {
    if (!map || selected.size === 0) return null
    let params = 0
    let adapterParams = 0
    for (const layer of map.layers) {
      if (!selected.has(layer.index)) continue
      params += layer.params
      for (const block of layer.blocks) {
        if (block.adapter_target) adapterParams += block.params
      }
    }
    return { params, adapterParams, layers: selected.size }
  }, [map, selected])

  const selectLast = (count: number) => {
    if (!map) return
    const next = new Set<number>()
    for (let index = map.layer_count - count; index < map.layer_count; index += 1) {
      if (index >= 0) next.add(index)
    }
    setSelected(next)
    setFocused(map.layer_count - 1)
  }

  /* -- Render -------------------------------------------------------------- */

  const modelOptions = models.map((item) => item.folder)

  return (
    <div className="layers-pane">
      <Panel
        title="Model layer map"
        note={
          map
            ? `${map.architecture} · ${map.layer_count} layers · ${(map.total_params / 1e9).toFixed(2)}B parameters`
            : 'Read straight from the checkpoint headers.'
        }
        action={
          <div className="button-row">
            <label className="field inline-field">
              <span className="visually-hidden">Model folder</span>
              <select value={folder} onChange={(event) => onFolder(event.target.value)}>
                {modelOptions.length === 0 ? <option value="">No local models found</option> : null}
                {modelOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="switch inline-switch">
              <input
                type="checkbox"
                checked={sample}
                onChange={(event) => onSample(event.target.checked)}
              />
              <span className="switch-text">
                <span className="switch-name">Sample weights</span>
              </span>
            </label>
          </div>
        }
      >
        {notice ? <p className="field-hint">{notice}</p> : null}

        <div className="metric-row-controls">
          <div className="segmented" role="group" aria-label="Density metric">
            {METRICS.map((item) => (
              <button
                key={item.id}
                type="button"
                className="segment"
                aria-pressed={item.id === metricId}
                disabled={item.sampled && map !== null && !map.sampled}
                onClick={() => setMetricId(item.id)}
              >
                {item.label}
                {item.sampled ? <span className="segment-tag">sampled</span> : null}
              </button>
            ))}
          </div>
          <p className="field-hint">{metric.hint}</p>
        </div>

        <div className="scene-frame">
          <canvas
            ref={canvasRef}
            className="scene-canvas"
            // The canvas is a picture of data that is also present as text
            // below. Describing it here, and keeping every action it offers
            // available from the table, is what makes it safe to leave out of
            // the tab order.
            role="img"
            aria-label={
              map
                ? `Three-dimensional map of ${map.name}: ${map.layer_count} layers, each row a layer and each column a module role, extruded and coloured by ${metric.label.toLowerCase()}. The same values are listed in the layer table below.`
                : 'Model layer map. No model loaded.'
            }
          />
          {glReady === false ? (
            <div className="scene-fallback">
              <p className="empty-title">3D view unavailable</p>
              <p className="empty-note">
                This machine reports no WebGL2 context. The layer table below carries the same
                numbers and every selection the picture offers.
              </p>
            </div>
          ) : null}
          {loading ? <div className="scene-fallback"><p className="empty-title">Reading checkpoint…</p></div> : null}

          {/* World-anchored labels. Positioned by projecting their scene
              coordinates each frame, so they turn with the model instead of
              floating in a fixed corner. Marked aria-hidden because the same
              structure is in the table below as real headings; a screen reader
              reading loose numbers off a picture would be noise. */}
          <div className="scene-labels" aria-hidden="true">
            {labels.map((label) => (
              <span
                key={label.key}
                className="scene-label"
                data-kind={label.kind}
                ref={(element) => {
                  if (element) labelRefs.current.set(label.key, element)
                  else labelRefs.current.delete(label.key)
                }}
              >
                {label.text}
              </span>
            ))}
          </div>

          {/* Hover readout. Follows the cursor rather than sitting in a corner,
              so the number and the block it describes are read together. */}
          {hover ? (
            <div
              className="scene-tooltip"
              style={{ transform: `translate(${hover.x}px, ${hover.y}px)` }}
              role="status"
            >
              <strong>Layer {hover.layer}</strong>
              <span>{hover.role}</span>
              <span className="scene-tooltip-value">{hover.text}</span>
            </div>
          ) : null}
        </div>

        <div className="scene-legend">
          <span className="legend-label">{metric.label}</span>
          <span className="legend-ramp" aria-hidden="true" />
          <span className="legend-ends">
            <span>low</span>
            <span>high</span>
          </span>
          <span className="field-hint">
            Drag to rotate, scroll to zoom, click a block to select its layer.
            {metric.sampled ? ' Sampled values, not exact measurements.' : ''}
          </span>
        </div>
      </Panel>

      <Panel
        title="Training scope"
        note="Pick the layers to adapt, then apply them to the run."
        action={
          selectedTotals ? (
            <Chip tone="neutral">
              {selectedTotals.layers} of {map?.layer_count ?? 0} layers
            </Chip>
          ) : null
        }
      >
        <div className="button-row">
          <button type="button" className="button" onClick={() => selectLast(4)} disabled={!map}>
            Last 4
          </button>
          <button type="button" className="button" onClick={() => selectLast(8)} disabled={!map}>
            Last 8
          </button>
          <button type="button" className="button" onClick={() => selectLast(12)} disabled={!map}>
            Last 12
          </button>
          <button
            type="button"
            className="button"
            onClick={() => map && selectLast(map.layer_count)}
            disabled={!map}
          >
            All
          </button>
          <button
            type="button"
            className="button"
            data-role="quiet"
            onClick={() => setSelected(new Set())}
            disabled={selected.size === 0}
          >
            Clear
          </button>
        </div>

        {selectedTotals && map ? (
          <>
            <dl className="definition-list scope-summary">
              <div>
                <dt className="definition-term">Layers selected</dt>
                <dd className="definition-value">{selectedTotals.layers}</dd>
              </div>
              <div>
                <dt className="definition-term">Parameters in scope</dt>
                <dd className="definition-value">
                  {(selectedTotals.params / 1e6).toFixed(1)}M
                  <span className="field-hint">
                    {((selectedTotals.params / map.total_params) * 100).toFixed(1)}% of the model
                  </span>
                </dd>
              </div>
              <div>
                <dt className="definition-term">Adapter-targetable</dt>
                <dd className="definition-value">
                  {(selectedTotals.adapterParams / 1e6).toFixed(1)}M
                  <span className="field-hint">projections a LoRA can attach to</span>
                </dd>
              </div>
            </dl>

            {trailing === null ? (
              <p className="field-hint">
                This selection is not a run of layers ending at the last one. The trainer expresses
                partial training as “the last N layers”, so it cannot be applied as-is — use one of
                the buttons above, or select a trailing range.
              </p>
            ) : (
              <div className="button-row">
                <button
                  type="button"
                  className="button"
                  data-role="primary"
                  onClick={() => onApplyScope(trailing)}
                >
                  Train the last {trailing} layer{trailing === 1 ? '' : 's'}
                </button>
                <span className="field-hint">
                  Sets tune scope to “Last layers” and the layer count on the Configure pane.
                </span>
              </div>
            )}
          </>
        ) : (
          <Empty
            title="No layers selected"
            note="Click a block in the map, use a quick range above, or choose layers from the table below."
          />
        )}
      </Panel>

      <Panel
        title="Layers"
        note={
          map?.sampled
            ? 'Sampled weight statistics. Structure is exact; magnitude and activity are estimated from a bounded read.'
            : 'Structure only. Turn on weight sampling for magnitude and activity.'
        }
      >
        {!map ? (
          <Empty title="No model loaded" note="Choose a local checkpoint above." />
        ) : (
          <div className="table-scroll">
            <table className="compare-table layer-table">
              <caption className="visually-hidden">
                Every layer with its parameter count and sampled weight statistics. Use the select
                button in each row to add that layer to the training scope.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Layer</th>
                  <th scope="col">Parameters</th>
                  <th scope="col">Size</th>
                  <th scope="col">Weight RMS</th>
                  <th scope="col">Active</th>
                  <th scope="col">In scope</th>
                </tr>
              </thead>
              <tbody>
                {map.layers.map((layer) => (
                  <LayerRow
                    key={layer.index}
                    layer={layer}
                    selected={selected.has(layer.index)}
                    onToggle={() => {
                      setFocused(layer.index)
                      setSelected((current) => {
                        const next = new Set(current)
                        if (next.has(layer.index)) next.delete(layer.index)
                        else next.add(layer.index)
                        return next
                      })
                    }}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {map && map.terminals.length > 0 ? (
        <Panel
          title="Outside the stack"
          note="Embedding and output blocks. Not interchangeable with a layer, and not adapter targets."
        >
          <div className="dependency-grid">
            {map.terminals.map((terminal) => (
              <div className="dependency" key={terminal.kind}>
                <span className="dependency-name">{terminal.label}</span>
                <span className="dependency-version">{(terminal.params / 1e6).toFixed(1)}M</span>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}
    </div>
  )
}

/** One row of the layer table. Split out to keep the table body readable. */
function LayerRow(props: { layer: ModelLayer; selected: boolean; onToggle: () => void }) {
  const { layer, selected, onToggle } = props
  return (
    <tr data-selected={selected || undefined}>
      <th scope="row">{layer.label}</th>
      <td>{(layer.params / 1e6).toFixed(2)}M</td>
      <td>{(layer.bytes / 1e6).toFixed(1)} MB</td>
      <td>{layer.stats ? layer.stats.rms.toFixed(4) : '—'}</td>
      <td>{layer.stats ? `${(layer.stats.active * 100).toFixed(1)}%` : '—'}</td>
      <td>
        <button
          type="button"
          className="button"
          data-role={selected ? undefined : 'quiet'}
          aria-pressed={selected}
          onClick={onToggle}
        >
          {selected ? 'In scope' : 'Add'}
        </button>
      </td>
    </tr>
  )
}
