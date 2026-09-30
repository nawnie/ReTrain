/**
 * Shared presentation primitives.
 *
 * Each of these exists because the same shape appears in more than one pane.
 * None of them own state or fetch anything, so they can be read top to bottom
 * without tracing a data flow.
 */

import type { ReactNode } from 'react'

/* -- Panel: the one grouping container ------------------------------------- */

export function Panel(props: {
  title: string
  note?: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{props.title}</h2>
          {props.note ? <p className="panel-note">{props.note}</p> : null}
        </div>
        {props.action}
      </div>
      <div className="panel-body">{props.children}</div>
    </section>
  )
}

/* -- Status chip -----------------------------------------------------------
   Status is carried by a word first and a color second. The glyph is a third,
   non-color signal, so a reader who cannot separate the hues still gets two
   independent cues.
   -------------------------------------------------------------------------- */

export type Tone = 'ok' | 'warn' | 'risk' | 'neutral'

const TONE_MARK: Record<Tone, string> = {
  ok: '✓', // check
  warn: '!',
  risk: '×', // cross
  neutral: '–', // dash
}

export function Chip(props: { tone: Tone; children: ReactNode }) {
  return (
    <span className="chip" data-tone={props.tone}>
      <span className="chip-mark" aria-hidden="true">
        {TONE_MARK[props.tone]}
      </span>
      {props.children}
    </span>
  )
}

/**
 * Map the server's state vocabulary onto the app's three tones.
 *
 * Matched case-insensitively and by prefix, because the same concept arrives
 * capitalised differently depending on the endpoint: gates report `ready`,
 * runtime reports `Ready`, and the engine-degraded case reports
 * `Ready (LoRA)`. An exact lowercase compare silently falls through to
 * neutral, which reads as "unknown" for a machine that is in fact fine.
 */
export function toneForState(state: string): Tone {
  const value = state.trim().toLowerCase()
  if (!value) return 'neutral'
  if (value.startsWith('ready') || value === 'safe' || value === 'completed') return 'ok'
  if (value === 'warning' || value === 'tight' || value.startsWith('training') || value === 'running') {
    return 'warn'
  }
  if (value === 'blocked' || value === 'unsafe' || value === 'failed') return 'risk'
  return 'neutral'
}

/* -- Form controls ---------------------------------------------------------- */

export function SelectField(props: {
  label: string
  hint?: string
  value: string
  options: { value: string; label: string }[]
  onChange: (value: string) => void
}) {
  return (
    <label className="field">
      <span className="field-label">{props.label}</span>
      <select value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {props.hint ? <span className="field-hint">{props.hint}</span> : null}
    </label>
  )
}

export function NumberField(props: {
  label: string
  hint?: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <label className="field">
      <span className="field-label">{props.label}</span>
      <input
        type="number"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step}
        onChange={(event) => {
          // An empty or half-typed field parses as NaN. Holding the previous
          // value keeps the projection stable while the operator is still
          // typing, instead of flashing a plan for a nonsense configuration.
          const next = Number(event.target.value)
          props.onChange(Number.isFinite(next) ? next : props.value)
        }}
      />
      {props.hint ? <span className="field-hint">{props.hint}</span> : null}
    </label>
  )
}

/** A native checkbox with a name and an explanation of what it costs or saves. */
export function Switch(props: {
  name: string
  note: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="switch">
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(event) => props.onChange(event.target.checked)}
      />
      <span className="switch-text">
        <span className="switch-name">{props.name}</span>
        <span className="switch-note">{props.note}</span>
      </span>
    </label>
  )
}

/* -- Feedback --------------------------------------------------------------- */

export function Notice(props: { tone: Tone; children: ReactNode; onDismiss?: () => void }) {
  return (
    <div className="notice" data-tone={props.tone} role="status">
      <span className="chip-mark" aria-hidden="true">
        {TONE_MARK[props.tone]}
      </span>
      <span className="notice-text">{props.children}</span>
      {props.onDismiss ? (
        <button type="button" className="button" data-role="quiet" onClick={props.onDismiss}>
          Dismiss
        </button>
      ) : null}
    </div>
  )
}

export function Empty(props: { title: string; note: string; action?: ReactNode }) {
  return (
    <div className="empty">
      <p className="empty-title">{props.title}</p>
      <p className="empty-note">{props.note}</p>
      {props.action ? <div className="button-row">{props.action}</div> : null}
    </div>
  )
}

/* -- Meter: a labelled proportion ------------------------------------------
   The number is always printed next to the bar. The bar is for speed of
   reading, the number is the value the operator can act on.
   -------------------------------------------------------------------------- */

export function Meter(props: { name: string; value: string; percent: number; tone: Tone }) {
  const clamped = Math.max(0, Math.min(100, Math.round(props.percent)))
  return (
    <div className="meter">
      <span className="meter-name">{props.name}</span>
      <span className="meter-value">{props.value}</span>
      <div
        className="meter-track"
        role="meter"
        aria-label={props.name}
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={props.value}
      >
        <div
          className="meter-fill"
          data-tone={props.tone === 'risk' ? 'warn' : props.tone}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  )
}
