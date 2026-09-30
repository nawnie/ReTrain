/**
 * The Live pane: watch a training run while it happens.
 *
 * At the top, the run's state and headline numbers. Below, in the order a person asks the questions:
 * is the loss falling, is the model actually getting better at answering, which layers are doing the learning,
 * and what is the machine doing. The pane only reads; nothing here starts or stops a run.
 *
 * The pieces live in src/live/ (data hook, chart wrapper, one card per question).
 */

import * as api from '../api'
import { clock } from '../live/format'
import { Empty, Panel, SelectField } from '../ui'
import { useLiveData } from '../live/useLive'
import { Hero } from '../live/cards/Hero'
import { LossCard } from '../live/cards/LossCard'
import { LearningCard } from '../live/cards/LearningCard'
import { LayersCard } from '../live/cards/LayersCard'
import { SystemCard } from '../live/cards/SystemCard'
import { AnswersCard } from '../live/cards/AnswersCard'
import '../live/live.css'

export function Live() {
  const d = useLiveData()
  const { run } = d

  if (!d.runs.length && !d.problem) {
    return (
      <Empty
        title="No training run to watch yet"
        note="Start a run from Configure, or register one started elsewhere. It appears here as soon as its log exists."
      />
    )
  }

  return (
    <div className="live">
      <div className="live-bar">
        <SelectField
          label="Run"
          value={d.selected}
          options={d.runs.map((r) => ({ value: r.id, label: `${r.demo ? '[demo] ' : ''}${r.title}` }))}
          onChange={d.select}
        />
        <button type="button" className="button" onClick={() => d.setPaused(!d.paused)} aria-pressed={d.paused}>
          {d.paused ? 'Resume updates' : 'Pause updates'}
        </button>
        {run ? (
          <span className="live-strip" data-status={run.status} aria-live="polite">
            <i className="live-dot" data-live={run.status === 'training' ? 'on' : 'off'} data-tone={run.status === 'failed' || run.status === 'stopped' ? 'risk' : run.status === 'loading' ? 'warn' : 'ok'} />
            <b>{run.status === 'loading' ? 'Loading…' : `${Math.round(run.progress * 100)}%`}</b>
            <span>{run.status === 'training' ? 'training' : run.status}</span>
            {run.lossNow !== null ? <span>loss <b>{run.lossNow.toFixed(3)}</b></span> : null}
            {run.status === 'training' && run.etaSeconds !== null ? <span>~{clock(run.etaSeconds)} left</span> : null}
          </span>
        ) : null}
      </div>

      {d.problem ? <p className="live-problem">Could not read the run: {d.problem}</p> : null}

      {run ? (
        <>
          <Hero run={run} gpu={d.gpu} />

          <div className="live-grid">
            <LossCard run={run} />
            <LearningCard run={run} />
          </div>

          <div className="live-grid">
            <LayersCard layers={d.layers} />
            <SystemCard run={run} gpu={d.gpu} />
          </div>

          <AnswersCard samples={d.samples} />

          <Panel title="Trainer messages" note={run.log ? `Log file: ${run.log}` : undefined}>
            <pre className="live-log" aria-label="Latest trainer messages">
              {run.error ? `${run.error}\n` : ''}
              {run.messages.length ? run.messages.join('\n') : '(nothing yet)'}
            </pre>
          </Panel>
        </>
      ) : (
        <p className="muted">Reading the run…</p>
      )}
    </div>
  )
}

// Kept so a caller can tell whether an error came from the API layer.
export const readableError = api.readableError
