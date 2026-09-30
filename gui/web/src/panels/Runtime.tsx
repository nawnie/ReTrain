/**
 * The Runtime pane: the machine, not the run.
 *
 * Answers the questions an operator asks when something will not start — what
 * GPU does the API actually see, how much of it is free right now, which
 * training packages are importable, and is the QLoRA engine present.
 */

import type { RuntimeStatus, TensorBoardStatus } from '../types'
import { Chip, Meter, Panel, type Tone } from '../ui'

/** The API's own tone vocabulary, mapped onto this app's three tones. */
function meterTone(tone: string): Tone {
  if (tone === 'mint') return 'ok'
  if (tone === 'amber') return 'warn'
  return 'neutral'
}

export function Runtime(props: {
  runtime: RuntimeStatus
  tensorboard: TensorBoardStatus | null
  busy: string
  onRefresh: () => void
  onStartTensorBoard: () => void
}) {
  const { runtime, tensorboard, busy, onRefresh, onStartTensorBoard } = props

  return (
    <div>
      <Panel
        title="Device"
        note="What the local API reports for this machine."
        action={
          <button type="button" className="button" onClick={onRefresh} disabled={busy === 'runtime'}>
            {busy === 'runtime' ? 'Reading' : 'Refresh'}
          </button>
        }
      >
        <dl className="definition-list">
          <div>
            <dt className="definition-term">GPU</dt>
            <dd className="definition-value">{runtime.device}</dd>
          </div>
          <div>
            <dt className="definition-term">State</dt>
            <dd className="definition-value">{runtime.state}</dd>
          </div>
          <div>
            <dt className="definition-term">VRAM free</dt>
            <dd className="definition-value">
              {runtime.vramFreeGb.toFixed(1)} of {runtime.vramTotalGb.toFixed(1)} GB
            </dd>
          </div>
          <div>
            <dt className="definition-term">Backend</dt>
            <dd className="definition-value">{runtime.backend}</dd>
          </div>
        </dl>
      </Panel>

      <Panel title="Resources" note="Live, from the API's own polling.">
        <div className="meter-list">
          {runtime.resources.map((resource) => (
            <Meter
              key={resource.label}
              name={resource.label}
              value={resource.value}
              percent={resource.percent}
              tone={meterTone(resource.tone)}
            />
          ))}
        </div>
      </Panel>

      {/* A dependency that will not import is the most common reason a run is
          blocked, so each one is listed by name rather than summarised as a
          count the operator would then have to go and investigate. */}
      <Panel
        title="Training packages"
        note="Import-checked in the API's own interpreter."
        action={
          <Chip tone={runtime.engine.ready ? 'ok' : 'warn'}>
            {runtime.engine.ready ? 'QLoRA engine ready' : 'QLoRA engine incomplete'}
          </Chip>
        }
      >
        <div className="dependency-grid">
          {runtime.dependencies.map((dependency) => (
            <div className="dependency" key={dependency.package}>
              <span className="dependency-name">{dependency.label}</span>
              {dependency.available ? (
                <span className="dependency-version">{dependency.version ?? 'present'}</span>
              ) : (
                <Chip tone="risk">missing</Chip>
              )}
            </div>
          ))}
        </div>
        {runtime.engine.missing && runtime.engine.missing.length > 0 ? (
          <p className="field-hint">Engine is missing: {runtime.engine.missing.join(', ')}</p>
        ) : null}
      </Panel>

      <Panel
        title="TensorBoard"
        note="Serves scalars from the workspace log directory."
        action={
          <button
            type="button"
            className="button"
            onClick={onStartTensorBoard}
            disabled={busy === 'tensorboard'}
          >
            {busy === 'tensorboard' ? 'Starting' : 'Start service'}
          </button>
        }
      >
        {tensorboard?.url ? (
          <p className="definition-value">
            Running at <code className="run-path">{tensorboard.url}</code>
          </p>
        ) : (
          <p className="field-hint">
            {tensorboard?.message ?? tensorboard?.status ?? 'Not started.'}
          </p>
        )}
      </Panel>
    </div>
  )
}
