/**
 * The projection column: what the current configuration will actually cost,
 * and whether the server will let it start.
 *
 * This is the reason the Configure pane is two columns. The estimate is not a
 * report produced after the fact; it is the feedback that makes the controls
 * meaningful. Moving a control and seeing this number move is the whole
 * interaction.
 *
 * Presentational only. The shell owns the request and passes the result down.
 */

import type { TrainingPlan } from '../types'
import { Chip, Empty, Panel, toneForState } from '../ui'

/** Plain-language reading of the server's fit verdict. */
const FIT_CAPTION: Record<string, string> = {
  safe: 'Fits with room to spare.',
  tight: 'Fits, but with little headroom. A longer sequence may fail mid-run.',
  unsafe: 'Exceeds the safety limit. This will run out of memory.',
}

export function Projection(props: { plan: TrainingPlan | null; pending: boolean }) {
  const { plan, pending } = props

  if (!plan) {
    return (
      <Panel title="Projected fit" note="Server-validated memory estimate.">
        <Empty
          title={pending ? 'Estimating' : 'No estimate yet'}
          note={
            pending
              ? 'Asking the local API what this configuration will cost.'
              : 'Change any control, or reconnect to the local API, to get an estimate.'
          }
        />
      </Panel>
    )
  }

  const estimate = plan.validation.estimate
  const fit = estimate.fit_state
  // The bar shows the estimate as a share of the limit, so "how close am I to
  // the ceiling" is readable without doing arithmetic.
  const usedPercent = Math.max(
    0,
    Math.min(100, Math.round((estimate.estimated_gb / Math.max(0.1, estimate.limit_gb)) * 100)),
  )
  // Each line remains visible as text below. The bar is a second, faster way
  // to see which allocation dominates the current estimate, not a replacement
  // for the numbers or a claim that the lines add up to an exact peak.
  const estimateFloor = Math.max(0.1, estimate.estimated_gb)

  return (
    <>
      <Panel
        title="Projected fit"
        note="Server-validated memory estimate."
        action={<Chip tone={toneForState(fit)}>{fit}</Chip>}
      >
        <div className="figure">
          <span className="figure-value">{estimate.estimated_gb.toFixed(1)}</span>
          <span className="figure-unit">GB estimated</span>
        </div>
        <p className="figure-caption">
          {FIT_CAPTION[fit] ?? 'The server did not classify this configuration.'}
        </p>

        <div className="headroom">
          <div className="headroom-track">
            <div className="headroom-fill" data-fit={fit} style={{ width: `${usedPercent}%` }} />
          </div>
          <div className="headroom-scale">
            <span>
              {estimate.headroom_gb >= 0
                ? `${estimate.headroom_gb.toFixed(1)} GB headroom`
                : `${Math.abs(estimate.headroom_gb).toFixed(1)} GB over`}
            </span>
            <span>{estimate.limit_gb.toFixed(1)} GB limit</span>
          </div>
        </div>

        {/* A total tells the operator there is a problem. The breakdown tells
            them which control to move, which is the part they can act on. */}
        <div className="breakdown" aria-label="Estimated VRAM allocation by training component">
          {estimate.breakdown.map((line) => (
            <div className="breakdown-row" key={line.item}>
              <div className="breakdown-item">
                <div className="breakdown-label">
                  <span>{line.item}</span>
                  <span className="breakdown-value">{line.gb.toFixed(2)} GB</span>
                </div>
                <div
                  className="allocation-track"
                  role="img"
                  aria-label={`${line.item}: ${line.gb.toFixed(2)} GB, ${Math.round((line.gb / estimateFloor) * 100)} percent of the estimate.`}
                >
                  <span
                    className="allocation-fill"
                    style={{ width: `${Math.max(0, Math.min(100, (line.gb / estimateFloor) * 100))}%` }}
                  />
                </div>
                <span className="breakdown-detail">{line.detail}</span>
              </div>
            </div>
          ))}
        </div>

        {estimate.warnings.length > 0 ? (
          <ul className="gate-list">
            {estimate.warnings.map((warning) => (
              <li className="gate" key={warning}>
                <span className="gate-name">
                  <span className="gate-detail">{warning}</span>
                </span>
                <Chip tone="warn">note</Chip>
              </li>
            ))}
          </ul>
        ) : null}
      </Panel>

      {/* The gates are the server's own preconditions. They are shown in full
          rather than reduced to a single verdict, because "blocked" without
          the reason leaves the operator with nothing to fix. */}
      <Panel
        title="Safety gates"
        note="Checked by the API before a run is allowed to start."
        action={<Chip tone={toneForState(plan.status)}>{plan.status}</Chip>}
      >
        <ul className="gate-list">
          {plan.validation.gates.map((gate) => (
            <li className="gate" key={gate.gate}>
              <span className="gate-name">
                {gate.gate}
                <span className="gate-detail">{gate.detail}</span>
              </span>
              <Chip tone={toneForState(gate.state)}>{gate.state}</Chip>
            </li>
          ))}
        </ul>
      </Panel>
    </>
  )
}
