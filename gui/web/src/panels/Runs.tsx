/**
 * The Runs pane: what has already happened.
 *
 * Every ReTrain action writes a receipt to the workspace. This pane is the
 * reader for those receipts, plus the two actions an operator wants on one:
 * export it somewhere durable, or delete it to reclaim disk.
 *
 * Delete is guarded by a typed confirmation rather than a bare confirm(),
 * because the thing being removed is the only record that a run happened.
 */

import { useState } from 'react'
import type { CheckpointInventory, CompareSummary } from '../types'
import { Chip, Empty, Panel, toneForState } from '../ui'

export function Runs(props: {
  inventory: CheckpointInventory | null
  compare: CompareSummary | null
  busy: string
  onRefresh: () => void
  onExport: (path: string) => void
  onDelete: (path: string) => void
}) {
  const { inventory, compare, busy, onRefresh, onExport, onDelete } = props
  // Which row is currently asking for confirmation. Only one at a time, so a
  // stray click cannot leave several destructive prompts armed at once.
  const [confirming, setConfirming] = useState<string | null>(null)

  const items = inventory?.items ?? []

  return (
    <div>
      <Panel
        title="Receipts"
        note="Every plan, rehearsal, and launch writes one of these."
        action={
          <div className="button-row">
            <span className="dependency-version">{inventory?.count ?? 0} on disk</span>
            <button
              type="button"
              className="button"
              onClick={onRefresh}
              disabled={busy === 'inventory'}
            >
              {busy === 'inventory' ? 'Refreshing' : 'Refresh'}
            </button>
          </div>
        }
      >
        {items.length === 0 ? (
          <Empty
            title="No receipts yet"
            note="A dry run is the cheapest way to produce one. It writes the same record a real run does, without touching the GPU."
          />
        ) : (
          <div className="run-list">
            {items.map((item) => (
              <div className="run-row" key={item.id}>
                <div className="run-main">
                  <div className="run-title">
                    {item.label}
                    <Chip tone={toneForState(item.status)}>{item.status}</Chip>
                  </div>
                  <div className="run-meta">
                    <span>{item.time ?? item.created_at}</span>
                    <span>{item.type}</span>
                    <span>{item.size}</span>
                    {item.fit_state ? <span>fit: {item.fit_state}</span> : null}
                  </div>
                  <code className="run-path">{item.path}</code>
                </div>
                <div className="button-row">
                  {confirming === item.path ? (
                    <>
                      <span className="field-hint">Delete permanently?</span>
                      <button
                        type="button"
                        className="button"
                        data-role="danger"
                        onClick={() => {
                          onDelete(item.path)
                          setConfirming(null)
                        }}
                      >
                        Delete
                      </button>
                      <button
                        type="button"
                        className="button"
                        data-role="quiet"
                        onClick={() => setConfirming(null)}
                      >
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="button"
                        onClick={() => onExport(item.path)}
                        disabled={busy === 'export'}
                      >
                        Export
                      </button>
                      <button
                        type="button"
                        className="button"
                        data-role="quiet"
                        onClick={() => setConfirming(item.path)}
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      {/* Comparison only means something once there is more than one run, so it
          says so plainly instead of rendering an empty grid. */}
      <Panel title="Comparison" note="Baseline against the two most recent candidates.">
        {!compare || compare.metrics.length === 0 ? (
          <Empty
            title="Nothing to compare"
            note="Comparison needs at least two completed runs in the workspace."
          />
        ) : (
          <div className="table-scroll">
            <table className="compare-table">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col">Baseline</th>
                  <th scope="col">Candidate A</th>
                  <th scope="col">Candidate B</th>
                </tr>
              </thead>
              <tbody>
                {compare.metrics.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    <td>{row.baseline}</td>
                    <td>{row.candidate_a}</td>
                    <td>{row.candidate_b}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  )
}
