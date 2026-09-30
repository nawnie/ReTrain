/**
 * The most direct proof of learning: one held-out question, and what the model answered to it at every check.
 * Pick a question on the left; on the right the reference answer is shown first, then the model's answer at the baseline
 * (before any training), after epoch 1, after epoch 2, and so on - so a wrong answer turning into a right one is visible.
 */

import { useMemo, useState } from 'react'
import type { LiveSampleItem, LiveSamples } from '../../types'
import { Panel } from '../../ui'
import { parseAnswer } from '../format'

/** The buttons of one answer drawn as small keys: "START ·8f". */
function Chips({ raw, tone }: { raw: string; tone?: 'ok' | 'risk' | 'neutral' }) {
  const a = parseAnswer(raw)
  if (!a.ok) return <span className="answer-bad">not valid JSON</span>
  if (!a.actions.length) return <span className="answer-bad">no actions</span>
  return (
    <span className="chips" data-tone={tone ?? 'neutral'}>
      {a.actions.map((x, i) => (
        <span className="chip-key" key={i}>
          {x.buttons.length ? x.buttons.join(' + ') : 'wait'} <small>{x.frames}f</small>
        </span>
      ))}
    </span>
  )
}

function verdict(item: LiveSampleItem, tag: string): 'right' | 'wrong' | 'invalid' {
  const a = item.answers[tag]
  if (!a) return 'invalid'
  if (!a.valid_json) return 'invalid'
  return a.actions_match ? 'right' : 'wrong'
}

const WORD = { right: 'Right action', wrong: 'Wrong action', invalid: 'Invalid reply' } as const

export function AnswersCard({ samples }: { samples: LiveSamples | null }) {
  const [picked, setPicked] = useState(0)
  const [category, setCategory] = useState('all')
  const [mistakesOnly, setMistakesOnly] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  const tags = samples?.tags ?? []
  const lastTag = tags[tags.length - 1]
  const items = samples?.items ?? []
  const categories = useMemo(() => Array.from(new Set(items.map((i) => i.category).filter(Boolean))).sort(), [items])
  const visible = useMemo(
    () => items.filter((i) => (category === 'all' || i.category === category) && (!mistakesOnly || (lastTag && verdict(i, lastTag) !== 'right'))),
    [items, category, mistakesOnly, lastTag],
  )
  const current = visible.find((i) => i.i === picked) ?? visible[0]

  if (!samples || !samples.available || !items.length) {
    return (
      <Panel title="Question by question" note="Each learning check saves the model's answers, so you can read them side by side.">
        <p className="muted">No answers saved yet. They appear when the first learning check (the baseline, before training) finishes.</p>
      </Panel>
    )
  }

  const reference = current ? parseAnswer(current.expected) : null

  return (
    <Panel
      title="Question by question"
      note="The same held-out question, answered before training and after each epoch."
      action={
        <div className="chart-tools">
          <select aria-label="Filter by skill" value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="all">All skills ({items.length})</option>
            {categories.map((c) => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
          </select>
          <label><input type="checkbox" checked={mistakesOnly} onChange={(e) => setMistakesOnly(e.target.checked)} /> only mistakes now</label>
        </div>
      }
    >
      <div className="answers">
        <ul className="answer-list" aria-label="Held-out questions">
          {visible.map((i) => (
            <li key={i.i}>
              <button type="button" data-active={current?.i === i.i} onClick={() => setPicked(i.i)}>
                <span className="answer-cat">{i.category.replace(/_/g, ' ') || 'question'}</span>
                <span className="answer-dots" aria-hidden="true">
                  {tags.map((t) => <i key={t} data-v={verdict(i, t)} />)}
                </span>
              </button>
            </li>
          ))}
          {!visible.length ? <li className="muted">Nothing matches this filter.</li> : null}
        </ul>

        {current ? (
          <div className="answer-detail">
            <h3 className="sub">Question</h3>
            <pre className="answer-prompt">{current.prompt || '(question text not saved)'}</pre>

            <h3 className="sub">Reference answer</h3>
            <Chips raw={current.expected} tone="ok" />
            {reference?.think ? <p className="think">“{reference.think}”</p> : null}

            <h3 className="sub">Model's answer over time</h3>
            <ol className="timeline">
              {tags.map((t) => {
                const a = current.answers[t]
                const v = verdict(current, t)
                const parsed = a ? parseAnswer(a.got) : null
                return (
                  <li key={t} data-v={v}>
                    <div className="tl-head"><b>{t}</b><span className="tl-verdict" data-v={v}>{WORD[v]}</span></div>
                    {a ? <Chips raw={a.got} tone={v === 'right' ? 'ok' : 'risk'} /> : null}
                    {parsed?.think ? <p className="think">“{parsed.think}”</p> : null}
                    {showRaw && a ? <pre className="answer-raw">{a.got}</pre> : null}
                    {!parsed?.ok && a && !showRaw ? <p className="think">“{a.got.slice(0, 140)}”</p> : null}
                  </li>
                )
              })}
            </ol>
            <label className="raw-toggle"><input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} /> show the raw text</label>
          </div>
        ) : null}
      </div>
    </Panel>
  )
}
