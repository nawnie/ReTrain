/** The top of the Live pane: state, a progress ring, and the numbers that matter with a tiny history line under each. */

import type { GpuSample, LiveRun, LiveStatus } from '../../types'
import { Chip, type Tone } from '../../ui'
import { clock, latest, speedSeries, thin } from '../format'

const STATUS_WORD: Record<LiveStatus, string> = {
  loading: 'Loading the model',
  training: 'Training',
  done: 'Finished',
  failed: 'Failed',
  stopped: 'Stopped',
}

/** "trainable params: 65,568,768 || all params: 12,025,298,944 ..." -> "65.6M of 12.0B parameters train" */
function params(text: string): string {
  const m = text.match(/trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)/)
  if (!m) return ''
  const a = Number(m[1].replace(/,/g, ''))
  const b = Number(m[2].replace(/,/g, ''))
  return `${(a / 1e6).toFixed(1)}M of ${(b / 1e9).toFixed(1)}B parameters train`
}

function tone(status: LiveStatus): Tone {
  if (status === 'training' || status === 'done') return 'ok'
  if (status === 'loading') return 'warn'
  return 'risk'
}

/** A small line chart with no axes: shape only. */
function Spark({ values, label }: { values: number[]; label: string }) {
  if (values.length < 2) return <svg className="spark" viewBox="0 0 100 24" aria-hidden="true" />
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo || 1
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${((i / (values.length - 1)) * 100).toFixed(1)},${(22 - ((v - lo) / span) * 20).toFixed(1)}`).join(' ')
  return (
    <svg className="spark" viewBox="0 0 100 24" preserveAspectRatio="none" role="img" aria-label={label}>
      <path d={d} fill="none" />
    </svg>
  )
}

function Ring({ fraction, status }: { fraction: number; status: LiveStatus }) {
  const r = 46
  const c = 2 * Math.PI * r
  return (
    <div className="ring" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fraction * 100)} aria-label="Training progress">
      <svg viewBox="0 0 108 108">
        <circle className="ring-track" cx="54" cy="54" r={r} />
        <circle className="ring-fill" data-tone={tone(status)} cx="54" cy="54" r={r} strokeDasharray={c} strokeDashoffset={c * (1 - fraction)} />
      </svg>
      <div className="ring-text">
        <b>{Math.round(fraction * 100)}</b>
        <span>%</span>
      </div>
    </div>
  )
}

export function Hero({ run, gpu }: { run: LiveRun; gpu: GpuSample[] }) {
  const t = tone(run.status)
  const losses = run.loss.map((p) => p[1])
  const speeds = speedSeries(run).map((s) => s[1])
  const now = latest(gpu)
  const lossDrop = run.loss.length > 1 && run.loss[0][1] > 0 ? 1 - run.loss[run.loss.length - 1][1] / run.loss[0][1] : null
  const done = run.epoch * run.trainExamples

  return (
    <section className="hero" aria-label="Run summary">
      <Ring fraction={run.progress} status={run.status} />

      <div className="hero-main">
        <div className="hero-status">
          <span className="live-dot" data-live={run.status === 'training' ? 'on' : 'off'} data-tone={t} />
          <Chip tone={t}>{run.status === 'loading' ? (run.loadingPercent > 0 ? `${STATUS_WORD.loading} ${run.loadingPercent}%` : `${STATUS_WORD.loading}…`) : STATUS_WORD[run.status]}</Chip>
          {run.demo ? <span className="demo-badge" title="Simulated data written by scripts/make_live_demo.py">DEMO DATA · not a real run</span> : null}
        </div>
        <h2 className="hero-title">{run.title}</h2>
        <p className="hero-sub">
          epoch {run.epoch.toFixed(2)}
          {run.totalEpochs ? ` of ${run.totalEpochs}` : ''} · {Math.round(done)} of {run.trainExamples ? Math.round(run.totalEpochs * run.trainExamples) : '—'} examples
          {params(run.trainable) ? ` · ${params(run.trainable)}` : ''}
        </p>
        {run.savedTo ? <p className="hero-saved">Adapter saved to <code>{run.savedTo}</code></p> : null}
        {run.error ? <p className="hero-error">{run.error}</p> : null}
      </div>

      <dl className="hero-tiles">
        <div className="tile">
          <dt>Loss</dt>
          <dd>{run.lossNow === null ? '—' : run.lossNow.toFixed(4)}</dd>
          <small>{lossDrop === null ? 'first reading soon' : `down ${Math.round(lossDrop * 100)}% since the start`}</small>
          <Spark values={thin(losses, 40)} label="Loss over the run" />
        </div>
        <div className="tile">
          <dt>Speed</dt>
          <dd>{run.examplesPerSecond ? run.examplesPerSecond.toFixed(2) : '—'}<em> ex/s</em></dd>
          <small>{run.status === 'done' ? 'average' : 'recent'}</small>
          <Spark values={thin(speeds, 40)} label="Speed over the run" />
        </div>
        <div className="tile">
          <dt>Time</dt>
          <dd>{clock(run.elapsedSeconds)}</dd>
          <small>{run.status === 'done' ? 'total' : run.etaSeconds === null ? 'estimating…' : `~${clock(run.etaSeconds)} left`}</small>
        </div>
        <div className="tile">
          <dt>GPU memory</dt>
          <dd>{now ? now.usedGb.toFixed(1) : '—'}<em>{now ? ` / ${now.totalGb.toFixed(0)} GB` : ''}</em></dd>
          <small>{run.peakVram ? `peak allocated ${run.peakVram}` : now ? `${Math.round(now.util)}% busy · ${now.tempC}°C` : ' '}</small>
          <Spark values={thin(gpu.map((g) => g.usedGb), 40)} label="GPU memory recently" />
        </div>
      </dl>
    </section>
  )
}
