/**
 * The Configure pane: every control that defines a training run.
 *
 * The controls are grouped by the question they answer, not by data type:
 * what am I training, how am I training it, how much memory does each step
 * cost, and what am I trading away to make it fit. That ordering matches how
 * an operator actually narrows down a run.
 *
 * This component is presentational. It holds no state and performs no
 * requests; the shell owns the settings and the plan.
 */

import type { Bootstrap, TrainingSettings } from '../types'
import { NumberField, Panel, SelectField, Switch } from '../ui'

/** Methods the backend maps to a runner (gui/api/app.py METHOD_MAP). */
const METHODS = [
  'QLoRA',
  'LoRA',
  'Full fine-tune',
  'DPO',
  'GRPO',
  'Reward model',
  'KTO',
  'RLOO',
  'PPO',
]

const PRECISIONS = ['4-bit', '8-bit', 'BF16', 'FP16']
const TUNE_SCOPES = ['Last layers', 'Full model', 'Output head only']
const OPTIMIZERS = ['Paged AdamW', 'AdamW 8-bit', 'Lion', 'CPU AdamW']
const SCHEDULERS = ['Cosine', 'Linear', 'Constant with warmup']

/** Adapter rank only means something for the methods that train an adapter. */
const ADAPTER_METHODS = new Set(['LoRA', 'QLoRA'])

export function Configure(props: {
  boot: Bootstrap
  settings: TrainingSettings
  onChange: <K extends keyof TrainingSettings>(key: K, value: TrainingSettings[K]) => void
}) {
  const { boot, settings, onChange } = props
  const model = boot.models.find((item) => item.id === settings.modelId)
  const dataset = boot.datasets.find((item) => item.id === settings.datasetId)
  const trainsAdapter = ADAPTER_METHODS.has(settings.method)
  const isFullFineTune = settings.method === 'Full fine-tune'

  return (
    <div>
      {/* What is being trained, and on what. */}
      <Panel
        title="Run definition"
        note="The base weights and the data they will be adapted with."
      >
        <div className="field-grid">
          <SelectField
            label="Base model"
            hint={model ? `${model.family} · ${model.sizeB}B · ${model.backend}` : undefined}
            value={settings.modelId}
            options={boot.models.map((item) => ({ value: item.id, label: item.name }))}
            onChange={(value) => onChange('modelId', value)}
          />
          <SelectField
            label="Dataset"
            hint={dataset ? `${dataset.status} · ${dataset.source}` : undefined}
            value={settings.datasetId}
            options={boot.datasets.map((item) => ({ value: item.id, label: item.name }))}
            onChange={(value) => onChange('datasetId', value)}
          />
        </div>
        {model ? <p className="field-hint">{model.status}</p> : null}
      </Panel>

      {/* How it is being trained. Method drives most of the memory model, so it
          leads, and the fields it makes irrelevant are disabled rather than
          hidden -- a control that vanishes reads as a bug. */}
      <Panel title="Method" note="Determines the training loop and most of the memory profile.">
        <div className="field-grid">
          <SelectField
            label="Training method"
            value={settings.method}
            options={METHODS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => onChange('method', value)}
          />
          <SelectField
            label="Precision"
            hint={
              settings.precision === '4-bit'
                ? 'Smallest footprint; pairs with QLoRA.'
                : 'Higher precision costs roughly 3.5x the weight memory of 4-bit.'
            }
            value={settings.precision}
            options={PRECISIONS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => onChange('precision', value)}
          />
          <SelectField
            label="Tune scope"
            hint={
              isFullFineTune
                ? 'Applies to full fine-tuning only.'
                : 'Adapter methods train the adapter, not the base weights.'
            }
            value={settings.tuneScope}
            options={TUNE_SCOPES.map((item) => ({ value: item, label: item }))}
            onChange={(value) => onChange('tuneScope', value)}
          />
          {/* Only meaningful once "Last layers" is chosen -- shown next to that
              choice rather than buried in a separate panel, since the Layers
              pane sets this pair together and an operator adjusting it by hand
              needs to see both at once. */}
          {settings.tuneScope === 'Last layers' ? (
            <NumberField
              label="Layers from the end"
              hint="Set here, or by selecting a trailing range on the Layers pane."
              value={settings.lastNLayers}
              min={1}
              step={1}
              onChange={(value) => onChange('lastNLayers', value)}
            />
          ) : null}
        </div>
      </Panel>

      {/* The three numbers that move the activation memory the most. */}
      <Panel
        title="Capacity"
        note="Sequence length and batch size drive activation memory almost linearly."
      >
        <div className="field-grid">
          <NumberField
            label="Context length"
            hint="Tokens per sample."
            value={settings.contextLength}
            min={128}
            step={128}
            onChange={(value) => onChange('contextLength', value)}
          />
          <NumberField
            label="Micro batch"
            hint="Samples held on the GPU at once."
            value={settings.microBatch}
            min={1}
            step={1}
            onChange={(value) => onChange('microBatch', value)}
          />
          <NumberField
            label="Gradient accumulation"
            hint="Raises effective batch size at no VRAM cost."
            value={settings.gradAccum}
            min={1}
            step={1}
            onChange={(value) => onChange('gradAccum', value)}
          />
          <NumberField
            label="LoRA rank"
            hint={trainsAdapter ? 'Adapter capacity.' : 'Not used by this method.'}
            value={settings.loraRank}
            min={1}
            step={1}
            onChange={(value) => onChange('loraRank', value)}
          />
        </div>
      </Panel>

      {/* Every switch here buys memory with either speed or fidelity. The note
          on each one says which, so the trade is visible at the point of the
          decision rather than in documentation. */}
      <Panel title="Memory strategy" note="Each of these trades something for VRAM headroom.">
        <div className="switch-list">
          <Switch
            name="Gradient checkpointing"
            note="Recomputes activations in the backward pass. Roughly 40% less activation memory, noticeably slower."
            checked={settings.gradientCheckpointing}
            onChange={(value) => onChange('gradientCheckpointing', value)}
          />
          <Switch
            name="Flash attention"
            note="Memory-efficient attention kernel. Faster and smaller where the GPU supports it."
            checked={settings.flashAttention}
            onChange={(value) => onChange('flashAttention', value)}
          />
          <Switch
            name="CPU optimizer offload"
            note="Moves optimizer state to system RAM. Large VRAM saving, bound by PCIe bandwidth."
            checked={settings.cpuOffload}
            onChange={(value) => onChange('cpuOffload', value)}
          />
          <Switch
            name="Low VRAM mode"
            note="Conservative allocation throughout. Use when the estimate is close to the limit."
            checked={settings.lowVramMode}
            onChange={(value) => onChange('lowVramMode', value)}
          />
          <Switch
            name="TensorBoard logging"
            note="Writes scalars to the workspace log directory for the Runtime pane to serve."
            checked={settings.tensorboard}
            onChange={(value) => onChange('tensorboard', value)}
          />
          <Switch
            name="Trust remote code"
            note="Allows model repositories to execute their own Python. Leave off unless the source is known."
            checked={settings.trustRemoteCode}
            onChange={(value) => onChange('trustRemoteCode', value)}
          />
        </div>
      </Panel>

      {/* Rarely changed, so it sits last rather than competing for attention. */}
      <Panel title="Optimizer" note="Defaults suit most local runs.">
        <div className="field-grid">
          <SelectField
            label="Optimizer"
            value={settings.optimizer}
            options={OPTIMIZERS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => onChange('optimizer', value)}
          />
          <SelectField
            label="Scheduler"
            value={settings.scheduler}
            options={SCHEDULERS.map((item) => ({ value: item, label: item }))}
            onChange={(value) => onChange('scheduler', value)}
          />
        </div>
      </Panel>
    </div>
  )
}
