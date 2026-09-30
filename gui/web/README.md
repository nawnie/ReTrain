# ReTrain console

The frontend for the local ReTrain API (`gui/api/app.py`), and the desktop shell
that runs it.

## Running it

**As the desktop app**, from the repository root:

```bash
launch_retrain_web.bat
```

The shell starts and supervises the backend itself, so nothing else needs to be
running first.

**In a browser, for development:**

```bash
npm ci
npm run dev        # http://localhost:4173, proxies /api/retrain to the backend
```

The backend then has to be started separately:

```bash
python -m uvicorn gui.api.app:app --host 127.0.0.1 --port 8787
```

Run the backend command from the repository root. Point the dev proxy elsewhere with `RETRAIN_API_TARGET`; point the desktop shell
elsewhere with `RETRAIN_API_HOST` and `RETRAIN_API_PORT`.

## The desktop shell (`electron/`)

**The renderer uses the same relative paths everywhere.** In dev, Vite proxies
`/api/retrain/*`; in the desktop app, the `retrain://` protocol handler forwards
it to the same local API. The client contains no deployment knowledge and there
is no separate desktop code path through the data layer.

**Native caption buttons, custom header.** `titleBarStyle: 'hidden'` with
`titleBarOverlay` keeps the app's own header while leaving minimise, maximise,
snap layouts, and their keyboard and accessibility behaviour to Windows. The
renderer reports its resolved `--surface` and `--ink` so the buttons match the
header in both themes. The header reserves space for them using
`env(titlebar-area-width)`, which collapses to normal padding in a browser.

**The backend is supervised, and only ours is stopped.** If something already
answers on the API port it is adopted — an operator's own uvicorn, or one
orphaned by a shell that was force-killed. Only a backend this shell started is
ever terminated, together with its child processes.

**Failures are diagnosable.** The standing backend must run under
the project's `pythonw`, which has no console standard handles. Instead the
import is preflighted in a normal short-lived interpreter whose stderr *can* be
read, and uvicorn is given a `--log-config` that writes
`training/gui_runs/desktop-api.log` itself.

**A backend that will not start does not block the window.** The UI has a
designed offline state that names the problem and recovers on its own once the
API answers.

## What the design is trying to do

The expensive mistake in this app is not a mis-click, it is a misconfigured run
that burns hours of GPU before failing. So the console is built around one idea:
**make the cost of a configuration visible at the moment you configure it**.

**The setting and its consequence share a screen.** Configure is two columns —
the controls on the left, the server-validated VRAM projection on the right.
Move a control and the estimate, the fit verdict, the headroom bar, and the
breakdown all move with it, debounced and abortable so an older estimate can
never land after a newer one. Splitting those across pages would hide the
consequence from the decision.

**A total is not actionable; a breakdown is.** "48.5 GB" tells you there is a
problem. "Activations 44.9 GB — 16384 ctx x 32 batch" tells you which control to
move.

**Launching is a shell row, not page content.** Dry run, arm, start, and stop
live in a bar pinned below the work area. They are present on every pane at
every window size and cannot scroll out of reach.

**Arming is separate from starting.** A real run costs hours, so one button
should not be able to spend that on a mis-click. Arming is consent for one
specific configuration and is revoked automatically the moment any control
changes.

**Status never rides on colour alone.** Every chip carries a glyph, a word, and
a colour — three independent signals, so a reader with a colour vision
deficiency reads the same result.

## Layout and theme rules this follows

- **One token layer.** `theme.css` is the only file containing a colour
  literal. Every component rule references a role token. A hex inside a
  component rule is identical in every theme, so it can only ever be right in
  one of them.
- **No orphan tokens.** All 17 role tokens are redefined in both themes.
  A token defined in only one layer keeps a foreign value on the other's
  surface, and it lands on `::selection`, scrollbars and focus rings, where a
  component review never looks.
- **`color-scheme` matches the paint** in both themes, so native selects,
  number spinners, checkboxes and scrollbars render in the right chrome.
- **The shell owns the viewport; panes scroll.** `.app` is a three-row grid and
  only the work area flexes. Each pane is an explicit `overflow-y: auto`, which
  is what makes a fixed header and launch bar safe.
- **No fixed pixel layout.** Every track is `rem`, `fr`, `minmax`, or `clamp`,
  and every flexible grid track is floored with `minmax(0, 1fr)`.
- **One spacing scale**, in `rem`, so a reader who raises their browser font
  size gets a layout that grows with them.
- **Panes are addressable** (`#configure`, `#runs`, `#runtime`). Deep links and
  the back button work — and, less obviously, an automated review can visit
  every view. A single-page app whose views live only in client state ships
  every pane but the first unverified.

## Verifying it

The original local design review recorded a worst contrast pair of
`--ink-faint` on `--surface-sunken` at 4.62:1. It also recorded the three optional
checks below passing; these are historical results, not fresh-checkout checks:

```bash
# Render legibility: clipped roots, rigid layout, hardcoded colour, spacing scale
python <wren>/scripts/validate_wren_design.py src/theme.css src/app.css

# Canonical frontend audit, including the WLEG render-legibility family
python <wren>/skills/wren-avoid-ai-design/scripts/audit_frontend.py src --profile dashboard

# Reachability: every control the reader must be able to get to, at real window sizes
py -3.12 <wren>/skills/wren-avoid-ai-design/scripts/render_frontend.py \
  http://localhost:4173/ ./render-out \
  --viewports 400x900,768x1024,1280x800,1480x940,1920x1080,1050x700 \
  --paths "#configure,#runs,#runtime"
```

`<wren>` is an optional Wren specialist installation. These design-review tools
are not required to install or launch the console. The standard build check is
`npm run build` in `gui/web`.

## Files

| File | Holds |
|---|---|
| `src/theme.css` | The only token layer. Every colour literal in the app. |
| `src/app.css` | Component styles, ordered outer shell inwards. No colour literals. |
| `src/App.tsx` | The shell. All state, all requests, the launch bar, pane routing. |
| `src/api.ts` | Typed client. Every call cancellable, every failure typed. |
| `src/types.ts` | The backend's payload shapes, mirrored from `gui/api/app.py`. |
| `src/ui.tsx` | Presentational primitives shared across panes. |
| `src/panels/` | One file per pane. Presentational; no state, no requests. |
