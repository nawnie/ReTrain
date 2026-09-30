/**
 * Chart colours, read from the app's own theme tokens.
 *
 * theme.css is the only file allowed to hold a colour literal, so the charts
 * ask the browser for the resolved value of each role token instead of writing
 * their own. When the reader switches theme (or the system does), the tokens
 * change and this hook returns a fresh palette, which redraws the charts.
 */

import { useEffect, useState } from 'react'

export interface Palette {
  ink: string
  inkMuted: string
  inkFaint: string
  line: string
  surface: string
  sunken: string
  accent: string
  ok: string
  warn: string
  risk: string
  ramp0: string
  ramp1: string
  ramp2: string
}

function read(): Palette {
  const css = getComputedStyle(document.documentElement)
  const v = (name: string) => css.getPropertyValue(name).trim()
  return {
    ink: v('--ink'), inkMuted: v('--ink-muted'), inkFaint: v('--ink-faint'), line: v('--line'), surface: v('--surface'), sunken: v('--surface-sunken'),
    accent: v('--accent'), ok: v('--ok'), warn: v('--warn'), risk: v('--risk'), ramp0: v('--ramp-0'), ramp1: v('--ramp-1'), ramp2: v('--ramp-2'),
  }
}

export function usePalette(): Palette {
  const [palette, setPalette] = useState<Palette>(read)
  useEffect(() => {
    const update = () => setPalette(read())
    // An explicit theme choice changes the data-theme attribute; the system choice changes the media query.
    const observer = new MutationObserver(update)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    media.addEventListener('change', update)
    return () => {
      observer.disconnect()
      media.removeEventListener('change', update)
    }
  }, [])
  return palette
}

/** Options every chart shares: fonts, text colour, and a quiet tooltip. */
export function baseOption(p: Palette) {
  return {
    animationDuration: 500,
    textStyle: { color: p.inkMuted, fontFamily: 'var(--font-ui), "Segoe UI", sans-serif' },
    tooltip: { backgroundColor: p.surface, borderColor: p.line, textStyle: { color: p.ink }, extraCssText: 'box-shadow: var(--shadow-raised);' },
  }
}
