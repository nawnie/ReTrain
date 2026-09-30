/**
 * The surface the desktop shell's preload script exposes.
 *
 * Optional on purpose: the same build runs in a browser, where this is
 * undefined and every caller must handle its absence.
 */

export interface RetrainDesktopBridge {
  isDesktop: true
  /** Report resolved header colours so the native caption buttons match. */
  setTitleBarTheme(colors: { color: string; symbolColor: string; dark: boolean }): void
}

declare global {
  interface Window {
    retrainDesktop?: RetrainDesktopBridge
  }
}

export {}
