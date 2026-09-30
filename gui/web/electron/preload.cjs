/**
 * The only bridge between the renderer and the main process.
 *
 * Deliberately one-way and one-purpose: the renderer reports the theme colours
 * it resolved so the native caption buttons can match the header. Nothing here
 * exposes the filesystem, the shell, or ipcRenderer itself, so a compromised
 * renderer gains one cosmetic call and nothing else.
 */

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('retrainDesktop', {
  /** True only inside the desktop shell; the web build leaves this undefined. */
  isDesktop: true,

  /**
   * @param {{ color: string, symbolColor: string, dark: boolean }} colors
   * Resolved CSS colours for the header background and its foreground text.
   */
  setTitleBarTheme(colors) {
    ipcRenderer.send('retrain:title-bar-theme', colors)
  },
})
