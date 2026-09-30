/**
 * Entry point.
 *
 * theme.css is imported before app.css so the token layer is defined before
 * anything references it. There is exactly one token layer in this app; a
 * second stylesheet declaring :root variables would make import order decide
 * the theme, and any token defined in only one of them would keep a foreign
 * value on the other's surface.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './theme.css'
import './app.css'
import App from './App'

const container = document.getElementById('root')
if (!container) throw new Error('Root element #root is missing from index.html')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
