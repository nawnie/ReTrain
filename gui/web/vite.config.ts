import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The FastAPI backend in gui/api serves every /api/retrain route. In dev the
// browser talks to Vite and Vite forwards those calls, so the app never needs a
// cross-origin request and the backend's CORS allowlist is not involved.
const backend =
  process.env.RETRAIN_API_TARGET ??
  process.env.VITE_RETRAIN_API_TARGET ??
  'http://127.0.0.1:8787'

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    // The Electron shell loads the build from a custom protocol at the app
    // root, so assets must be referenced relatively rather than from '/'.
    assetsDir: 'assets',
  },
  server: {
    host: '127.0.0.1',
    port: 4173,
    proxy: {
      '/api/retrain': { target: backend, changeOrigin: true },
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    proxy: {
      '/api/retrain': { target: backend, changeOrigin: true },
    },
  },
})
