import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        // Three.js is the bulk of the bundle and changes far less often than
        // app code, so splitting it keeps it cached across deploys and lets the
        // shell paint without waiting on the 3D engine.
        manualChunks: {
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          motion: ['framer-motion'],
        },
      },
    },
  },
  server: {
    port: 5173,
    // The backend sets CORS, but proxying in dev keeps the browser on one
    // origin so microphone permission and cookies behave predictably.
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
