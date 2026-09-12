import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = process.env.CPL_BACKEND ?? 'http://127.0.0.1:8848'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: BACKEND,
        changeOrigin: true,
        // Server-sent events must not be buffered by the dev proxy, or progress
        // arrives in one lump at the end instead of streaming.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (String(proxyRes.headers['content-type']).includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
            }
          })
        },
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
})
