/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        surface:  '#0f172a',   // slate-950
        panel:    '#1e293b',   // slate-800
        border:   '#334155',   // slate-700
        muted:    '#64748b',   // slate-500
      },
      animation: {
        'pulse-fast': 'pulse 0.8s cubic-bezier(0.4,0,0.6,1) infinite',
      },
    },
  },
  plugins: [],
}
