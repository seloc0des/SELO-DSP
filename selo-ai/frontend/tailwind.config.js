/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0f0f0f',
        'darker-bg': '#0a0a0a',
        'neon-cyan': '#00ffff',
        'neon-magenta': '#ff00ff',
        'neon-pink': '#ff1493',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow-cyan': 'glow-cyan 1.5s ease-in-out infinite alternate',
        'glow-magenta': 'glow-magenta 1.5s ease-in-out infinite alternate',
      },
      keyframes: {
        'glow-cyan': {
          '0%': { 'text-shadow': '0 0 5px #00ffff, 0 0 10px #00ffff' },
          '100%': { 'text-shadow': '0 0 20px #00ffff, 0 0 30px #00ffff, 0 0 40px #00ffff' },
        },
        'glow-magenta': {
          '0%': { 'text-shadow': '0 0 5px #ff00ff, 0 0 10px #ff00ff' },
          '100%': { 'text-shadow': '0 0 20px #ff00ff, 0 0 30px #ff00ff, 0 0 40px #ff00ff' },
        },
      },
    },
  },
  plugins: [],
}
