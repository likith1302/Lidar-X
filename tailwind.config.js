/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#06080C',
          900: '#0B0E14',
          850: '#10141D',
          800: '#161B26',
          750: '#1C2331',
          700: '#242C3D',
          600: '#343E54',
          500: '#4B5563',
          400: '#9CA3AF',
          300: '#D1D5DB',
          200: '#E5E7EB',
          100: '#F3F4F6',
        },
        brand: {
          cyan: '#00E5FF',
          blue: '#0070F3',
          purple: '#8A2BE2',
          purpleLight: '#A855F7',
          orange: '#FF8A00',
          red: '#EF4444',
          emerald: '#10B981',
          amber: '#F59E0B',
        },
        semantic: {
          drivable: '#00E5FF',     // Cyan / Electric blue
          nondrivable: '#64748B',  // Slate
          staticObstacle: '#F59E0B',// Amber / Orange
          dynamicObstacle: '#EF4444',// Red / Coral
          vegetation: '#10B981',   // Emerald
          infrastructure: '#8B5CF6',// Purple / Violet
          unknown: '#475569',      // Muted slate
        }
      },
      fontFamily: {
        sans: ['Inter', 'DM Sans', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 20px -5px rgba(0, 229, 255, 0.3)',
        'glow-purple': '0 0 20px -5px rgba(138, 43, 226, 0.3)',
        'glow-orange': '0 0 20px -5px rgba(255, 138, 0, 0.3)',
        'inner-dark': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.6)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 4s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%, 100%': { transform: 'translateY(0%)' },
          '50%': { transform: 'translateY(100%)' },
        }
      }
    },
  },
  plugins: [],
}
