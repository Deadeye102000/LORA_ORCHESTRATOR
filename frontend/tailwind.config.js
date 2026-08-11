/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: "#0d1117",
          800: "#161b22",
          700: "#21262d",
          600: "#30363d",
        },
        teal: {
          400: "#00b4d8",
          500: "#0096c7",
          600: "#0077b6",
        },
        status: {
          idle:    "#2ea043",
          busy:    "#f0a500",
          offline: "#f85149",
          pending: "#8b949e",
          completed: "#2ea043",
          failed:  "#f85149",
          training:"#00b4d8",
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      animation: {
        pulse_slow: "pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
}
