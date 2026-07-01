/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      gridTemplateColumns: {
        "32": "repeat(32, minmax(0, 1fr))",
      },
    },
  },
  plugins: [],
};
