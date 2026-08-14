import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#09090B",
        surface: "#111113",
        card: "#18181B",
        accent: "#38BDF8",
      },
      boxShadow: { glow: "0 12px 42px rgba(14, 165, 233, .13)" },
    },
  },
  plugins: [],
};

export default config;
