import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        calm: "#B6CECE",
        mindful: "#6A8D92",
      },
      backgroundImage: (theme) => ({
        "gradient-calm": "linear-gradient(to top, #B6CECE, #6A8D92)",
      }),
    },
  },
  plugins: [],
};
export default config;
