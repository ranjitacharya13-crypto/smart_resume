import type { Config } from 'tailwindcss';
const config: Config = { content: ['./app/**/*.{js,ts,jsx,tsx,mdx}'], theme: { extend: { boxShadow: { glow: '0 24px 70px rgba(15,23,42,.22)' } } }, plugins: [] };
export default config;
