import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { tanstackRouter } from "@tanstack/router-plugin/vite";

export default defineConfig({
  plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss(), tsconfigPaths()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  test: { environment: "jsdom", setupFiles: ["./src/test/setup.ts"] },
});
