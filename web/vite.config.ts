import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const webApiKey = env.CAREER_AGENT_WEB_API_KEY?.trim();
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": {
          target: env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
          headers: webApiKey
            ? { Authorization: `Bearer ${webApiKey}` }
            : undefined,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  };
});
