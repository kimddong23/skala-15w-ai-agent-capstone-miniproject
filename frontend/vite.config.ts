import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// 빌드·실행 환경변수(Node). 이 설정 파일에서만 읽으므로 @types/node 없이 필요한 모양만 선언한다.
declare const process: { env: Record<string, string | undefined> };

// /api -> FastAPI(기본 8100, VITE_PROXY_TARGET 으로 변경). `vite preview` 도 같은 프록시를 쓴다. VITE_API_BASE 를 주면 프록시 대신 해당 주소로 직접 호출한다(CORS는 백엔드가 8600·5173 허용).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8600,
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_TARGET ?? "http://localhost:8100",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  test: { environment: "node" },
});
