/**
 * 애플리케이션 진입점. 번들 폰트를 등록하고 `<App />` 을 마운트한다.
 * 폰트는 CDN 없이 `@fontsource` 로 번들해 오프라인(내부망·시연장)에서도 동일하게 보인다. 한글 본문 400/600, 숫자용 모노 400/500.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/ibm-plex-sans-kr/400.css";
import "@fontsource/ibm-plex-sans-kr/600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
