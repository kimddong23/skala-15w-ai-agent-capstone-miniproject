# Vital Sentinel 프론트엔드: React 빌드 → nginx(비루트)로 정적 서빙 + /api 를 백엔드로 리버스 프록시.
#   docker build -t vital-sentinel-frontend frontend/
# 같은 출처(/api)로 호출하므로 브라우저 CORS 설정이 필요 없다.
# 베이스 이미지는 태그(읽는 사람용)와 digest(실제 고정값)를 함께 적어 고정한다(2026-10-20 기준 최신). 같은 태그가 나중에 다른
# 내용을 가리켜도 빌드가 바뀌지 않는다. 갱신할 때: 새 digest 를 확인하고(`docker buildx imagetools inspect <이미지:태그>` 의 Digest)
# 이 줄을 바꾼 뒤 `docker compose build` 와 /health·/chat 확인까지 통과시킨다. tests/test_packaging.py 가 digest 누락을 잡는다.
FROM node:22-alpine@sha256:0a7108bf6c7bf5de370ffb1a3ed6be93d405b43ff159f681a8d18c0e2bc2e402 AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
# tsc --noEmit + vite build. 타입 오류가 있으면 이미지 빌드가 실패한다.
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine@sha256:65e3e85dbaed8ba248841d9d58a899b6197106c23cb0ff1a132b7bfe0547e4c0
# BACKEND_URL 은 기동 시 envsubst 로 nginx 설정에 들어간다(공식 이미지의 /etc/nginx/templates 기능).
ENV BACKEND_URL=http://backend:8100 \
    NGINX_ENVSUBST_FILTER=^BACKEND_
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY nginx-security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
