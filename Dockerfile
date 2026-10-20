# Vital Sentinel 백엔드 (FastAPI + MCP stdio subprocess + LangGraph).
#   docker build -t vital-sentinel-backend .
# - 비루트 사용자(uid 10001)로 실행한다.
# - 가공 데이터(data/processed, 공개 FitBit 데이터(CC BY 4.0)에서 만든 약 8MB)를 이미지에 포함한다. 원본 CSV(data/raw)는 넣지 않는다.
# - Ollama 는 이미지 밖에 있다: OLLAMA_BASE_URL 로 가리킨다. 없어도 기본 리포트(폴백) 모드로 동작한다.
# 베이스 이미지는 태그(읽는 사람용)와 digest(실제 고정값)를 함께 적어 고정한다(2026-10-20 기준 최신). 같은 태그가 나중에 다른
# 내용을 가리켜도 빌드가 바뀌지 않는다. 갱신할 때: 새 digest 를 확인하고(`docker buildx imagetools inspect <이미지:태그>` 의 Digest)
# 이 줄을 바꾼 뒤 `docker compose build` 와 /health·/chat 확인까지 통과시킨다. tests/test_packaging.py 가 digest 누락을 잡는다.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 의존성 레이어를 소스보다 먼저 두어 코드만 바뀌면 재설치하지 않는다.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# 코드와 가공 데이터는 root 소유 + 쓰기 금지로 둔다: 실행 사용자(app)가 침해돼도 코드를 바꿔치기할 수 없다.
COPY app ./app
COPY data/processed ./data/processed
# 쓸 수 있는 곳은 두 군데뿐이다.
#  - /app/state: 대화 기억(sqlite 체크포인터). compose 가 이름 있는 볼륨을 붙인다.
#  - /app/data/processed/rag: 가이드라인 임베딩 캐시(index.npz). 서버가 캐시를 다시 쓸 수 있어야 한다.
RUN chmod -R a+rX,go-w /app/app /app/data \
    && mkdir -p /app/state /app/data/processed/rag \
    && chown -R app:app /app/state /app/data/processed/rag

USER app

ENV DATA_DIR=/app/data/processed \
    SQLITE_PATH=/app/state/checkpoints.sqlite \
    CHECKPOINTER=memory \
    OLLAMA_BASE_URL=http://host.docker.internal:11434

EXPOSE 8100

# /health 는 MCP 연결까지 확인한다. 기동 직후 MCP subprocess 가 올라올 시간을 start-period 로 준다.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8100"]
