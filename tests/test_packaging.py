"""패키징 산출물(sh/, docker-compose.yml)이 앱 설정과 어긋나지 않는지 확인한다. LLM·네트워크·Docker 데몬 없이 파일만 읽는다."""
import re
import subprocess
from pathlib import Path

from app.config import settings

ROOT = Path(__file__).resolve().parent.parent


def test_ollama_pull_defaults_match_app_config():
    """compose 의 ollama-pull 기본 모델이 app/config.py 기본값과 같다(기본 모델을 바꿀 때 한 곳만 고치는 실수를 막는다)."""
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    fast = re.search(r"FAST_MODEL: \$\{FAST_MODEL:-([^}]+)\}", text)
    quality = re.search(r"QUALITY_MODEL: \$\{QUALITY_MODEL:-([^}]+)\}", text)
    assert fast and quality, "ollama-pull 의 모델 기본값 줄을 찾지 못했다"
    assert fast.group(1) == settings.fast_model
    assert quality.group(1) == settings.quality_model


def test_setup_ollama_reads_default_models_from_config(monkeypatch):
    """setup_ollama.sh 가 app/config.py 기본값을 그대로 읽고, base 철자(OLLAMA_FAST_MODEL)가 우선한다. (dry-run: 아무것도 받지 않는다)"""
    def plan(extra_env: dict) -> str:
        env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(ROOT), "OLLAMA_DRY_RUN": "1", "OLLAMA_BASE_URL": "http://127.0.0.1:9", **extra_env}
        out = subprocess.run(["bash", str(ROOT / "sh" / "setup_ollama.sh")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        return out.stdout + out.stderr

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        import pytest
        pytest.skip("ollama CLI 없음")
    default = plan({})
    assert f"fast={settings.fast_model}" in default and f"quality={settings.quality_model}" in default
    override = plan({"OLLAMA_FAST_MODEL": "qwen3:0.6b"})
    assert "fast=qwen3:0.6b" in override


def test_shell_scripts_have_valid_syntax():
    for f in sorted((ROOT / "sh").glob("*.sh")):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


def test_images_are_pinned_by_digest():
    """베이스·외부 이미지는 digest 로 고정돼 있다(태그만 쓰면 같은 태그가 바뀔 때 빌드가 조용히 달라진다)."""
    lines = []
    for f in ("Dockerfile", "frontend/Dockerfile"):
        lines += [l for l in (ROOT / f).read_text(encoding="utf-8").splitlines() if l.startswith("FROM ")]
    lines += [l.strip() for l in (ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines() if l.strip().startswith("image: ollama/")]
    assert len(lines) >= 4
    for l in lines:
        assert re.search(r"@sha256:[0-9a-f]{64}", l), f"digest 없음: {l}"


def test_ollama_service_publishes_no_host_port():
    """인증 없는 Ollama 가 호스트·LAN 에 노출되지 않는다: ollama 서비스에는 ports 가 없다(compose 네트워크 안에서만 접근)."""
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    block = re.search(r"^  ollama:\n(.*?)(?=^  \S|\Z)", text, re.S | re.M)
    assert block, "ollama 서비스를 찾지 못했다"
    assert not re.search(r"^\s+ports:", block.group(1), re.M)
    assert "OLLAMA_HOST=0.0.0.0" not in text.replace("권하지 않는다", "")  # 안내 문구는 허용, 설정값으로 쓰지 않는다


def test_runtime_user_cannot_write_code_dirs():
    """Dockerfile: 코드·가공 데이터는 root 소유 + 쓰기 금지, 쓰기 가능한 곳은 state 와 rag 캐시뿐이다."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=app:app" not in text
    assert "chmod -R a+rX,go-w /app/app /app/data" in text
    assert "chown -R app:app /app/state /app/data/processed/rag" in text
