#!/usr/bin/env bash
# pre-commit 프레임워크 없이 로컬 gitleaks 바이너리로 커밋 훅을 설치한다.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
command -v gitleaks >/dev/null || { echo "gitleaks 미설치: brew install gitleaks"; exit 1; }
cat > .git/hooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
# 스테이징된 변경분만 검사한다. 유출이 발견되면 커밋을 중단한다.
exec gitleaks git --pre-commit --staged --redact --no-banner
HOOK
chmod +x .git/hooks/pre-commit
echo "gitleaks pre-commit 훅 설치 완료"
