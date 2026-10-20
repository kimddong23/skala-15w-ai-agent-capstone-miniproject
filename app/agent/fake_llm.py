"""결정론적 Fake LLM 백엔드 (`LLM_BACKEND=fake`): 실제 모델 없이 LLM 경로 전체를 재현 가능하게 돌린다.

책임: 노드가 LLM 에 보내는 프롬프트에서 질문·사실을 읽어, 규칙(`gates`)과 템플릿으로 *LLM 처럼 보이는* 응답을 만든다.
`NullLLM`(항상 None → 폴백 경로)과 달리 `enabled=True` 이므로 구조화 출력·토큰 스트리밍·사용량 계량·검증 경로가 모두 실제로 실행된다.
`tests/fakes.py::FakeLLM`(스크립트된 응답·실패 주입용 테스트 더블)과는 목적이 다르다: 이쪽은 앱 안에서 쓰는 *백엔드*이다.
협력 모듈: `llm.make_llm`(선택), `nodes.py`(프롬프트 형식 제공), `gates.py`(규칙), `usage.py`(토큰 추정 기록).
한계: 응답 품질이 템플릿 수준이고 프롬프트 형식에 의존한다 — 프롬프트 형식을 바꾸면 이 파일의 파서도 함께 바꿔야 한다
(`tests/test_deps_backends.py` 가 계약을 고정한다).
"""
import re
import time

from app.agent import gates
from app.agent import report as rpt
from app.agent.usage import estimate_tokens, record
from app.agent.validator import NO_FINDING_SENTENCE

_NUM = r"([\d.]+)"


class FakeBackend:
    """프롬프트 기반 결정론 응답기. fast/quality 티어를 모두 처리한다."""

    enabled = True

    async def structured(self, tier, name, schema, system, user):
        """route/plan/judge 단계의 구조화 응답을 규칙으로 만든다 (알 수 없는 단계는 None)."""
        started = time.perf_counter()
        out = None
        if name == "route":
            msg = _between(user, "질문: ", "\n사용자 지정됨")
            route, reason = gates.route_fallback(msg, "이전 분석 결과 있음: 예" in user)
            out = schema(route=route, reasoning=f"fake: {reason}")
        elif name == "plan":
            out = schema(**gates.rule_based_plan(_between(user, "사용자 요청: ", "\n\n표준 점검 절차")))
        elif name == "validator_judge":
            out = schema(passed=True, issues=[])
        record(name, tier, "fake", estimate_tokens(system, user), estimate_tokens(str(out)), started, estimated=True)
        return out

    async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
        """analysis/writing/simple_answer 단계의 텍스트를 템플릿으로 만들어 조각으로 흘린다."""
        started = time.perf_counter()
        if name == "analysis":
            text = _analysis(user)
        elif name in ("writing", "v1_writing"):
            text = _report(user)
        elif name == "simple_answer":
            text = _simple(user)
        elif name == "summarize":
            prior = _between(user, "기존 요약:\n", "\n\n새로 요약할 대화:")
            turns = [tuple(l.split(": ", 1)) for l in _between(user, "새로 요약할 대화:\n", "").splitlines() if ": " in l]
            text = rpt.rule_summary("" if prior == "(없음)" else prior, turns)
        elif name == "chat":
            earlier = [l[2:] for l in _between(user, "이전 사용자 발화:\n", "\n\n대화:").splitlines() if l.startswith("- ")]
            text = gates.personal_reply(_between(user, "질문: ", "\n\n이전 사용자 발화"), earlier) or "네, 알겠습니다."
        else:
            text = "fake 응답"
        if on_token:
            for i in range(0, len(text), 24):
                await on_token(text[i : i + 24])
        record(name, tier, "fake", estimate_tokens(system, user), estimate_tokens(text), started, estimated=True)
        return text

    async def health(self) -> dict:
        """항상 정상 (외부 의존 없음)."""
        return {"enabled": True, "backend": "fake", "reachable": True, "models": ["fake"], "missing": []}


def _between(text: str, start: str, end: str) -> str:
    """text 에서 start 와 end 사이(end 가 빈 문자열이면 끝까지) 문자열."""
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i) if end else -1
    return (text[i:j] if j >= 0 else text[i:]).strip()


def _facts_lines(user: str) -> list[str]:
    """FACTS 블록(압축 텍스트 형식)의 줄 목록."""
    return _between(user, "FACTS:\n", "\n\nANALYSIS:").splitlines() if "ANALYSIS:" in user else _between(user, "FACTS:\n", "").splitlines()


def _anomaly_bullets(lines: list[str]) -> list[str]:
    return [l for l in lines if l.startswith("- [")]


def _analysis(user: str) -> str:
    """해석: 이상 항목의 근거 문장을 이어 붙인 한두 문장 (수치는 FACTS 에서만)."""
    lines = _facts_lines(user)
    if any("이상 탐지 실패" in l for l in lines):
        return "이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다."
    bullets = _anomaly_bullets(lines)
    return ("감지된 구간: " + " / ".join(b[2:] for b in bullets[:3])) if bullets else NO_FINDING_SENTENCE


def _report(user: str) -> str:
    """리포트 본문: 요약 통계 한 줄 + 이상 목록(또는 허용된 무발견 문장) + 권고. 검증기를 통과하는 형태만 만든다.

    비교 요청(FACTS 가 `### 사용자 Uxx` 블록으로 나뉜 경우)에는 사용자마다 `## Uxx` 절을 따로 만든다 - 사실이 한 사용자 것으로 섞이지 않는다.
    """
    lines = _facts_lines(user)
    parts = re.split(r"^### 사용자 (\S+)\n", "\n".join(lines), flags=re.M)
    if len(parts) >= 5:   # [앞부분, 사용자1, 블록1, 사용자2, 블록2, ...]
        sections = [f"## {uid}\n" + _report_body(block.splitlines(), headings=False) for uid, block in zip(parts[1::2], parts[2::2])]
        return "\n\n".join(sections + ["## 권고\n증상이 있거나 걱정되면 의료진과 상담하세요."])
    return _report_body(lines) + "\n## 권고\n증상이 있거나 걱정되면 의료진과 상담하세요."


def _report_body(lines: list[str], headings: bool = True) -> str:
    """한 사용자의 FACTS 줄로 요약 한 줄 + 이상 목록을 만든다 (headings=False 면 `## 요약` 등 소제목 없이 사용자 절 안에 이어 붙인다)."""
    stat = " ".join(l for l in lines if l.startswith("통계:"))
    parts = []
    for label, pat in (("평균 심박", rf"평균_심박\(bpm\)={_NUM}"), ("휴식기 심박", rf"휴식기_평균_심박\(bpm\)={_NUM}")):
        if m := re.search(pat, stat):
            parts.append(f"{label} {m.group(1)} bpm")
    if m := re.search(rf"데이터_커버리지\(%\)={_NUM}", stat):
        parts.append(f"데이터 커버리지 {m.group(1)}%")
    bullets = _anomaly_bullets(lines)
    if any("이상 탐지 실패" in l for l in lines):
        findings = ["이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다."]
    else:
        findings = bullets or [NO_FINDING_SENTENCE]
    summary = "- " + (", ".join(parts) or "요약 통계를 확인하지 못했습니다.")
    return "\n".join([*(["## 요약"] if headings else []), summary, *(["## 감지된 이상"] if headings else []), *findings])


def _simple(user: str) -> str:
    """간단 답변: 질문과 가장 관련된 가이드라인 부분의 앞부분 (없으면 안내 문장)."""
    rel = _between(user, "GUIDELINES(질문과 가장 관련된 부분):\n", "\n\nGUIDELINES(전체)")
    body = " ".join(l for l in rel.splitlines() if l and not l.startswith("#"))
    return body[:300] or "심박 점검은 사용자 ID와 함께 '이상 점검'을 요청해 주세요."
