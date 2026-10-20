"""테스트용 가짜 도구/LLM. 그래프·API 테스트가 MCP 서버·Ollama 없이 결정론적으로 돈다."""
import copy
from typing import Any

from app.agent.toolbox import ToolBox

SUMMARY = {
    "user_id": "U01",
    "window": {"start": "2016-04-05T00:00:00", "end": "2016-04-12T00:00:00"},
    "summary": {
        "coverage_pct": 88.0, "hr_mean": 72.4, "hr_p05": 55.0, "hr_p95": 98.0,
        "resting_hr": 61.2, "sleep_hr": 54.0, "active_minutes": 120, "rest_minutes": 580, "n_minutes": 9000,
    },
    "daily_resting_hr": [
        {"date": "2016-04-10", "resting_hr": 61.0, "n_rest_minutes": 300},
        {"date": "2016-04-11", "resting_hr": 62.0, "n_rest_minutes": 280},
    ],
}
WARNING = {
    "anomaly_id": "A1", "type": "resting_tachycardia", "start": "2016-04-10T14:00:00", "end": "2016-04-10T14:25:00",
    "duration_min": 25, "severity": "warning", "value": 112.3, "baseline": 62.0, "context": "rest",
    "evidence": "휴식 중 25분간 평균 112.3 bpm (개인 기준선 62.0 bpm)",
}
CRITICAL = {**WARNING, "anomaly_id": "A2", "severity": "critical", "value": 135.0,
            "evidence": "휴식 중 25분간 평균 135.0 bpm (개인 기준선 62.0 bpm)"}
GUIDELINES = {
    "guideline://heart-rate-ranges": "성인 휴식기 심박 정상 범위는 분당 60~100회입니다. 운동 중에는 올라가는 것이 정상입니다.",
    "guideline://red-flags": "흉통, 호흡곤란, 실신이 동반되면 즉시 응급 진료가 필요합니다.",
    "guideline://disclaimer": "※ 이 내용은 웨어러블 심박 데이터에 대한 참고용 분석이며 의학적 진단이 아닙니다. 증상이 있거나 걱정되면 의료진과 상담하세요.",
}


class FakeTools:
    """호출 기록과 실패 주입(summary_error/detect_error: 예외 리스트, 호출마다 하나씩 소비)을 지원하는 ToolBox 공급자."""

    def __init__(self, anomalies: list[dict] | None = None, coverage: float | list[float] = 88.0):
        self.anomalies = anomalies if anomalies is not None else [WARNING]
        self.coverage = coverage if isinstance(coverage, list) else [coverage]
        self.summary_errors: list[BaseException] = []
        self.detect_errors: list[BaseException] = []
        self.summary_calls: list[tuple] = []
        self.detect_calls: list[tuple] = []
        self.search_hits: list[dict] | None = None  # None 이면 search 도구가 없는 서버(키워드 폴백 경로)
        self.search_error: BaseException | None = None
        self.search_calls: list[tuple] = []
        self.prompt_text: str | None = None  # 설정하면 MCP Prompt(anomaly_review) 를 제공하는 서버
        self.prompt_error: BaseException | None = None
        self.prompt_calls: list[tuple] = []
        self.resource_uris: list[str] | None = None  # 설정하면 list_resources 탐색을 지원하는 서버
        self.resource_reads: list[str] = []
        self.extra_users: list[dict] = []  # 다른 데이터셋 사용자 (list_users 에 추가로 보인다)
        self.events_by_user: dict[str, list[dict]] = {}

    async def _summary(self, uid, days, end):
        self.summary_calls.append((uid, days, end))
        if self.summary_errors:
            raise self.summary_errors.pop(0)
        out = copy.deepcopy(SUMMARY)
        out["user_id"] = uid
        out["summary"]["coverage_pct"] = self.coverage[min(len(self.summary_calls) - 1, len(self.coverage) - 1)]
        return out

    async def _detect(self, uid, days, end):
        self.detect_calls.append((uid, days, end))
        if self.detect_errors:
            raise self.detect_errors.pop(0)
        counts: dict[str, int] = {}
        for a in self.anomalies:
            counts[a["type"]] = counts.get(a["type"], 0) + 1
        return {"user_id": uid, "window": SUMMARY["window"], "anomalies": copy.deepcopy(self.anomalies), "counts": counts}

    async def _users(self, dataset=None):
        users = [{"user_id": f"U0{i}", "dataset": "fitbit", "has_sleep": True, "start": "2016-03-12", "end": "2016-05-12", "days": 61, "coverage_pct": 80.0}
                 for i in range(1, 4)] + self.extra_users
        return [u for u in users if dataset is None or u["dataset"] == dataset]

    async def _events(self, uid):
        """get_health_events 모사: extra_users 중 events_by_user 에 있는 사용자만 라벨이 있고, FitBit 은 빈 목록, 모르는 사용자는 error."""
        known = {u["user_id"]: u["dataset"] for u in await self._users()}
        if uid not in known:
            return {"error": f"알 수 없는 user_id: {uid}"}
        return {"user_id": uid, "dataset": known[uid], "events": list(self.events_by_user.get(uid, [])), "note": "연구 라벨이며 진단이 아닙니다."}

    async def _series(self, uid, days, end, resample):
        return {"user_id": uid, "points": [{"ts": "2016-04-11T00:00:00", "hr": 60.0, "steps": 0, "asleep": True}]}

    async def _resource(self, uri):
        self.resource_reads.append(uri)
        return GUIDELINES[uri]

    async def _prompt(self, name, args):
        """MCP Prompt 모사."""
        self.prompt_calls.append((name, dict(args)))
        if self.prompt_error:
            raise self.prompt_error
        return self.prompt_text

    async def _list_resources(self):
        return list(self.resource_uris or [])

    async def _search(self, query, k):
        """RAG 검색 모사: 준비된 hits 를 돌려주거나 준비된 오류를 던진다."""
        self.search_calls.append((query, k))
        if self.search_error:
            raise self.search_error
        return {"query": query, "method": "vector", "note": "", "hits": list(self.search_hits or [])[:k]}

    def toolbox(self) -> ToolBox:
        """ToolBox 로 묶어 그래프/API 에 주입한다 (search_hits 가 설정된 때만 search 도구 포함)."""
        return ToolBox(self._users, self._summary, self._detect, self._series, self._resource,
                       search=self._search if self.search_hits is not None else None,
                       list_resources=self._list_resources if self.resource_uris is not None else None,
                       prompt=self._prompt if self.prompt_text is not None else None, events=self._events)


class FakeLLM:
    """structured: {name: pydantic 객체 | None}. text: {name: 문자열 | 문자열 리스트(호출마다 하나씩, 마지막 반복) | None}."""

    enabled = True

    def __init__(self, structured: dict[str, Any] | None = None, text: dict[str, Any] | None = None):
        self._structured, self._text = structured or {}, text or {}
        self.prompts: dict[str, list[str]] = {}

    async def structured(self, tier, name, schema, system, user):
        """이름별로 준비된 구조화 응답(없으면 None=LLM 실패)을 돌려주고 프롬프트를 기록한다."""
        self.prompts.setdefault(name, []).append(user)
        return self._structured.get(name)

    async def text(self, tier, name, system, user, on_token=None, max_tokens=None):
        """이름별로 준비된 텍스트(리스트면 호출 순서대로)를 토큰 조각으로 흘리며 프롬프트를 기록한다."""
        self.prompts.setdefault(name, []).append(user)
        v = self._text.get(name)
        if callable(v):  # 프롬프트를 보고 답을 만드는 LLM 모사 (예: 이전 리포트를 베끼는 모델)
            v = v(user)
        if isinstance(v, list):
            v = v[min(len(self.prompts[name]) - 1, len(v) - 1)]
        if v is not None and on_token:
            for i in range(0, len(v), 10):
                await on_token(v[i : i + 10])
        return v


def assemble(events: list[dict]) -> str:
    """클라이언트 규칙대로 token 이벤트 조립: reset=True 를 만나면 지금까지의 텍스트를 버린다."""
    text = ""
    for e in events:
        if e["type"] == "token":
            text = e["text"] if e["reset"] else text + e["text"]
    return text
