"""리포트 관련 순수 헬퍼: LLM 프롬프트용 사실 요약, 결정론적 템플릿 리포트, 고정 문구 조립.

책임: 상태(state) → 텍스트 변환만 담당한다. LLM 호출·이벤트 방출·검증은 하지 않는다.
협력 모듈: `nodes.py`(호출), `validator.py`(템플릿 수치가 facts 에서 왔음을 전제), `gates.py`(DISCLAIMER).
설계 근거: LLM 에는 집계만 전달, 템플릿 폴백/graceful degradation, 고정 문구는 코드가 조립.
"""
import json
import re
from datetime import datetime
from typing import Any

from app.agent.gates import (
    MAX_USERS, CONTEXT_LABELS, DIAGNOSIS_NOTICE, DISCLAIMER, ESCALATION_MESSAGE, NO_ANOMALY_GUIDANCE, SEVERITY_LABELS, TYPE_LABELS, WARNING_GUIDANCE,
)
from app.config import settings

MAX_ANOMALIES_IN_PROMPT = 15
# 가이드라인 리소스를 불러오지 못했을 때 쓰는 번들 요약 (실패를 캐시하지 않고, 이 값으로 그 요청만 답한다)
FALLBACK_GUIDELINES = {
    "ranges": "## 안정시 심박수 (요약)\n일반 성인의 안정시 심박수 정상 범위는 분당 60~100회입니다. 운동 중에는 심박이 오르는 것이 정상이며, 수면 중에는 평소보다 낮아집니다.",
    "red_flags": "## 위험 신호 (요약)\n흉통, 호흡곤란, 실신, 심한 어지럼이 동반되면 심박 수치와 무관하게 즉시 119 또는 응급실에 연락하세요.",
}
CHECK_FAILED_ADVICE = "증상이 있거나 걱정되면 의료진과 상담하고, 흉통·호흡곤란·실신 같은 응급 증상이면 즉시 119에 연락하세요."
# LLM 이 필드 의미를 오해하지 않도록 키를 단위·정의가 담긴 한국어 라벨로 바꿔 전달한다.
# 실사례: n_minutes(심박 기록 분수)를 '활동 시간'으로 서술한 리포트가 검증을 통과했다 → 라벨 명확화 + validator.metric_mismatches 이중 방어.
_STAT_LABELS = {
    "coverage_pct": "데이터_커버리지(%)", "hr_mean": "평균_심박(bpm)", "hr_p05": "심박_하위5%(bpm)", "hr_p95": "심박_상위95%(bpm)",
    "resting_hr": "휴식기_평균_심박(bpm)", "sleep_hr": "수면_중_평균_심박(bpm)",
    "active_minutes": "활동_분수(걸음이_있었던_분, 단위 분)", "rest_minutes": "휴식_분수(걸음이_없던_분, 단위 분)",
    "n_minutes": "심박_기록이_있는_분수(측정된_총_분, 활동·휴식·수면 시간이_아님, 단위 분)",
}
_SEV_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _hhmm(ts: Any) -> str:
    """ISO 시각 문자열을 분 단위("YYYY-MM-DD HH:MM")로 줄인다."""
    return str(ts)[:16].replace("T", " ")


def type_label(t: Any) -> str:
    """이상 type 코드를 화면과 같은 한국어 유형명으로 (사전에 없으면 원문)."""
    return TYPE_LABELS.get(t, str(t))


def context_label(c: Any) -> str:
    """발생 맥락 코드(rest/sleep/active)를 한국어로 (사전에 없으면 원문)."""
    return CONTEXT_LABELS.get(c, str(c))


# 센서 결측이 조회 구간의 이 비율 이상일 때만 '데이터 한계'로 본다. 착용 중 센서가 몇 시간 놓치는 일은 흔해서(커버리지 85~90% 사용자도
# 1~2건) 건수만으로 한계 안내를 붙이면 critical 에스컬레이션 문구 바로 아래에 항상 붙는다. 전체 커버리지·휴식 분 부족은
# 별도 기준(< 50%, < 60분)으로 이미 잡으므로, 결측 기준은 '구간의 5분의 1'로 두었다: 계획안의 10%는 7일 구간에서 509분짜리 결측
# 2건(10.1%)처럼 흔한 경우에도 걸려 오탐이 남는다.
SENSOR_GAP_LIMIT_FRACTION = 0.20


def _window_minutes(state: dict) -> float:
    """조회 구간 길이(분): 탐지 결과의 window 가 있으면 그것, 없으면 lookback_days (둘 다 없으면 0)."""
    w = (state.get("detection") or {}).get("window") or {}
    try:
        return (datetime.fromisoformat(w["end"]) - datetime.fromisoformat(w["start"])).total_seconds() / 60
    except (KeyError, TypeError, ValueError):
        return float(state.get("lookback_days") or 0) * 1440


def user_views(state: dict) -> list[tuple[str, dict]]:
    """상태를 *사용자별 단일 상태*로 나눈다. 단일 요청이면 [(사용자, state)], 비교 요청(`state["users"]`)이면 사용자마다 그 사용자의 결과만 담은 상태.

    단일 사용자 전용 함수(요약·템플릿·한계 판단)를 비교 요청에서도 그대로 재사용하기 위한 어댑터이다 — 사용자 간 사실이 섞이지 않는다.
    """
    users = state.get("users")
    if not users:
        return [(state.get("analyzed_user") or state.get("user_id"), state)]
    return [
        (uid, {**state, "user_id": uid, "analyzed_user": uid, "summary": u.get("summary"), "detection": u.get("detection"),
               "anomalies": u.get("anomalies") or [], "research_errors": u.get("errors") or [], "users": None})
        for uid, u in users.items()
    ]


def data_limits(state: dict) -> list[str]:
    """이번 조회의 데이터 한계 목록 (비교 요청이면 사용자 ID 를 앞에 붙여 합친다)."""
    views = user_views(state)
    if len(views) == 1:
        return _data_limits_one(state)
    return [f"{uid} {l}" for uid, v in views for l in _data_limits_one(v)]


def _data_limits_one(state: dict) -> list[str]:
    """한 사용자의 데이터 한계 목록 (비어 있으면 한계 없음). 거짓 안심을 막고 한계를 밝히는 데 쓴다.

    한계: 커버리지 < 50%, 휴식 분 < 60, 센서 결측 시간 합이 조회 구간의 SENSOR_GAP_LIMIT_FRACTION(20%) 이상.
    센서 결측은 건수가 아니라 비율로 판단한다(위 상수 주석). 탐지 자체의 실패는 별도(`state["detection"] is None`)로 다룬다.

    Args:
        state: 에이전트 상태 (summary, anomalies, detection, lookback_days 를 읽는다).
    """
    s = ((state.get("summary") or {}).get("summary")) or {}
    out: list[str] = []
    if (cov := s.get("coverage_pct")) is not None and cov < 50:
        out.append(f"데이터 커버리지 {round(cov, 1):g}%")
    if (rest := s.get("rest_minutes")) is not None and rest < 60:
        out.append(f"휴식 구간 {rest}분")
    gaps = [a for a in state.get("anomalies") or [] if a.get("type") == "sensor_gap"]
    window = _window_minutes(state)
    if gaps and window > 0 and sum(a.get("duration_min") or 0 for a in gaps) >= SENSOR_GAP_LIMIT_FRACTION * window:
        out.append(f"센서 결측 {len(gaps)}건")
    return out


def limitation_notice(state: dict) -> str:
    """데이터 한계·탐지 실패 시 코드가 리포트에 강제하는 고정 안내 한 문장 (없으면 빈 문자열).

    에스컬레이션 안내와 같은 방식이다: LLM 이 쓰거나 빠뜨리는 것에 의존하지 않고 `compose` 가 넣고, 수치는 도구 결과(`data_limits`)에서 온다.
    """
    if any(v.get("detection") is None for _, v in user_views(state)):
        return "※ 이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다."
    limits = data_limits(state)
    return f"※ 이번 기간은 기록이 제한적이어서({', '.join(limits)}) 이상이 없다고 판단할 수 없습니다." if limits else ""


def limitation_sentence(state: dict) -> str:
    """데이터 한계가 있을 때 결과 서술에 덧붙이는 고정 한계 문장 (없으면 빈 문자열)."""
    limits = data_limits(state)
    return f"다만 조회 구간의 데이터가 제한적이어서({', '.join(limits)}) 감지되지 않은 구간까지 안전하다고 볼 수 없습니다. 증상이 있거나 걱정되면 의료진과 상담하세요." if limits else ""


def top_anomalies(anomalies: list[dict], n: int = MAX_ANOMALIES_IN_PROMPT) -> list[dict]:
    """심각도(critical → warning → info), 시작 시각 순으로 상위 n건. 프롬프트·SSE 크기를 제한한다."""
    return sorted(anomalies, key=lambda a: (_SEV_ORDER.get(a.get("severity"), 3), str(a.get("start"))))[:n]


def compact_brief(state: dict) -> str:
    """`facts_brief` 와 같은 정보를 JSON 대신 압축 텍스트로 만든다 (프롬프트 토큰 절감, 측정상 지연 -13s).

    Args:
        state: AgentState (user_id, summary, anomalies, research_errors 를 읽는다).

    Returns:
        여러 줄 텍스트. 원시 시계열은 포함하지 않는다 (데이터 최소화).
    """
    summary = state.get("summary") or {}
    anomalies = state.get("anomalies") or []
    stats = ", ".join(f"{_STAT_LABELS.get(k, k)}={v}" for k, v in (summary.get("summary") or {}).items())
    win = summary.get("window") or {}
    daily = ", ".join(f"{str(d['date'])[5:]}={d.get('resting_hr')}" for d in summary.get("daily_resting_hr") or [])
    detection_ok = state.get("detection") is not None
    lines = [f"사용자 {state.get('analyzed_user') or state.get('user_id')} 조회기간 {_hhmm(win.get('start'))}~{_hhmm(win.get('end'))}", f"통계: {stats or '없음'}",
             f"일별 휴식기 심박(bpm): {daily or '없음'}",
             f"이상 총 {len(anomalies)}건 (상위):" if detection_ok else "이상 탐지 실패: 이상 여부를 판단할 수 없음 (0건이 아니다 — '이상 없음'이라고 쓰지 말 것)"]
    lines += [f"- [{SEVERITY_LABELS.get(a.get('severity'), a.get('severity'))}] {type_label(a.get('type'))} {_hhmm(a.get('start'))}~{_hhmm(a.get('end'))} "
              f"({a.get('duration_min')}분, {context_label(a.get('context'))}): {a.get('evidence')}" for a in top_anomalies(anomalies)]
    if limits := data_limits(state):
        lines.append("데이터 한계: " + ", ".join(limits) + " — 이상이 감지되지 않아도 '정상/걱정 없음/추가 검사 불필요'라고 쓰지 말 것")
    if state.get("research_errors"):
        lines.append("조회 실패: " + "; ".join(state["research_errors"]))
    return "\n".join(lines)


def facts_brief(state: dict) -> str:
    """LLM 프롬프트용 사실 요약. 비교 요청이면 사용자별 블록을 `### 사용자 Uxx` 머리글로 구분해 이어 붙인다 (사실이 섞이지 않게)."""
    views = user_views(state)
    if len(views) == 1:
        return _facts_brief_one(state)
    return "\n\n".join(f"### 사용자 {uid}\n{_facts_brief_one(v)}" for uid, v in views)


def _facts_brief_one(state: dict) -> str:
    """LLM 프롬프트용 사실 요약 (설정 COMPACT_FACTS 이면 `compact_brief` 로 위임).

    Args:
        state: AgentState.

    Returns:
        JSON 문자열. 요약 통계·일별 휴식 심박·상위 이상 N건만 담는다 (원시 분 단위 시계열은 절대 포함하지 않음).
    """
    if settings.compact_facts:
        return compact_brief(state)
    summary = state.get("summary") or {}
    anomalies = state.get("anomalies") or []
    brief = {
        "사용자": state.get("analyzed_user") or state.get("user_id"),
        "조회기간": summary.get("window") or {"lookback_days": state.get("lookback_days")},
        "요약통계": {_STAT_LABELS.get(k, k): v for k, v in (summary.get("summary") or {}).items()},
        "일별_휴식기_심박": summary.get("daily_resting_hr"),
        "이상_총건수": len(anomalies) if state.get("detection") is not None else None,
        "탐지_상태": "성공" if state.get("detection") is not None else "실패: 이상 여부 판단 불가 ('이상 없음'이라고 쓰지 말 것)",
        "이상_목록(상위)": [
            {"유형명": type_label(a.get("type")), "심각도명": SEVERITY_LABELS.get(a.get("severity"), a.get("severity")),
             **{k: (_hhmm(v) if k in ("start", "end") else (context_label(v) if k == "context" else v)) for k, v in a.items() if k != "anomaly_id"}}
            for a in top_anomalies(anomalies)
        ],
        "데이터_한계": data_limits(state),
        "조회_실패": state.get("research_errors") or [],
    }
    return json.dumps(brief, ensure_ascii=False, default=str)


def _fmt(v: Any, unit: str = "") -> str:
    """수치를 소수 1자리로 표기한다 (None 이면 "측정 없음"). 반올림 오차는 validator 허용 범위(±0.5) 안이다."""
    return "측정 없음" if v is None else f"{round(v, 1):g}{unit}"


def rule_analysis(anomalies: list[dict], detection_ok: bool = True, limits: list[str] | None = None) -> str:
    """analysis LLM 실패/비활성/생략(MERGE_ANALYSIS) 시 쓰는 결정론적 해석 문장.

    Args:
        anomalies: 탐지된 이상 목록 (type, evidence 사용).
        detection_ok: 이상 탐지가 성공했는가. False 면 "이상 없음"이 아니라 판단 불가를 서술한다 (거짓 안심 방지).
        limits: `data_limits` 결과. 있으면 한계 문장을 덧붙인다.

    Returns:
        유형별 건수와 상위 3건의 evidence 를 이은 문장. 이상이 없으면 그 사실을 알리는 문장.
    """
    if not detection_ok:
        return "이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다."
    if not anomalies:
        base = "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다."
        return base + (f" 다만 데이터가 제한적입니다({', '.join(limits)})." if limits else "")
    counts: dict[str, int] = {}
    for a in anomalies:
        counts[a["type"]] = counts.get(a["type"], 0) + 1
    kinds = ", ".join(f"{t} {n}건" for t, n in counts.items())
    return f"감지된 이상은 총 {len(anomalies)}건입니다 ({kinds}). 주요 근거: " + " / ".join(a["evidence"] for a in top_anomalies(anomalies, 3) if a.get("evidence"))


def template_body(state: dict) -> str:
    """템플릿 본문. 비교 요청이면 사용자별 템플릿 + 유형별 건수 비교 절(공통/차이)을 만든다 (수치는 모두 도구 결과의 건수)."""
    views = user_views(state)
    if len(views) == 1:
        return _template_body_one(state)
    parts = [_template_body_one(v) for _, v in views]
    types = sorted({a["type"] for _, v in views for a in v.get("anomalies") or []})
    lines = ["## 사용자 비교"]
    if types:
        for t in types:
            per = {uid: sum(a["type"] == t for a in v.get("anomalies") or []) for uid, v in views}
            lines.append(f"- {type_label(t)}: " + ", ".join(f"{uid} {n}건" for uid, n in per.items()))
        common = [type_label(t) for t in types if all(any(a["type"] == t for a in v.get("anomalies") or []) for _, v in views)]
        lines.append("- 모든 사용자에게 나타난 유형: " + (", ".join(common) if common else "없음"))
    else:
        lines.append("- 비교한 모든 사용자에서 이번 기간 데이터에 임계값을 넘는 구간이 감지되지 않았습니다.")
    failed = [uid for uid, v in views if v.get("detection") is None]
    if failed:
        lines.append(f"- {', '.join(failed)}: 탐지 결과를 확인하지 못해 비교에서 판단할 수 없습니다.")
    return "\n\n".join(parts + ["\n".join(lines)])


def _template_body_one(state: dict) -> str:
    """결정론적 템플릿 리포트 본문 — LLM 실패·비활성 폴백이자 검증 재시도 소진 시의 graceful degradation 결과물.

    모든 수치와 문장은 도구 결과(summary/anomalies/evidence)에서만 오므로 validator 를 항상 통과한다.

    Args:
        state: AgentState (user_id, summary, detection, anomalies, research_errors 를 읽는다).

    Returns:
        마크다운 본문 (면책/에스컬레이션 문구는 `compose` 가 붙인다). 조회가 전부 실패했다면 그 사실만 알린다.
    """
    uid = state.get("user_id")
    summary, anomalies = state.get("summary"), state.get("anomalies") or []
    lines = [f"## 심박 점검 결과 ({uid})"]
    if summary is None and state.get("detection") is None:
        lines.append("데이터를 조회하지 못해 점검 결과를 제공할 수 없습니다. 잠시 후 다시 시도해 주세요. " + CHECK_FAILED_ADVICE)
        lines += [f"- {e}" for e in state.get("research_errors") or []]
        return "\n".join(lines)
    if summary:
        s = summary.get("summary") or {}
        lines += [
            "### 요약",
            f"- 평균 심박 {_fmt(s.get('hr_mean'), ' bpm')}, 휴식기 심박 {_fmt(s.get('resting_hr'), ' bpm')}, 수면 중 심박 {_fmt(s.get('sleep_hr'), ' bpm')}",
            f"- 데이터 커버리지 {_fmt(s.get('coverage_pct'), '%')}",
        ]
    lines.append("### 감지된 이상")
    if state.get("detection") is None:  # 탐지 실패는 '이상 없음'이 아니다: 확인하지 못했음을 밝히고 증상 시 진료를 권한다
        lines.append(f"- 탐지 결과를 확인하지 못했습니다. {CHECK_FAILED_ADVICE}")
    elif anomalies:
        # 유형명을 수치와 같은 줄에 둔다: 화면의 각주(citation)가 줄 안의 유형 단어와 수치로 이상을 찾는다.
        lines += [
            f"- [{SEVERITY_LABELS.get(a['severity'], a['severity'])}] {type_label(a['type'])} ({context_label(a.get('context'))}): {a['evidence']}"
            for a in top_anomalies(anomalies)
        ]
        if len(anomalies) > MAX_ANOMALIES_IN_PROMPT:
            lines.append(f"- 그 외 {len(anomalies) - MAX_ANOMALIES_IN_PROMPT}건 생략")
    else:
        lines.append("- 이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다.")
    if state.get("research_errors"):
        lines.append("### 참고")
        lines += [f"- 일부 조회 실패: {e}" for e in state["research_errors"]]  # research_errors 는 내부 예외 문구가 없는 사용자용 문장이다
    return "\n".join(lines)


_LIMIT_NOTICE_RE = re.compile(r"※ 이번 기간은 [^\n]*?판단할 수 없습니다\.|※ 이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다\.")
_CONTEXT_NOTE_RE = re.compile(r"※ (?:직전 점검 대상: \S+ 기준 답변|한 번에 최대 \d명)[^\n]*")


def excluded_note(state: dict) -> str:
    """비교 요청이 인원 한도를 넘었을 때 제외된 사용자를 밝히는 고정 문구 (없으면 빈 문자열). 조용히 무시하지 않는다는 원칙."""
    ex = state.get("excluded_users") or []
    return f"※ 한 번에 최대 {MAX_USERS}명까지 비교합니다. 제외된 사용자: {', '.join(ex)} — 따로 요청해 주세요." if ex else ""


def context_note(analyzed_user: str, current_user: str) -> str:
    """요청 본문의 사용자와 직전 분석 대상이 다를 때 붙이는 한 줄 표지 (다른 사용자의 사실을 현재 사용자로 오인 방지)."""
    return f"※ 직전 점검 대상: {analyzed_user} 기준 답변입니다. {current_user} 의 점검이 필요하면 '{current_user} 점검해줘'라고 요청하세요."


def strip_fixed(text: str, *extra: str) -> str:
    """코드가 조립하는 고정 문구(에스컬레이션·진단 거절·면책)와 "중요 안내" 표지를 텍스트에서 제거한다.

    두 곳에서 쓴다: (1) 이전 리포트·대화 기록을 프롬프트에 넣기 전 — LLM 이 고정 문구를 베껴 쓰는 것을 막는다,
    (2) LLM 본문을 조립하기 전 — 베껴 쓴 문구가 있어도 `compose` 결과에 각 문구가 정확히 한 번만 남게 한다(멱등).

    Args:
        text: 정리할 텍스트.
        *extra: 추가로 제거할 고정 문구 (예: 리소스에서 읽은 면책 문단).
    """
    for fixed in (ESCALATION_MESSAGE, DIAGNOSIS_NOTICE, DISCLAIMER, WARNING_GUIDANCE, NO_ANOMALY_GUIDANCE, *extra):
        if fixed:
            text = text.replace(fixed, "")
    text = _NOTIFY_RE.sub("", _SOURCES_RE.sub("", _CONTEXT_NOTE_RE.sub("", _LIMIT_NOTICE_RE.sub("", text))))
    return re.sub(r"\n{3,}", "\n\n", text.replace("**중요 안내**", "")).strip()


_SOURCES_RE = re.compile(r"※ 출처: [^\n]*")
_NOTIFY_RE = re.compile(r"※ 보호자 알림[^\n]*")


def guardian_preview(state: dict, reason: str) -> str:
    """보호자 알림(시뮬레이션) 미리보기 문구 — 가명 사용자 ID 와 위험 요약만 담는다(데이터 최소화). 실제 발송은 일어나지 않는다."""
    who = ", ".join(sorted({str(a["user_id"]) for a in state.get("anomalies") or [] if a.get("user_id") and a.get("severity") == "critical"})) or str(
        state.get("analyzed_user") or state.get("user_id"))
    return f"[Vital Sentinel] {who} 님의 심박에서 위험 신호({reason})가 감지되었습니다. 증상을 확인하고 의료진 상담 또는 119 연락을 권해 주세요. (시뮬레이션: 실제 발송 아님)"


def notification_note(approved: bool) -> str:
    """승인 결과를 리포트 끝에 덧붙이는 한 줄 (코드가 쓰는 고정 문구)."""
    return "※ 보호자 알림 발송됨(시뮬레이션)" if approved else "※ 보호자 알림을 발송하지 않았습니다(승인되지 않음)"


def sources_line(hits: list[dict]) -> str:
    """검색 근거의 출처 한 줄 (코드가 조립 — LLM 이 출처를 지어내거나 빠뜨리지 않게). 근거가 없으면 빈 문자열."""
    seen: list[str] = []
    for h in hits:
        # 내부 문서 ID(heart-rate-ranges 등)는 사용자에게 의미가 없으므로 문서 제목만 쓴다. 제목이 없을 때만 ID 로 대신한다.
        label = (h.get("title") or h.get("doc_id") or "") + (f" > {h['heading']}" if h.get("heading") else "")
        if label and label not in seen:
            seen.append(label)
    return "※ 출처: " + ", ".join(seen) if seen else ""


def head(escalation_message: str = "", notice: str = "", note: str = "", limit_notice: str = "") -> str:
    """리포트 머리말: 에스컬레이션 안내(강조) → 진단 거절 안내 → 데이터 한계 안내 → 직전 점검 대상 표지. 모두 없으면 빈 문자열."""
    parts = ([f"**중요 안내** {escalation_message}"] if escalation_message else []) + [p for p in (notice, limit_notice, note) if p]
    return "\n\n".join(parts) + "\n\n" if parts else ""


def guidance_for(state: dict) -> str:
    """코드가 쓰는 대응 안내: 주의(warning) 이상이 있고 에스컬레이션이 아니면 `WARNING_GUIDANCE`, 이상이 하나도 없으면 `NO_ANOMALY_GUIDANCE`, 그 밖에는 빈 문자열.

    critical 이면 에스컬레이션 안내문이, 센서 결측 같은 참고(info)뿐이면 데이터 한계 안내가 그 역할을 하므로 중복해서 붙이지 않는다.
    """
    from app.agent import gates  # 순환 import 회피

    anomalies = state.get("anomalies") or []
    if not anomalies:
        return NO_ANOMALY_GUIDANCE
    return WARNING_GUIDANCE if not gates.safety_gate(anomalies).escalation and any(a.get("severity") == "warning" for a in anomalies) else ""


def compose(
    body: str, escalation_message: str = "", disclaimer: str = DISCLAIMER, notice: str = "", note: str = "", limit_notice: str = "",
    sources: str = "", guidance: str = "",
) -> str:
    """고정 문구를 코드가 조립한다: [에스컬레이션 안내] + [진단 거절 안내] + 본문 + [경고 수준 대응 안내] + 면책.

    LLM 이 문구를 빠뜨리거나 바꿔 쓸 수 없도록 본문과 분리해 코드가 붙인다.

    Args:
        body: LLM 또는 템플릿이 만든 본문.
        escalation_message: 비어 있지 않으면 맨 앞에 강조해 붙인다.
        disclaimer: 맨 끝에 붙일 면책 문구.
        notice: 진단·확진 요청에 대한 거절 안내 (요청이 없으면 빈 문자열).
        note: 직전 점검 대상 표지 (`context_note`, 해당 없으면 빈 문자열).
        limit_notice: 데이터 한계·탐지 실패 안내 (`limitation_notice`, 해당 없으면 빈 문자열).
        sources: 검색 근거 출처 한 줄 (`sources_line`, 본문과 면책 사이에 둔다).
        guidance: 경고 수준 대응 안내 (`guidance_for`, 본문 바로 뒤).

    Returns:
        완성된 리포트 마크다운.
    """
    body = strip_fixed(body, escalation_message, notice, disclaimer, limit_notice)  # 멱등: 본문에 고정 문구가 섞여 있어도 한 번만 남긴다
    mid = (f"\n\n{guidance}" if guidance else "") + (f"\n\n{sources}" if sources else "")
    return f"{head(escalation_message, notice, note, limit_notice)}{body}{mid}\n\n{disclaimer}"


def strip_fences(text: str) -> str:
    """LLM 이 리포트 전체를 ```markdown 코드블록으로 감싼 경우 바깥 펜스를 벗겨낸다 (그 외 텍스트는 그대로)."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def first_paragraph(markdown: str) -> str:
    """리소스 문서(제목 + 여러 문단)에서 첫 본문 문단만 한 줄로 — 리포트 말미 면책 문구로 쓴다."""
    body = "\n".join(l for l in markdown.splitlines() if not l.lstrip().startswith("#"))
    para = next((p for p in body.split("\n\n") if p.strip()), "")
    return " ".join(para.split())


def guideline_excerpt(docs: list[str], question: str, limit: int = 700) -> str:
    """LLM 없이 후속/일반 질문에 답하는 폴백: 질문과 가장 관련 있는 가이드라인 섹션(## 단위)을 발췌.

    한국어 조사·어미 때문에 "운동할/운동하면" 이 "운동" 과 안 맞는 문제를 피하려고 어절의 앞 두 글자도 단서로 쓴다.
    """
    sections = [sec.strip() for d in docs for sec in ("\n" + d).split("\n## ")[1:] if sec.strip()]
    if not sections:
        return ""
    tokens = {w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)}
    cues = tokens | {t[:2] for t in tokens if len(t) >= 3 and re.match(r"[가-힣]", t)}
    best = max(sections, key=lambda sec: sum(w in sec for w in cues))
    return "## " + best[:limit].rstrip()


SUMMARY_MAX_CHARS = 700

# 대화 요약은 모든 이후 턴의 프롬프트에 들어가는 *영속 상태*라서 프롬프트 주입 통로가 된다. 지시문·역할 표지·안전 문구 조작·링크·코드 블록처럼
# "사실"이 아닌 것이 보이면 그 요약은 통째로 버린다(부분 삭제는 문장을 조각내 뜻을 바꾸므로 하지 않는다).
_SUMMARY_UNSAFE_RE = re.compile(
    r"이전\s*(?:의\s*)?(?:지시|명령|지침|규칙|대화)|(?:지시|명령|지침|규칙)(?:사항|을|를)?\s*(?:무시|따르|잊|어기)|무시\s*(?:하|해|할)|시스템\s*(?:프롬프트|메시지|지시)"
    r"|ignore|disregard|forget\s+(?:all|previous|the)|override|jailbreak|system\s*(?:prompt|message)|you\s+are|assistant\s*:|system\s*:"
    r"|(?:항상|반드시|절대|무조건)[^.\n]{0,40}(?:답하|말하|출력|해야|하세요|하라|해라|하십시오|쓰세요)"
    r"|(?:119|응급|안내문|면책|경고)[^.\n]{0,20}(?:하지\s*마|생략|삭제|빼|금지|말고)|당신은|너는\s|역할(?:을|은)?\s*(?:바꿔|변경|무시)"
    r"|\[이전 대화 요약|```|<\s*/?\s*[a-zA-Z가-힣][^>]{0,30}>|https?://|\{\{|\}\}"
    # 요약에는 사용자가 말한 *사실*만 들어간다: 도우미가 어떻게 답할지 정하는 문장(지시·형식·말투·시작/끝 문구)은 사실이 아니다.
    r"|(?:하라|해라|하세요|하십시오|할\s*것|하도록|해야\s*(?:한다|합니다|함)|시오)(?:\W|$)"
    r"|(?:답|답변|응답|대답|출력|말투|어조|형식|양식|문체)(?:을|를|은|는|이|가|도)?[^.\n]{0,30}(?:시작|끝|항상|반드시|(?:으)?로\s*하|만\s*(?:하|쓰|말)|처럼|붙)"
    r"|모든\s*(?:답|응답|답변|대답)|정상입니다|문제\s*없습니다|(?:앞으로|이후|이제부터|다음부터)[^.\n]{0,20}(?:답|응답|말|써|넣|붙)"
    r"|(?:쓰지|넣지|붙이지|하지)\s*말(?:\s*것|아|라|고)|말아\s*(?:달|주)|해\s*달라|(?:요청|원)함"
    r"|^\s*(?:사용자|도우미|human|ai)\s*[:：]",   # 줄 머리의 역할 표지만 (규칙 요약의 "이전에 다룬 사용자: U01" 은 허용)
    re.IGNORECASE | re.MULTILINE,
)


def safe_summary(text: str) -> str | None:
    """대화 요약 후보가 *사실만* 담고 있으면 정리한 텍스트를, 지시문·역할 표지·진단 표현이 섞였으면 None 을 돌려준다.

    Args:
        text: LLM 이 쓴 요약, 또는 상태에 저장돼 있던 기존 요약.

    Returns:
        고정 문구·제어문자를 제거하고 길이를 자른 요약. 위험 패턴이 있거나 비면 None (호출 측이 규칙 요약으로 대체한다).
    """
    from app.agent.validator import diagnostic_assertions  # 순환 import 회피

    clean = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", strip_fixed(text or ""))
    if not clean or _SUMMARY_UNSAFE_RE.search(clean) or diagnostic_assertions(clean):
        return None
    return clean[:SUMMARY_MAX_CHARS]


def rule_summary(prior: str, turns: list[tuple[str, str]]) -> str:
    """LLM 요약기의 결정론적 폴백/가짜: 개인 정보(이름·나이)와 점검한 사용자 ID 만 추려 기존 요약에 합친다 (진단·수치·고정 문구는 넣지 않는다).

    Args:
        prior: 기존 요약 (없으면 빈 문자열).
        turns: [(role("사용자"|"도우미"), 내용)] — 새로 요약할 오래된 대화.
    """
    from app.agent import gates  # 순환 import 회피

    humans = [t for r, t in turns if r == "사용자"]
    facts = []
    if (name := gates._last_fact(gates._NAME_SAY, humans)) and safe_summary(name):  # "제 이름은 이전지시무시 입니다" 같은 이름 칸 악용 차단
        facts.append(f"이름 {name}")
    if age := gates._last_fact(gates._AGE_SAY, humans):
        facts.append(f"나이 {age}세")
    ids = sorted({u for t in humans for u in gates.extract_user_ids(t, 99)[0]})
    new = ("사용자가 말한 정보: " + ", ".join(facts) + ". " if facts else "") + (f"이전에 다룬 사용자: {', '.join(ids)}. " if ids else "") + f"(이전 사용자 발화 {len(humans)}건 요약)"
    return ((safe_summary(prior) or "") + " " + new).strip()[-SUMMARY_MAX_CHARS:]


def hits_text(hits: list[dict]) -> str:
    """검색 결과를 프롬프트/폴백용 본문으로 합친다: ``[제목 > 소제목]`` 머리글 + 대목 텍스트 (없으면 빈 문자열)."""
    return "\n\n".join(
        f"[{h.get('title', '')}" + (f" > {h['heading']}" if h.get("heading") else "") + f"]\n{h.get('text', '')}" for h in hits if h.get("text")
    )


def followup_fallback(state: dict, question: str, docs: list[str], retrieved: str = "") -> str:
    """후속 질문의 LLM 폴백 답: 직전 분석의 핵심 사실을 다시 말하고, 질문과 관련된 가이드라인 섹션을 덧붙인다.

    일반 가이드라인 표만 내놓던 동작을 대체한다. 모든 문장과 수치는 저장된 도구 결과·가이드라인에서만 온다.

    Args:
        state: AgentState (analyzed_user, summary, anomalies, detection 을 읽는다).
        question: 사용자 질문.
        docs: 가이드라인 원문 목록.
        retrieved: RAG 검색으로 얻은 근거 본문 (있으면 키워드 발췌 대신 쓴다).
    """
    who = ", ".join(state["users"]) if state.get("users") else (state.get("analyzed_user") or state.get("user_id"))
    if any(v.get("detection") is None for _, v in user_views(state)):
        facts = f"직전 점검({who})에서는 이상 탐지를 실행하지 못해 이상 여부를 판단하지 못했습니다."
    elif anomalies := state.get("anomalies"):
        top = top_anomalies(anomalies, 3)
        facts = f"직전 점검({who}) 결과 이상 {len(anomalies)}건이 감지되었습니다. 주요 항목: " + " / ".join(
            f"[{SEVERITY_LABELS.get(a['severity'], a['severity'])}] {type_label(a['type'])} ({context_label(a.get('context'))}) {a['evidence']}" for a in top
        )
    else:
        facts = f"직전 점검({who}) 결과 이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다. {limitation_sentence(state)}".strip()
    excerpt = retrieved or guideline_excerpt(docs, question)  # 검색 대목은 서버가 이미 문장 경계로 줄였다(text) - 여기서 다시 자르지 않는다
    return facts + (f"\n\n질문과 관련된 가이드라인:\n{excerpt}" if excerpt else "")
