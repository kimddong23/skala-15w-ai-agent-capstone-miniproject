"""vitals MCP 서버 (FastMCP, stdio): 사용자 심박 데이터의 조회·탐지·요약 도구와 점검 절차 Prompt.

책임 경계:
    사용자별로 값이 달라지는 조회/계산(Tool)과 점검 절차(Prompt)만 제공한다. 정적 가이드라인과 의미 검색은
    `guidelines_server.py` 의 책임이다 - 서버를 둘로 나눠 데이터 서버는 가이드라인/임베딩에, 가이드라인 서버는 환자 데이터에 의존하지 않게 한다.
    계산은 `app.detection`, 데이터 접근은 `app.data.store` 에 위임하며 이 모듈에는 입력 검증과 직렬화만 둔다.

설계 근거:
    - Tool/Resource 분리: 호출 시점·인자마다 값이 달라지는 조회/계산은 Tool. 다중 서버 분리는 클라이언트가 이름으로 탐색한다.
    - 이상 탐지는 결정론적 코드이며 LLM 은 결과만 해석한다.
    - 응답에는 원시 1분 데이터를 대량으로 싣지 않는다 (시계열은 5분 이상 간격, 점 수 상한 - 데이터 최소화).
    - 독립 subprocess + stdio 로 기동한다: `.venv/bin/python -m app.mcp_server.vitals_server`.

협력 모듈: `app.data.store.VitalsStore`(프로세스당 싱글턴; FitBit + 있을 때만 외부 연구 데이터셋), `app.detection.detector`, `app.agent.toolbox`(클라이언트).

계약:
    모든 Tool 은 잘못된 입력에 예외 대신 {"error": 사유} 를 반환한다 (에이전트가 오류를 데이터로 받아 격리 처리).
    모든 시각 인자/응답은 ISO 8601 이며 기기 로컬 시간의 timezone-naive 값이다 (tz 가 붙어 오면 제거).
"""
from datetime import datetime

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from app.data.store import VitalsStore
from app.detection.detector import daily_resting_hr, detect, summarize

MAX_LOOKBACK_DAYS = 90        # 탐지·요약 조회 상한
MAX_SERIES_DAYS = 30          # 시계열 조회 상한 (프런트 30일 보기)
MAX_SERIES_POINTS = 2000      # 응답 점 수 상한 - 넘으면 resample 간격을 5분 단위로 자동 확대
MIN_RESAMPLE = pd.Timedelta(minutes=5)   # 원시 1분 데이터 대량 노출 금지

mcp = FastMCP(
    "vitals",
    instructions=(
        "웨어러블 심박 데이터에서 이상 징후를 탐지·요약하는 서버입니다. 숫자는 반드시 도구 결과에서만 인용하세요. "
        "user_id 는 list_users 로 먼저 확인합니다. 가이드라인 문서는 별도의 guidelines 서버가 제공합니다."
    ),
)
store = VitalsStore()          # 프로세스당 한 번만 로드 (싱글턴)


def _resolve(user_id: str, lookback_days: float, end: str | None, max_days: float = MAX_LOOKBACK_DAYS):
    """도구 공통 입력 검증과 조회 구간 계산.

    Args:
        user_id: 가명 사용자 ID.
        lookback_days: 조회 기간(일). 0 초과 max_days 이하.
        end: ISO 8601 끝 시각 또는 None (None 이면 사용자의 마지막 심박 시각).
        max_days: lookback_days 상한(일).

    Returns:
        (end_ts, start_ts) — 조회 구간은 (start_ts, end_ts].

    Raises:
        ValueError: 기간 범위 위반 또는 end 형식 오류.
        KeyError: 알 수 없는 user_id.
    """
    if not 0 < lookback_days <= max_days:
        raise ValueError(f"lookback_days 는 0 초과 {max_days} 이하여야 합니다 (받은 값: {lookback_days})")
    if end is None:
        end_ts = pd.Timestamp(store.data_range(user_id)[1])      # 알 수 없는 user_id 면 KeyError
    else:
        try:
            end_ts = pd.Timestamp(datetime.fromisoformat(end)).tz_localize(None)
        except ValueError:
            raise ValueError(f"end 는 ISO 8601 형식(예: 2016-04-20T12:00:00)이어야 합니다 (받은 값: {end!r})") from None
        store.data_range(user_id)
    return end_ts, end_ts - pd.Timedelta(days=lookback_days)


def _error(e: Exception) -> dict:
    """검증 예외를 {"error": 사유} 로 변환한다 (KeyError 는 repr 따옴표 없이 메시지만)."""
    return {"error": str(e.args[0]) if isinstance(e, KeyError) else str(e)}


@mcp.tool(annotations={"readOnlyHint": True})
def list_users(dataset: str | None = None) -> list[dict]:
    """분석 가능한 사용자 목록을 반환한다.

    각 항목: user_id(가명: FitBit 서비스 데이터는 'U01'.., 외부 연구 데이터셋은 'S001'..), dataset('fitbit' | 'stanford'),
    has_sleep(수면 구간이 하나라도 기록됐는가 - false 면 수면 중 값은 신뢰할 수 없다), start/end(심박 기록 시작·끝, ISO 8601, 기기 로컬시간),
    days(심박이 기록된 날 수), coverage_pct(기록 구간 중 심박이 있는 분의 비율, %).
    다른 도구를 부르기 전에 유효한 user_id 를 확인할 때 사용한다. 두 데이터셋은 서로 섞지 않으며 사용자 단위로 따로 분석한다.

    Args:
        dataset: 'fitbit' 또는 'stanford' 로 거르기. 생략하면 연결된 모든 데이터셋(외부 데이터셋은 있을 때만). 알 수 없는 이름이면 빈 목록.
    """
    return store.list_users(dataset)


EVENTS_NOTE = ("연구 데이터셋에 붙은 라벨(참여자가 보고한 증상 시작·진단·회복 날짜)이며 진단이 아닙니다. 탐지 결과가 이 시점 근처에서 "
               "반응했는지 *검증용으로 비교*하는 데만 쓰고, 사용자의 질병 여부를 단정하거나 원인으로 설명하지 마세요.")


@mcp.tool(annotations={"readOnlyHint": True})
def get_health_events(user_id: str) -> dict:
    """사용자에게 붙은 연구용 건강 사건 라벨(증상 시작일, 진단일, 회복일)을 반환한다.

    외부 연구 데이터셋(stanford)의 참여자에게만 있고 FitBit 서비스 사용자는 항상 빈 목록이다. 이 라벨은 *검증을 위해 화면에 표시하는
    연구 라벨*이지 진단이 아니다: 이상 탐지 결과(detect_anomalies)가 이 날짜 전후에 반응했는지 비교하는 용도로만 언급하고, 질병을 단정하거나
    원인을 추정하지 말 것. 날짜는 개인정보 보호를 위해 이동된 값이다.

    반환: {user_id, dataset('fitbit'|'stanford'), events:[{type('symptom_onset'|'diagnosis'|'recovery'), date(YYYY-MM-DD),
    illness('covid'|'other'), first_warning(symptom_onset 에만, 있을 때)}] (날짜순), note(해석 주의 문구)}.
    first_warning = 그 증상 시작일 -7~+3일 창에서 시스템이 처음 알람을 낸 날: {evaluable(창에 평가 가능한 날이 있는가), hit(알람이 있었는가),
    first_alarm(YYYY-MM-DD|null), lead_days(증상일보다 며칠 먼저인가, 양수=먼저·0=당일·음수=늦음 | null), types(그날 알람 유형들), counted_in_eval}.
    참여자 전체 기록으로 사건마다 미리 계산한 값이라 조회 구간(lookback/end)과 무관하다. 이 값도 연구 검증용 비교 결과일 뿐 질병 예측이 아니다. 알 수 없는 user_id 면 {"error": 사유}.

    Args:
        user_id: 사용자 가명 ID (예: 'S004'). list_users 로 확인.
    """
    try:
        return {"user_id": user_id, "dataset": store.dataset_of(user_id),
                "events": [{k: e[k] for k in ("type", "date", "illness", "first_warning") if k in e} for e in store.events(user_id)], "note": EVENTS_NOTE}
    except KeyError as e:
        return _error(e)


@mcp.tool(annotations={"readOnlyHint": True})
def get_vitals_summary(user_id: str, lookback_days: float = 7, end: str | None = None) -> dict:
    """사용자의 최근 구간 심박 요약 통계와 일별 안정시 심박을 계산한다.

    반환: {user_id, window:{start,end}, summary:{coverage_pct(조회 구간 대비 심박 있는 분 %), hr_mean, hr_p05, hr_p95
    (모두 bpm), resting_hr(안정시 심박, bpm), sleep_hr(수면 중 중앙값, bpm), active_minutes(활동 분),
    rest_minutes(휴식 분: 신뢰 가능한 요약인지 판단하는 근거), n_minutes(심박이 있는 분 수)},
    daily_resting_hr:[{date, resting_hr(bpm, 휴식 분이 너무 적은 날은 null), n_rest_minutes}]}.
    실패하면 {"error": 사유}.

    Args:
        user_id: 사용자 가명 ID (예: 'U03'). list_users 로 확인.
        lookback_days: 조회 기간(일). 0 초과 90 이하, 기본 7. 조회 구간은 (end - lookback_days, end].
        end: 조회 끝 시각(ISO 8601, 예 '2016-04-20T12:00:00'). 생략하면 그 사용자의 마지막 기록 시각.
    """
    try:
        end_ts, start_ts = _resolve(user_id, lookback_days, end)
        df = store.window(user_id, end_ts.to_pydatetime(), lookback_days)
        return {"user_id": user_id, "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
                "summary": summarize(df, start_ts, end_ts), "daily_resting_hr": daily_resting_hr(df)}
    except (KeyError, ValueError) as e:
        return _error(e)


@mcp.tool(annotations={"readOnlyHint": True})
def detect_anomalies(user_id: str, lookback_days: float = 7, end: str | None = None) -> dict:
    """규칙 기반 탐지기로 조회 구간의 심박 이상 이벤트를 찾는다 (LLM 판단 없음, 결정론적).

    유형: resting_tachycardia(휴식 중 100bpm 초과 10분+), bradycardia(40bpm 미만, 수면 35 미만),
    personal_deviation(개인 평소 대비 통계적 이탈), elevated_resting_trend(안정시 심박이 7일 중앙값보다
    10bpm 이상 높은 날이 2일+ 연속), sensor_gap(착용 중 심박 신호 2시간+ 결측).
    운동 중·직후(걸음이 있던 때) 심박 상승은 이상으로 치지 않지만, 활동으로 설명되지 않는 극단 심박(210 bpm 이상, 또는 걷지 않는데 180 bpm 이상이 10분+)은 critical 로 알린다. 개인 기준선은 end 이전 전체 이력으로 계산한다.

    반환: {user_id, window:{start,end}, anomalies:[{anomaly_id, type, start, end(마지막 분 포함), duration_min,
    severity('info'|'warning'|'critical'), value(대표 bpm 등), baseline(개인 기준 bpm 또는 null),
    context('rest'|'sleep'|'active'), evidence(쉬운 한국어 한 줄 근거, 사용자에게 그대로 보여도 됨), stats(통계 세부값 dict —
    사용자 설명에는 쓰지 말고 필요할 때만 참고)}], counts:{유형:건수}}.
    severity 가 critical 인 이벤트가 하나라도 있으면 증상 확인과 의료진 상담 안내가 필요하다.
    실패하면 {"error": 사유}.

    Args:
        user_id: 사용자 가명 ID (예: 'U03').
        lookback_days: 조회 기간(일). 0 초과 90 이하, 기본 7.
        end: 조회 끝 시각(ISO 8601). 생략하면 그 사용자의 마지막 기록 시각.
    """
    try:
        end_ts, start_ts = _resolve(user_id, lookback_days, end)
        df = store.window(user_id, end_ts.to_pydatetime(), lookback_days)
        found = detect(df, store.history(user_id, end_ts.to_pydatetime()))
        counts: dict[str, int] = {}
        for a in found:
            counts[a.type] = counts.get(a.type, 0) + 1
        return {"user_id": user_id, "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
                "anomalies": [a.to_dict() for a in found], "counts": counts}
    except (KeyError, ValueError) as e:
        return _error(e)


@mcp.tool(annotations={"readOnlyHint": True})
def get_hr_series(user_id: str, lookback_days: float = 1, end: str | None = None, resample: str = "5min") -> dict:
    """차트용으로 다운샘플한 심박 시계열을 반환한다 (원시 1분 데이터 대량 노출 방지: 5분 이상 간격만 허용).

    반환: {user_id, window:{start,end}, resample, points:[{ts(구간 시작, ISO 8601), hr(구간 평균 bpm), steps(구간 걸음 합),
    asleep(구간의 과반이 수면이면 true)}]}. 데이터가 없는 구간은 생략된다. 실패하면 {"error": 사유}.

    Args:
        user_id: 사용자 가명 ID (예: 'U03').
        lookback_days: 조회 기간(일). 0 초과 30 이하, 기본 1.
        end: 조회 끝 시각(ISO 8601). 생략하면 그 사용자의 마지막 기록 시각.
        resample: 집계 간격. pandas 주기 표기('5min', '15min', '1h'). 5분 미만은 거부. 점이 2000개를 넘으면 5분 단위로 자동 확대되며
            실제 적용된 간격이 응답의 resample 에 들어간다.
    """
    try:
        end_ts, start_ts = _resolve(user_id, lookback_days, end, MAX_SERIES_DAYS)
        try:
            step = pd.to_timedelta(pd.tseries.frequencies.to_offset(resample))
        except (ValueError, TypeError):
            raise ValueError(f"resample 형식이 올바르지 않습니다: {resample!r} (예: '5min', '15min', '1h')") from None
        if step < MIN_RESAMPLE:
            raise ValueError("resample 은 5min 이상이어야 합니다 (원시 데이터 최소 노출 원칙)")
        if lookback_days * 1440 / (step / pd.Timedelta(minutes=1)) > MAX_SERIES_POINTS:
            step = pd.Timedelta(minutes=5 * int(np.ceil(lookback_days * 1440 / MAX_SERIES_POINTS / 5)))
            resample = f"{int(step / pd.Timedelta(minutes=1))}min"
        df = store.window(user_id, end_ts.to_pydatetime(), lookback_days)
        pts = []
        if not df.empty:
            r = df.astype({"asleep": "float64"}).set_index("ts").resample(resample)      # 수면 정보가 없는 참여자(NA)는 NaN -> 수면 아님
            agg = pd.DataFrame({"hr": r["hr_mean"].mean(), "steps": r["steps"].sum(), "asleep": r["asleep"].mean()})
            agg = agg[r["steps"].count() > 0]
            pts = [{"ts": ts.isoformat(), "hr": None if pd.isna(row.hr) else round(float(row.hr), 1),
                    "steps": int(row.steps), "asleep": bool(row.asleep > 0.5)} for ts, row in agg.iterrows()]
        return {"user_id": user_id, "window": {"start": start_ts.isoformat(), "end": end_ts.isoformat()},
                "resample": resample, "points": pts}
    except (KeyError, ValueError) as e:
        return _error(e)


@mcp.prompt()
def anomaly_review(user_id: str, lookback_days: str = "7") -> str:
    """사용자의 심박 이상 징후 점검 절차를 지시하는 프롬프트 (요약 → 탐지 → 가이드라인 대조 → 리포트)."""
    return (
        f"사용자 {user_id} 의 최근 {lookback_days}일 심박 이상 징후를 다음 순서로 점검하세요.\n"
        f"1. get_vitals_summary(user_id='{user_id}', lookback_days={lookback_days}) 로 요약과 coverage_pct 를 확인합니다. "
        "coverage_pct 가 50 미만이거나 rest_minutes 가 60 미만이면 데이터 부족을 명시하고 lookback_days 를 늘려 다시 조회합니다.\n"
        f"2. detect_anomalies(user_id='{user_id}', lookback_days={lookback_days}) 로 이상 이벤트를 조회합니다.\n"
        "3. guideline://heart-rate-ranges 와 guideline://red-flags 를 읽고 각 이벤트의 심각도와 맥락(rest/sleep)을 대조합니다.\n"
        "4. 한국어 리포트를 작성합니다: 핵심 요약, 이벤트별 근거(숫자는 도구 결과에서만 인용), 권장 행동. "
        "critical 이벤트가 있으면 증상 확인과 119/의료진 상담 안내를 먼저 쓰고, 진단을 단정하지 않으며, "
        "guideline://disclaimer 의 면책 문구로 끝맺습니다."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
