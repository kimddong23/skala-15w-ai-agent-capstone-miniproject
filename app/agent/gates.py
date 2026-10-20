"""게이트: LLM 의 판단을 코드 규칙이 감독하는 지점들. 모두 순수 함수이며 LLM·네트워크에 의존하지 않는다.

책임:
    - 라우팅 폴백/보정 (simple vs analysis, 후속 질문 판별)
    - 이중 게이트 (계획의 needs_analysis·태스크·기간을 코드가 최종 확정)
    - 안전(권위) 게이트 (critical 이상 → 고정 안내문 강제)
    - 재계획 판단 (데이터 부족 감지)
협력 모듈: `nodes.py` 가 호출하고, 결과를 SSE 이벤트와 상태로 옮긴다. 순수 함수라서 `tests/test_gates.py` 가 LLM 없이 전부 검증한다.
설계 근거: 라우팅 게이트, 이중 게이트, 재계획, 안전 게이트.
설계 원칙: 안전·정확성이 걸린 결정은 LLM 이 아니라 여기의 코드가 최종 권한을 가진다. 소형 모델의 오분류가 평가에서 실제로 관측되었다.
"""
import re
from dataclasses import dataclass, field

DISCLAIMER = (
    "※ 이 내용은 웨어러블 심박 데이터에 대한 참고용 분석이며 의학적 진단이 아닙니다. "
    "증상이 있거나 걱정되면 의료진과 상담하세요."
)
ESCALATION_MESSAGE = (
    "위험 수준의 심박 이상이 감지되었습니다. 이 서비스는 진단 도구가 아닙니다. 흉통·호흡곤란·어지럼·실신 같은 "
    "증상이 있거나 지속되면 즉시 119 또는 가까운 응급실에 연락하고, 증상이 없더라도 가능한 한 빨리 의료진의 확인을 받으세요."
)

# 프런트엔드 labels.ts 의 TYPE_LABEL 과 같은 표기를 쓴다: 리포트의 유형명이 화면의 각주(citation) 매칭 단서가 된다.
TYPE_LABELS = {
    "resting_tachycardia": "안정 시 빈맥",  # 수면 중에도 발생하므로 "휴식 중"이라 쓰지 않는다. 발생 맥락은 CONTEXT_LABELS 로 따로 표기
    "bradycardia": "서맥",
    "personal_deviation": "개인 기준선 이탈",
    "elevated_resting_trend": "안정시 심박 상승 추세",
    "sensor_gap": "센서 결측",
}
CONTEXT_LABELS = {"rest": "휴식 중", "sleep": "수면 중", "active": "활동 중"}  # detector 의 context 코드를 리포트에 노출하지 않기 위한 한국어 표기
SEVERITY_LABELS = {"critical": "심각", "warning": "주의", "info": "참고"}

# 경고(주의) 수준 이상이 있고 에스컬레이션은 아닐 때 코드가 쓰는 대응 안내. 이상이 있는 리포트에서 LLM 은 조언·평가를 쓰지 못하므로(허용 목록 필터)
# 대응 안내는 이 고정 문장이 전담한다. critical 은 ESCALATION_MESSAGE 가, 데이터 한계는 limitation_notice 가 맡는다.
WARNING_GUIDANCE = "비슷한 패턴이 반복되거나 증상(가슴 두근거림·어지럼·숨참)이 있으면 의료진과 상담하세요."

# 이상이 감지되지 않은 리포트·일반 건강 답변에 코드가 붙이는 안내. LLM 은 진료·복약·증상·건강 판정 문장을 쓸 수 없으므로(모든 경로에서 삭제) 이 고정 문장이 전담한다.
NO_ANOMALY_GUIDANCE = "데이터에서 이상이 보이지 않아도 가슴 통증·어지럼·숨참 같은 증상이 있으면 의료진과 상담하고, 증상이 심하면 119에 연락하세요."

DIAGNOSIS_NOTICE = (
    "요청하신 진단·확진은 이 서비스가 할 수 없습니다. 심박 데이터만으로는 질환을 판단할 수 없으니 "
    "정확한 진단은 반드시 의료진(내과·심장내과)과 상담하세요."
)
GREETING_TEXT = (
    "안녕하세요. 웨어러블 심박 데이터에서 이상 징후를 점검해 드리는 Vital Sentinel 입니다. "
    "예: 'U05 최근 7일 심박 이상 점검해줘', '휴식기 심박 정상 범위가 어떻게 돼?', 점검 후 '그럼 운동 중엔 괜찮은 거야?' 같은 후속 질문도 가능합니다."
)
OUT_OF_SCOPE_TEXT = (
    "이 서비스는 심박·건강 점검과 관련된 질문에만 답할 수 있습니다. "
    "예: 'U05 최근 7일 심박 이상 점검해줘', '휴식기 심박 정상 범위가 어떻게 돼?'"
)

# 점검 의도 키워드 (이상/점검/감지/위험/괜찮 등). STRONG 은 단독으로 analysis 라우팅 근거가 되고,
# WEAK 은 사용자 지정이 함께 있거나 이전 분석이 없을 때만 근거가 된다 (일반 상식 질문과의 겹침을 줄이기 위함).
STRONG_KW = ("비교", "공통", "이상", "점검", "감지", "분석", "리포트", "체크", "탐지", "확인해", "살펴", "최근", "지난", "이상징후", "평가")
WEAK_KW = ("위험", "괜찮", "문제", "걱정")  # "정상 범위" 같은 일반 지식 질문과 겹치는 "정상"은 제외
# 점검을 새로 요구하는 행위 키워드 ("이상"/"괜찮" 같은 상태 표현은 후속 질문에도 흔하므로 제외)
ACTION_KW = ("점검", "다시", "재분석", "재점검", "분석해", "체크", "감지해", "탐지해", "확인해", "리포트", "살펴")
SUMMARY_TOOL_KW = ("요약", "통계", "summary", "vitals")
DETECT_TOOL_KW = ("탐지", "이상", "detect", "anomal")
MIN_LOOKBACK, MAX_LOOKBACK, DEFAULT_LOOKBACK = 1.0, 30.0, 7.0

# \b 는 한글을 단어 문자로 보아 "U05의/U05는" 에서 경계가 없다 → 영숫자 lookaround 로 대체한다.
# U01 형식은 FitBit 서비스 사용자, S001 형식(세 자리)은 외부 연구 데이터셋(Stanford) 참여자. "S24" 같은 모델명과 헷갈리지 않게 S 는 세 자리만 받는다.
_USER_RE = re.compile(r"(?<![A-Za-z0-9])(?:U\d{2,}|S\d{3})(?!\d)", re.IGNORECASE)
# 기간이 아닌 횟수 표현("1일 3회", "하루 2번")은 기간 파싱 전에 제거한다.
_FREQUENCY_RE = re.compile(r"(?:\d+\s*일|하루)?\s*\d+\s*(?:회|번)(?:씩)?|(?:하루|매일)\s*(?:\d+\s*)?(?:회|번|씩)")
_DISEASES = "부정맥|심방세동|서맥증|빈맥증|심근경색|협심증|심부전|심장마비"
_DIAG_REQUEST_RE = re.compile(rf"진단\s*(해|좀|하|부탁|내려)|확진|무슨\s*병|병명|(?:{_DISEASES})\S{{0,3}}\s*(?:맞|인가|이야|이지|이에요|아니|이라고|라고)")
# 심박·건강 도메인 단어: 하나라도 있으면 가이드라인으로 답할 수 있는 질문으로 본다.
_HR_KW = ("심박", "심장", "맥박", "bpm", "빈맥", "서맥", "부정맥", "심방", "가슴", "두근", "호흡", "어지럼", "실신", "응급", "운동", "수면", "휴식",
          "정상", "위험", "이상", "센서", "서비스", "데이터", "가이드", "기준", "범위", "건강", "증상", "의료", "진료", "병원", "점검", "탐지",
          "착용", "측정", "워치", "웨어러블", "결과", "리포트")
_GREETING_RE = re.compile(r"^\s*(안녕|하이|헬로|hello|hi\b|반가|고마워|감사|수고|뭐\s*할\s*수|도움|사용법|기능)", re.IGNORECASE)


def extract_user_id(message: str) -> str | None:
    """메시지에서 사용자 가명 ID(U01·S001 형식)를 찾아 대문자로 반환한다. 없으면 None."""
    m = _USER_RE.search(message)
    return m.group(0).upper() if m else None


MAX_USERS = 3  # 한 번에 비교할 수 있는 사용자 수 (프롬프트 크기·지연 상한)


def extract_user_ids(message: str, limit: int = MAX_USERS) -> tuple[list[str], list[str]]:
    """메시지 속 사용자 ID 를 등장 순서대로 중복 없이 뽑는다 (비교 요청 지원).

    Returns:
        (포함할 ID 최대 `limit`개, 한도 초과로 제외된 ID).
    """
    ids: list[str] = []
    for m in _USER_RE.finditer(message):
        if (u := m.group(0).upper()) not in ids:
            ids.append(u)
    return ids[:limit], ids[limit:]


def extract_lookback_days(message: str) -> float | None:
    """질문 속 기간 표현을 일 단위로 환산한다.

    Args:
        message: 사용자 발화. "지난 3일", "2주", "일주일", "이틀", "한 달", "36시간" 등을 인식한다. "어제"·"오늘"은 인식하지 않는다.

    Returns:
        일 수(float). 기간 표현이 없으면 None. 날짜("4월 12일")는 기간이 아니므로 무시한다. clamp 는 하지 않는다.
    """
    t = _FREQUENCY_RE.sub(" ", re.sub(r"\d{1,2}\s*월\s*\d{1,2}\s*일", " ", message))  # 날짜('3월 12일')·횟수('1일 3회')는 기간이 아님
    for pat, mul in ((r"(\d+(?:\.\d+)?)\s*개월", 30), (r"(\d+(?:\.\d+)?)\s*주", 7), (r"(\d+(?:\.\d+)?)\s*일", 1), (r"(\d+(?:\.\d+)?)\s*시간", 1 / 24)):
        if m := re.search(pat, t):
            return float(m.group(1)) * mul
    for word, days in (("한달", 30), ("한 달", 30), ("일주일", 7), ("이주일", 14), ("사흘", 3), ("이틀", 2), ("하루", 1)):  # "어제/오늘"은 달력 날짜가 아니라 마지막 기록 기준 24시간이 되어 오해를 부르므로 뺐다
        if word in t:
            return float(days)
    return None


def clamp_lookback(days: float | None) -> float:
    """조회 기간을 [MIN_LOOKBACK, MAX_LOOKBACK] 일로 제한한다 (None 이면 기본 7일). 상한 30일은 비용·지연·데이터 최소화 때문이다."""
    return min(MAX_LOOKBACK, max(MIN_LOOKBACK, float(days if days is not None else DEFAULT_LOOKBACK)))


def has_strong_intent(message: str) -> bool:
    """강한 점검 의도 키워드(STRONG_KW) 포함 여부."""
    return any(k in message for k in STRONG_KW)


def has_check_intent(message: str) -> bool:
    """강/약 점검 의도 키워드 중 하나라도 포함하는지 여부."""
    return has_strong_intent(message) or any(k in message for k in WEAK_KW)


# 직전 결과를 가리키는 표현: "아까 결과 다시 요약해줘" 의 '다시'는 재점검이 아니라 이미 나온 결과를 다시 말해 달라는 뜻이다.
_PRIOR_REF = re.compile(
    r"(?:아까|이전|지난번|앞서|방금|직전|위)(?:의|에)?\s*(?:점검\s*)?(?:결과|리포트|분석|내용|답변|보고서|말)"
    r"|(?:결과|리포트|분석|내용|답변)(?:를|을|도)?\s*(?:다시\s*)?(?:요약|정리|설명|풀어)"
)


def asks_new_check(message: str, prev_user: str | None = None) -> bool:
    """메시지가 '새로운 점검'을 요구하는지 판별한다.

    새 점검 요구 = 다른 사용자 지정, 명시 기간, 또는 점검 행위 키워드(ACTION_KW).
    이전 분석이 있는 thread 에서 False 면 후속 질문으로 보고 저장된 사실로만 답한다. 3b 라우터가 후속 질문을 재분석으로
    오분류해 기본 7일 창으로 다시 돌리면서 이전의 critical 안내가 사라진 사고를 막기 위한 코드 규칙이다.

    Args:
        message: 현재 사용자 발화.
        prev_user: 직전 분석의 대상 사용자. 같은 사용자를 다시 언급하는 것만으로는 새 점검이 아니다. 요청 본문의 user_id(UI 선택)는
            여기서 비교하지 않는다: 본문 사용자가 다르면 후속 답변이 직전 분석 대상(analyzed_user) 기준임을 명시해 답한다 (nodes.simple_answer).

    Returns:
        새 점검 요구이면 True.
    """
    uid = extract_user_id(message)
    new_user, new_period = uid is not None and uid != prev_user, extract_lookback_days(message) is not None
    if _PRIOR_REF.search(message) and not (new_user or new_period):
        return False  # 이미 나온 결과를 다시 말해 달라는 요청 - '다시' 가 있어도 재조회하지 않는다
    return new_user or new_period or any(k in message for k in ACTION_KW)


# --- 대화 기억(이름·나이): 같은 thread 에서 사용자가 말한 개인 정보를 다시 물으면 기록에서 답한다 -----------------------------------------
_NAME_SAY = re.compile(r"(?:제|내|저의|나의)\s*이름(?:은|이|를)?\s*([가-힣A-Za-z]{1,12}?)\s*(?:입니다|이에요|예요|이야|야|이라고|라고|입니당|임)")
_AGE_SAY = re.compile(r"(?:제|내|저의|나의)?\s*나이(?:는|가)?\s*(\d{1,3})\s*(?:살|세)|(\d{1,3})\s*(?:살|세)\s*(?:입니다|이에요|예요|이야|야)")
_NAME_ASK = re.compile(r"(?:제|내|저의|나의)\s*이름|(?:제가|내가)\s*(?:누구|이름)")
_AGE_ASK = re.compile(r"(?:제|내|저의|나의)\s*나이|(?:제가|내가)\s*몇\s*(?:살|세)")
_RECALL = re.compile(r"뭐였|무엇이었|뭐라고|기억|아까|알아|누구|몇|뭔지|뭐지|뭐야|뭐예요|뭡니까|어떻게\s*되")


def _last_fact(pattern: re.Pattern, human_messages: list[str]) -> str | None:
    """사용자 발화들(오래된 순)에서 패턴에 처음 걸리는 사실을 *가장 최근 것부터* 찾는다."""
    for m in reversed(human_messages):
        if hit := pattern.search(m):
            return next(g for g in hit.groups() if g)
    return None


def is_personal_chat(message: str) -> bool:
    """이름·나이를 말하거나 되묻는 대화인가 (심박 점검과 무관한 개인 정보 대화 — 기록에서 답한다)."""
    return bool(_NAME_SAY.search(message) or _AGE_SAY.search(message) or ((_NAME_ASK.search(message) or _AGE_ASK.search(message)) and _RECALL.search(message)))


def personal_fact(message: str, earlier_human_messages: list[str]) -> str | None:
    """개인 정보 대화에서 답에 *반드시 들어가야 하는 사실*(이름 또는 나이 값). LLM 답이 기억을 틀렸는지 코드가 대조하는 기준이다."""
    if m := _NAME_SAY.search(message):
        return m.group(1)
    if m := _AGE_SAY.search(message):
        return next(g for g in m.groups() if g)
    if _NAME_ASK.search(message) and _RECALL.search(message):
        return _last_fact(_NAME_SAY, earlier_human_messages)
    if _AGE_ASK.search(message) and _RECALL.search(message):
        return _last_fact(_AGE_SAY, earlier_human_messages)
    return None


def personal_reply(message: str, earlier_human_messages: list[str]) -> str | None:
    """개인 정보 대화의 *결정론적* 답 (LLM 이 없거나 기억을 틀릴 때의 기준). 해당 없으면 None.

    - 이름/나이를 알려 주면 확인 인사
    - 되물으면 같은 thread 의 이전 사용자 발화에서 가장 최근 값을 찾아 답한다 (체크포인터에 메시지가 저장되므로 서버 재시작 후에도 동작)
    - 말한 적이 없으면 없다고 솔직히 답한다

    Args:
        message: 현재 사용자 발화.
        earlier_human_messages: 이전 사용자 발화들 (현재 발화 제외, 오래된 순).
    """
    if m := _NAME_SAY.search(message):
        return f"반갑습니다, {m.group(1)}님. 기억해 둘게요. 심박 점검이 필요하면 'U05 최근 7일 점검해줘'처럼 말씀해 주세요."
    if m := _AGE_SAY.search(message):
        return f"{next(g for g in m.groups() if g)}세시군요, 기억해 둘게요. 심박 점검이 필요하면 'U05 최근 7일 점검해줘'처럼 말씀해 주세요."
    if _NAME_ASK.search(message) and _RECALL.search(message):
        name = _last_fact(_NAME_SAY, earlier_human_messages)
        return f"{name}님이라고 말씀하셨어요." if name else "아직 이름을 말씀해 주지 않으셨어요."
    if _AGE_ASK.search(message) and _RECALL.search(message):
        age = _last_fact(_AGE_SAY, earlier_human_messages)
        return f"{age}세라고 말씀하셨어요." if age else "아직 나이를 말씀해 주지 않으셨어요."
    return None


def classify_simple(message: str, followup: bool) -> str:
    """simple 경로 질문의 종류를 코드로 가른다 (가이드라인 덤프가 인사·무관 질문에 나가는 것을 막는다).

    Args:
        message: 현재 사용자 발화.
        followup: 이전 분석이 있고 새 점검 요구가 없는 후속 질문인가.

    Returns:
        "personal"(이름·나이 등 대화 기억) | "followup"(직전 분석 기반 답변) | "greeting"(인사·기능 문의) | "hr"(심박·건강 지식 질문) | "out_of_scope"(범위 밖).
    """
    if is_personal_chat(message):
        return "personal"
    if followup:
        return "followup"
    if _GREETING_RE.match(message) and not any(k in message for k in _HR_KW):
        return "greeting"
    return "hr" if any(k in message for k in _HR_KW) else "out_of_scope"


def is_diagnosis_request(message: str) -> bool:
    """진단·확진을 요구하는 발화인가 (예: "진단해줘", "확진해줘", "심방세동 맞지?").

    True 면 어느 경로든 리포트에 DIAGNOSIS_NOTICE 가 코드로 강제된다 (LLM 이 거절 문장을 쓸지에 의존하지 않는다).
    """
    return bool(_DIAG_REQUEST_RE.search(message))


def effective_lookback(plan_days: float | None, message: str, carry: float | None = None) -> tuple[float, str]:
    """계획 단계에서 *실제로 쓰일* 조회 기간과 사유를 계산한다 (게이트와 같은 규칙: 명시 기간 > 직전 창 이어받기 > 계획값).

    plan_created 이벤트가 게이트가 확정할 기간과 다른 값을 보여주던 문제를 없애려고 plan 노드가 방출 전에 사용한다.
    """
    explicit = extract_lookback_days(message)
    if explicit is not None:
        return clamp_lookback(explicit), "질문의 명시 기간"
    if carry:
        return clamp_lookback(carry), "직전 분석 창 이어받음"
    return clamp_lookback(plan_days), "계획값"


# "내 심박 괜찮아?" 처럼 *본인 데이터*를 가리키는 표지 (사용자 ID 없이도 점검 요청으로 본다 → 사용자 선택 안내)
_OWN_DATA = re.compile(r"(?:내|제|저의|나의)\s*(?:심박|데이터|상태|건강|맥박)|오늘|최근|이번|지난")


def route_fallback(message: str, has_context: bool, prev_user: str | None = None, has_user: bool = True) -> tuple[str, str]:
    """LLM 라우팅 실패(타임아웃·파싱 오류·LLM 비활성) 시의 키워드 기반 폴백.

    Args:
        message: 현재 사용자 발화.
        has_context: 같은 thread 에 이전 분석 결과가 있는가.
        prev_user: 직전 분석의 대상 사용자.
        has_user: 점검 대상 사용자가 정해져 있는가(질문 속 ID 또는 요청 본문). 없으면 약한 키워드("괜찮")만으로는 분석으로 보내지 않는다.

    Returns:
        (route, 사유). route 는 "simple" | "analysis".
    """
    if is_personal_chat(message):
        return "simple", "키워드 폴백: 이름·나이 등 대화 기억"
    if has_context and not asks_new_check(message, prev_user):
        return "simple", "키워드 폴백: 이전 분석에 대한 후속 질문"
    if has_strong_intent(message):
        return "analysis", "키워드 폴백: 점검 의도 키워드"
    if not has_context and any(k in message for k in WEAK_KW):
        if has_user or _OWN_DATA.search(message):
            return "analysis", "키워드 폴백: 이전 분석이 없는 상태에서의 상태 질문"
        return "simple", "키워드 폴백: 본인 데이터를 가리키지 않는 일반 질문(예: '운동할 때는 괜찮아?')"
    return "simple", "키워드 폴백: 점검 의도 없음"


def route_override(route: str, message: str, has_context: bool, prev_user: str | None = None) -> tuple[str, str | None]:
    """코드 보정(권위): LLM 라우팅 결과를 규칙으로 덮어쓴다.

    - 이전 분석이 있고 새 점검 요구가 없으면 항상 simple (저장된 사실로 답변, 에스컬레이션 문구 유지)
    - 그 외에 simple 인데 점검 의도가 명백하면(강한 키워드, 또는 사용자 명시 + 약한 키워드) analysis 로 상향

    Args:
        route: LLM(또는 폴백)이 정한 route.
        message: 현재 사용자 발화.
        has_context: 같은 thread 에 이전 분석 결과가 있는가.
        prev_user: 직전 분석의 대상 사용자.

    Returns:
        (최종 route, 보정 사유). 보정이 없으면 사유는 None.
    """
    if is_personal_chat(message):  # 이름·나이 대화는 라우터가 어떻게 분류하든 대화 기억 경로(simple)로 간다
        return "simple", None if route == "simple" else "코드 보정: 이름·나이 등 대화 기억 → simple"
    if has_context and not asks_new_check(message, prev_user):
        return "simple", None if route == "simple" else "코드 보정: 이전 분석에 대한 후속 질문 → 재분석 없이 답변"
    if route == "simple" and (has_strong_intent(message) or (extract_user_id(message) and has_check_intent(message))):
        return "analysis", "코드 보정: 점검 의도 → analysis 로 상향"
    return route, None


def task_tool(task: dict) -> str | None:
    """research 태스크의 id·설명 키워드로 대응 도구를 추정한다.

    Returns:
        "summary" | "detect" | "both"(둘 다 언급) | None(판별 불가).
    """
    d = f"{task.get('id', '')} {task.get('description', '')}".lower()
    det, summ = any(k in d for k in DETECT_TOOL_KW), any(k in d for k in SUMMARY_TOOL_KW)
    return "both" if det and summ else "detect" if det else "summary" if summ else None


def rule_based_plan(message: str) -> dict:
    """plan LLM 실패/비활성 시의 결정론적 계획 (research 2개 + 선택적 analysis + writing).

    Returns:
        {"tasks", "needs_analysis", "lookback_days"} — LLM 계획과 같은 형태.
    """
    needs = has_check_intent(message)
    tasks = [
        {"id": "r_summary", "type": "research", "description": "요약 통계 조회 (get_vitals_summary)"},
        {"id": "r_detect", "type": "research", "description": "이상 탐지 실행 (detect_anomalies)"},
    ]
    if needs:
        tasks.append({"id": "a1", "type": "analysis", "description": "탐지 결과의 패턴·맥락 해석"})
    tasks.append({"id": "w1", "type": "writing", "description": "한국어 리포트 작성"})
    return {"tasks": tasks, "needs_analysis": needs, "lookback_days": extract_lookback_days(message) or DEFAULT_LOOKBACK}


@dataclass
class GateResult:
    """`apply_gate` 결과.

    Attributes:
        tasks: 정규화된 최종 태스크 (research → analysis → writing 순, id 유일).
        needs_analysis: analysis 단계 실행 여부 (코드가 확정).
        lookback_days: 최종 조회 기간(일, 1~30).
        removed_tasks: 제거된 태스크 id.
        forced_analysis: LLM 이 불필요하다 했지만 코드가 analysis 를 강제했는가.
        reason: 변경 사유 요약 (SSE gate_applied 로 노출).
    """

    tasks: list[dict]
    needs_analysis: bool
    lookback_days: float
    removed_tasks: list[str] = field(default_factory=list)
    forced_analysis: bool = False
    reason: str = ""


def apply_gate(plan: dict, message: str, carry_lookback: float | None = None) -> GateResult:
    """이중 게이트. 1차=LLM 의 needs_analysis, 2차=코드가 최종 결정.

    - 점검 의도가 있으면 LLM 이 False 라 해도 analysis 강제
    - needs_analysis=False 면 analysis 태스크 제거
    - detect 는 안전 게이트 입력이므로 research 에 항상 포함, research·writing 최소 1개 보장
    - 질문에 명시된 기간이 있으면 LLM 기간보다 우선, 1~30일 clamp
    - 기간 명시가 없고 같은 사용자의 직전 분석 창(carry_lookback)이 있으면 그 창을 이어받는다 (재분석이 더 짧은 창으로
      달라져 이전에 본 위험이 사라지는 것을 방지)

    Args:
        plan: {"tasks", "needs_analysis", "lookback_days"} (LLM 계획 또는 규칙 계획).
        message: 현재 사용자 발화.
        carry_lookback: 같은 사용자의 직전 분석 조회 기간(일). 없으면 None.

    Returns:
        GateResult.
    """
    tasks = [dict(t) for t in plan.get("tasks", [])]
    needs = bool(plan.get("needs_analysis"))
    reasons: list[str] = []

    forced = has_check_intent(message) and not needs
    if forced:
        needs = True
        reasons.append("점검 의도 감지 → analysis 강제")

    removed: list[str] = []
    if not needs:
        removed = [t["id"] for t in tasks if t["type"] == "analysis"]
        tasks = [t for t in tasks if t["type"] != "analysis"]
        if removed:
            reasons.append("needs_analysis=False → analysis 태스크 제거")
    elif not any(t["type"] == "analysis" for t in tasks):
        tasks.append({"id": "a1", "type": "analysis", "description": "탐지 결과의 패턴·맥락 해석"})

    # 한 태스크에 도구 둘이 섞이면 분리한다: research 노드가 태스크:도구 = 1:1 로 이벤트를 내보내 UI 진행 표시가 정확해진다.
    if any(t["type"] == "research" and task_tool(t) == "both" for t in tasks):
        tasks = [t for t in tasks if not (t["type"] == "research" and task_tool(t) == "both")]
        reasons.append("복합 research 태스크를 summary/detect 로 분리")
    have = {task_tool(t) for t in tasks if t["type"] == "research"}
    if "summary" not in have:
        tasks.append({"id": "r_summary", "type": "research", "description": "요약 통계 조회 (get_vitals_summary)"})
        reasons.append("summary 태스크 보강")
    if "detect" not in have:  # LLM 이 detect 를 빼먹어도 안전 게이트 입력은 반드시 확보한다
        tasks.append({"id": "r_detect", "type": "research", "description": "이상 탐지 실행 (detect_anomalies)"})
        reasons.append("detect 태스크 보강(안전 게이트 입력)")
    if not any(t["type"] == "writing" for t in tasks):
        tasks.append({"id": "w1", "type": "writing", "description": "한국어 리포트 작성"})
        reasons.append("writing 태스크 보강")

    order = {"research": 0, "analysis": 1, "writing": 2}
    tasks.sort(key=lambda t: order[t["type"]])  # 안정 정렬: 같은 타입은 원래 순서 유지
    seen: set[str] = set()
    for t in tasks:  # LLM 이 id 를 중복하거나 비워도 SSE task_id 가 유일하도록 정규화
        base = str(t.get("id") or t["type"])
        t["id"] = base if base not in seen else f"{base}_{len(seen)}"
        seen.add(t["id"])

    lookback = clamp_lookback(plan.get("lookback_days"))
    if (explicit := extract_lookback_days(message)) is not None:
        explicit = clamp_lookback(explicit)
        if explicit != lookback:
            reasons.append(f"질문의 명시 기간 {explicit:g}일 우선")
        lookback = explicit
    elif carry_lookback:
        lookback = clamp_lookback(carry_lookback)
        reasons.append(f"기간 미지정 → 직전 분석 창 {lookback:g}일 이어받음")
    return GateResult(tasks, needs, lookback, removed, forced, "; ".join(reasons) or "변경 없음")


@dataclass
class SafetyResult:
    """`safety_gate` 결과. escalation=False 면 reason/message 는 빈 문자열."""

    escalation: bool
    reason: str
    message: str


def safety_gate(anomalies: list[dict]) -> SafetyResult:
    """권위 게이트: critical 이상이 하나라도 있으면 LLM 판단과 무관하게 에스컬레이션 + 고정 안내문을 강제한다.

    고정 문구를 LLM 이 쓰게 하면 빠뜨리거나 바꿔 쓸 수 있으므로, 문구는 상수이고 리포트 조립은 코드가 한다.

    Args:
        anomalies: 탐지 결과 이상 목록 (각 항목에 severity, type).

    Returns:
        SafetyResult.
    """
    crit = [a for a in anomalies if a.get("severity") == "critical"]
    if not crit:
        return SafetyResult(False, "", "")
    kinds = ", ".join(sorted({TYPE_LABELS.get(str(a.get("type")), str(a.get("type"))) for a in crit}))   # 내부 코드(resting_tachycardia)가 승인 카드에 노출되지 않게 한국어 유형명
    # 여러 사용자를 한 번에 볼 때는 이상 항목에 user_id 가 붙는다 → critical 이 있는 사용자를 안내문에 밝힌다 (다른 사용자가 안심하는 일을 막는다).
    who = sorted({str(a["user_id"]) for a in crit if a.get("user_id")})
    suffix = f" (해당 사용자: {', '.join(who)})" if who else ""
    return SafetyResult(True, f"critical 이상 {len(crit)}건 ({kinds}){suffix}", ESCALATION_MESSAGE + suffix)


def replan_reason(summary: dict | None, round_: int, max_rounds: int, lookback_days: float) -> str | None:
    """데이터가 부족해 조회 기간을 늘려야 하는지 판단한다 (커버리지 < 50% 또는 휴식 분 < 60).

    Args:
        summary: `get_vitals_summary` 결과. None(조회 실패)이면 기간을 늘려도 소용없으므로 재계획하지 않는다.
        round_: 지금까지의 재계획 횟수.
        max_rounds: 재계획 상한 (루프 가드).
        lookback_days: 현재 조회 기간(일). 상한(30일)에 도달했으면 더 늘릴 수 없다.

    Returns:
        재계획 사유 문자열, 불필요/불가능하면 None.
    """
    if summary is None or round_ >= max_rounds or lookback_days >= MAX_LOOKBACK:
        return None
    cov = (summary.get("summary") or {}).get("coverage_pct")
    if cov is not None and cov < 50:
        return f"데이터 커버리지 {cov:g}% < 50% → 조회 기간 확대"
    rest = (summary.get("summary") or {}).get("rest_minutes")  # 요약이 직접 주는 휴식 분 (없으면 일별 합으로 대체)
    if rest is None and "daily_resting_hr" in summary:
        rest = sum(d.get("n_rest_minutes") or 0 for d in summary["daily_resting_hr"])
    if rest is not None and rest < 60:
        return f"휴식 구간 {rest}분 < 60분 → 조회 기간 확대"
    return None
