"""리포트 검증기: 결정론적 규칙(필수) + 선택적 LLM judge.

책임: writing 이 만든 리포트가 (1) 도구 결과에 없는 수치를 지어내지 않았는지, (2) 진단을 단정하지 않는지, (3) 지표 의미를
혼동하지 않았는지, (4) 면책/에스컬레이션 고정 문구를 포함하는지 검사하고 지적사항을 돌려준다. 재시도·템플릿 대체 결정은 하지 않는다
(그건 `nodes.py` 의 validation_check/finalize 몫).
협력 모듈: `nodes.py`(validator 노드), `gates.py`(DISCLAIMER), `llm.py`(judge 용 LLM), 평가 스크립트(numbers_check 재사용).
설계 근거: 규칙 필수 + judge 선택, 수치는 도구 결과에서만.
왜 규칙이 주 장치인가: 3b judge 는 정상 리포트도 반려해(실측) 재작성 지연만 늘렸다. 규칙은 오탐 원인을 추적·수정할 수 있다.
"""
import math
import re
from typing import Any

from pydantic import BaseModel

from app.agent.gates import DISCLAIMER
from app.agent.llm import LLM
from app.config import settings

# --- 숫자 출처 검증 -------------------------------------------------------------------------------
_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?|\d{1,2}:\d{2}(?::\d{2})?|\d{4}\s*년|\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?"
    r"|\d{1,2}\s*시(?!간)(?:\s*\d{1,2}\s*분)?|\b[US]\d+\b|^\s*\d+[.)]\s",
    re.MULTILINE,
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# '60~100회', '130 bpm', '12분', '95%' 처럼 단위가 붙은 수치만 인용 검사 대상 (순번·'2가지' 같은 잡음 제외). 소수는 단위 없어도 검사.
_CITED_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*[~\-–]\s*(\d+(?:\.\d+)?))?\s*(bpm|BPM|분|시간|%|건|회|일|개|번)")
_DECIMAL_RE = re.compile(r"\d+\.\d+")


def numbers_in(text: str) -> list[float]:
    """텍스트에서 날짜·시각·사용자 ID·목록 번호를 제외한 모든 숫자를 추출한다."""
    return [float(x) for x in _NUM_RE.findall(_DATE_RE.sub(" ", text))]


def collect_facts(*sources: Any, extra_text: str = "") -> list[float]:
    """리포트가 인용해도 되는 '사실 수치' 집합을 만든다.

    Args:
        *sources: 도구 결과·가이드라인 등 중첩 dict/list/str/숫자. 문자열 값은 날짜를 제거한 뒤 숫자를 추출한다.
        extra_text: 허용할 추가 텍스트. 사용자 발화 전체를 넣지 말 것 — 사용자가 입력한 숫자("250 bpm 이라고 써줘")가 사실로
            둔갑한다. 조회 기간처럼 시스템이 정한 값만 넣는다.

    Returns:
        허용 수치 리스트. 파생값(지속시간의 시간 환산, 기준선 대비 차이)과 자명한 상수(0, 24)를 포함한다.
    """
    facts: list[float] = []

    def walk(x: Any) -> None:
        """중첩 구조에서 숫자를 재귀적으로 수집한다 (bool·None 제외)."""
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            facts.append(float(x))
        elif isinstance(x, str):
            facts.extend(numbers_in(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    for s in sources:
        walk(s)
    facts.extend(numbers_in(extra_text))
    facts.extend([0.0, 24.0])  # '0건', '24시간' 같은 자명한 상수는 수치 환각이 아니다 (벤치마크에서 오탐 확인)
    # 파생값: 지속시간(시간 단위), 기준선 대비 차이는 리포트에서 자주 인용된다.
    for src in sources:
        for a in (src or {}).get("anomalies", []) if isinstance(src, dict) else []:
            if isinstance(a.get("duration_min"), (int, float)):
                facts.append(round(a["duration_min"] / 60, 1))
            if isinstance(a.get("value"), (int, float)) and isinstance(a.get("baseline"), (int, float)):
                facts.append(abs(round(a["value"] - a["baseline"], 1)))
    return facts


def _in_facts(n: float, facts: list[float]) -> bool:
    """n 이 사실값 하나의 표기로 볼 수 있는가. 정수 표기는 사실값의 버림/반올림과 같아야 하고, 소수 표기는 0.05 이내여야 한다.

    예전에는 ±0.5 또는 ±1% 를 허용했는데, 실데이터에서 40~140 정수의 18~31% 가 우연히 통과해 환각을 못 잡았다.
    """
    if float(n).is_integer():
        return any(n in (math.floor(f), math.floor(f + 0.5)) for f in facts)
    return any(abs(n - f) <= 0.051 for f in facts)


# 단위 없이 쓴 수치도 심박 문맥에서는 검사한다: "심박이 145까지 올랐습니다", "평균 72".
_BARE_RE = re.compile(r"(?:심박수?|맥박|평균|최고|최대|최소|기준선)[^\d\n]{0,6}?(\d+(?:\.\d+)?)(?!\s*(?:bpm|BPM|분|시간|%|건|회|일|개|번)|\d|\.\d)")


def numbers_check(report: str, facts: list[float]) -> tuple[int, list[float]]:
    """리포트가 인용한 수치의 개수와 facts 로 뒷받침되지 않는 수치 목록.

    검사 대상: 단위가 붙은 수치(bpm·분·%·건 …), 소수, 심박 문맥의 단위 없는 수치.

    Args:
        report: 검사할 텍스트 (고정 문구는 호출 전에 제거할 것).
        facts: `collect_facts` 결과.

    Returns:
        (인용 수치 개수, facts 로 뒷받침되지 않는 수치 리스트).
    """
    text = _DATE_RE.sub(" ", report)
    cited: list[float] = []
    for m in _CITED_RE.finditer(text):
        cited.extend(float(g) for g in (m.group(1), m.group(2)) if g is not None)
    cited.extend(float(m.group(1)) for m in _BARE_RE.finditer(text))
    cited.extend(float(m.group()) for m in _DECIMAL_RE.finditer(text) if float(m.group()) not in cited)
    return len(cited), [n for n in cited if not _in_facts(n, facts)]


def unsupported_numbers(report: str, facts: list[float]) -> list[float]:
    """`numbers_check` 의 두 번째 값만: facts 에 없는 수치 목록 (개수가 필요 없는 호출용 단축)."""
    return numbers_check(report, facts)[1]


# --- 진단 단정·복약 지시 금지 ---------------------------------------------------------------------
# 금지 대상은 '질병 진단'이다. 탐지 유형 이름(빈맥·서맥·개인 기준선 이탈 …)을 서비스가 서술하는 것은 허용한다 → 사전에 넣지 않는다.
# 판정은 절 단위이며 공백을 모두 제거한 문자열에서 한다("심방 세동", "심방세 동" 우회 방지).
_DX_KO = ("부정맥", "심방세동", "심방조동", "심실세동", "심실빈맥", "서맥증", "빈맥증", "심근경색", "협심증", "심부전", "심장마비", "심장병", "심장질환",
          "심혈관질환", "갑상선기능항진증", "갑상선항진증", "갑상선기능저하증", "패혈증", "감염증", "빈혈", "고혈압", "저혈압", "당뇨", "수면무호흡")
_DX_EN = re.compile(r"afib|a-fib|atrialfibrillation|arrhythmia|myocardialinfarction|heartattack|heartfailure|angina|tachycardia|bradycardia")
_ASSERT_KO = re.compile(
    r"입니다|이다|이에요|이네요|임이|이맞|맞습니다|있습니다|있어요|있음|있다|확인(?:됩|되|됨|했|합)|판단(?:됩|되|됨|합)|진단(?:됩|되|됨|합|했|입|하)|확진|소견|분명|확실|으로보|로보입|보입니다|이라고|있고|있으며|라고할수있|겪고계|에해당|진단할수있|발생했|나타났|나타나고|확인됐"
)
_ASSERT_EN = re.compile(r"youhave|youhad|youare|you're|diagnosedwith|diagnosis|suffer|confirm|thisis|itis|shows|indicates|patienthas")
_GENERIC_DX = re.compile(r"진단(?:됩|합|했|입)니다|진단됐|확진|진단할수있|진단해드")
# 헤지는 '해당 절 안'에서 질병 주장을 약화·부정할 때만 인정한다. "상담/진료" 권고는 헤지가 아니다.
# 헤지는 *질병 용어 바로 옆*에서만 인정한다. 예전의 절 전체 화이트리스트("없/않/못/수있" 한 글자 포함)는 "부정맥이 있고 걱정할 수 없는 상태입니다"
# 같은 문장을 통과시켰다. 질병 뒤 10자 안의 한정어 / 앞 8자 안의 의심 표현 / 소견·증상 부재 서술 / 서비스의 거절 서술만 허용한다.
_HEDGE_NEAR = re.compile(r"^.{0,10}?(?:가능성|의심|배제|아닐|아닙|아니|아님|여부|인지|단정|확정|감별|추정)")
_HEDGE_PRE = re.compile(r"(?:의심|가능성|추정|배제)\S{0,6}$")
_NEGATED_FINDING = re.compile(r"^.{0,4}?(?:소견|증상|징후|진단)(?:은|이|도)?(?:없|아닙|아니)")
_REFUSAL = re.compile(r"거절|불가|제공하지|드릴수없|수행할수없")
_HEDGE_KO = re.compile(r"가능성|의심|수도|아닐|여부|인지|추정|단정|확정|감별|아닙|아니|아님|거절|불가|제공하지")  # 질병 용어 없는 일반 진단 문구용 (한 글자 헤지 제외)
_HEDGE_EN = re.compile(r"maybe|might|could|possib|suspect|cannot|can't|unlikely|donot|doesnot|isnot|arenot|notdiagnosed|ruleout")  # 공백 제거 문자열이라 'no' 같은 짧은 단어는 쓸 수 없다
_MED_NOUN = r"(?:약|아스피린|와파린|베타차단제|항부정맥제|진통제|항생제|약물|영양제|타이레놀|이부프로펜|항응고제|아세트아미노펜|니트로글리세린)"
_MED_RE = re.compile(rf"{_MED_NOUN}\S{{0,6}}?(?:드세요|드십시오|드시|복용|먹어|먹으세요|먹는|투여|섭취)|(?:takeaspirin|takemedication|youshouldtake)")
# 복약 *중단·감량·조절* 지시도 복약 지시다 (공백이 제거된 절에 적용): "약은 끊으셔도 됩니다", "혈압약은 그만 드셔도 돼요", "약을 줄이셔도 됩니다"
_MED_STOP = re.compile(rf"{_MED_NOUN}\S{{0,6}}?(?:끊|그만드|줄이|줄여|중단|건너뛰|거르|바꾸|늘리|늘려|조절|빼)|(?:끊|줄이|중단|거르|그만).{{0,4}}{_MED_NOUN}")
# 복약 *생략* 지시 (공백 제거된 절): "약은 증상이 없으면 안 드셔도 됩니다", "복용하지 않으셔도", "복용 안 해도", "생략해도". 여기서는 부정어(않·안)가 곧 지시이므로
# 일반 복약 부정 예외(_MED_NEGATION)를 쓰지 않고, 의료진 상의·임의 중단 금지를 말하는 문장만 예외로 둔다.
_MED_OMIT = re.compile(rf"{_MED_NOUN}\S{{0,14}}?(?:안드|드시지않|안먹|먹지않|복용하지않|복용안|생략|안챙|챙기지않)|복용(?:하지않|안하|을생략|생략)")
_MED_OMIT_SAFE = re.compile(r"임의로|함부로|상의|문의|삼가|금지|안됩니다|안돼|마세요|말고")
_MED_NEGATION = re.compile(r"마세요|말고|지말|하지|않|금지|삼가|불가|바꾸지|임의로|문의|상의")
_CLAUSE_SPLIT = re.compile(r"[.!?。\n;,，]|(?<=니다)만|지만|는데|으나")


def _dx_asserted(c: str) -> bool:
    """공백 제거된 절 c 에서, 질병 용어가 헤지·부재 서술 없이 단정 서술과 함께 쓰였는가."""
    if not _ASSERT_KO.search(c) or _REFUSAL.search(c):
        return False
    for d in _DX_KO:
        for m in re.finditer(re.escape(d), c):
            if _HEDGE_NEAR.match(c[m.end():]) or _NEGATED_FINDING.match(c[m.end():]) or _HEDGE_PRE.search(c[: m.start()]):
                continue
            return True
    return False


def diagnostic_assertions(report: str) -> list[str]:
    """진단 단정·복약 지시를 포함한 절을 찾는다.

    - 질병 사전(KO+EN)의 용어가 있는 절에서 단정 서술(입니다/있습니다/확인되/소견/is·have …)이 있고 같은 절에 헤지가 없으면 위반
    - 복약 지시(약·아스피린 … + 드세요/복용 …)가 있고 부정(마세요/하지 …)이 없으면 위반
    - 질병 용어 없이 "진단됩니다/확진" 만 있어도 위반 (헤지 제외)

    절 분리는 `. ! ? 。 줄바꿈 ; ,` 와 "~니다만/지만/는데/으나" 에서 하며 공백이 없어도 나눈다.

    Args:
        report: 검사할 텍스트 (고정 문구는 호출 전에 제거할 것).

    Returns:
        위반 절 원문(최대 80자) 리스트.
    """
    out: list[str] = []
    for raw in _CLAUSE_SPLIT.split(report):
        c = re.sub(r"\s+", "", raw).lower()
        if not c:
            continue
        bad = (
            _dx_asserted(c)
            or (_DX_EN.search(c) and (_ASSERT_EN.search(c) or _ASSERT_KO.search(c)) and not _HEDGE_EN.search(c) and not _HEDGE_KO.search(c))
            or (_GENERIC_DX.search(c) and not _HEDGE_KO.search(c))
            or ((_MED_RE.search(c) or _MED_STOP.search(c)) and not _MED_NEGATION.search(c))
            or (_MED_OMIT.search(c) and not _MED_OMIT_SAFE.search(c))
        )
        if bad:
            out.append(raw.strip()[:80])
    return out


# 단정적 안심: 데이터 상태와 무관하게 금지한다. 허용 문구는 "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다" + 한계 문장.
# "운동 중 심박 상승은 정상입니다" 같은 일반 지식은 허용해야 하므로 '개인 건강 전반'을 가리키는 주어가 있는 경우만 막는다.
_REASSURANCE = re.compile(
    r"걱정(?:하지|할\s*필요|마세요|안\s*하)|안심(?:하세요|하셔도|해도|하십시오|하셔요)|추가\s*(?:검사|진료|검진|확인)\S{0,4}\s*(?:필요\s*없|불필요|필요하지\s*않)"
    r"|(?:전혀|완전히|모두|전부|전반적으로|전체적으로|종합적으로)\s*(?:정상|문제\s*없|이상\s*없)|(?:심박|건강|상태|결과|수치)(?:은|는|이|가)?\s*(?:모두\s*)?정상(?:입니다|이에요|이므로|이라)"
    r"|건강(?:합니다|하십니다|한\s*상태)"
    # "이상 없으므로 추가적인 주의가 필요하지 않습니다" — 데이터 한계·이상 유무와 무관하게 항상 금지(허용 형태는 범위를 한정한 무발견 문장 + 진료 권고)
    r"|추가적?(?:인)?\s*(?:주의|조치|관찰|관리|검사|진료|검진|확인)(?:는|가|이)?\s*(?:필요\s*(?:없|하지\s*않)|불필요)"
    r"|주의(?:할|하실)?\s*(?:필요|것)(?:는|이|가)?\s*(?:전혀\s*)?없|신경\s*(?:을\s*)?(?:쓰지|안\s*써도|쓰실\s*필요|쓸\s*필요)"
    # 절대적 건강 보증: "완벽합니다", "앞으로도 걱정은 안 하셔도", "You are perfectly healthy"
    r"|걱정(?:은|이|도)?\s*(?:안|않|하지\s*않)|완벽(?:합니다|하게|한\s*건강|해요)|건강(?:은|이)\s*(?:완벽|최상|매우\s*좋)|앞으로도[^.\n]{0,20}(?:걱정|신경)"
    r"|(?i:perfectly\s+(?:healthy|fine)|completely\s+(?:healthy|fine|safe)|you\s+are\s+(?:healthy|fine|safe)|nothing\s+to\s+worry|no\s+need\s+to\s+(?:worry|see))"
)
def is_definitive_reassurance(sentence: str) -> bool:
    """문장이 단정적 안심 표현(`_REASSURANCE`)을 담고 있는가. 스트리밍 가드가 검증 전에 화면에서 보류하는 데 쓴다."""
    return _REASSURANCE.search(sentence) is not None


# 내부 코드가 사용자 문장에 새어 나온 경우
_RAW_CODE = re.compile(r"\b(?:rest|sleep|active|resting_tachycardia|bradycardia|personal_deviation|elevated_resting_trend|sensor_gap|critical|warning)\b")
# 탐지가 실패했거나 데이터가 제한적인데 "이상 없음"으로 읽히는 서술 (거짓 안심 방지)
_ALL_CLEAR = re.compile(r"이상(?:이|은|도)?\s*(?:전혀\s*)?(?:없|발견되지|감지되지|관찰되지)|이상\s*구간[이가]?\s*(?:감지|발견)되지|문제(?:가|는|도)?\s*없")  # 개인 건강 전반의 '정상입니다'는 _REASSURANCE 가 맡는다 — 일반 지식('운동 중 상승은 정상')은 허용해야 한다


# 사용자 데이터를 가리키는 '정상' 서술: "심박 기록은 정상 범위 내에 있습니다", "수치는 정상 수준입니다".
# 일반 지식("성인의 정상 범위는 60~100회", "일반적으로 정상 범위에 해당")은 허용해야 하므로 일반 표지가 앞에 있으면 제외한다.
_NORMAL_DATA = re.compile(r"(?:심박(?:수|\s*기록)?|기록|데이터|수치|결과|상태)[^.\n]{0,12}?정상\s*(?:범위\s*(?:내|안)?\s*에\s*(?:있|해당|속)|수준|적(?:입니다|이에요|이다|인\s*범위))")
_GENERAL_MARK = re.compile(r"일반적|일반\s*성인|성인|보통|대부분|사람")
_NEGATED = r".{0,16}?(?:아닙니다|단정|수\s*없|보장|아니)"


# 사용자 ID 표기 (gates._USER_RE 와 같은 형식 - validator 는 gates 의 정규식 세부에 기대지 않으려고 따로 둔다)
_UID = re.compile(r"(?<![A-Za-z0-9])(?:U\d{2,}|S\d{3})(?!\d)", re.IGNORECASE)


def _owner(text: str, pos: int) -> str | None:
    """text[pos] 위치의 서술이 *누구에 관한 것*인가: 그 앞에서 마지막으로 언급된 사용자 ID (제목 줄 "## U01" 도 범위를 정한다). 없으면 None."""
    ids = _UID.findall(text[:pos])
    return ids[-1].upper() if ids else None


def _normal_data_hits(body: str) -> list[int]:
    """사용자 데이터가 '정상 범위/수준'이라는 서술의 위치 목록 (일반 지식 표지가 앞에 있거나 곧바로 부정되면 제외)."""
    return [m.start() for m in _NORMAL_DATA.finditer(body)
            if not _GENERAL_MARK.search(body[max(0, m.start() - 20): m.end()]) and not re.match(_NEGATED, body[m.end():])]


def _normal_data_claim(body: str) -> bool:
    """사용자 데이터가 '정상 범위/수준'이라는 서술이 있는가."""
    return bool(_normal_data_hits(body))


# "문제가 없어 보이더라도 상담을 권합니다": 양보절 뒤에 진료를 권하는 말이 이어지면 '이상 없음' 선언이 아니라 격려다.
_CONCEDED_THEN_CARE = re.compile(r".{0,12}?(?:더라도|어도|아도)\s+[^.\n]*(?:받으세요|받아\s*보세요|권합니다|권해|권장|상담|진료|확인|검사|병원|119)")


def _all_clear_hits(body: str) -> list[int]:
    """'이상 없음'(또는 데이터가 정상 범위라는 서술)으로 읽히는 서술의 위치 목록. 곧바로 부정·한정되면("~없다고 단정할 수 없습니다", "~없었다는 뜻은 아닙니다") 한계 서술이므로 제외한다."""
    return [m.start() for m in _ALL_CLEAR.finditer(body)
            if not re.match(_NEGATED, body[m.end():]) and not _CONCEDED_THEN_CARE.match(body[m.end():])] + _normal_data_hits(body)


def _all_clear_claim(body: str) -> bool:
    """'이상 없음'(또는 데이터가 정상 범위라는 서술)으로 읽히는 문장이 있는가."""
    return bool(_all_clear_hits(body))


# --- 근거 없는 맥락별 안심 ----------------------------------------------------------------
# 서비스는 운동·활동 중 심박을 평가하지 않는다(운동 중 상승은 탐지 대상이 아님). 따라서 "운동 중 심박은 정상/괜찮다"는 사용자 데이터에
# 근거가 없다. 수면·휴식은 해당 요약 통계(sleep_hr / resting_hr)가 있을 때만 그 맥락의 안심 서술을 허용한다.
_CONTEXT_STAT = {"운동": None, "활동": None, "걷": None, "달리": None, "수면": "sleep_hr", "잠잘": "sleep_hr", "휴식": "resting_hr"}
_CONTEXT_ASSURE = re.compile(
    r"괜찮(?:습니다|아요|았|다|으)|문제\s*(?:가\s*)?(?:없|ㅇ)|정상\s*범위\s*(?:내|안)?\s*에\s*(?:있|해당|속)|정상(?:이었|적이었|적입니다)|양호|안전(?:합니다|해요|했)"
)
_GENERAL_RISE = re.compile(r"상승|올라|오르|높아|증가|자연스|반응")   # '운동하면 심박이 올라도 괜찮다'는 일반 생리 지식은 허용


def unsupported_context_claims(body: str, stats: dict | None) -> list[str]:
    """요약 통계에 근거가 없는 맥락(운동/활동/수면/휴식)별 안심 문장을 찾는다.

    Args:
        body: 고정 문구를 제거한 본문.
        stats: `get_vitals_summary` 의 summary 블록. 수면은 sleep_hr, 휴식은 resting_hr 가 있어야 그 맥락의 안심 서술이 허용된다
            (None 이면 아무 맥락도 근거 없음). 운동·활동은 서비스가 평가하지 않아 항상 근거 없음.

    Returns:
        위반 문장 원문(최대 80자) 리스트.
    """
    out = []
    for sent in re.split(r"[.!?。\n]", body):
        if not sent.strip() or not _CONTEXT_ASSURE.search(sent) or _GENERAL_RISE.search(sent):
            continue
        for word, key in _CONTEXT_STAT.items():
            if word in sent and not (key and isinstance((stats or {}).get(key), (int, float))):
                out.append(sent.strip()[:80])
                break
    return out


# 에스컬레이션이 켜진 동안 진료·119 필요성을 깎아내리는 서술. 질병 진단이 아니라 '안전 안내를 약화'하는 문장이다.
_DOWNPLAY = re.compile(
    r"병원에\s*(?:갈|가실)\s*필요(?:는|가)?\s*없|119(?:에)?\s*(?:연락|신고)?\S{0,4}\s*필요(?:는)?\s*없|응급\s*(?:상황|증상)(?:은|이)?\s*아닙"
    r"|걱정(?:할|하실)?\s*일(?:은|이)?\s*아닙|괜찮은\s*편|문제(?:가)?\s*없어\s*보|양호(?:합니다|한)|조치(?:할|하실)?\s*(?:것|필요)(?:은|이|는)?\s*없"
    r"|큰\s*문제(?:는|가)?\s*(?:아니|없)|심각(?:한|하지)\s*(?:상황|문제)?(?:은|이)?\s*(?:아닙|않)|크게\s*걱정|안전(?:합니다|한\s*상태)"
)
# 진료를 받지 않아도 된다는 서술: 이상이 하나라도 감지된 뒤에는 항상 금지한다 (에스컬레이션 중에는 _DOWNPLAY 로도 막는다).
_NO_VISIT = re.compile(
    r"(?:병원|의원|응급실|의료기관|진료)(?:에|을|를|은|는)?\s*(?:굳이\s*)?(?:가지|가시지|방문하지|찾지|받지)\s*않(?:으셔도|아도|으시고도)\s*(?:됩니다|돼요|되|괜찮)"
    r"|(?:검사|상담|진료)(?:를|을)?\s*(?:굳이\s*)?받지\s*않(?:으셔도|아도)\s*(?:됩니다|돼요|되|괜찮)|굳이\s*(?:병원|진료|검사|상담)"
    r"|(?:병원|진료|검사|상담)(?:에|을|를|은|는|이)?[^.\n]{0,8}(?:필요\s*(?:없|하지\s*않)|불필요)"
)


# --- 구조적 차단: '진료·응급 용어 또는 심각도·우려 용어' + '부정·축소 표지' 가 같은 절에 있으면 어떤 표현이든 막는다 --------------
# 문구 목록(_DOWNPLAY·_NO_VISIT)은 새 어순·말투를 놓친다("119는 부르지 않으셔도 됩니다", "응급실은 안 가셔도 될 것 같아요", "걱정할 만한 구간은 없었습니다").
# 그래서 문구가 아니라 *구조*를 본다. 이상이 감지된 맥락(에스컬레이션·이상 목록이 있는 사용자)에서만 적용한다.
_CARE_TERMS = re.compile(r"119|응급|병원|진료|의사|의료진|의료기관|검사|검진|진찰|상담|내원|입원|치료|(?i:doctor|hospital|clinic|emergency|\bER\b)")
_CONCERN_TERMS = re.compile(r"심각|위험|위급|위중|걱정|우려|염려|괜찮|문제|이상|안전|급하|급박|시급|서두|별일|대수|신경|(?i:serious|worr|urgent|danger|concern|alarm)")
_NEGATION = re.compile(r"(?:않|불필요|없(?!이(?!도))|아니|아닙|말고|마세요|마십시오|말아|무방|굳이|(?<![가-힣])안\s|(?i:\bno\b|\bnot\b|n't|\bnothing\b))(?!으면|다면|더라도|을\s*때|을\s*경우)")
# 에스컬레이션 중에는 '지켜보라/쉬면 나아진다/그대로 지내도 된다' 식의 경과 관찰·방치 권유도 막는다 (부정어가 없어도 진료를 미루게 하는 말이다).
_WAIT_AND_SEE = re.compile(r"지켜보(?:기|시|면|세요|는|고)|쉬시면|쉬면\s*(?:나아|좋아)|그대로\s*(?:하셔도|지내|생활|유지)|무방|나아질|저절로|자연스럽게\s*(?:좋아|회복|나아)|하셔도\s*(?:충분|괜찮|됩니다|돼요)|(?i:wait and see|will (?:go away|pass|resolve))")
# 부정이 있어도 경고·한정 표현인 것: "안전하지 않을 수 있습니다", "~없다고 단정할 수 없습니다", "배제할 수 없습니다"
_HEDGED = re.compile(
    r"(?:망설이|주저하|미루|지체하|무시하|방치하|참|피하|꺼리|외면하)지\s*(?:마|말)\S*|없이[^.\n]{0,20}(?:마세요|마십시오|말아)|안전하지\s*않|괜찮지\s*않|좋지\s*않|배제할\s*수\s*없|(?:없|괜찮|안전|정상|아니)(?:다|하다|라)?(?:고|는\s*것)?\s*(?:단정|보장|확신|판단|말|볼|여길|생각)\S{0,6}\s*(?:수\s*)?(?:없|않|아닙|못|어렵)"
)
# 절 경계: 문장부호·줄바꿈, 그리고 뒤에 공백이 오는 접속 어미. 접속 어미는 앞 절에 남겨 "않으면" 같은 조건절을 앞 절 안에서 판정한다.
_CLAUSE_END = re.compile(r"[,;:·、\n.!?。]+|(?:더라도|다면|라면|으면|하면|되면|시면|지만|으나|니까|므로|는데|은데|면서)(?=\s)")


# 양보절("증상이 없어도 / 괜찮아 보여도 / 문제가 없어 보이더라도")은 뒤에 진료를 *권하는* 말이 오면 격려이지 축소가 아니다.
_CONCESSIVE = re.compile(r"(?:어도|아도|여도|더라도|라도)")
_ENCOURAGE = re.compile(
    r"받으세요|받아\s*보세요|받아보세요|받으시|받는\s*것이\s*좋|권합니다|권해|권장|권고|상담하세요|상담을\s*받|확인하세요|확인을\s*받|문의하세요|연락하세요|방문하세요|가세요|가\s*보세요|바랍니다"
)


def _concession_then_care(clause: str, following: str) -> bool:
    """양보절 뒤에 진료를 권하는 말이 이어지는가 ("증상이 없어도 진료를 받으세요", "문제가 없어 보이더라도 / 상담을 권합니다").

    양보절 자체에 진료 용어가 있으면("119는 부르지 않아도 …") 격려가 아니라 축소이므로 해당하지 않는다. 같은 절의 뒷부분이 있으면 그것을, 없으면 다음 절을 본다.
    """
    m = _CONCESSIVE.search(clause)
    if not m or _CARE_TERMS.search(clause[: m.end()]):
        return False
    rest = clause[m.end():] if clause[m.end():].strip(" ,.") else following
    return bool(_CARE_TERMS.search(rest) and _ENCOURAGE.search(rest) and not _NEGATION.search(rest))


def _clause_dismissal_pos(clause: str, wait_and_see: bool, care_only: bool = False) -> int | None:
    """한 절이 진료·응급 또는 심각도·우려를 부정·축소하는가: 그 표지의 위치, 아니면 None. wait_and_see 면 경과 관찰·방치 권유도 같은 취급."""
    c = _HEDGED.sub(lambda m: " " * len(m.group()), clause)   # 위치를 유지하려고 같은 길이의 공백으로 지운다
    if wait_and_see and (w := _WAIT_AND_SEE.search(c)):
        return w.start()
    if not (_CARE_TERMS.search(c) or (not care_only and _CONCERN_TERMS.search(c))):
        return None
    neg = _NEGATION.search(c)
    return neg.start() if neg else None


def dismisses_care(
    text: str, *, context: str = "", clean: frozenset[str] = frozenset(), wait_and_see: bool = False, care_only: bool = False,
) -> bool:
    """진료·응급·심각도·우려를 부정하거나 축소하는 절이 있는가 (표현과 무관한 구조 규칙).

    같은 절에 (진료·응급·병원·119·의사·검사·상담 등) 또는 (심각·위험·걱정·괜찮·문제·이상·안전 등)과 부정·축소 표지(않·없·안·아니·말고·굳이 …)가 함께 있으면 True.
    "증상이 없더라도 진료를 권합니다"는 절이 나뉘어 통과하고, "119는 부르지 않으셔도 됩니다"는 한 절이라 걸린다. 소유 사용자(바로 앞 ID)가 `clean`
    (이상이 없는 사용자)이면 그 절은 건너뛴다 - 비교 리포트에서 이상 없는 사용자에 대한 서술은 사실이다.

    Args:
        text: 검사할 문장 또는 본문 (고정 문구는 미리 제거).
        context: 이 text 앞에 이미 나온 본문 (스트리밍 가드가 제목 줄의 사용자 범위를 이어받게 한다).
        clean: 이상이 없는 사용자 ID 집합 (비교 요청일 때만).
        care_only: True 면 진료·응급 용어와 부정만 본다(심각도·우려 용어는 제외) - 이상 유무와 무관하게 적용하는 최소 규칙이다.
        wait_and_see: True 면 "지켜보세요/쉬시면 나아질 거예요/그대로 하셔도 무방" 같은 경과 관찰·방치 권유도 막는다 (에스컬레이션 중).
    """
    bounds, start = [], 0
    for m in [*_CLAUSE_END.finditer(text), None]:
        end = m.end() if m else len(text)
        bounds.append((start, end))
        start = end
    for i, (start, end) in enumerate(bounds):
        pos = _clause_dismissal_pos(text[start:end], wait_and_see, care_only)
        following = text[bounds[i + 1][0]: bounds[i + 1][1]] if i + 1 < len(bounds) else ""
        if pos is not None and not _concession_then_care(text[start:end], following) and _owner(context + text, len(context) + start + pos) not in clean:
            return True
    return False


# --- 허용 목록: 이상이 있는 맥락에서 LLM 본문은 *사실 서술*만 남긴다 ------------------------------------------------
# 금지 문구를 늘리는 방식은 새 말투에 계속 진다(조사·어순·완곡어법이 무한하다). 그래서 방향을 뒤집는다: 시각·지속·수치·유형·맥락·데이터 한계 같은 *사실*을
# 말하는 문장만 통과시키고 나머지는 지운다. 조언·평가·안심·진료 언급은 모두 코드가 쓴다(에스컬레이션 안내문, WARNING_GUIDANCE, 데이터 한계 안내).
_SEV_TOKEN = re.compile(r"\[(?:심각|주의|참고)\]|\((?:심각|주의|참고)\)|심각도\s*[:：]?\s*(?:심각|주의|참고)?|(?:심각|주의|참고)\s*(?:단계|등급)")   # 리포트가 쓰는 심각도 표기는 사실이다
_FACT_MARK = re.compile(
    r"\d|bpm|BPM|구간|감지|기록|측정|결측|커버리지|조회|평균|지속|분간|시간|기준선|추세|센서|임계값|데이터|휴식|수면|활동|빈맥|서맥|이탈|심박|이상\s*(?:구간|징후|항목|건)"
)
_ADVICE_END = re.compile(
    r"세요|십시오|십시요|바랍니다|바래요|권합니다|권해|권장|권고|좋습니다|좋겠|좋아요|면\s*됩니다|셔도|해도\s*(?:됩니다|돼|괜찮)|도\s*(?:됩니다|돼요|돼\b)|야\s*(?:합니다|해요|한다)"
    r"|할\s*것|하는\s*것이|수\s*있|듯|같아|같습니다|보입니다|보여요|보이네|해\s*보|거예요|겁니다|걸요|죠[.\s]|네요|군요|인\s*것\s*같"
)
_EVAL_STEM = re.compile(
    r"괜찮|충분|해결|나아|나을|좋아|좋은|흔[하한히]|일시적|정상|걱정|염려|우려|문제|지장|무리|급[하한박히]|필요|위험|안전|안심|심각|양호|다행|별일|별거|별\s*거|대수|신경"
    r"|가볍|경미|사소|무방|그냥|단순|대체로|대부분|보통|일반적|누구|자연|저절로|회복|호전|만하|정도는|수준은|수준입|추정|뿐|그쳤|범주|범위|돌아왔|돌아|짜리|(?i:fine|rest|normal|okay|probably|nothing|serious|harmless)"
)
_CARE_ALL = re.compile(_CARE_TERMS.pattern + r"|구급|방문|의료|전문의|약\s*복용|복약")
_LIMIT_WORD = re.compile(r"결측|끊|기록|측정|조회|커버리지|데이터|임계값|감지|탐지|센서|구간")
_BAN_HEADING = re.compile(r"권고|권장|대응|조치|조언|제안|추천|행동|해석|평가|결론|종합")
_LEADING_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


# 닫힌 어휘 검사: 사실 서술 문장의 모든 한글 어절은 (1) 고정 사실 어휘, (2) 이번 분석의 FACTS 에 실제로 나온 단어 중 하나로 시작해야 한다. 그 밖의 말("계단", "착용", "작은",
# "때문")이 하나라도 있으면 사실을 가장한 평가·원인 추정·상황 지어내기로 보고 문장을 지운다. 금지어를 늘리는 대신 *허용 어휘*를 닫아 두는 쪽이다.
_FUNCTION_TOKENS = frozenset({
    "한", "달", "이", "그", "각", "총", "약", "등", "및", "또", "중", "간", "내", "외", "전", "후", "시", "차", "째", "만", "도", "은", "는", "가", "을", "를", "의", "에", "로", "과", "와",
    "으로", "에서", "에는", "에서는", "까지", "부터", "씩", "마다", "이며", "이고", "이다", "으며", "며", "고", "입니다", "이었습니다", "였습니다", "이었", "였", "이", "인",
})
_UNIT_TOKEN = re.compile(r"^(?:분간|분|건|회|번|초|시간|시|일|주|개|보|곳)(?:은|는|이|가|을|를|의|에|에서|부터|까지|으로|로|과|와|도|씩|간|입니다|이며|이고|째|차)?$")
_FACT_LEXICON = tuple(sorted({
    # 측정·기록 어휘
    "심박", "심박수", "평균", "최고", "최저", "중앙", "기준선", "개인", "추세", "상승", "하락", "증가", "감소", "변화", "이탈", "지속", "시작", "종료", "시각", "시간대", "구간",
    "이상", "이벤트", "항목", "유형", "맥락", "심각도", "건수", "합계", "커버리지", "데이터", "기록", "측정", "조회", "기간", "결측", "센서", "신호", "끊", "누락", "걸음", "걸음수",
    "활동", "수면", "휴식", "안정", "빈맥", "서맥", "사용자", "비교", "공통", "차이", "동안", "사이", "오전", "오후", "새벽", "저녁", "아침", "낮", "밤", "월", "요일", "처음", "마지막",
    "가장", "이번", "최근", "지난", "해당", "전체", "일부", "임계값", "탐지", "감지", "발견", "관찰", "확인", "계산", "나타", "보였", "높", "낮았", "넘", "벗어", "유지", "발생",
    "포함", "있었", "있습", "없었", "없습", "않았", "않습", "않은", "않는", "않고", "않아", "않으", "없는", "없고", "없던", "되었", "됐", "되지", "됩니다", "되어", "하였", "했습", "합니다", "하며", "하고", "이루어", "대비", "폭", "수치", "값", "횟수",
    "분포", "비율", "퍼센트", "주의", "참고", "심각", "보다", "상태", "수준", "무렵", "올라", "올랐", "내려", "일별", "이어", "크게", "계속", "쉬는", "같은", "같이", "다시", "모두", "다만", "또한", "그리고", "그러나", "하지만", "반면", "이후", "이전", "직전",
    "평소", "변동", "기준", "중단", "결과", "분석", "일간", "누적", "연속", "반복", "횟수", "정확", "부족", "제한", "한계", "범위를", "개의", "여러", "하나", "두", "세", "네", "첫", "둘째",
}, key=len, reverse=True))
def _detector_vocab() -> frozenset[str]:
    """탐지기(`detector._evidence`)와 리포트 템플릿이 스스로 쓰는 고정 서술의 한글 단어 - 이 문구를 그대로 옮겨 쓴 문장이 어휘 검사에 걸려 지워지지 않게 한다.

    소스를 import 하지 않고 AST 로 읽는다(탐지기는 pandas 등을 끌고 오는 무거운 모듈이다). 읽기에 실패하면 빈 집합 - 고정 어휘만으로 동작한다.
    """
    import ast
    from pathlib import Path

    from app.agent import gates

    words = {w for d in (gates.TYPE_LABELS, gates.CONTEXT_LABELS, gates.SEVERITY_LABELS) for v in d.values() for w in re.findall(r"[가-힣]{2,}", v)}
    root = Path(__file__).resolve().parent.parent
    for rel, funcs in (("detection/detector.py", {"_evidence"}), ("agent/report.py", {"_template_body_one", "_data_limits_one"})):
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name in funcs:
                body = fn.body[1:] if ast.get_docstring(fn) else fn.body   # 독스트링의 설명 용어는 제외
                for node in body:
                    words.update(w for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str) for w in re.findall(r"[가-힣]{2,}", n.value))
    return frozenset(words)


_DETECTOR_VOCAB = _detector_vocab()
_GRAM = 6
_MIN_RUN = 13   # 겹침이 이 글자 수 이상 이어질 때만 '탐지기 문장을 옮겨 쓴 것'으로 인정한다 - 짧은 조각("일반적인 ")만 빌려 뒤에 평가를 붙이는 문장을 막는다


def evidence_grams(texts: list[str]) -> frozenset[str]:
    """탐지기가 만든 근거 문장(evidence)의 6글자 조각 집합 - 문장 안에서 이 조각과 겹치는 부분은 탐지기 자신의 서술이므로 평가어 검사에서 제외한다."""
    grams: set[str] = set()
    for t in texts:
        t = re.sub(r"\s+", " ", t)
        grams.update(t[i: i + _GRAM] for i in range(max(0, len(t) - _GRAM + 1)))
    return frozenset(grams)


def _covered(s: str, grams: frozenset[str]) -> list[bool]:
    """(공백을 하나로 줄인) 문장에서 탐지기 근거 문장과 6글자 이상 그대로 겹치는 글자 위치."""
    covered = [False] * len(s)
    for i in range(len(s) - _GRAM + 1 if grams else 0):
        if s[i: i + _GRAM] in grams:
            covered[i: i + _GRAM] = [True] * _GRAM
    for m in re.finditer(r"(?:T)+", "".join("T" if c else "F" for c in covered)):   # 짧게만 겹친 구간은 취소
        if len(s[m.start(): m.end()].strip()) < _MIN_RUN:
            covered[m.start(): m.end()] = [False] * (m.end() - m.start())
    return covered


def _mask_evidence(sentence: str, grams: frozenset[str]) -> str:
    """탐지기 근거 문장과 6글자 이상 그대로 겹치는 구간을 공백으로 가린다 (길이 유지)."""
    s = re.sub(r"\s+", " ", sentence)
    return "".join(" " if c else ch for ch, c in zip(s, _covered(s, grams)))


_LATIN_OK = re.compile(r"^(?:bpm|BPM|U\d+|S\d{3}|SpO2|HR)$")


def unknown_words(sentence: str, vocab: frozenset[str] = frozenset(), evidence: frozenset[str] = frozenset()) -> list[str]:
    """사실 서술이라고 보기에 *허용 어휘에 없는* 한글 어절과 영단어 목록 (비어 있으면 모두 허용 어휘).

    어절은 숫자·영문을 뺀 한글 덩어리이고, 고정 사실 어휘(2자 이상)나 `vocab`(이번 FACTS 에 나온 단어)으로 시작하거나 조사·단위 토큰이면 통과한다.
    영문은 bpm·사용자 ID 만 허용한다 ("harmless", "probably" 같은 평가어를 막는다).
    """
    t = re.sub(r"\s+", " ", _SEV_TOKEN.sub(" ", sentence))
    cov = _covered(t, evidence)
    bad = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", t) if not _LATIN_OK.match(w)]
    known = tuple(w for w in vocab | _DETECTOR_VOCAB if len(w) >= 2)
    for m in re.finditer(r"[가-힣]+", re.sub(r"[A-Za-z0-9.]", " ", t)):
        tok = m.group()
        if tok in _FUNCTION_TOKENS or _UNIT_TOKEN.match(tok) or all(cov[m.start(): m.end()]):   # 탐지기 자신의 서술에서 그대로 옮긴 어절은 통과
            continue
        if any(tok.startswith(w) for w in _FACT_LEXICON) or any(tok.startswith(w) for w in known):
            continue
        bad.append(tok)
    return bad


# --- 모든 경로 공통: LLM 이 쓰지 못하는 주제 (이상 유무와 무관) -------------------------------------------------------------------------------
# 진료·응급, 복약, 증상, 사람 전체의 건강 판정은 코드가 쓰는 고정 안내문(에스컬레이션·WARNING_GUIDANCE·NO_ANOMALY_GUIDANCE)의 몫이다. LLM 문장에 이런 말이 있으면
# 어떤 말투든 *문장째 지운다* (금지 문구를 하나씩 막는 방식은 새 말투에 진다). 영문 문장도 같다.
_SENSITIVE = re.compile(
    r"병원|진료|의사|의료진|의료기관|응급|119|구급|검사|검진|상담|내원|입원|치료"
    r"|(?<![가-힣])약(?=[을은는이가도과와의에]|\s*(?:복용|물|드|먹|끊|줄|중단|바꾸))|약물|복용|처방|혈압약|진통제|아스피린|영양제|타이레놀"
    r"|증상|아프|통증|두근|어지럼|어지러|숨\s*참|숨\s*찬|숨차|숨\s*가쁘|가쁜|숨이|숨을|호흡|가슴|실신|쓰러"
    r"|건강(?:하|합|한|은|이|을)|건강\s*(?:상태|문제)|문제(?:는|가|도)?\s*없|심장|걱정|염려|안심|완벽|무방|(?i:healthy|doctor|hospital|medic)"
)


# 양보/허가 활용형 자체 (-셔도/-해도/-어도/-아도/-여도/-와도/-워도/-러도/-져도/-겨도): 뒤에 무엇이 오든 LLM 문장에서 지운다. "~해도 좋아요", "~셔도 문제 될 건 없어요" 처럼
# 허가를 건네는 말투는 끝말이 무한하므로 끝말을 쫓지 않고 *활용형*을 막는다. 일반 지식 문장이 함께 지워지는 것은 안전한 쪽의 비용이며 고정 안내문이 대신한다.
# 주의: 코드가 쓰는 고정 안내("보이지 않아도 …")는 이 필터 바깥(strip_fixed 후 필터링, 필터 후 compose)에서 붙으므로 걸리지 않는다. validate_rules 에는 넣지 않는다.
_CONCESSIVE_ANY = re.compile(r"[가-힣](?:셔도|해도|어도|아도|여도|와도|워도|러도|져도|겨도)")


# 조언·예측은 코드의 몫이다: LLM 은 서술문만 쓴다. 명령·권유 서술어와 앞날을 단정하는 예측·안심 서술어가 든 문장은 어떤 내용이든 지운다.
# 예외는 없다(안녕하세요 제외): 절제 표현을 예외로 두면 "천천히 소주 한 병 드세요" 같은 우회가 된다. 코드가 쓰는 고정 안내는 이 필터 바깥에서 붙는다.
_DIRECTIVE = re.compile(
    r"(?<!안녕하)세요|십시오|십시요|(?:하라|하자|보자|가자|합시다|바랍니다|권합니다|권장합니다|권장해요|권해요|추천합니다|추천해요|좋겠습니다|해야\s*(?:합니다|해요|한다)|하시면\s*됩니다|하시길)(?![가-힣])"
)
_PREDICTION = re.compile(r"거예요|거에요|겁니다|것입니다|것이다|테니|거라고|거야\b|질\s*거")


def is_directive_or_prediction(sentence: str) -> bool:
    """명령·권유 서술어이거나 앞날을 단정하는 예측·안심 서술어가 든 문장인가 (예외 없음)."""
    t = _SEV_TOKEN.sub(" ", sentence)
    return bool(_PREDICTION.search(t) or _DIRECTIVE.search(t))


def mentions_sensitive(sentence: str) -> bool:
    """문장이 진료·응급·복약·증상·건강 판정을 언급하는가 (LLM 이 쓸 수 없는 주제; 안내는 코드가 쓴다)."""
    return bool(_SENSITIVE.search(_SEV_TOKEN.sub(" ", sentence)))


def is_english_sentence(sentence: str) -> bool:
    """bpm·사용자 ID 밖의 영단어(3자 이상)가 있는가 - 영어 조언·평가는 어디서나 지운다."""
    return any(not _LATIN_OK.match(w) for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", sentence) if len(w) >= 3)


def drop_general(sentence: str, facts: list[float] | None = None) -> bool:
    """모든 경로의 공통 삭제 규칙: 민감 주제, 영어 문장, 근거 없는 수치가 있으면 True (제목 줄은 대상이 아니다)."""
    t = sentence.strip()
    if not t or t.startswith("#"):
        return False
    return (mentions_sensitive(t) or is_english_sentence(t) or unsafe_everywhere(t) or bool(_CONCESSIVE_ANY.search(t)) or is_directive_or_prediction(t)
            or bool(facts is not None and unsupported_numbers(t, facts)))


def sanitize_general(text: str, facts: list[float] | None = None) -> str:
    """이상이 없는 리포트·일반 답변·후속 답변의 LLM 본문에서 `drop_general` 에 걸리는 문장을 지운다 (구조·제목 유지, 일반 지식 문장은 남는다)."""
    out = []
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        lead = _LEADING_MARK.match(line)
        prefix = lead.group(0) if lead else ""
        kept = [s for s in _SENT_SPLIT.split(line[len(prefix):]) if not drop_general(s, facts)]
        out.append(prefix + " ".join(kept) if kept else "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def has_advice_or_evaluation(sentence: str) -> bool:
    """문장이 조언·권유·평가·안심·진료 언급을 담고 있는가 (심각도 표기 `[심각]` 등은 사실로 보고 제외)."""
    t = _SEV_TOKEN.sub(" ", sentence)
    return bool(_ADVICE_END.search(t) or _EVAL_STEM.search(t) or _CARE_ALL.search(t))


def is_fact_narration(
    sentence: str, vocab: frozenset[str] = frozenset(), facts: list[float] | None = None, evidence: frozenset[str] = frozenset(),
) -> bool:
    """허용 목록 판정: 사실(시각·지속·수치·유형·맥락·데이터 한계)을 서술하고 조언·평가·진료 언급이 없는 문장인가.

    조건: (1) 조언/권유/평가/안심/진료 표지가 없고, (2) 사실 표지(숫자·bpm·구간·감지·측정·유형명 등)가 있으며, (3) 부정 표현(않·없·아니·못)은 데이터 한계
    서술(결측·기록·측정·감지 …)에서만 허용하고, (4) 모든 어절이 허용 어휘(고정 사실 어휘 + 이번 FACTS 에 나온 단어 `vocab`)에 속한다 - 사실의 모양을 한
    원인 추정·평가("착용 위치 때문에", "작은 편")를 막는다. 제목 줄은 짧고 평가가 없는 것만 통과한다. 알려진 말투가 아니라 *사실처럼 보이는 것만* 허용하는 방식이라
    처음 보는 완곡어법도 통과하지 못한다.
    """
    t = sentence.strip()
    if not t or not re.search(r"[가-힣A-Za-z0-9]", t):
        return True   # 빈 줄·구분선
    if t.startswith("#"):
        h = t.lstrip("#").strip()
        return len(h) <= 30 and not _BAN_HEADING.search(h) and not has_advice_or_evaluation(h)
    t = _LEADING_MARK.sub("", t)
    m = _mask_evidence(t, evidence)   # 탐지기 자신의 서술과 겹치는 부분은 평가어 검사에서 제외 (단 추측·권유 어미는 제외하지 않는다: 근거 문장을 다른 구간에 옮겨 붙이는 재귀속을 막는다)
    if _CONCESSIVE_ANY.search(t) or _ADVICE_END.search(_SEV_TOKEN.sub(" ", t)) or has_advice_or_evaluation(m) or mentions_sensitive(m) or not _FACT_MARK.search(_SEV_TOKEN.sub(" ", t)) or unknown_words(t, vocab, evidence):
        return False
    if facts is not None and unsupported_numbers(t, facts):
        return False   # 도구 결과에 없는 수치("140 bpm에 그쳤습니다")는 스트리밍 단계에서도 내보내지 않는다
    if not re.search(r"[가-힣]", t):
        return False   # 한글 없는 문장(영어 조언 등)은 사실 서술로 보지 않는다
    return not (re.search(r"않|없|아니|아닙|못|말[고아라]", m) and not _LIMIT_WORD.search(t))


def is_allowed_sentence(
    sentence: str, *, context: str = "", clean: frozenset[str] = frozenset(), vocab: frozenset[str] = frozenset(), facts: list[float] | None = None,
    evidence: frozenset[str] = frozenset(),
) -> bool:
    """이상이 있는 맥락에서 이 문장을 내보내도 되는가: 사실 서술이거나, 이상이 없는 사용자에 관한 문장(비교 요청)이면 True.

    이상 없는 사용자의 문장은 허용 목록을 면제하고 기존 규칙(안심 차단·사용자별 모순 검사)만 받는다.
    """
    if is_fact_narration(sentence, vocab, facts, evidence):
        return True
    ids = {i.upper() for i in _UID.findall(sentence)}
    return bool(clean) and bool(ids) and ids <= clean and _owner(context + sentence, len(context) + len(sentence)) in clean and not drop_general(sentence, facts)


_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def narration_only(
    text: str, *, context: str = "", clean: frozenset[str] = frozenset(), vocab: frozenset[str] = frozenset(), facts: list[float] | None = None,
    evidence: frozenset[str] = frozenset(),
) -> str:
    """본문에서 사실 서술 문장만 남긴다 (줄바꿈·글머리표·제목 구조는 유지, 내용이 모두 지워진 제목 줄은 함께 지운다).

    Args:
        text: 고정 문구를 제거한 LLM 본문.
        context: text 앞에 이미 나온 본문.
        clean: 이상이 없는 사용자 ID (비교 요청일 때만; 그 사용자 문장은 면제).
        vocab: 이번 분석의 FACTS 에 나온 단어 (닫힌 어휘 검사의 동적 허용 어휘).
        facts: 인용이 허용된 수치 집합 - 주어지면 근거 없는 수치가 있는 문장도 지운다.
        evidence: 탐지기 근거 문장의 6글자 조각 집합(`evidence_grams`) - 겹치는 부분은 평가어 검사에서 제외한다.

    Returns:
        사실 서술만 남은 본문. 모두 지워졌으면 빈 문자열.
    """
    out: list[str] = []
    seen = context
    for line in text.split("\n"):
        if not line.strip():
            out.append(line)
            seen += line + "\n"
            continue
        if line.lstrip().startswith("#"):
            keep = is_fact_narration(line)
            out.append(line if keep else "")
            seen += line + "\n"
            continue
        lead = _LEADING_MARK.match(line)
        prefix = lead.group(0) if lead else ""
        kept = []
        for sent in _SENT_SPLIT.split(line[len(prefix):]):
            if is_allowed_sentence(sent, context=seen, clean=clean, vocab=vocab, facts=facts, evidence=evidence):
                kept.append(sent)
            seen += sent + " "
        seen += "\n"
        out.append(prefix + " ".join(kept) if kept else "")
    # 내용 없는 제목 줄 제거
    lines, result = out, []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not nxt or nxt.lstrip().startswith("#"):
                continue
        result.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


# 이상 유무와 무관하게 항상 금지: 증상이 있어도 괜찮다는 말, 운동·음주 등의 허가, 진료·응급 부정 (이상이 없는 경로의 누수)
_SYMPTOM = re.compile(r"증상|통증|아프|가슴|어지럼|어지러|호흡|숨|실신|두근|쓰러")
_SYMPTOM_BRUSH_OFF = re.compile(r"(?:아도|어도|더라도|여도)[^.\n]{0,25}(?:넘기|무시|괜찮|무방|(?<!안\s)(?<!안)됩니다|(?<!안\s)(?<!안)돼요|관계\s*없|상관\s*없|문제\s*없|아닙|아니)")
_CLEARANCE = re.compile(
    r"(?:운동|활동|훈련|달리기|조깅|러닝|등산|수영|헬스|자전거|음주|술|맥주|와인|소주|커피|카페인|담배|흡연|사우나)[^.\n]{0,20}"
    r"(?:하셔도|드셔도|마셔도|피우셔도|계속하셔도|계속해도|해도)\s*(?:무방|됩니다|돼요|괜찮|상관\s*없)"
)
# 일반화된 허가 어미 ("~하셔도 됩니다/상관없어요"): 의학적 허가·미루기·넘기기를 건네는 말투 전반
# 구조 규칙: 어떤 동사든 -어도/-아도/-셔도/-해도 뒤에 됩니다·돼요·괜찮·상관없·무방·문제없 이 오는 허가 문법은 일반 지식에 필요 없다 (활동·음료 목록에 기대지 않는다).
_PERMISSION = re.compile(r"[가-힣]도\s*(?:됩|돼|괜찮|좋|상관\s*없|무방|문제\s*(?:될\s*(?:건|것은?)\s*)?없|지장\s*없|무리\s*없)|상관\s*없(?:어요|습니다)|그냥\s*(?:두세요|넘어가|지나가)|참고\s*넘어가")


def unsafe_everywhere(text: str) -> bool:
    """이상이 감지되지 않은 경우에도 언제나 허용하지 않는 서술인가: 진료·응급 부정, 증상을 넘기라는 말, 운동·음주 허가.

    일반 지식("운동 중 심박 상승은 정상입니다")은 건드리지 않도록 심각도·우려 용어 기반 규칙(`dismisses_care` 전체)은 쓰지 않고 *진료·응급 용어 + 부정*만 본다.
    """
    return bool(
        dismisses_care(text, care_only=True)
        or any(_SYMPTOM.search(c) and _SYMPTOM_BRUSH_OFF.search(c) for c in re.split(r"[.!?。\n]", text))
        or _CLEARANCE.search(text)
        or _PERMISSION.search(text)
    )


def dismisses_visit(text: str) -> bool:
    """진료·검사를 받을 필요가 없다는 서술이 있는가 (이상이 감지된 뒤에는 금지)."""
    return bool(_NO_VISIT.search(text))


def downplays_care(text: str) -> bool:
    """진료·119 의 필요성을 낮추는 서술이 있는가 (에스컬레이션 활성 중에는 금지)."""
    return any(not _CONCEDED_THEN_CARE.match(text[m.end():]) for rx in (_DOWNPLAY, _NO_VISIT) for m in rx.finditer(text))


# "임계값을 넘는 구간이 감지되지 않았습니다": 임계값 초과 이상(센서 결측 제외)이 있으면 거짓이다.
_NO_FINDING_RE = re.compile(r"임계값을\s*넘는\s*구간[이가]?\s*(?:감지|발견)되지\s*않")


def contradicts_findings(
    text: str, *, any_anomalies: bool, threshold_anomalies: bool, no_anomaly_users: frozenset[str] = frozenset(),
    no_threshold_users: frozenset[str] = frozenset(), context: str = "",
) -> bool:
    """이상이 있는데 '없다'고 말하는가: 이상(센서 결측 포함)이 있으면 '이상 없음' 계열, 임계값 초과 이상이 있으면 '구간이 감지되지 않았다'도 모순이다.

    여러 사용자 비교에서는 *사용자별*로 본다: 서술이 가리키는 사용자(바로 앞에 언급된 ID, `_owner`)가 이상이 없는 사용자라면 그 서술은 사실이다.
    ("U01 은 이상이 없고 U05 는 3건" 은 통과, "U01 과 U05 모두 이상이 없습니다" 는 U05 때문에 반려.) 소유자를 알 수 없는 서술은 전체 기준으로 본다.

    Args:
        no_anomaly_users / no_threshold_users: 이상이 하나도 없는 / 임계값 초과 이상이 없는 사용자 ID (비교 요청일 때만 채운다).
        context: 이 text 앞에 이미 나온 본문 (문장 단위 스트리밍 가드가 제목 줄의 사용자 범위를 이어받게 한다).
    """
    def contradicted(hits: list[int], clean: frozenset[str]) -> bool:
        """서술 위치 중 하나라도 이상이 있는 사용자(또는 소유자 불명)에 관한 것인가."""
        return any(_owner(context + text, len(context) + pos) not in clean for pos in hits)

    return (any_anomalies and contradicted(_all_clear_hits(text), no_anomaly_users)) or (
        threshold_anomalies and contradicted([m.start() for m in _NO_FINDING_RE.finditer(text)], no_threshold_users))


def blocked_sentence(
    sentence: str, *, escalation: bool, any_anomalies: bool, threshold_anomalies: bool | None = None, no_anomaly_users: frozenset[str] = frozenset(),
    no_threshold_users: frozenset[str] = frozenset(), context: str = "",
) -> bool:
    """스트리밍 가드와 검증기가 공유하는 문장 단위 차단 규칙.

    진단 단정·복약 지시는 항상, 에스컬레이션 중에는 진료 필요성 축소, 이상이 하나라도 있으면 '이상 없음'·'진료 불필요' 서술을 막는다.
    """
    return bool(
        diagnostic_assertions(sentence)
        or unsafe_everywhere(sentence)
        or (escalation and downplays_care(sentence))
        or (any_anomalies and dismisses_visit(sentence))
        or ((escalation or any_anomalies) and dismisses_care(sentence, context=context, clean=no_anomaly_users, wait_and_see=escalation))
        or contradicts_findings(
            sentence, any_anomalies=any_anomalies, threshold_anomalies=any_anomalies if threshold_anomalies is None else threshold_anomalies,
            no_anomaly_users=no_anomaly_users, no_threshold_users=no_threshold_users, context=context)
    )


NO_FINDING_SENTENCE = "이번 기간 데이터에서는 임계값을 넘는 구간이 감지되지 않았습니다."
DETECTION_FAILED_SENTENCE = "이상 탐지를 실행하지 못해 이상 여부를 판단할 수 없습니다."


def repair_all_clear(body: str, detection_ok: bool) -> str:
    """한계가 있는 데이터에서 LLM 이 쓴 '이상 없음' 문장을 사실 진술 문장으로 *결정론적으로* 바꾼다.

    8b 모델이 "정확히 이 문장을 쓰라"는 지시를 자주 무시해(측정: 저커버리지 2건 모두 재시도 후 템플릿 대체) 거짓 안심 문장 때문에
    리포트 전체를 버리는 일이 잦았다. 그 문장만 코드가 알려진 안전한 문장으로 교체하면 나머지 서술은 살린다.
    교체 대상은 `_all_clear_claim` 이 걸리는 문장(줄 안에서 문장 단위)뿐이고, 한계가 없는 데이터에는 호출하지 않는다.

    Args:
        body: LLM 본문 (고정 문구 제거 후).
        detection_ok: 탐지 성공 여부. False 면 "판단할 수 없습니다" 문장으로 교체한다.
    """
    replacement = NO_FINDING_SENTENCE if detection_ok else DETECTION_FAILED_SENTENCE
    lines = []
    for line in body.split("\n"):
        segments = re.split(r"(?<=[.!?。])\s+", line)
        lines.append(" ".join(replacement if _all_clear_claim(seg) else seg for seg in segments))
    return "\n".join(lines)


# --- 지표 의미 혼동 검출 ---------------------------------------------------------------------------
# 지표값(분 단위)과 같은 숫자가 '의미가 다른 단어' 바로 뒤에 붙으면 오독이다. 예: n_minutes(심박 기록 분수)를 '활동 시간'으로 서술.
_METRIC_FORBIDDEN = {
    "n_minutes": ("활동", "운동", "휴식", "수면"),
    "active_minutes": ("휴식", "수면", "측정", "기록"),
    "rest_minutes": ("활동", "운동", "수면", "측정", "기록"),
}
_LABELED_MIN_RE = re.compile(r"(활동|운동|휴식|수면|측정|기록)[^0-9\n]{0,12}?(\d+(?:\.\d+)?)\s*(?:분|시간)")


def metric_mismatches(report: str, stats: dict | None) -> list[str]:
    """요약 통계의 분 단위 지표를 엉뚱한 라벨과 짝지어 서술했는지 검사한다.

    Args:
        report: 검사할 리포트 본문.
        stats: `get_vitals_summary` 의 summary 블록 (n_minutes/active_minutes/rest_minutes 사용).

    Returns:
        이슈 문자열 리스트 (없으면 빈 리스트).
    """
    out: list[str] = []
    for m in _LABELED_MIN_RE.finditer(report):
        word, num = m.group(1), float(m.group(2))
        for key, banned in _METRIC_FORBIDDEN.items():
            v = (stats or {}).get(key)
            if isinstance(v, (int, float)) and v >= 10 and abs(num - v) <= 0.5 and word in banned:
                out.append(f"지표 의미 혼동: {key}={v:g} 를 '{word}' 와 연결해 서술했습니다")
    return out


# --- 종합 -----------------------------------------------------------------------------------------
def validate_rules(
    report: str, facts: list[float], *, escalation: bool, escalation_message: str, disclaimer: str = DISCLAIMER, stats: dict | None = None,
    notice: str = "", detection_ok: bool = True, limited_data: bool = False, limit_notice: str = "", any_anomalies: bool = False,
    threshold_anomalies: bool | None = None, no_anomaly_users: frozenset[str] = frozenset(), no_threshold_users: frozenset[str] = frozenset(),
) -> list[str]:
    """규칙 기반 검증 (결정론적, 항상 실행).

    Args:
        report: 검사할 리포트 전문 (고정 문구 포함).
        facts: `collect_facts` 결과.
        escalation: 에스컬레이션이 켜져 있는가 (켜졌다면 안내문 포함 필수).
        escalation_message: 고정 에스컬레이션 안내문.
        disclaimer: 필수 면책 문구. 빈 문자열이면 면책 검사를 건너뛴다 (simple 답변 본문만 검사할 때).
        stats: 요약 통계 (지표 의미 혼동 검사용).
        notice: 진단 요청에 대한 거절 안내문. 비어 있지 않으면 리포트에 포함되어야 한다 (코드가 붙이므로 안전망 검사).
        detection_ok: 이상 탐지가 성공했는가. False 인데 "이상 없음"으로 읽히는 서술이 있으면 거짓 안심으로 지적한다.
        threshold_anomalies: 임계값 초과 이상(센서 결측 제외)이 있는가. None 이면 any_anomalies 와 같다.
        any_anomalies: 이상이 하나라도 감지되었는가. True 면 '이상 없음' 서술은 사실과 모순이므로 반려한다 (교체하지 않는다).
        limit_notice: 코드가 강제하는 데이터 한계 안내문. 비어 있지 않으면 리포트에 포함되어야 한다 (안전망 검사).
        no_anomaly_users / no_threshold_users: 비교 요청에서 이상(임계값 초과 이상)이 없는 사용자 ID - 그 사용자에 관한 '이상 없음' 서술은 허용한다 (사용자별 판정).
        limited_data: 데이터가 제한적인가(낮은 커버리지·센서 결측·휴식 구간 부족). True 인데 "이상 없음" 서술이 있으면 지적한다.

    Returns:
        지적사항 문자열 리스트. 비어 있으면 통과.
    """
    issues: list[str] = []
    if disclaimer and disclaimer not in report:
        issues.append("면책 문구가 없습니다")
    if escalation and escalation_message not in report:
        issues.append("에스컬레이션 안내문이 리포트에 없습니다")
    if notice and notice not in report:
        issues.append("진단 거절 안내문이 리포트에 없습니다")
    if limit_notice and limit_notice not in report:
        issues.append("데이터 한계 안내문이 리포트에 없습니다")
    body = report
    for fixed in (disclaimer, escalation_message, notice, limit_notice):  # 코드가 붙인 고정 문구는 숫자·문구 검사에서 제외
        if fixed:
            body = body.replace(fixed, " ")
    if bad := unsupported_numbers(body, facts):
        issues.append("tool 결과에 없는 수치 인용: " + ", ".join(f"{b:g}" for b in bad))
    for s in diagnostic_assertions(body):
        issues.append(f"진단 단정 표현: '{s}'")
    if unsafe_everywhere(body):
        issues.append("진료·응급을 부정하거나 증상을 넘기라고 하거나 운동·음주를 허가하는 서술이 있습니다 - 대응 안내는 시스템이 고정 문구로 붙입니다")
    issues.extend(metric_mismatches(body, stats))
    if not detection_ok and _all_clear_claim(body):
        issues.append("이상 탐지가 실패했는데 '이상 없음'으로 읽히는 서술이 있습니다")
    elif contradicts_findings(
        body, any_anomalies=any_anomalies, threshold_anomalies=any_anomalies if threshold_anomalies is None else threshold_anomalies,
        no_anomaly_users=no_anomaly_users, no_threshold_users=no_threshold_users,
    ):
        issues.append("이상이 감지되었는데 '이상 없음/구간이 감지되지 않았다'로 읽히는 서술이 있습니다")
    elif limited_data and _all_clear_claim(body):
        issues.append("데이터가 제한적인데(낮은 커버리지·센서 결측) '이상 없음'으로 읽히는 서술이 있습니다 — '임계값을 넘는 구간이 감지되지 않았다'와 한계를 함께 쓰세요")
    for s in unsupported_context_claims(body, stats):
        issues.append(f"근거 없는 맥락별 안심 서술: '{s}' — 운동·활동 중 심박은 평가하지 않으며 수면·휴식은 해당 요약 통계가 있을 때만 말할 수 있습니다")
    if escalation and downplays_care(body):
        issues.append("에스컬레이션 중인데 진료·119 필요성을 낮추는 서술이 있습니다")
    elif any_anomalies and dismisses_visit(body):
        issues.append("이상이 감지되었는데 진료·검사가 필요 없다는 서술이 있습니다")
    elif (escalation or any_anomalies) and dismisses_care(body, clean=no_anomaly_users, wait_and_see=escalation):
        issues.append("이상이 감지되었는데 진료·응급·심각도를 부정하거나 낮추는 서술이 있습니다 - 대응 안내는 시스템이 고정 문구로 붙입니다")
    if m := _REASSURANCE.search(body):
        issues.append(f"단정적 안심 표현: '{m.group()}' — 데이터로 건강을 보증할 수 없습니다")
    if m := _RAW_CODE.search(body):
        issues.append(f"내부 코드 노출: '{m.group()}' — 한국어 표기를 쓰세요")
    return issues


class JudgeVerdict(BaseModel):
    """judge LLM 의 구조화 응답. passed=False 이면 issues 에 짧은 사유."""

    passed: bool
    issues: list[str] = []


_JUDGE_SYSTEM = (
    "당신은 건강 리포트 검수자입니다. 리포트가 (1) FACTS 에 없는 수치나 사실을 지어내지 않았는지, "
    "(2) 질병을 단정 진단하지 않고 가능성/진료 권고 수준으로만 말하는지 확인하세요. "
    '문제가 없으면 {"passed": true, "issues": []}, 문제가 있으면 passed=false 와 짧은 issues 를 JSON 으로 답하세요.'
)


async def judge(llm: LLM, report: str, facts_text: str) -> list[str]:
    """보조 LLM judge (선택). 규칙을 통과한 리포트를 fast 모델이 한 번 더 본다.

    Args:
        llm: LLM 구현.
        report: 검사할 리포트.
        facts_text: 모델에게 보여줄 사실 요약.

    Returns:
        지적사항 리스트. LLM 비활성·호출 실패·설정 off 면 빈 리스트 (규칙 검증만 신뢰하고 judge 장애가 서비스를 막지 않게 한다).
    """
    if not (llm.enabled and settings.validator_judge):
        return []
    v = await llm.structured("fast", "validator_judge", JudgeVerdict, _JUDGE_SYSTEM, f"FACTS:\n{facts_text}\n\nREPORT:\n{report}")
    return [] if v is None or v.passed else [f"judge: {i}" for i in (v.issues or ["판정 실패"])]
