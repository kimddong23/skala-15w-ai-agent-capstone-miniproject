"""guidelines MCP 서버 (FastMCP, stdio): 정적 가이드라인 Resource 와 의미 검색 Tool(RAG).

책임 경계:
    가이드라인 문서(`guidelines/*.md`)를 Resource 3종(전문 로드용)과 `search_guidelines` Tool(대목 검색)로 제공한다.
    사용자 데이터에는 접근하지 않는다 (그 반대편이 `vitals_server.py`). 임베딩/검색은 `app.rag` 에 위임한다.

설계 근거:
    - 문서 전문은 Resource(바뀌지 않는 정적 문서), 질문별 근거 대목은 Tool(질의마다 결과가 다름).
    - 서버를 분리해 클라이언트가 `list_tools/list_resources` 로 이름을 탐색하게 하고, 이 서버가 죽어도 데이터 조회는 계속된다 (부분 성능 저하).
    - 임베딩(Ollama)은 첫 검색 때만 지연 접속한다. 서버 기동은 Ollama 와 무관하다.
    - 독립 subprocess + stdio: `.venv/bin/python -m app.mcp_server.guidelines_server` (폴더 `guidelines/` 와 이름이 겹치지 않게 `_server` 접미).

협력 모듈: `app.rag.search`(검색), `guidelines/*.md`(원문), `app.agent.toolbox`(클라이언트).
"""
from pathlib import Path

from fastmcp import FastMCP

GUIDE_DIR = Path(__file__).parent / "guidelines"

mcp = FastMCP(
    "guidelines",
    instructions=(
        "심박수 관련 일반 건강 가이드라인 문서 서버입니다. 질문에 맞는 근거는 search_guidelines 로 찾고, "
        "리포트 면책 문구는 guideline://disclaimer 를 따르세요. 진단하지 않습니다."
    ),
)


MAX_SEARCH_K = 8


@mcp.tool(annotations={"readOnlyHint": True})
def search_guidelines(query: str, k: int = 3) -> dict:
    """심박수·건강 가이드라인 문서(약 20개)에서 질문과 가장 관련 있는 대목을 의미 기반(RAG)으로 찾아 돌려준다.

    일반 지식 질문("안정시 심박수 정상 범위는?", "카페인이 심박에 영향을 주나요?", "센서 결측은 왜 생겨요?", "언제 병원에 가야 하나요?")에
    답할 근거를 가져올 때 쓴다. 특정 사용자의 수치·이상 여부는 이 도구가 아니라 get_vitals_summary / detect_anomalies 로 조회한다.
    답변에는 hits[].text 의 내용만 근거로 쓰고, 여러 문서가 걸리면 doc_id 와 title 로 출처를 밝힌다. 진단하지 말 것.

    반환: {query, method('vector'=임베딩 검색 | 'keyword'=임베딩을 못 써서 키워드로 대체), note(대체 사유, 없으면 빈 문자열),
    hits:[{doc_id, title(문서 제목), heading(소제목), text(문장 경계에서 300자 이하로 줄인 근거 발췌; 잘렸으면 끝에 …), full_text(청크 전문 - 더 읽어야 할 때만 쓰고 화면에 그대로 내보이지 말 것), score(vector 면 코사인 유사도, keyword 면 단서 일치 비율),
    source(출처 표기)}]} — hits 는 점수 내림차순이며 같은 문서의 다른 대목이 여러 개 나올 수 있다. 입력이 잘못되면 {"error": 사유}.

    Args:
        query: 자연어 질문 (한국어, 영어 혼용 가능). 비어 있으면 안 된다.
        k: 돌려줄 대목 수. 1 이상 8 이하, 기본 3.
    """
    from app.rag.search import get_search   # 지연 import: 서버 기동 시 Ollama 에 접속하지 않는다

    if not query or not query.strip():
        return {"error": "query 가 비어 있습니다"}
    if not 1 <= k <= MAX_SEARCH_K:
        return {"error": f"k 는 1 이상 {MAX_SEARCH_K} 이하여야 합니다 (받은 값: {k})"}
    r = get_search().search(query.strip(), k)
    return {"query": r.query, "method": r.method, "note": r.note, "hits": [h.to_dict() for h in r.hits]}


def _doc(name: str) -> str:
    """guidelines/{name}.md 원문을 UTF-8 로 읽는다. 호출마다 읽으므로 문서 수정이 재기동 없이 반영된다."""
    return (GUIDE_DIR / f"{name}.md").read_text(encoding="utf-8")


@mcp.resource("guideline://heart-rate-ranges", mime_type="text/markdown")
def heart_rate_ranges() -> str:
    """성인 심박수 정상 범위, 이 서비스의 이상 유형별 탐지 기준, 운동 중 심박 해석 (정적 문서)."""
    return _doc("heart-rate-ranges")


@mcp.resource("guideline://red-flags", mime_type="text/markdown")
def red_flags() -> str:
    """119/응급실이 필요한 위험 신호와 진료 권고 기준, 심각도 등급별 권장 행동 (정적 문서)."""
    return _doc("red-flags")


@mcp.resource("guideline://disclaimer", mime_type="text/markdown")
def disclaimer() -> str:
    """리포트에 반드시 포함해야 하는 면책 문구 (정적 문서)."""
    return _doc("disclaimer")


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
