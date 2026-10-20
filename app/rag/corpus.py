"""가이드라인 코퍼스 로더.

책임 경계:
    `app/mcp_server/guidelines/*.md` 를 읽어 문서 단위(Doc)로 만든다. 청킹/임베딩/검색은 다른 모듈의 책임이다.

협력 모듈: `chunking.py`(Doc 소비), `search.py`(코퍼스 위치 기본값), `eval/rag_eval.py`(정답 문서 ID).

데이터 가정:
    - 문서는 UTF-8 markdown 이다. 선택적 front-matter(`---` 로 감싼 `key: value` 줄: id, title, topics, source)를 가진다.
      front-matter 가 없는 기존 문서(heart-rate-ranges, red-flags, disclaimer)는 파일명을 id 로, 첫 `# ` 제목을 title 로 쓴다.
      (MCP Resource 로 그대로 서빙되는 문서라 front-matter 를 강제하지 않는다 - 면책 문구 등 원문 보존.)
    - topics 는 `[a, b, c]` 또는 쉼표 구분 문자열이다.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

GUIDELINES_DIR = Path(__file__).resolve().parent.parent / "mcp_server" / "guidelines"
DEFAULT_SOURCE = "서비스 가이드라인"
_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Doc:
    """가이드라인 문서 하나.

    Attributes:
        id: 문서 식별자 (front-matter id 또는 파일명).
        title: 문서 제목. topics: 주제어 목록. source: 출처 표기 (예: 자체 작성한 일반 상식 요약 - 특정 기관 문서를 인용·대조하지 않음).
        body: front-matter 를 제외한 markdown 본문. path: 원본 파일 경로.
    """
    id: str
    title: str
    body: str
    topics: tuple[str, ...] = field(default_factory=tuple)
    source: str = DEFAULT_SOURCE
    path: Path | None = None


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """markdown 앞의 front-matter 를 분리한다.

    Args:
        raw: 파일 전문.

    Returns:
        (메타 dict, 본문). front-matter 가 없으면 ({}, raw). 값은 앞뒤 공백만 제거한 문자열이다.
    """
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _topics(value: str) -> tuple[str, ...]:
    """'[a, b]' 또는 'a, b' 형태의 주제어 문자열을 튜플로."""
    return tuple(t.strip() for t in value.strip("[]").split(",") if t.strip())


def load_corpus(directory: Path | None = None) -> list[Doc]:
    """디렉터리의 모든 *.md 를 Doc 목록으로 읽는다 (파일명 순, 결과 순서는 항상 같다).

    Args:
        directory: 문서 폴더. 생략하면 서비스 기본 폴더(GUIDELINES_DIR).

    Returns:
        Doc 리스트. 같은 id 가 두 번 나오면 ValueError.

    Raises:
        ValueError: id 중복.
        FileNotFoundError: 폴더가 없을 때.
    """
    docs: list[Doc] = []
    seen: set[str] = set()
    for path in sorted((directory or GUIDELINES_DIR).glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        title_line = next((ln[2:].strip() for ln in body.splitlines() if ln.startswith("# ")), path.stem)
        doc = Doc(id=meta.get("id", path.stem), title=meta.get("title", title_line), body=body.strip(),
                  topics=_topics(meta.get("topics", "")), source=meta.get("source", DEFAULT_SOURCE), path=path)
        if doc.id in seen:
            raise ValueError(f"가이드라인 문서 id 중복: {doc.id}")
        seen.add(doc.id)
        docs.append(doc)
    return docs
