"""제목 인식(heading-aware) 청킹.

책임 경계:
    Doc 하나를 검색 단위 Chunk 들로 자른다. 임베딩/인덱싱은 하지 않는다.

설계 근거:
    - 문서는 `## ` 소제목 단위로 주제가 나뉘므로 소제목을 청크 경계의 1순위로 쓴다 (주제가 섞이면 임베딩이 흐려진다).
    - 소제목 구간이 MAX_CHARS 를 넘으면 문장 단위로 다시 나누고, 너무 짧은 구간(MIN_CHARS 미만)은 앞 청크에 붙인다.
    - 임베딩 입력에는 '문서 제목 > 소제목' 접두를 붙인다: 본문만으로는 모호한 짧은 구간("회복" 등)이 어느 문서 소속인지 알려 준다.
    - 크기 기준(약 300~500자)은 bge-m3 의 입력 한계보다 훨씬 작지만, 한 청크가 한 가지 사실에 집중되게 하려는 것이다.

협력 모듈: `corpus.py`(Doc), `index.py`(Chunk.embed_text 와 content_hash 를 캐시 키로 사용).
"""
import hashlib
import re
from dataclasses import dataclass

from app.rag.corpus import Doc

MAX_CHARS = 500
MIN_CHARS = 150
# 문서 끝의 메타 정보 섹션(출처·검증 상태·면책 한 줄)은 검색 대상이 아니다: 사용자에게 보일 근거 본문이 아니고, 이 섹션을 고쳐도 청크 해시가
# 바뀌지 않아 임베딩 인덱스가 낡지 않는다. 출처 표기는 front-matter `source` 가 검색 결과(Hit.source)로 전달한다.
INFO_HEADING = "문서 정보"
_SENTENCE = re.compile(r"(?<=[.!?。])\s+")


@dataclass(frozen=True)
class Chunk:
    """검색 단위.

    Attributes:
        chunk_id: "{doc_id}#{순번}". doc_id/title/source: 소속 문서 정보. heading: 소제목 (없으면 빈 문자열).
        text: 사용자에게 보여 주는 본문 조각. embed_text: 임베딩 입력 ("제목 > 소제목\\n본문").
    """
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str
    embed_text: str
    source: str = ""

    @property
    def content_hash(self) -> str:
        """embed_text 의 sha256 앞 16자. 임베딩 캐시 키이며 내용이 바뀐 청크만 다시 임베딩하게 한다."""
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()[:16]


def _split_long(text: str, limit: int) -> list[str]:
    """limit 자를 넘는 단락을 문장 경계에서 나눈다 (한 문장이 limit 보다 길면 그 문장은 그대로 둔다)."""
    out, cur = [], ""
    for sent in _SENTENCE.split(text):
        if cur and len(cur) + 1 + len(sent) > limit:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def _sections(body: str) -> list[tuple[str, str]]:
    """본문을 [(소제목, 본문)] 으로 나눈다. `# ` 문서 제목 줄은 버리고, 첫 `## ` 앞의 도입부는 소제목 ''로 둔다. INFO_HEADING 섹션은 제외한다."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in body.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not line.startswith("# "):
            sections[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in sections if h != INFO_HEADING and "\n".join(ls).strip()]


def chunk_doc(doc: Doc, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[Chunk]:
    """Doc 를 Chunk 목록으로 자른다.

    Args:
        doc: 대상 문서.
        max_chars: 청크 본문 최대 길이(자). 소제목 구간이 넘으면 문장 단위로 분할.
        min_chars: 이보다 짧은 구간은 같은 문서의 앞 청크에 합친다 (합친 길이가 max_chars 를 넘지 않을 때만).

    Returns:
        Chunk 리스트 (문서 순서). 본문이 비어 있으면 빈 리스트.
    """
    pieces: list[tuple[str, str]] = []
    for heading, text in _sections(doc.body):
        for part in (_split_long(text, max_chars) if len(text) > max_chars else [text]):
            if pieces and len(part) < min_chars and len(pieces[-1][1]) + len(part) + 1 <= max_chars:
                pieces[-1] = (pieces[-1][0], f"{pieces[-1][1]}\n{part}")      # 짧은 구간은 앞 청크에 합침(앞 소제목 유지)
            else:
                pieces.append((heading, part))
    return [
        Chunk(f"{doc.id}#{i}", doc.id, doc.title, h, t, f"{doc.title} > {h}\n{t}" if h else f"{doc.title}\n{t}", doc.source)
        for i, (h, t) in enumerate(pieces)
    ]


def chunk_corpus(docs: list[Doc], **kw) -> list[Chunk]:
    """모든 문서를 순서대로 청킹한다 (chunk_doc 의 인자를 그대로 전달)."""
    return [c for d in docs for c in chunk_doc(d, **kw)]
