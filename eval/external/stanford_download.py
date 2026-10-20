"""Stanford COVID-19 웨어러블 데이터에서 필요한 참여자 파일만 HTTP Range 로 내려받는다.

책임 경계:
    378 MB zip 전체를 받지 않고 중앙 디렉터리와 필요한 멤버(라벨이 있는 47명 + 라벨 없는 대조군 표본)만 받아
    `data/external/stanford/raw/` 에 저장하고, 라벨 엑셀(Supplementary Data 1)도 내려받는다. 변환은 `stanford_adapter.py` 의 책임이다.

데이터 이용 조건 (중요): 원본 zip 과 논문 Data availability 문구에 데이터 라이선스 표기가 없다 -> '공개 다운로드 가능'이지 '재배포 허가'가 아니다.
    따라서 내려받은 원시 파일과 분 단위 변환본은 절대 커밋하지 않는다 (`data/external/` 는 gitignore). 레포에는 이 스크립트와 집계 지표만 둔다.

데이터 가정: zip 안 파일명은 `COVID-19-Wearables/{ID}_{hr|steps|sleep}.csv` (ID 7자, 일부는 `{ID}_1_...` 처럼 조각 번호가 붙는다).
    라벨 엑셀의 날짜는 개인정보 보호를 위해 이동(shift)된 값이며 zip 의 날짜와 같은 이동을 쓴다 -> 라벨은 반드시 이 zip 과 짝지어 쓴다
    (GitHub gireeshkbogu/AnomalyDetect 의 샘플 데이터는 날짜 이동이 달라 섞으면 안 된다).

실행: `.venv/bin/python -m eval.external.stanford_download [--controls 30] [--dry-run]`
    표준 라이브러리 + openpyxl(pandas.read_excel 용)만 쓴다. 이미 받은 파일은 건너뛴다(멱등).
"""
import argparse
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "external" / "stanford"
ZIP_URL = "https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip"
XLSX_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41551-020-00640-6/MediaObjects/41551_2020_640_MOESM3_ESM.xlsx"
SEED = 20261020
MEMBER_RE = re.compile(r"^COVID-19-Wearables/(?P<stem>[A-Z0-9]{7}(?:_\d)?)_(?P<kind>hr|steps|sleep)\.csv$")


class RangeFile(io.RawIOBase):
    """HTTP Range 로 읽는 읽기 전용 파일 객체 (zipfile 이 중앙 디렉터리와 필요한 멤버만 읽게 한다).

    Attributes:
        size: 원격 파일 크기(bytes). fetched: 지금까지 실제로 내려받은 bytes (대역폭 확인용).
    """

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.fetched = url, 0, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        """buf 길이만큼 (또는 파일 끝까지) 내려받아 채운다. 일시 오류는 3회까지 재시도한다."""
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=120).read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        buf[:len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


def labeled_ids(xlsx: Path) -> list[str]:
    """라벨 엑셀(SuppTable3)의 참여자 ID 목록 (중복 제거, 등장 순서 유지).

    한 사람이 질병을 여러 번 앓은 경우 엑셀에 `AA0HAI1_Illness_1`, `_2`, `_3` 처럼 사건별 행이 있다 -> 데이터 파일 ID(`AA0HAI1`)로 합친다.
    """
    import pandas as pd

    df = pd.read_excel(xlsx, sheet_name="SuppTable3_Fig2a_COVID-19", header=3)
    return list(dict.fromkeys(str(p).split("_")[0] for p in df["ParticipantID"].dropna()))


def index_members(zf: zipfile.ZipFile) -> dict[str, dict[str, list[str]]]:
    """zip 멤버를 {참여자 ID: {종류: [멤버 이름 ...]}} 로 정리한다 (조각 번호 `_1`, `_2` 는 같은 참여자로 묶는다)."""
    out: dict[str, dict[str, list[str]]] = {}
    for info in zf.infolist():
        m = MEMBER_RE.match(info.filename)
        if m:
            out.setdefault(m.group("stem").split("_")[0], {}).setdefault(m.group("kind"), []).append(info.filename)
    return {k: {kind: sorted(v) for kind, v in d.items()} for k, d in out.items()}


def pick_controls(members: dict, labeled: list[str], n: int, seed: int = SEED) -> list[str]:
    """라벨 없는 참여자(= 논문의 건강 대조군) 중 HR 과 걸음 파일이 모두 있는 n 명을 고정 시드로 뽑는다."""
    import random

    pool = sorted(k for k, d in members.items() if k not in set(labeled) and "hr" in d and "steps" in d)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def download(controls: int = 30, dry_run: bool = False) -> dict:
    """라벨 엑셀과 대상 참여자의 hr/steps/sleep CSV 를 내려받는다.

    Args:
        controls: 라벨 없는 대조군 표본 수. dry_run: True 면 목록과 예상 크기만 출력하고 받지 않는다.

    Returns:
        {"labeled": [...], "controls": [...], "missing_labeled": [...], "fetched_mb": 내려받은 용량, "files": 파일 수}.
    """
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    xlsx = OUT / "SuppData1.xlsx"
    if not xlsx.exists():
        urllib.request.urlretrieve(XLSX_URL, xlsx)       # 라벨 목록이 있어야 계획을 세울 수 있어 드라이런에서도 받는다 (4 MB)
    labeled = labeled_ids(xlsx)
    rf = RangeFile(ZIP_URL)
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    members = index_members(zf)
    missing = [p for p in labeled if p not in members]
    controls_ids = pick_controls(members, labeled, controls)
    chosen = [p for p in labeled if p in members] + controls_ids
    plan = [(name, zf.getinfo(name).compress_size) for p in chosen for names in members[p].values() for name in names]
    total_mb = sum(c for _, c in plan) / 1e6
    print(f"라벨 참여자 {len(labeled)}명(zip 에 없음 {missing}), 대조군 {len(controls_ids)}명, 파일 {len(plan)}개, 압축 기준 약 {total_mb:.0f} MB")
    files = 0
    if not dry_run:
        for name, _ in plan:
            dest = raw / Path(name).name                           # 멤버 이름의 마지막 요소만 쓴다 (경로 탈출 방지)
            if dest.exists() and dest.stat().st_size == zf.getinfo(name).file_size:
                continue
            with zf.open(name) as src, open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            files += 1
            print(f"  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return {"labeled": labeled, "controls": controls_ids, "missing_labeled": missing,
            "fetched_mb": round(rf.fetched / 1e6, 1), "files": files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    res = download(a.controls, a.dry_run)
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in res.items()})
