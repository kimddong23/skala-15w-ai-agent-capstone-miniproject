"""분 단위 생체 데이터 저장소 (데이터 계층의 읽기 측): FitBit 서비스 데이터 + 선택적 외부 연구 데이터셋.

책임 경계:
    prepare.py 가 만든 parquet(FitBit)와, 있을 때만 외부 연구 데이터셋(Stanford COVID-19, `eval/external/stanford_adapter.py` 산출물)을
    프로세스당 한 번 메모리에 올려 사용자·기간 단위로 잘라 준다. 탐지/요약 계산은 하지 않는다 (`app.detection`), 쓰기도 하지 않는다.

여러 데이터셋 (섞지 않는다):
    - FitBit 은 항상 로드한다 (사용자 U01..). 외부 데이터셋은 `settings.external_dir/<이름>/vitals_minute.parquet` 가 있을 때만 로드한다
      (사용자 S001..). 없으면 조용히 건너뛴다 - CI/데모 환경에는 외부 데이터가 없다.
    - 사용자 ID 접두가 서로 달라 한 사전에 담아도 충돌하지 않는다. 각 사용자는 자기 데이터셋에 속하며(`dataset_of`) 탐지·기준선은
      사용자 단위로 계산되므로 데이터셋 간에 통계가 섞이지 않는다.
    - 외부 데이터셋의 사건 라벨(`events.json`)은 `events(user_id)` 로만 조회한다. 탐지기 입력에는 들어가지 않는다 (라벨 누수 방지).

협력 모듈:
    `app.mcp_server.vitals_server`(싱글턴으로 사용), `eval/*`(직접 parquet 를 읽어 라벨과 대조), `app.config`(경로).

데이터 가정:
    - ts 는 기기 로컬 시간의 timezone-naive 값이다. 조회 구간은 (start, end] - 시작 제외, 끝 포함.
    - 희소 형식이다: 심박이 있는 분과 걸음이 있는 분만 행이 있고, 행이 없는 분은 걸음 0 으로 해석한다.
    - asleep 은 FitBit 은 bool, 외부 데이터셋은 nullable boolean(수면 정보가 없는 참여자는 NA)일 수 있다. NA 는 탐지기가 '깨어 있음'으로 취급한다.
    - FitBit 의 평가용 정답 라벨 컬럼(injected)은 여기서 제거되어 호출자에게 노출되지 않는다 (라벨 누수 방지).
    - 사용자 ID 는 가명이다.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.config import settings

COLUMNS = ["ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep"]
EXTERNAL_DATASETS = ("stanford",)       # settings.external_dir 아래 폴더 이름 = 데이터셋 이름
FITBIT = "fitbit"


class VitalsStore:
    """parquet 기반 읽기 전용 저장소. 인스턴스는 생성 시점의 데이터를 스냅샷으로 들고 있다 (스레드 안전한 읽기)."""

    def __init__(self, path: Path | None = None, external_dir: Path | None = None):
        """parquet 를 읽어 사용자별로 분리한다.

        Args:
            path: FitBit parquet 경로. 생략하면 settings.data_dir/vitals_minute.parquet.
            external_dir: 외부 데이터셋 루트. 생략하면 settings.external_dir. 그 아래 <이름>/vitals_minute.parquet 가 있는 데이터셋만 로드한다.

        Raises:
            FileNotFoundError: FitBit parquet 가 없을 때 (먼저 `python -m app.data.prepare` 실행). 외부 데이터셋이 없는 것은 오류가 아니다.
        """
        self.path = Path(path) if path else settings.data_dir / "vitals_minute.parquet"
        df = pd.read_parquet(self.path)
        self._by_user = {u: p[COLUMNS].sort_values("ts").reset_index(drop=True) for u, p in df.groupby("user_id")}
        self._dataset = {u: FITBIT for u in self._by_user}
        self._events: dict[str, list[dict]] = {}
        self._first_warning: dict[tuple[str, str], dict] = {}
        root = Path(external_dir) if external_dir else settings.external_dir
        for name in EXTERNAL_DATASETS:
            self._load_external(name, root / name)

    def _load_external(self, name: str, folder: Path) -> None:
        """외부 데이터셋 하나를 읽는다 (vitals_minute.parquet 가 없으면 건너뜀; events.json 은 선택)."""
        pq = folder / "vitals_minute.parquet"
        if not pq.exists():
            return
        ext = pd.read_parquet(pq)
        for u, p in ext.groupby("user_id"):
            if u in self._by_user:                      # ID 충돌 시 FitBit 우선 (접두가 달라 실제로는 없음)
                continue
            self._by_user[u] = p[COLUMNS].sort_values("ts").reset_index(drop=True)
            self._dataset[u] = name
        ev = folder / "events.json"
        if ev.exists():
            for e in json.loads(ev.read_text(encoding="utf-8")):
                if self._dataset.get(e["user_id"]) == name:
                    self._events.setdefault(e["user_id"], []).append(e)
        fw = folder / "first_warnings.json"       # 평가 스크립트가 전체 기록에서 사건마다 미리 계산해 둔 '첫 경고' (조회 구간과 무관)
        if fw.exists():
            for uid, rows in json.loads(fw.read_text(encoding="utf-8")).items():
                if self._dataset.get(uid) == name:
                    self._first_warning.update({(uid, r["onset"]): {k: v for k, v in r.items() if k != "onset"} for r in rows})

    def _user(self, user_id: str) -> pd.DataFrame:
        """사용자 데이터를 반환한다. 없으면 사용 가능한 ID 를 포함한 KeyError (메시지는 호출자가 그대로 노출해도 안전)."""
        if user_id not in self._by_user:
            ids = sorted(self._by_user)
            shown = ", ".join(ids[:20]) + (f" ... 외 {len(ids) - 20}명" if len(ids) > 20 else "")
            raise KeyError(f"알 수 없는 user_id: {user_id!r} (사용 가능: {shown})")
        return self._by_user[user_id]

    def datasets(self) -> list[str]:
        """로드된 데이터셋 이름 목록 (항상 'fitbit' 이 먼저, 그 뒤 외부 데이터셋)."""
        return [FITBIT] + [n for n in EXTERNAL_DATASETS if n in set(self._dataset.values())]

    def dataset_of(self, user_id: str) -> str:
        """사용자가 속한 데이터셋 이름. Raises: KeyError (알 수 없는 user_id)."""
        self._user(user_id)
        return self._dataset[user_id]

    def events(self, user_id: str) -> list[dict]:
        """사용자의 연구 라벨 사건 [{"user_id","type"(symptom_onset|diagnosis|recovery),"date"(ISO),"illness"(covid|other)}] (날짜 순).

        symptom_onset 항목에는 `first_warning`(그 증상일 -7~+3일 창의 가장 이른 알람; 전체 기록으로 미리 계산해 조회 구간과 무관)이 붙는다 -
        `{"evaluable","hit","first_alarm"(ISO|None),"lead_days"(앞서면 양수|None),"types","counted_in_eval","pre_days","post_days","method"}`.
        계산 결과 파일(first_warnings.json)이 없으면 이 키는 없다.
        FitBit 사용자와 라벨이 없는 외부 참여자는 빈 목록이다 (FitBit 의 주입 이상 라벨은 평가 전용이라 여기로 새지 않는다).

        Raises:
            KeyError: 알 수 없는 user_id.
        """
        self._user(user_id)
        rows = []
        for e in sorted(self._events.get(user_id, []), key=lambda e: (e["date"], e["type"])):
            fw = self._first_warning.get((user_id, e["date"])) if e["type"] == "symptom_onset" else None
            rows.append({**e, "first_warning": fw} if fw else e)
        return rows

    def list_users(self, dataset: str | None = None) -> list[dict]:
        """사용자 목록과 기록 요약을 반환한다.

        Args:
            dataset: 'fitbit' | 'stanford' 로 필터. 생략하면 로드된 모든 데이터셋. 알 수 없는 이름이면 빈 목록.

        Returns:
            [{"user_id", "dataset", "has_sleep"(수면 구간이 하나라도 있는가), "start", "end"(첫·마지막 심박 시각, ISO 8601 naive),
            "days"(심박이 있는 달력일 수), "coverage_pct"(첫~마지막 구간 분 수 대비 심박이 있는 분의 비율, %, 소수 1자리)}].
            데이터셋 순서(fitbit 먼저), 그 안에서 user_id 정렬순.
        """
        order = {n: i for i, n in enumerate(self.datasets())}
        out = []
        for u in sorted(self._by_user, key=lambda x: (order[self._dataset[x]], x)):
            if dataset is not None and self._dataset[u] != dataset:
                continue
            p = self._by_user[u]
            hr = p[p["hr_n"] > 0]
            start, end = hr["ts"].min(), hr["ts"].max()
            span_min = (end - start) / pd.Timedelta(minutes=1) + 1
            out.append({"user_id": u, "dataset": self._dataset[u], "has_sleep": bool(p["asleep"].fillna(False).astype(bool).any()),
                        "start": start.isoformat(), "end": end.isoformat(), "days": int(hr["ts"].dt.date.nunique()),
                        "coverage_pct": round(100 * len(hr) / span_min, 1)})
        return out

    def data_range(self, user_id: str) -> tuple[datetime, datetime]:
        """사용자의 (첫 심박 시각, 마지막 심박 시각). 걸음만 있는 분은 포함하지 않는다.

        Raises:
            KeyError: 알 수 없는 user_id.
        """
        p = self._user(user_id)
        hr = p[p["hr_n"] > 0]
        return hr["ts"].min().to_pydatetime(), hr["ts"].max().to_pydatetime()

    def window(self, user_id: str, end: datetime | None, lookback_days: float) -> pd.DataFrame:
        """(end - lookback_days, end] 구간의 행을 반환한다.

        Args:
            user_id: 가명 사용자 ID.
            end: 구간 끝 (naive 로컬 시각, 포함). None 이면 그 사용자의 마지막 심박 시각.
            lookback_days: 조회 기간(일, 소수 허용). 기간 검증은 호출자(MCP 서버)의 책임이다.

        Returns:
            컬럼 ts, hr_mean, hr_min, hr_max, hr_n, steps, asleep (ts 오름차순, 인덱스 0..n-1). 구간에 자료가 없으면 빈 프레임.

        Raises:
            KeyError: 알 수 없는 user_id.
        """
        p = self._user(user_id)
        end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(self.data_range(user_id)[1])
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
        return p[(p["ts"] > start_ts) & (p["ts"] <= end_ts)].reset_index(drop=True)

    def history(self, user_id: str, end: datetime) -> pd.DataFrame:
        """end 이전(포함) 전체 이력. 개인 기준선·일별 추세·시간 단위 모델 적합에 쓴다.

        Raises:
            KeyError: 알 수 없는 user_id.
        """
        p = self._user(user_id)
        return p[p["ts"] <= pd.Timestamp(end)].reset_index(drop=True)
