"""테스트 공용: 처리된 데이터(parquet / labels)가 없으면 prepare 를 실행해 보장한다.

책임 경계: 존재 확인과 생성만 한다 (내용 검증은 각 테스트의 몫). 협력: `app.data.prepare`, `app.config.settings`.
데이터 가정: data/raw 에 원본 CSV 가 있어야 한다 (없으면 prepare 가 FileNotFoundError).
"""
from app.config import settings


def ensure_prepared() -> None:
    """vitals_minute.parquet 과 labels.json 이 없으면 `prepare.main()` 으로 만든다 (약 15초)."""
    if not all((settings.data_dir / f).exists() for f in ("vitals_minute.parquet", "labels.json")):
        from app.data import prepare
        prepare.main()
