"""데이터 계약(parquet / labels / id_map)과 VitalsStore 조회 테스트.

범위: `app.data.prepare` 산출물의 스키마·재현성·주입 라벨의 일관성, `app.data.store.VitalsStore` 의 조회 의미.
협력: `tests/_prepared.py`(산출물 보장), raw CSV(`data/raw`, determinism 테스트가 prepare 전체를 임시 폴더에 다시 돌린다).
데이터 가정: ts 는 naive 로컬 시각, 조회 구간은 (start, end], 가명 ID 는 U01..Unn, 정답 라벨은 store 를 통해 노출되지 않는다.
"""
import dataclasses
import json
import re

import pandas as pd
import pytest

from app.config import settings
from app.data import prepare
from app.data.store import VitalsStore
from tests._prepared import ensure_prepared

ensure_prepared()
# 원본 Kaggle CSV(data/raw)는 저장소에 없다. 받은 사람만 prepare 재현 테스트를 돌린다 (README 데이터 절).
needs_raw = pytest.mark.skipif(
    not any(settings.raw_dir.glob("*/*/heartrate_seconds_merged.csv")),
    reason="data/raw 원본 CSV 없음: kaggle datasets download -d arashnic/fitbit --unzip -p data/raw",
)

TYPES = {"resting_tachycardia", "bradycardia", "personal_deviation", "elevated_resting_trend", "sensor_gap"}


@pytest.fixture(scope="module")
def store() -> VitalsStore:
    """VitalsStore 를 모듈당 한 번 만든다 (parquet 로드가 느리다)."""
    return VitalsStore()


def test_parquet_schema_and_pseudonyms():
    """출력 스키마(컬럼·dtype)와 가명 ID 형식, 심박 값 범위, 결측 규칙이 계약과 같다."""
    df = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    assert list(df.columns) == ["user_id", "ts", "hr_mean", "hr_min", "hr_max", "hr_n", "steps", "asleep", "injected"]
    assert str(df["ts"].dtype) == "datetime64[ns]" and df["asleep"].dtype == bool
    assert all(re.fullmatch(r"U\d{2}", u) for u in df["user_id"].unique())
    assert not df.duplicated(["user_id", "ts"]).any()
    hr = df.loc[df["hr_n"] > 0, "hr_mean"]
    assert hr.between(20, 230).all() and df.loc[df["hr_n"] == 0, "hr_mean"].isna().all()


def test_id_map_is_pseudonymization_table():
    """id_map.json 은 U01.. 연번 -> 숫자 원본 ID 매핑이다."""
    m = json.loads((settings.data_dir / "id_map.json").read_text())
    assert sorted(m) == [f"U{i:02d}" for i in range(1, len(m) + 1)]
    assert all(v.isdigit() for v in m.values())


def test_labels_cover_all_types_and_enough_users():
    """라벨은 5개 유형을 모두 포함하고 6명 이상, 유형당 1~3건이며 injected 컬럼과 정합한다."""
    labels = json.loads((settings.data_dir / "labels.json").read_text())
    assert {lb["type"] for lb in labels} == TYPES
    assert len({lb["user_id"] for lb in labels}) >= 6
    for t in TYPES:
        assert 1 <= sum(lb["type"] == t for lb in labels) <= 3
    assert all(set(lb) == {"event_id", "user_id", "type", "start", "end", "severity", "params"} for lb in labels)
    df = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    assert set(df["injected"].dropna()) <= {lb["event_id"] for lb in labels}


def test_injected_data_equals_clean_outside_injection():
    """주입 구간 밖은 원본과 같다 (배경은 진짜 데이터)."""
    inj = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    clean = pd.read_parquet(settings.data_dir / "vitals_minute_clean.parquet")
    assert clean["injected"].isna().all()
    m = inj[inj["injected"].isna()].merge(clean, on=["user_id", "ts"], suffixes=("", "_c"))
    for c in ["hr_mean", "hr_n", "steps", "asleep"]:
        same = (m[c] == m[f"{c}_c"]) | (m[c].isna() & m[f"{c}_c"].isna())
        assert same.all(), c


def test_injected_minutes_are_at_rest_context():
    """분 단위 이벤트의 주입 분은 걸음 0 이고 수면 이벤트만 asleep=True 이다."""
    inj = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    labels = {lb["event_id"]: lb for lb in json.loads((settings.data_dir / "labels.json").read_text())}
    point = {e for e, lb in labels.items() if lb["params"]["kind"] in ("awake", "sleep")}
    rows = inj[inj["injected"].isin(point) & (inj["hr_n"] > 0)]
    assert len(rows) > 0 and (rows["steps"] == 0).all()
    sleep_ids = {e for e in point if labels[e]["params"]["kind"] == "sleep"}
    assert rows.loc[rows["injected"].isin(sleep_ids), "asleep"].all()
    assert not rows.loc[~rows["injected"].isin(sleep_ids), "asleep"].any()


@needs_raw
def test_prepare_is_deterministic(tmp_path, monkeypatch):
    """prepare 를 임시 폴더에 다시 돌리면 라벨과 parquet 가 비트 단위로 같다 (시드 고정, 멱등)."""
    monkeypatch.setattr(prepare, "settings", dataclasses.replace(settings, data_dir=tmp_path))
    prepare.main()
    assert (tmp_path / "labels.json").read_text() == (settings.data_dir / "labels.json").read_text()
    a = pd.read_parquet(tmp_path / "vitals_minute.parquet")
    b = pd.read_parquet(settings.data_dir / "vitals_minute.parquet")
    pd.testing.assert_frame_equal(a, b)


@needs_raw
def test_no_inject_option_writes_empty_labels(tmp_path, monkeypatch):
    """--no-inject 이면 labels 는 빈 목록이고 주입본이 clean 과 같다."""
    monkeypatch.setattr(prepare, "settings", dataclasses.replace(settings, data_dir=tmp_path))
    prepare.main(no_inject=True)
    assert json.loads((tmp_path / "labels.json").read_text()) == []
    a = pd.read_parquet(tmp_path / "vitals_minute.parquet")
    b = pd.read_parquet(tmp_path / "vitals_minute_clean.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_list_users(store):
    """list_users 는 계약 키를 가진 JSON 직렬화 가능한 목록이다."""
    users = store.list_users()
    assert len(users) >= 10 and users[0]["user_id"] == "U01"
    u = users[0]
    assert set(u) == {"user_id", "dataset", "has_sleep", "start", "end", "days", "coverage_pct"} and 0 < u["coverage_pct"] <= 100 and u["dataset"] == "fitbit"
    json.dumps(users)


def test_window_bounds_and_default_end(store):
    """window 는 (end-lookback, end] 이며 end 생략 시 마지막 심박 시각을 쓴다."""
    start, end = store.data_range("U03")
    w = store.window("U03", None, 1)
    assert not w.empty and w["ts"].max() <= pd.Timestamp(end)
    assert w["ts"].min() > pd.Timestamp(end) - pd.Timedelta(days=1)
    w2 = store.window("U03", pd.Timestamp(start) + pd.Timedelta(days=2), 0.5)
    assert w2["ts"].between(pd.Timestamp(start) + pd.Timedelta(days=1.5), pd.Timestamp(start) + pd.Timedelta(days=2)).all()


def test_label_column_not_exposed(store):
    """store 가 반환하는 프레임에는 정답 라벨(injected)이 없다."""
    assert "injected" not in store.window("U03", None, 3).columns


def test_unknown_user_raises_keyerror(store):
    """알 수 없는 user_id 는 KeyError 이다 (MCP 서버가 error 응답으로 변환)."""
    with pytest.raises(KeyError):
        store.window("U99", None, 1)
    with pytest.raises(KeyError):
        store.data_range("nope")
