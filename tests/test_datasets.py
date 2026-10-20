"""다중 데이터셋(FitBit + 선택적 외부 연구 데이터셋) 테스트 - 작은 합성 픽스처만 사용한다 (실제 외부 데이터는 필요 없고 CI 에도 없다).

범위: `VitalsStore` 가 외부 데이터셋을 있을 때만 읽는지, list_users 의 dataset/has_sleep, 사건 라벨 조회(FitBit 은 항상 빈 목록),
MCP 도구 `get_health_events`/`list_users(dataset)`/`get_hr_series`(수면 NA 안전), 외부 데이터가 없을 때의 동작.
협력: `app.data.store`, `app.mcp_server.vitals_server`(store 를 픽스처로 교체), fastmcp 인메모리 Client.
"""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client

from app.data.store import VitalsStore
from app.mcp_server import vitals_server as vs


def minutes(user, start, days, hr=62.0, sleep="bool", seed=0):
    """하루 1440분짜리 합성 분 단위 프레임. sleep: 'bool'(전부 False 에 0~6시 True) | 'na'(수면 정보 없음, NA)."""
    ts = pd.date_range(start, periods=days * 1440, freq="min")
    rng = np.random.default_rng(seed)
    h = hr + rng.normal(0, 1, len(ts))
    asleep = (ts.hour < 6)
    return pd.DataFrame({"user_id": user, "ts": ts, "hr_mean": h, "hr_min": h - 1, "hr_max": h + 1, "hr_n": np.int32(12), "steps": np.int32(0),
                         "asleep": pd.array(asleep, dtype="boolean") if sleep == "bool" else pd.array([pd.NA] * len(ts), dtype="boolean")})


@pytest.fixture()
def dirs(tmp_path):
    """tmp/fitbit/vitals_minute.parquet (U01, U02) + tmp/ext/stanford/{vitals_minute.parquet, events.json} (S001 수면 NA, S002 수면 있음)."""
    fit = pd.concat([minutes("U01", "2016-04-01", 3), minutes("U02", "2016-04-01", 3, 70.0, seed=1)]).assign(injected=None)
    (tmp_path / "fitbit").mkdir()
    fit.to_parquet(tmp_path / "fitbit" / "vitals_minute.parquet", index=False)
    ext = tmp_path / "ext" / "stanford"
    ext.mkdir(parents=True)
    pd.concat([minutes("S001", "2024-01-01", 12, 58.0, sleep="na", seed=2).assign(source="stanford"),
               minutes("S002", "2024-02-01", 12, 64.0, seed=3).assign(source="stanford")]).to_parquet(ext / "vitals_minute.parquet", index=False)
    (ext / "events.json").write_text(json.dumps([
        {"user_id": "S001", "type": "recovery", "date": "2024-01-09", "illness": "covid"},
        {"user_id": "S001", "type": "symptom_onset", "date": "2024-01-05", "illness": "covid"},
        {"user_id": "U01", "type": "symptom_onset", "date": "2016-04-02", "illness": "covid"}]))      # 잘못 섞여 들어온 라벨은 FitBit 사용자에게 붙지 않는다
    (ext / "first_warnings.json").write_text(json.dumps({
        "S001": [{"onset": "2024-01-05", "counted_in_eval": True, "pre_days": 7, "post_days": 3, "method": "causal_daily_replay_frozen_config",
                  "evaluable": True, "hit": True, "first_alarm": "2024-01-03", "lead_days": 2, "types": ["resting_tachycardia"]}],
        "U01": [{"onset": "2016-04-02", "hit": True, "lead_days": 9}]}))      # FitBit 사용자에게는 붙지 않는다
    return tmp_path


def store_of(dirs, with_external=True):
    return VitalsStore(dirs / "fitbit" / "vitals_minute.parquet", dirs / ("ext" if with_external else "none"))


def test_external_dataset_is_loaded_only_when_present(dirs):
    both, only = store_of(dirs), store_of(dirs, with_external=False)
    assert both.datasets() == ["fitbit", "stanford"] and only.datasets() == ["fitbit"]
    assert [u["user_id"] for u in only.list_users()] == ["U01", "U02"]
    with pytest.raises(KeyError):
        only.window("S001", None, 1)
    assert only.events("U01") == []


def test_list_users_has_dataset_and_sleep_flags_and_filter(dirs):
    s = store_of(dirs)
    users = {u["user_id"]: u for u in s.list_users()}
    assert list(users) == ["U01", "U02", "S001", "S002"]                       # fitbit 먼저
    assert [users[k]["dataset"] for k in users] == ["fitbit", "fitbit", "stanford", "stanford"]
    assert users["U01"]["has_sleep"] and not users["S001"]["has_sleep"] and users["S002"]["has_sleep"]      # S001: 수면 NA
    assert [u["user_id"] for u in s.list_users("stanford")] == ["S001", "S002"] and s.list_users("nope") == []
    assert users["S001"]["days"] == 12 and 99 < users["S001"]["coverage_pct"] <= 100
    json.dumps(list(users.values()))


def test_events_are_sorted_belong_to_external_users_only_and_fitbit_is_empty(dirs):
    s = store_of(dirs)
    assert [(e["type"], e["date"]) for e in s.events("S001")] == [("symptom_onset", "2024-01-05"), ("recovery", "2024-01-09")]
    assert s.events("S002") == [] and s.events("U01") == [] and s.dataset_of("S002") == "stanford"
    with pytest.raises(KeyError):
        s.events("nope")


def test_datasets_do_not_mix_statistics_each_user_keeps_its_own_window(dirs):
    s = store_of(dirs)
    w = s.window("S001", None, 1)
    assert "injected" not in w.columns and len(w) == 1440 and abs(w["hr_mean"].mean() - 58.0) < 1
    assert str(w["asleep"].dtype) == "boolean" and w["asleep"].isna().all()
    assert abs(s.window("U02", None, 1)["hr_mean"].mean() - 70.0) < 1


@pytest.fixture()
def server(dirs, monkeypatch):
    monkeypatch.setattr(vs, "store", store_of(dirs))
    return vs.mcp


def call(server, name, args=None):
    async def run():
        async with Client(server) as c:
            return (await c.call_tool(name, args or {})).data
    return asyncio.run(run())


def test_mcp_get_health_events_contract(server):
    r = call(server, "get_health_events", {"user_id": "S001"})
    assert set(r) == {"user_id", "dataset", "events", "note"} and r["dataset"] == "stanford"
    first = r["events"][0]
    assert {k: first[k] for k in ("type", "date", "illness")} == {"type": "symptom_onset", "date": "2024-01-05", "illness": "covid"}
    assert first["first_warning"]["lead_days"] == 2 and first["first_warning"]["first_alarm"] == "2024-01-03" and first["first_warning"]["types"] == ["resting_tachycardia"]
    assert "first_warning" not in r["events"][1]                               # 회복 사건에는 붙지 않는다
    assert "진단이 아닙니다" in r["note"] and "검증" in r["note"]
    f = call(server, "get_health_events", {"user_id": "U01"})
    assert f["dataset"] == "fitbit" and f["events"] == []
    assert "error" in call(server, "get_health_events", {"user_id": "S999"})


def test_mcp_list_users_dataset_argument_and_tool_descriptions(server):
    assert [u["user_id"] for u in call(server, "list_users", {"dataset": "stanford"})] == ["S001", "S002"]
    assert len(call(server, "list_users")) == 4

    async def tools():
        async with Client(server) as c:
            return {t.name: t for t in await c.list_tools()}
    t = asyncio.run(tools())
    assert "get_health_events" in t and "진단이 아니다" in t["get_health_events"].description and "dataset" in t["list_users"].inputSchema["properties"]


def test_series_and_detection_work_for_external_user_without_sleep_info(server):
    r = call(server, "get_hr_series", {"user_id": "S001", "lookback_days": 2})
    assert r["points"] and all(p["asleep"] is False for p in r["points"])       # 수면 NA -> 수면 아님
    d = call(server, "detect_anomalies", {"user_id": "S001", "lookback_days": 5})
    assert "anomalies" in d and "error" not in d
    s = call(server, "get_vitals_summary", {"user_id": "S001", "lookback_days": 5})
    assert s["summary"]["sleep_hr"] is None


def test_application_default_store_works_without_external_data(tmp_path, monkeypatch):
    """EXTERNAL_DIR 이 비어 있는 환경(CI/데모): FitBit 만으로 정상 동작한다."""
    from app.config import settings
    import dataclasses
    from app.data import store as store_mod
    monkeypatch.setattr(store_mod, "settings", dataclasses.replace(settings, external_dir=tmp_path / "empty"))
    s = store_mod.VitalsStore()
    assert s.datasets() == ["fitbit"] and all(u["dataset"] == "fitbit" for u in s.list_users())


def test_first_warning_is_precomputed_per_event_and_independent_of_any_query_window(server):
    """첫 경고는 전체 기록에서 사건마다 미리 계산한 값이라 조회 구간(lookback/end)을 바꿔도 get_health_events 결과가 같다."""
    before = call(server, "get_health_events", {"user_id": "S001"})["events"]
    call(server, "detect_anomalies", {"user_id": "S001", "lookback_days": 3, "end": "2024-01-06T00:00:00"})
    call(server, "get_hr_series", {"user_id": "S001", "lookback_days": 2, "end": "2024-01-09T00:00:00"})
    assert call(server, "get_health_events", {"user_id": "S001"})["events"] == before
    assert before[0]["first_warning"]["lead_days"] == 2


def test_events_have_no_first_warning_when_the_precomputed_file_is_absent(dirs):
    (dirs / "ext" / "stanford" / "first_warnings.json").unlink()
    s = store_of(dirs)
    assert all("first_warning" not in e for e in s.events("S001")) and s.events("U01") == []


def test_fitbit_users_never_get_first_warning_even_if_the_file_names_them(dirs):
    s = store_of(dirs)
    assert s.events("U01") == []                 # first_warnings.json 이 U01 을 적어 놓았어도 FitBit 사용자에게는 붙지 않는다
