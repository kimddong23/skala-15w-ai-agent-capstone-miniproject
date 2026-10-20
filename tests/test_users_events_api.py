"""`GET /users?dataset=` 필터, `GET /users/{id}/events`, `/version.datasets` (FakeTools 로 실데이터 없이 검증).

핵심 불변식: 사건 라벨은 UI 전용이다 - 에이전트가 만드는 리포트·LLM 사실에 라벨이 들어가지 않는다.
"""
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.deps import AgentDeps
from app.agent.llm import NullLLM
from app.api.main import create_app
from app.config import settings
from tests.fakes import FakeTools

LABELS = [{"type": "symptom_onset", "date": "2021-01-05", "illness": "covid"}, {"type": "diagnosis", "date": "2021-01-07", "illness": "covid"}]


def client(tools: FakeTools | None = None) -> TestClient:
    tools = tools or FakeTools()
    tools.extra_users = [{"user_id": "S001", "dataset": "stanford", "has_sleep": False, "start": "2020-12-01", "end": "2021-02-01", "days": 60, "coverage_pct": 70.0}]
    tools.events_by_user = {"S001": LABELS}
    return TestClient(create_app(deps=AgentDeps(NullLLM(), NullLLM(), tools.toolbox(), InMemorySaver(), settings, "v2")))


def test_users_dataset_query_filters_and_unknown_dataset_is_empty():
    with client() as c:
        assert {u["dataset"] for u in c.get("/users").json()} == {"fitbit", "stanford"}
        assert [u["user_id"] for u in c.get("/users", params={"dataset": "stanford"}).json()] == ["S001"]
        assert {u["dataset"] for u in c.get("/users", params={"dataset": "fitbit"}).json()} == {"fitbit"}
        assert c.get("/users", params={"dataset": "nope"}).json() == []


def test_events_endpoint_returns_labels_for_external_users():
    with client() as c:
        r = c.get("/users/S001/events")
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == "S001" and body["dataset"] == "stanford" and body["events"] == LABELS and "진단이 아닙니다" in body["note"]


def test_events_endpoint_is_empty_200_for_fitbit_and_404_for_unknown():
    with client() as c:
        r = c.get("/users/U01/events")
        assert r.status_code == 200 and r.json()["events"] == [] and r.json()["dataset"] == "fitbit"
        assert c.get("/users/U99/events").status_code == 404


def test_events_endpoint_without_an_events_tool_checks_the_user_and_returns_empty():
    tools = FakeTools()
    box = tools.toolbox()
    box.events = None  # get_health_events 를 제공하지 않는 서버
    with TestClient(create_app(deps=AgentDeps(NullLLM(), NullLLM(), box, InMemorySaver(), settings, "v2"))) as c:
        assert c.get("/users/U01/events").json()["events"] == []
        assert c.get("/users/U99/events").status_code == 404


def test_version_lists_available_datasets():
    with client() as c:
        assert c.get("/version").json()["datasets"] == ["fitbit", "stanford"]
    only = FakeTools()  # 외부 데이터셋이 없는 환경
    with TestClient(create_app(deps=AgentDeps(NullLLM(), NullLLM(), only.toolbox(), InMemorySaver(), settings, "v2"))) as c:
        assert c.get("/version").json()["datasets"] == ["fitbit"]


@pytest.mark.parametrize("uid", ["S001", "U01"])
def test_event_labels_never_reach_the_report(uid):
    """라벨이 있어도 리포트에는 나오지 않는다 (UI 전용, 진단 단정의 근거가 될 수 없음)."""
    with client() as c:
        body = c.post("/chat", json={"message": f"{uid} 최근 7일 심박 이상 점검해줘"}).json()
        assert "covid" not in body["report"].lower() and "2021-01-05" not in body["report"] and "symptom_onset" not in body["report"]


def test_stanford_ids_are_recognized_in_messages_but_phone_models_are_not():
    from app.agent import gates

    assert gates.extract_user_id("s004 최근 7일 점검해줘") == "S004"
    assert gates.extract_user_ids("U05랑 S001 비교해줘") == (["U05", "S001"], [])
    assert gates.extract_user_id("갤럭시 S24 워치로 측정했어요") is None
