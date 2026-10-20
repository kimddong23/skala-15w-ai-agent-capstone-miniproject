"""과금 확인(OPENAI_CONFIRM)은 .env 가 아니라 프로세스 환경에서만 받는다 (공유 OpenAI 키 사고 방지)."""
import os

from app.config import load_env_file


def test_openai_confirm_in_dotenv_is_ignored_but_other_keys_load(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_CONFIRM=yes\nSOME_OTHER_KEY_FOR_TEST=abc\n")
    monkeypatch.delenv("OPENAI_CONFIRM", raising=False)
    monkeypatch.delenv("SOME_OTHER_KEY_FOR_TEST", raising=False)
    try:
        assert load_env_file(env) is True
        assert os.environ.get("OPENAI_CONFIRM") is None and os.environ["SOME_OTHER_KEY_FOR_TEST"] == "abc"
    finally:
        os.environ.pop("SOME_OTHER_KEY_FOR_TEST", None)


def test_openai_confirm_from_the_process_environment_is_kept(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_CONFIRM=no\n")
    monkeypatch.setenv("OPENAI_CONFIRM", "yes")
    load_env_file(env)
    assert os.environ["OPENAI_CONFIRM"] == "yes"
