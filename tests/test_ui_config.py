from memetalk.app.ui_config import DEFAULT_API_BASE_URL, resolve_api_base_url


def test_resolve_api_base_url_prefers_secrets(monkeypatch) -> None:
    monkeypatch.setenv("MEMETALK_API_BASE_URL", "http://env.example.com")

    value = resolve_api_base_url({"api_base_url": "http://secret.example.com/"})

    assert value == "http://secret.example.com"


def test_resolve_api_base_url_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("MEMETALK_API_BASE_URL", "http://env.example.com/")

    value = resolve_api_base_url(None)

    assert value == "http://env.example.com"


def test_resolve_api_base_url_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.delenv("MEMETALK_API_BASE_URL", raising=False)

    value = resolve_api_base_url(None)

    assert value == DEFAULT_API_BASE_URL
