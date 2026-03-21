from pathlib import Path

from memetalk.app.settings_io import load_settings, merge_settings, save_settings
from memetalk.config import AppSettings


def test_merge_settings_preserves_meme_folder_when_other_fields_change() -> None:
    base = AppSettings(
        provider_backend="lmstudio",
        vector_backend="memory",
        ocr_backend="mock",
        meme_folder="D:/Memes",
        lmstudio_chat_model="qwen",
    )

    merged = merge_settings(
        base,
        {
            "provider_backend": "openai",
            "openai_chat_model": "gpt-4.1-mini",
        },
    )

    assert merged.provider_backend == "openai"
    assert merged.openai_chat_model == "gpt-4.1-mini"
    assert merged.meme_folder == "D:/Memes"
    assert merged.lmstudio_chat_model == "qwen"


def test_save_and_load_settings_round_trip_meme_folder(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MEMETALK_MEME_FOLDER", raising=False)
    config_path = tmp_path / "memetalk_config.toml"
    settings = AppSettings(
        provider_backend="mock",
        vector_backend="memory",
        ocr_backend="mock",
        meme_folder="D:/Memes",
    )

    save_settings(settings, config_path)
    loaded = load_settings(config_path)

    assert loaded.meme_folder == "D:/Memes"
    assert loaded.provider_backend == "mock"


def test_from_env_reads_meme_folder(monkeypatch) -> None:
    monkeypatch.setenv("MEMETALK_MEME_FOLDER", "D:/EnvMemes")

    settings = AppSettings.from_env()

    assert settings.meme_folder == "D:/EnvMemes"
