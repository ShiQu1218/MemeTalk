from __future__ import annotations

import streamlit as st

from memetalk.app.settings_io import DEFAULT_CONFIG_PATH, load_settings, merge_settings, save_settings
from memetalk.app.ui import format_path, render_notice, render_section, setup_page
from memetalk.config import AppSettings


def _clean_optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def _clean_required_text(value: str, fallback: str) -> str:
    cleaned = value.strip()
    return cleaned or fallback


def _clear_runtime_cache() -> None:
    st.session_state.pop("container", None)


settings = load_settings()

setup_page(
    page_title="MemeTalk - 設定",
    page_icon="⚙️",
    title="設定",
    subtitle="把 Provider、模型與預設梗圖資料夾整理在同一頁，保存時不再遺失既有設定。",
    eyebrow="Settings",
    chips=(
        f"Provider: {settings.provider_backend}",
        f"Vector: {settings.vector_backend}",
        f"OCR: {settings.ocr_backend}",
    ),
)

flash_message = st.session_state.pop("settings_flash_message", None)
if flash_message:
    st.success(flash_message)

render_section("Provider 與模型", "先決定本次執行要用哪個 Provider，相關模型設定會一起保留。")
left_col, right_col = st.columns((1.45, 0.95))

provider_options = ["openai", "lmstudio", "ollama", "llama_cpp", "gemini", "claude", "mock"]
vector_options = ["chroma", "memory"]
ocr_options = ["paddleocr", "mock"]

with left_col:
    with st.container(border=True):
        provider_index = provider_options.index(settings.provider_backend) if settings.provider_backend in provider_options else 0
        provider_backend = st.selectbox("Provider Backend", provider_options, index=provider_index)

        if provider_backend == "openai":
            openai_api_key = st.text_input("OpenAI API Key", value=settings.openai_api_key or "", type="password")
            openai_base_url = st.text_input("OpenAI Base URL", value=settings.openai_base_url or "", placeholder="留空使用官方預設")
            model_col1, model_col2, model_col3 = st.columns(3)
            openai_chat_model = model_col1.text_input("Chat Model", value=settings.openai_chat_model)
            openai_vision_model = model_col2.text_input("Vision Model", value=settings.openai_vision_model)
            openai_embedding_model = model_col3.text_input("Embedding Model", value=settings.openai_embedding_model)

        elif provider_backend == "lmstudio":
            lmstudio_base_url = st.text_input("LM Studio Base URL", value=settings.lmstudio_base_url)
            lmstudio_api_key = st.text_input("LM Studio API Key", value=settings.lmstudio_api_key or "", type="password")
            model_col1, model_col2, model_col3 = st.columns(3)
            lmstudio_chat_model = model_col1.text_input("Chat Model", value=settings.lmstudio_chat_model or "")
            lmstudio_vision_model = model_col2.text_input("Vision Model", value=settings.lmstudio_vision_model or "")
            lmstudio_embedding_model = model_col3.text_input("Embedding Model", value=settings.lmstudio_embedding_model or "")

        elif provider_backend == "ollama":
            render_notice("Ollama", "使用本機 Ollama 服務，需先安裝並啟動 Ollama。", tone="info")
            ollama_base_url = st.text_input("Ollama Base URL", value=settings.ollama_base_url)
            model_col1, model_col2, model_col3 = st.columns(3)
            ollama_chat_model = model_col1.text_input("Chat Model", value=settings.ollama_chat_model or "", placeholder="llama3")
            ollama_vision_model = model_col2.text_input("Vision Model", value=settings.ollama_vision_model or "", placeholder="llava")
            ollama_embedding_model = model_col3.text_input("Embedding Model", value=settings.ollama_embedding_model or "", placeholder="nomic-embed-text")

        elif provider_backend == "llama_cpp":
            render_notice("llama.cpp", "使用本機 llama.cpp server，模型由 server 載入決定。", tone="info")
            llama_cpp_base_url = st.text_input("llama.cpp Base URL", value=settings.llama_cpp_base_url)

        elif provider_backend == "gemini":
            render_notice("Gemini", "使用 Google Gemini API，透過 OpenAI 相容端點。", tone="info")
            gemini_api_key = st.text_input("Gemini API Key", value=settings.gemini_api_key or "", type="password")
            model_col1, model_col2 = st.columns(2)
            gemini_chat_model = model_col1.text_input("Chat / Vision Model", value=settings.gemini_chat_model)
            gemini_embedding_model = model_col2.text_input("Embedding Model", value=settings.gemini_embedding_model)

        elif provider_backend == "claude":
            render_notice(
                "Claude",
                "使用 Anthropic Claude API。Claude 不提供 Embedding，需搭配 OpenAI 或 Gemini 的 embedding。",
                tone="info",
            )
            claude_api_key = st.text_input("Claude API Key", value=settings.claude_api_key or "", type="password")
            model_col1, model_col2 = st.columns(2)
            claude_chat_model = model_col1.text_input("Chat Model", value=settings.claude_chat_model)
            claude_vision_model = model_col2.text_input("Vision Model", value=settings.claude_vision_model)
            claude_embedding_options = ["openai", "gemini"]
            claude_emb_idx = claude_embedding_options.index(settings.claude_embedding_provider) if settings.claude_embedding_provider in claude_embedding_options else 0
            claude_embedding_provider = st.selectbox(
                "Embedding 來源",
                claude_embedding_options,
                index=claude_emb_idx,
                help="Claude 沒有 embedding API，需借用其他服務。請確認對應的 API Key 與模型已設定。",
            )

        else:
            render_notice("Mock 模式", "適合先驗證 UI 與流程，不需要外部 API。", tone="info")

    render_section("儲存與索引預設", "這裡的梗圖資料夾是正式預設值，Index 頁只會拿來預填，不會自動覆寫。")
    with st.container(border=True):
        vector_index = vector_options.index(settings.vector_backend) if settings.vector_backend in vector_options else 0
        vector_backend = st.selectbox("Vector Backend", vector_options, index=vector_index)

        ocr_index = ocr_options.index(settings.ocr_backend) if settings.ocr_backend in ocr_options else 0
        ocr_backend = st.selectbox("OCR Backend", ocr_options, index=ocr_index)

        meme_folder = st.text_input(
            "預設梗圖資料夾",
            value=settings.meme_folder,
            placeholder="例如：D:/MemeData",
            help="作為 Index 頁的預設路徑。Index 頁臨時修改不會回寫這個設定。",
        )

with right_col:
    with st.container(border=True):
        st.subheader("設定檔")
        st.code(str(DEFAULT_CONFIG_PATH))
        st.caption("設定來源優先順序固定為環境變數 > TOML 設定檔 > Pydantic 預設值。")
        st.caption(f"目前預設梗圖資料夾：`{format_path(settings.meme_folder)}`")

    with st.container(border=True):
        st.subheader("保存行為")
        st.markdown(
            """
- 儲存時會以現有設定為基底，只覆蓋你在表單中修改的欄位。
- `meme_folder` 不會再因為儲存其他 Provider 欄位而被清空。
- 如果剛改完設定，會一併清掉快取中的 runtime container，讓其他頁面讀到最新設定。
"""
        )

    with st.container(border=True):
        st.subheader("使用建議")
        st.markdown(
            """
- 想固定索引來源：在這頁儲存預設梗圖資料夾。
- 想臨時跑另一包素材：到 Index 頁直接輸入路徑即可。
- 模型或 API 切換後，先回 Dashboard 看健康檢查是否正常。
"""
        )

st.divider()
if st.button("儲存設定", type="primary", use_container_width=True):
    updates: dict[str, object] = {
        "provider_backend": provider_backend,
        "vector_backend": vector_backend,
        "ocr_backend": ocr_backend,
        "meme_folder": meme_folder.strip(),
    }

    if provider_backend == "openai":
        updates.update(
            {
                "openai_api_key": _clean_optional_text(openai_api_key),
                "openai_base_url": _clean_optional_text(openai_base_url.rstrip("/")),
                "openai_chat_model": _clean_required_text(openai_chat_model, settings.openai_chat_model),
                "openai_vision_model": _clean_required_text(openai_vision_model, settings.openai_vision_model),
                "openai_embedding_model": _clean_required_text(openai_embedding_model, settings.openai_embedding_model),
            }
        )
    elif provider_backend == "lmstudio":
        fallback_settings = AppSettings()
        updates.update(
            {
                "lmstudio_base_url": _clean_required_text(
                    lmstudio_base_url.rstrip("/"),
                    settings.lmstudio_base_url or fallback_settings.lmstudio_base_url,
                ),
                "lmstudio_api_key": _clean_optional_text(lmstudio_api_key),
                "lmstudio_chat_model": _clean_optional_text(lmstudio_chat_model),
                "lmstudio_vision_model": _clean_optional_text(lmstudio_vision_model),
                "lmstudio_embedding_model": _clean_optional_text(lmstudio_embedding_model),
            }
        )
    elif provider_backend == "ollama":
        fallback_settings = AppSettings()
        updates.update(
            {
                "ollama_base_url": _clean_required_text(
                    ollama_base_url.rstrip("/"),
                    settings.ollama_base_url or fallback_settings.ollama_base_url,
                ),
                "ollama_chat_model": _clean_optional_text(ollama_chat_model),
                "ollama_vision_model": _clean_optional_text(ollama_vision_model),
                "ollama_embedding_model": _clean_optional_text(ollama_embedding_model),
            }
        )
    elif provider_backend == "llama_cpp":
        fallback_settings = AppSettings()
        updates.update(
            {
                "llama_cpp_base_url": _clean_required_text(
                    llama_cpp_base_url.rstrip("/"),
                    settings.llama_cpp_base_url or fallback_settings.llama_cpp_base_url,
                ),
            }
        )
    elif provider_backend == "gemini":
        updates.update(
            {
                "gemini_api_key": _clean_optional_text(gemini_api_key),
                "gemini_chat_model": _clean_required_text(gemini_chat_model, settings.gemini_chat_model),
                "gemini_embedding_model": _clean_required_text(gemini_embedding_model, settings.gemini_embedding_model),
            }
        )
    elif provider_backend == "claude":
        updates.update(
            {
                "claude_api_key": _clean_optional_text(claude_api_key),
                "claude_chat_model": _clean_required_text(claude_chat_model, settings.claude_chat_model),
                "claude_vision_model": _clean_required_text(claude_vision_model, settings.claude_vision_model),
                "claude_embedding_provider": claude_embedding_provider,
            }
        )

    new_settings = merge_settings(settings, updates)
    save_settings(new_settings)
    _clear_runtime_cache()
    st.session_state["settings_flash_message"] = f"設定已儲存至 `{DEFAULT_CONFIG_PATH}`"
    st.rerun()
