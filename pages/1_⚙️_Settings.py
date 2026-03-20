from __future__ import annotations

from pathlib import Path

import streamlit as st

from memetalk.app.settings_io import DEFAULT_CONFIG_PATH, load_settings, save_settings
from memetalk.config import AppSettings

st.set_page_config(page_title="MemeTalk - 設定", page_icon="⚙️", layout="wide")
st.title("⚙️ 設定")
st.caption("選擇 Provider、輸入 API Key、設定各項參數。儲存後即可在其他頁面使用。")

# Load current settings
settings = load_settings()

# --- Provider Backend ---
st.subheader("Provider 設定")

provider_options = ["openai", "lmstudio", "mock"]
provider_index = provider_options.index(settings.provider_backend) if settings.provider_backend in provider_options else 0
provider_backend = st.selectbox("Provider Backend", provider_options, index=provider_index)

if provider_backend == "openai":
    openai_api_key = st.text_input("OpenAI API Key", value=settings.openai_api_key or "", type="password")
    openai_base_url = st.text_input("OpenAI Base URL（留空使用預設）", value=settings.openai_base_url or "")
    openai_chat_model = st.text_input("Chat Model", value=settings.openai_chat_model)
    openai_vision_model = st.text_input("Vision Model", value=settings.openai_vision_model)
    openai_embedding_model = st.text_input("Embedding Model", value=settings.openai_embedding_model)
elif provider_backend == "lmstudio":
    lmstudio_base_url = st.text_input("LM Studio Base URL", value=settings.lmstudio_base_url)
    lmstudio_api_key = st.text_input("LM Studio API Key（選填）", value=settings.lmstudio_api_key or "", type="password")
    lmstudio_chat_model = st.text_input("Chat Model", value=settings.lmstudio_chat_model or "")
    lmstudio_vision_model = st.text_input("Vision Model", value=settings.lmstudio_vision_model or "")
    lmstudio_embedding_model = st.text_input("Embedding Model", value=settings.lmstudio_embedding_model or "")

# --- Other backends ---
st.subheader("其他設定")

vector_options = ["chroma", "memory"]
vector_index = vector_options.index(settings.vector_backend) if settings.vector_backend in vector_options else 0
vector_backend = st.selectbox("Vector Backend", vector_options, index=vector_index)

ocr_options = ["paddleocr", "mock"]
ocr_index = ocr_options.index(settings.ocr_backend) if settings.ocr_backend in ocr_options else 0
ocr_backend = st.selectbox("OCR Backend", ocr_options, index=ocr_index)

# --- Save ---
st.divider()

if st.button("💾 儲存設定", type="primary", use_container_width=True):
    kwargs: dict = {
        "provider_backend": provider_backend,
        "vector_backend": vector_backend,
        "ocr_backend": ocr_backend,
        "sqlite_path": settings.sqlite_path,
        "chroma_path": settings.chroma_path,
        "chroma_collection": settings.chroma_collection,
        "search_candidate_k_default": settings.search_candidate_k_default,
        "search_top_n_default": settings.search_top_n_default,
    }

    if provider_backend == "openai":
        kwargs["openai_api_key"] = openai_api_key or None
        kwargs["openai_base_url"] = openai_base_url.strip().rstrip("/") or None
        kwargs["openai_chat_model"] = openai_chat_model
        kwargs["openai_vision_model"] = openai_vision_model
        kwargs["openai_embedding_model"] = openai_embedding_model
    elif provider_backend == "lmstudio":
        kwargs["lmstudio_base_url"] = lmstudio_base_url.strip().rstrip("/")
        kwargs["lmstudio_api_key"] = lmstudio_api_key or None
        kwargs["lmstudio_chat_model"] = lmstudio_chat_model or None
        kwargs["lmstudio_vision_model"] = lmstudio_vision_model or None
        kwargs["lmstudio_embedding_model"] = lmstudio_embedding_model or None

    new_settings = AppSettings(**kwargs)
    save_settings(new_settings)
    st.success(f"設定已儲存至 `{DEFAULT_CONFIG_PATH}`")
    # Clear cached container so next page load picks up new settings
    if "container" in st.session_state:
        del st.session_state["container"]
