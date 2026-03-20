from __future__ import annotations

import streamlit as st

from memetalk.app.settings_io import load_settings

st.set_page_config(page_title="MemeTalk", page_icon="🎭", layout="wide")
st.title("🎭 MemeTalk")
st.caption("梗圖語意搜尋系統 — 在左側選單切換頁面。")

# --- Load settings ---
settings = load_settings()

# --- Settings status ---
st.subheader("⚙️ 目前設定")
col1, col2, col3 = st.columns(3)
col1.metric("Provider", settings.provider_backend)
col2.metric("Vector Backend", settings.vector_backend)
col3.metric("OCR Backend", settings.ocr_backend)

# --- Index status ---
st.subheader("📦 索引狀態")
try:
    from memetalk.storage.sqlite_store import SQLiteMemeRepository

    if settings.sqlite_path.exists():
        repo = SQLiteMemeRepository(settings.sqlite_path)
        repo.initialize()
        count = repo.count_assets()
        st.metric("已索引梗圖數量", f"{count} 張")
    else:
        st.info("尚未建立索引資料庫。請至「索引」頁面建立索引。")
except Exception as e:
    st.warning(f"無法讀取索引狀態：{e}")

# --- Health check ---
st.subheader("🏥 系統健康檢查")
try:
    from memetalk.app.container import build_container

    with st.spinner("檢查中..."):
        container = build_container(settings)
    st.success("系統正常，所有元件已就緒。")
    st.session_state["container"] = container

    provider_trace = container.providers.trace()
    with st.expander("Provider 詳細資訊"):
        for key, value in provider_trace.items():
            st.text(f"{key}: {value}")
except Exception as e:
    st.error(f"系統初始化失敗：{e}")
    st.info("請至「設定」頁面確認設定是否正確。")

# --- Navigation hints ---
st.divider()
st.markdown(
    """
**快速導航：**
- **⚙️ 設定** — 設定 Provider、API Key 等參數
- **📦 索引** — 建立或重新建立梗圖索引
- **🔍 搜尋** — 輸入情境搜尋最適合的梗圖
"""
)
