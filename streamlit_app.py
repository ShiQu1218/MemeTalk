from __future__ import annotations

import streamlit as st

from memetalk.app.settings_io import DEFAULT_CONFIG_PATH, load_settings
from memetalk.app.ui import format_path, render_notice, render_section, setup_page

setup_page(
    page_title="MemeTalk",
    page_icon="🎭",
    title="MemeTalk",
    subtitle="把梗圖索引、搜尋與設定集中在一個比較順手的工作台。",
    eyebrow="Dashboard",
    chips=("Direct Mode", "Streamlit", "Local Search"),
)

settings = load_settings()

render_section("系統快覽", "先確認目前的執行設定與預設索引來源。")
top_left, top_right = st.columns(2)
with top_left:
    provider_col, vector_col = st.columns(2)
    provider_col.metric("Provider", settings.provider_backend)
    vector_col.metric("Vector Backend", settings.vector_backend)
with top_right:
    ocr_col, folder_col = st.columns(2)
    ocr_col.metric("OCR Backend", settings.ocr_backend)
    folder_col.metric("預設梗圖資料夾", "已設定" if settings.meme_folder else "未設定")
st.caption(f"設定檔位置：`{DEFAULT_CONFIG_PATH}`")
st.caption(f"預設梗圖資料夾：`{format_path(settings.meme_folder)}`")

render_section("索引與健康", "首頁直接看到資料量與初始化狀態，減少來回切頁確認。")
status_left, status_right = st.columns(2)

with status_left:
    with st.container(border=True):
        st.subheader("索引狀態")
        try:
            from memetalk.storage.sqlite_store import SQLiteMemeRepository

            if settings.sqlite_path.exists():
                repo = SQLiteMemeRepository(settings.sqlite_path)
                repo.initialize()
                count = repo.count_assets()
                st.metric("已索引梗圖數量", f"{count} 張")
                render_notice("資料庫可讀", "索引資料已可供搜尋頁直接使用。", tone="success")
            else:
                st.metric("已索引梗圖數量", "0 張")
                render_notice("尚未建立索引", "先到 Index 頁指定資料夾並建立索引。", tone="warning")
        except Exception as exc:
            render_notice("無法讀取索引狀態", str(exc), tone="danger")

with status_right:
    with st.container(border=True):
        st.subheader("系統健康檢查")
        try:
            from memetalk.app.container import build_container

            with st.spinner("檢查元件中..."):
                container = build_container(settings)
            st.session_state["container"] = container
            st.success("系統正常，Provider 與儲存層已完成初始化。")

            provider_trace = container.providers.trace()
            with st.expander("Provider 詳細資訊"):
                for key, value in provider_trace.items():
                    st.write(f"**{key}**：{value}")
        except Exception as exc:
            render_notice("系統初始化失敗", str(exc), tone="danger")
            st.info("請到 Settings 頁確認 Provider、模型與 API 相關設定。")

render_section("快速操作", "常見流程維持三步，但每一頁都會帶同一套視覺與狀態呈現。")
with st.container(border=True):
    st.markdown(
        """
1. 到 `Settings` 設定 Provider、模型與預設梗圖資料夾。
2. 到 `Index` 以預設路徑或臨時路徑建立索引。
3. 到 `Search` 輸入情境與偏好語氣，挑選最適合的梗圖。
"""
    )
