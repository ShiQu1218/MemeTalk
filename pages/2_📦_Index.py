from __future__ import annotations

from pathlib import Path

import streamlit as st

from memetalk.app.container import build_container
from memetalk.app.settings_io import load_settings, save_settings

st.set_page_config(page_title="MemeTalk - 索引", page_icon="📦", layout="wide")
st.title("📦 建立索引")
st.caption("選擇梗圖資料夾，建立或重新建立語意索引。")

settings = load_settings()

meme_folder = st.text_input("梗圖資料夾路徑", value=settings.meme_folder, placeholder="例如：D:/MemeData", help="請輸入完整路徑，支援子資料夾")
reindex = st.checkbox("強制重新索引", value=False, help="勾選後會重新處理所有圖片，包含已索引的")

if st.button("🚀 開始建立索引", type="primary", use_container_width=True):
    folder_path = Path(meme_folder)
    if not folder_path.exists():
        st.error(f"資料夾不存在：`{folder_path.resolve()}`")
    elif not folder_path.is_dir():
        st.error(f"指定路徑不是資料夾：`{folder_path.resolve()}`")
    else:
        try:
            with st.spinner("正在建立容器..."):
                container = build_container(settings)
                st.session_state["container"] = container

            with st.spinner("正在索引梗圖，這可能需要一些時間..."):
                summary = container.indexing_service.build_index(folder_path, reindex=reindex)

            if settings.meme_folder != meme_folder:
                settings.meme_folder = meme_folder
                save_settings(settings)

            st.success("索引完成！")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("處理", f"{summary.processed_count} 張")
            col2.metric("已索引", f"{summary.indexed_count} 張")
            col3.metric("跳過", f"{summary.skipped_count} 張")
            col4.metric("失敗", f"{summary.failed_count} 張")

            if summary.errors:
                with st.expander(f"錯誤詳情（{len(summary.errors)} 筆）"):
                    for error in summary.errors:
                        st.text(f"❌ {error.file_path}")
                        st.code(error.error)
        except Exception as e:
            st.error(f"索引過程發生錯誤：{e}")
