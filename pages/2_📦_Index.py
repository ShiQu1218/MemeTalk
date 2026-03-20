from __future__ import annotations

from pathlib import Path

import streamlit as st

from memetalk.app.container import build_container
from memetalk.app.indexer import IndexProgress
from memetalk.app.settings_io import load_settings, save_settings

st.set_page_config(page_title="MemeTalk - 索引", page_icon="📦", layout="wide")
st.title("📦 建立索引")
st.caption("選擇梗圖資料夾，建立或重新建立語意索引。")

settings = load_settings()

meme_folder = st.text_input(
    "梗圖資料夾路徑",
    value=settings.meme_folder,
    placeholder="例如：D:/MemeData",
    help="請輸入完整路徑，支援子資料夾。路徑會自動儲存，下次不用重新輸入。",
)

# Auto-save folder path when changed
if meme_folder != settings.meme_folder:
    settings.meme_folder = meme_folder
    save_settings(settings)

reindex = st.checkbox("強制重新索引", value=False, help="勾選後會重新處理所有圖片，包含已索引的")

_STEP_LABELS = {
    "scan": "掃描檔案",
    "sha256": "計算雜湊",
    "skip": "已索引，跳過",
    "ocr": "OCR 文字辨識",
    "metadata": "AI 分析圖片",
    "embedding": "產生向量",
    "store": "寫入資料庫",
    "done": "完成",
    "error": "處理失敗",
}

if st.button("🚀 開始建立索引", type="primary", use_container_width=True):
    folder_path = Path(meme_folder)
    if not meme_folder.strip():
        st.warning("請先輸入梗圖資料夾路徑。")
    elif not folder_path.exists():
        st.error(f"資料夾不存在：`{folder_path.resolve()}`")
    elif not folder_path.is_dir():
        st.error(f"指定路徑不是資料夾：`{folder_path.resolve()}`")
    else:
        try:
            with st.spinner("正在建立容器..."):
                container = build_container(settings)
                st.session_state["container"] = container

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            detail_container = st.container()
            stats_placeholder = st.empty()

            def _on_progress(p: IndexProgress) -> None:
                if p.total == 0:
                    return
                fraction = p.current / p.total
                progress_bar.progress(fraction, text=f"{p.current} / {p.total} 張")

                step_label = _STEP_LABELS.get(p.step, p.step)
                if p.step == "scan":
                    status_text.info(f"找到 {p.total} 張圖片，開始索引...")
                else:
                    status_text.info(f"[{p.current}/{p.total}] **{p.file_name}** — {step_label}")

                with stats_placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("已索引", f"{p.indexed} 張")
                    c2.metric("跳過", f"{p.skipped} 張")
                    c3.metric("失敗", f"{p.failed} 張")

            summary = container.indexing_service.build_index(
                folder_path, reindex=reindex, on_progress=_on_progress,
            )

            progress_bar.progress(1.0, text="索引完成！")
            status_text.empty()

            if summary.failed_count:
                st.warning(f"索引完成，但有 {summary.failed_count} 張失敗。")
            else:
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
