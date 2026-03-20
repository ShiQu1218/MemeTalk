from __future__ import annotations

import threading
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

# Initialize cancel event
if "index_cancel_event" not in st.session_state:
    st.session_state.index_cancel_event = threading.Event()


def _request_stop() -> None:
    st.session_state.index_cancel_event.set()


if st.button("🚀 開始建立索引", type="primary", use_container_width=True):
    # Reset cancel event for new run
    st.session_state.index_cancel_event.clear()

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

            # Stop button — rendered before the blocking call so it's interactive
            stop_placeholder = st.empty()
            stop_placeholder.button(
                "⏹️ 停止索引",
                type="secondary",
                use_container_width=True,
                on_click=_request_stop,
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            stats_placeholder = st.empty()
            log_container = st.container()

            cancel_event = st.session_state.index_cancel_event
            _shown = {"warnings": 0, "errors": 0}

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
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("已索引", f"{p.indexed} 張")
                    c2.metric("跳過", f"{p.skipped} 張")
                    c3.metric("失敗", f"{p.failed} 張")
                    c4.metric("警告", f"{p.warnings} 筆")

                # Stream new warnings / errors as they appear
                with log_container:
                    for w in p.warning_records[_shown["warnings"]:]:
                        st.warning(f"⚠️ **{Path(w.file_path).name}** [{w.stage}] {w.warning}")
                    _shown["warnings"] = len(p.warning_records)

                    for e in p.error_records[_shown["errors"]:]:
                        st.error(f"❌ **{Path(e.file_path).name}** {e.error}")
                    _shown["errors"] = len(p.error_records)

            summary = container.indexing_service.build_index(
                folder_path,
                reindex=reindex,
                on_progress=_on_progress,
                cancel_check=cancel_event.is_set,
            )

            # Remove stop button after completion
            stop_placeholder.empty()

            if summary.status == "cancelled":
                progress_bar.progress(
                    summary.processed_count / max(summary.processed_count + 1, 1),
                    text="索引已中斷",
                )
                status_text.empty()
                st.warning("索引已由使用者中斷。已完成的索引已儲存。")
            else:
                progress_bar.progress(1.0, text="索引完成！")
                status_text.empty()

                if summary.failed_count:
                    st.warning(f"索引完成，但有 {summary.failed_count} 張失敗。")
                elif summary.warning_count:
                    st.warning(f"索引完成，但有 {summary.warning_count} 筆降級警告。")
                else:
                    st.success("索引完成！")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("處理", f"{summary.processed_count} 張")
            col2.metric("已索引", f"{summary.indexed_count} 張")
            col3.metric("跳過", f"{summary.skipped_count} 張")
            col4.metric("失敗", f"{summary.failed_count} 張")
            col5.metric("警告", f"{summary.warning_count} 筆")

            if summary.errors:
                with st.expander(f"錯誤詳情（{len(summary.errors)} 筆）"):
                    for error in summary.errors:
                        st.text(f"❌ {error.file_path}")
                        st.code(error.error)
            if summary.warnings:
                with st.expander(f"警告詳情（{len(summary.warnings)} 筆）"):
                    for warning in summary.warnings:
                        st.text(f"⚠️ {warning.file_path}")
                        st.code(f"[{warning.stage}] {warning.warning}")
        except Exception as e:
            # Re-raise Streamlit internal exceptions (RerunException, StopException)
            from streamlit.runtime.scriptrunner import RerunException, StopException
            if isinstance(e, (RerunException, StopException)):
                raise
            st.error(f"索引過程發生錯誤：{e}")
