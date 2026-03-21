from __future__ import annotations

import threading
from pathlib import Path

import streamlit as st

from memetalk.app.container import build_container
from memetalk.app.indexer import IndexProgress
from memetalk.app.settings_io import load_settings
from memetalk.app.ui import format_path, render_notice, render_section, setup_page
from memetalk.app.ui_state import resolve_index_folder_input

setup_page(
    page_title="MemeTalk - 索引",
    page_icon="📦",
    title="建立索引",
    subtitle="用預設資料夾快速開跑，也能臨時指定另一個資料夾做這次索引。",
    eyebrow="Index",
    chips=("Incremental Indexing", "Per-run Override", "Progress Trace"),
)

settings = load_settings()

saved_folder = settings.meme_folder
current_folder_input = st.session_state.get("index_folder_input")
last_saved_folder = st.session_state.get("index_folder_saved")
st.session_state["index_folder_input"] = resolve_index_folder_input(saved_folder, current_folder_input, last_saved_folder)
st.session_state["index_folder_saved"] = saved_folder

render_section("索引來源", "設定頁保存的是預設路徑，這裡的輸入只影響本次執行。")
top_left, top_right = st.columns((1.4, 1.0))
with top_left:
    meme_folder = st.text_input(
        "本次索引資料夾",
        key="index_folder_input",
        placeholder="例如：D:/MemeData",
        help="支援子資料夾遞迴掃描。若要變更預設值，請到 Settings 頁儲存。",
    )
    reindex = st.checkbox("強制重新索引", value=False, help="勾選後會重新處理所有圖片，包含已索引的")

with top_right:
    with st.container(border=True):
        st.subheader("目前預設")
        st.caption(f"預設梗圖資料夾：`{format_path(saved_folder)}`")
        render_notice("保存規則", "在這頁修改路徑不會覆寫 Settings 的預設值。", tone="info")

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

if "index_cancel_event" not in st.session_state:
    st.session_state.index_cancel_event = threading.Event()


def _request_stop() -> None:
    st.session_state.index_cancel_event.set()


def _render_progress_metrics(progress: IndexProgress) -> None:
    metric_row1 = st.columns(2)
    metric_row2 = st.columns(2)
    metric_row1[0].metric("已索引", f"{progress.indexed} 張")
    metric_row1[1].metric("跳過", f"{progress.skipped} 張")
    metric_row2[0].metric("失敗", f"{progress.failed} 張")
    metric_row2[1].metric("警告", f"{progress.warnings} 筆")


def _render_summary_metrics(processed: int, indexed: int, skipped: int, failed: int, warnings: int) -> None:
    summary_row1 = st.columns(3)
    summary_row2 = st.columns(2)
    summary_row1[0].metric("處理", f"{processed} 張")
    summary_row1[1].metric("已索引", f"{indexed} 張")
    summary_row1[2].metric("跳過", f"{skipped} 張")
    summary_row2[0].metric("失敗", f"{failed} 張")
    summary_row2[1].metric("警告", f"{warnings} 筆")


st.divider()
if st.button("開始建立索引", type="primary", use_container_width=True):
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
            with st.spinner("正在建立執行容器..."):
                container = build_container(settings)
                st.session_state["container"] = container

            stop_placeholder = st.empty()
            stop_placeholder.button(
                "停止索引",
                type="secondary",
                use_container_width=True,
                on_click=_request_stop,
            )

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            stats_placeholder = st.empty()
            log_container = st.container()

            cancel_event = st.session_state.index_cancel_event
            shown = {"warnings": 0, "errors": 0}

            def _on_progress(progress: IndexProgress) -> None:
                if progress.total == 0:
                    return

                fraction = progress.current / progress.total
                progress_bar.progress(fraction, text=f"{progress.current} / {progress.total} 張")

                step_label = _STEP_LABELS.get(progress.step, progress.step)
                if progress.step == "scan":
                    status_text.info(f"找到 {progress.total} 張圖片，開始索引。")
                else:
                    status_text.info(f"[{progress.current}/{progress.total}] {progress.file_name} - {step_label}")

                with stats_placeholder.container():
                    _render_progress_metrics(progress)

                with log_container:
                    for warning in progress.warning_records[shown["warnings"] :]:
                        st.warning(f"**{Path(warning.file_path).name}** [{warning.stage}] {warning.warning}")
                    shown["warnings"] = len(progress.warning_records)

                    for error in progress.error_records[shown["errors"] :]:
                        st.error(f"**{Path(error.file_path).name}** {error.error}")
                    shown["errors"] = len(progress.error_records)

            summary = container.indexing_service.build_index(
                folder_path,
                reindex=reindex,
                on_progress=_on_progress,
                cancel_check=cancel_event.is_set,
            )

            stop_placeholder.empty()
            render_section("索引結果", "完整摘要保留在同一頁，方便直接確認失敗與警告。")

            if summary.status == "cancelled":
                progress_bar.progress(summary.processed_count / max(summary.processed_count + 1, 1), text="索引已中斷")
                status_text.empty()
                render_notice("索引已中斷", "已完成的索引結果會保留下來，之後可從相同或不同資料夾續跑。", tone="warning")
            else:
                progress_bar.progress(1.0, text="索引完成")
                status_text.empty()

                if summary.failed_count:
                    render_notice("索引完成但有失敗", f"共有 {summary.failed_count} 張圖片失敗，請檢查下方詳情。", tone="warning")
                elif summary.warning_count:
                    render_notice("索引完成但有降級警告", f"共有 {summary.warning_count} 筆警告，已記錄在下方。", tone="warning")
                else:
                    render_notice("索引完成", "全部圖片已順利處理並寫入索引。", tone="success")

            _render_summary_metrics(
                summary.processed_count,
                summary.indexed_count,
                summary.skipped_count,
                summary.failed_count,
                summary.warning_count,
            )

            if summary.errors:
                with st.expander(f"錯誤詳情（{len(summary.errors)} 筆）"):
                    for error in summary.errors:
                        st.text(error.file_path)
                        st.code(error.error)

            if summary.warnings:
                with st.expander(f"警告詳情（{len(summary.warnings)} 筆）"):
                    for warning in summary.warnings:
                        st.text(warning.file_path)
                        st.code(f"[{warning.stage}] {warning.warning}")
        except Exception as exc:
            from streamlit.runtime.scriptrunner import RerunException, StopException

            if isinstance(exc, (RerunException, StopException)):
                raise
            st.error(f"索引過程發生錯誤：{exc}")
