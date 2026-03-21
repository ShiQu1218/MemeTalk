from __future__ import annotations

from contextlib import nullcontext

import streamlit as st

from memetalk.app.container import build_container
from memetalk.app.query_image import temporary_query_image
from memetalk.app.settings_io import load_settings
from memetalk.app.ui import render_notice, render_section, setup_page
from memetalk.core.models import SearchMode

setup_page(
    page_title="MemeTalk - 搜尋",
    page_icon="🔍",
    title="搜尋梗圖",
    subtitle="輸入情境、對話或上傳參考圖片，讓系統回傳比較能直接拿來用的梗圖。",
    eyebrow="Search",
    chips=("Reply Mode", "Semantic Mode", "Preferred Tone", "Image Query"),
)


def _get_container():
    if "container" not in st.session_state:
        settings = load_settings()
        st.session_state["container"] = build_container(settings)
    return st.session_state["container"]


_MODE_OPTIONS = {"適合回覆": SearchMode.REPLY, "契合語意": SearchMode.SEMANTIC}

render_section("查詢條件", "回覆模式偏向能直接拿去回嘴，語意模式偏向找情境接近的梗圖。")
with st.container(border=True):
    mode_label = st.radio("搜尋模式", list(_MODE_OPTIONS.keys()), horizontal=True)
    search_mode = _MODE_OPTIONS[mode_label]
    query = st.text_area("想回什麼情境？", height=120, placeholder="例如：朋友說快到了但其實根本還沒出門")
    uploaded_image = st.file_uploader(
        "參考圖片（可選）",
        type=["png", "jpg", "jpeg", "webp"],
        help="可單獨上傳圖片搜尋，也可以搭配文字一起查。",
    )
    if uploaded_image is not None:
        st.image(uploaded_image.getvalue(), caption="查詢圖片預覽", use_container_width=True)
    preferred_tone = st.text_input(
        "偏好的梗圖語氣（可選）",
        placeholder="例如：嘴砲、冷淡、可憐、陰陽怪氣",
    )

if st.button("搜尋梗圖", type="primary", use_container_width=True):
    if not query.strip() and uploaded_image is None:
        st.warning("請先輸入查詢內容，或上傳一張參考圖片。")
    else:
        try:
            image_context = nullcontext(None)
            if uploaded_image is not None:
                image_context = temporary_query_image(
                    uploaded_image.getvalue(),
                    filename=uploaded_image.name,
                    media_type=uploaded_image.type,
                )
            with image_context as query_image_path:
                with st.spinner("搜尋中..."):
                    container = _get_container()
                    response = container.search_service.search(
                        query=query,
                        top_n=container.settings.search_top_n_default,
                        candidate_k=container.settings.search_candidate_k_default,
                        mode=search_mode,
                        preferred_tone=preferred_tone,
                        query_image_path=query_image_path,
                    )

            render_section("查詢分析", "把系統理解到的情境、情緒與語氣拆開顯示，方便判斷是否找對方向。")
            analysis = response.query_analysis
            with st.container(border=True):
                st.write(f"**情境**：{analysis.situation}")
                st.write(f"**情緒**：{'、'.join(analysis.emotions) or '無'}")
                st.write(f"**語氣**：{analysis.tone}")
                st.write(f"**回覆意圖**：{analysis.reply_intent}")
                st.write(f"**偏好梗圖語氣**：{analysis.preferred_tone or '未指定'}")
                st.write(f"**搜尋文字**：{analysis.query_embedding_text}")
                st.write(f"**查詢詞**：{'、'.join(analysis.query_terms) or '無'}")
                st.write(f"**模板提示**：{'、'.join(analysis.template_hints) or '無'}")

            render_section("搜尋追蹤", "保留最小但足夠的除錯資訊，方便判斷是查詢理解還是召回路由出了問題。")
            with st.container(border=True):
                st.write(f"**路由**：{'、'.join(response.search_trace.routes_used) or '無'}")
                st.write(f"**候選數**：{response.search_trace.candidate_counts}")
                st.write(f"**Rerank 策略**：{response.search_trace.rerank_strategy}")
                if response.search_trace.degraded_routes:
                    render_notice(
                        "路由已降級",
                        "、".join(response.search_trace.degraded_routes),
                        tone="warning",
                    )

            render_section("推薦結果", "結果改成逐張堆疊，避免固定多欄在窄版時變得難讀。")
            if not response.results:
                render_notice("沒有找到結果", "請先確認已完成索引，或換一種描述方式再試一次。", tone="warning")
            else:
                for rank, result in enumerate(response.results, start=1):
                    with st.container(border=True):
                        header_col, score_col = st.columns((1.3, 0.7))
                        header_col.subheader(f"#{rank} {result.template_name or '未知模板'}")
                        score_col.metric("分數", f"{result.score:.3f}")

                        asset = container.repository.get_asset_by_id(result.image_id)
                        if asset:
                            try:
                                st.image(asset.file_path, use_container_width=True)
                            except Exception:
                                st.warning(f"無法載入圖片：{asset.file_path}")

                        st.markdown("**推薦理由**")
                        st.write(result.reason)

                        tag_col1, tag_col2 = st.columns(2)
                        tag_col1.write("**情緒標籤**")
                        tag_col1.write("、".join(result.emotion_tags) or "無")
                        tag_col2.write("**意圖標籤**")
                        tag_col2.write("、".join(result.intent_tags) or "無")

                        st.caption(f"候選來源：{'、'.join(result.debug.candidate_sources) or '無'}")
                        if result.debug.degradation_flags:
                            st.caption(f"降級訊號：{'、'.join(result.debug.degradation_flags)}")
        except Exception as exc:
            st.error(f"搜尋失敗：{exc}")
