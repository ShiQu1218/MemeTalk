from __future__ import annotations

import streamlit as st

from memetalk.app.container import build_container
from memetalk.app.settings_io import load_settings
from memetalk.core.models import SearchMode

st.set_page_config(page_title="MemeTalk - 搜尋", page_icon="🔍", layout="wide")
st.title("🔍 搜尋梗圖")
st.caption("輸入一句情境、吐槽或情緒描述，回傳最適合的回覆梗圖。")


def _get_container():
    if "container" not in st.session_state:
        settings = load_settings()
        st.session_state["container"] = build_container(settings)
    return st.session_state["container"]


_MODE_OPTIONS = {"適合回覆": SearchMode.REPLY, "契合語意": SearchMode.SEMANTIC}
mode_label = st.radio("搜尋模式", list(_MODE_OPTIONS.keys()), horizontal=True)
search_mode = _MODE_OPTIONS[mode_label]

query = st.text_area("想回什麼情境？", height=120, placeholder="例如：朋友說快到了但其實根本還沒出門")
preferred_tone = st.text_input(
    "偏好的梗圖語氣（可選）",
    placeholder="例如：嘴砲、冷淡、可憐、陰陽怪氣",
)

if st.button("搜尋梗圖", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("請先輸入查詢內容。")
    else:
        try:
            with st.spinner("搜尋中..."):
                container = _get_container()
                response = container.search_service.search(
                    query=query,
                    top_n=container.settings.search_top_n_default,
                    candidate_k=container.settings.search_candidate_k_default,
                    mode=search_mode,
                    preferred_tone=preferred_tone,
                )

            st.subheader("查詢分析")
            analysis = response.query_analysis
            st.write(f"**情境：** {analysis.situation}")
            st.write(f"**情緒：** {'、'.join(analysis.emotions) or '無'}")
            st.write(f"**語氣：** {analysis.tone}")
            st.write(f"**回覆意圖：** {analysis.reply_intent}")
            st.write(f"**偏好梗圖語氣：** {analysis.preferred_tone or '未指定'}")
            st.write(f"**搜尋文字：** {analysis.query_embedding_text}")
            st.write(f"**查詢詞：** {'、'.join(analysis.query_terms) or '無'}")
            st.write(f"**模板提示：** {'、'.join(analysis.template_hints) or '無'}")

            st.subheader("搜尋追蹤")
            st.write(f"**路由：** {'、'.join(response.search_trace.routes_used) or '無'}")
            st.write(f"**候選數：** {response.search_trace.candidate_counts}")
            st.write(f"**Rerank 策略：** {response.search_trace.rerank_strategy}")
            if response.search_trace.degraded_routes:
                st.warning(f"降級路由：{'、'.join(response.search_trace.degraded_routes)}")

            st.subheader("推薦結果")
            if not response.results:
                st.info("沒有找到符合的梗圖。請確認已建立索引。")
            else:
                columns = st.columns(min(len(response.results), 3))
                for column, result in zip(columns, response.results, strict=False):
                    with column:
                        # Direct mode: read image from file_path stored in asset
                        asset = container.repository.get_asset_by_id(result.image_id)
                        if asset:
                            try:
                                st.image(asset.file_path, use_container_width=True)
                            except Exception:
                                st.warning(f"無法載入圖片：{asset.file_path}")
                        st.markdown(f"**推薦理由**\n\n{result.reason}")
                        st.caption(f"模板：{result.template_name or '未知模板'}")
                        st.write("情緒標籤：", "、".join(result.emotion_tags) or "無")
                        st.write("意圖標籤：", "、".join(result.intent_tags) or "無")
                        st.write("候選來源：", "、".join(result.debug.candidate_sources) or "無")
                        if result.debug.degradation_flags:
                            st.caption("降級訊號：" + "、".join(result.debug.degradation_flags))
        except Exception as e:
            st.error(f"搜尋失敗：{e}")
