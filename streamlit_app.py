from __future__ import annotations

import httpx
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from memetalk.app.ui_config import resolve_api_base_url


def _load_streamlit_secrets():
    try:
        return st.secrets
    except StreamlitSecretNotFoundError:
        return None


API_BASE_URL = resolve_api_base_url(_load_streamlit_secrets())


st.set_page_config(page_title="MemeTalk Demo", layout="wide")
st.title("MemeTalk")
st.caption("輸入一句情境、吐槽或情緒描述，回傳最適合的回覆梗圖。")

query = st.text_area("想回什麼情境？", height=120, placeholder="例如：朋友說快到了但其實根本還沒出門")

if st.button("搜尋梗圖", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("請先輸入查詢內容。")
    else:
        with st.spinner("搜尋中..."):
            response = httpx.post(
                f"{API_BASE_URL}/api/v1/search",
                json={"query": query, "top_n": 3, "candidate_k": 8},
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()

        st.subheader("查詢分析")
        st.write(payload["query_analysis"])

        st.subheader("推薦結果")
        columns = st.columns(3)
        for column, result in zip(columns, payload["results"], strict=False):
            with column:
                st.image(result["image_url"], use_container_width=True)
                st.markdown(f"**推薦理由**\n\n{result['reason']}")
                st.caption(f"模板：{result['template_name'] or '未知模板'}")
                st.write("情緒標籤：", "、".join(result["emotion_tags"]) or "無")
                st.write("意圖標籤：", "、".join(result["intent_tags"]) or "無")
