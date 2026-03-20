# CLAUDE.md

## Project Overview

MemeTalk 是本機單機版梗圖語意搜尋系統，使用 Streamlit 前端 + FastAPI 備用 API。

## Common Commands

```bash
# 安裝
pip install -e .[dev,openai,chroma]

# 啟動 Streamlit
streamlit run streamlit_app.py

# 跑測試
pytest

# 啟動 FastAPI（替代方式）
uvicorn memetalk.api.main:app --reload
```

## Key Architecture

- **Provider 抽象**：OCR / Metadata / Embedding / QueryAnalyzer / Reranker 皆可替換
- **雙 embedding 索引**：每張梗圖建 semantic + reply 兩份 embedding，搜尋時依模式篩選
- **搜尋模式**：`reply`（預設，OCR 文字權重 90%）和 `semantic`（整體語意匹配）
- **Direct Mode**：Streamlit 直接呼叫 Python 服務，不需啟動 API server

## Important Conventions

- 語言：metadata、prompt、UI 文字皆使用繁體中文
- 設定優先順序：環境變數 > TOML (`data/memetalk_config.toml`) > Pydantic 預設值
- Provider backend：openai / lmstudio / mock / local
- Vector backend：chroma / memory

## Workflow Rules

- **每次功能開發或修改完成後，必須同步更新 OpenSpec (`openspec/specs/meme-search-mvp/spec.md`)**，確保 spec 與實作一致。
- README 也應在有使用者可見變更時一併更新。
