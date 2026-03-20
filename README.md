# MemeTalk

MemeTalk 是一個本機單機版的梗圖語意搜尋系統。它會把靜態圖片做 OCR、影像語意分析、embedding 建立與索引儲存，並透過 Streamlit 多頁面介面提供一站式的設定、索引與搜尋體驗。

## 專案目標

- 讓一批梗圖圖片可被自然語言搜尋。
- 把每張圖片整理成統一 metadata schema。
- 同時保留結構化資料與向量資料，讓搜尋結果可以解釋為什麼推薦這張圖。
- 支援 OpenAI 與 OpenAI-compatible provider，包含 LM Studio。

## 快速開始

```bash
pip install -e .[dev,openai,chroma]
streamlit run streamlit_app.py
```

若要使用 PaddleOCR backend，請使用專案虛擬環境並安裝固定 OCR 依賴：

```bash
python -m pip install -e .[ocr]
```

Windows CPU 目前驗證可用的組合是：

- `paddleocr==2.10.0`
- `paddlepaddle==3.1.1`

若升到較新的 `paddlepaddle` 版本並出現 `OneDnnContext ... fused_conv2d`，請退回 `3.1.1`。

打開瀏覽器 `http://localhost:8501`，透過左側選單操作：

1. **⚙️ 設定** — 選擇 Provider、輸入 API Key、設定 Vector/OCR Backend
2. **📦 索引** — 指定梗圖資料夾，建立語意索引
3. **🔍 搜尋** — 輸入自然語言情境，搜尋最適合的梗圖

只需這一條指令，不需要手動設定環境變數、不需要另外啟動 API 伺服器。

## 架構：Multi-Page Streamlit + Direct Mode

所有頁面直接呼叫 Python 服務（Direct Mode），不經 HTTP API，無需另外啟動伺服器。

### 設定管理

設定存於 `data/memetalk_config.toml`（TOML 格式），優先順序：

1. 環境變數（最高優先）
2. TOML 設定檔
3. Pydantic 預設值

環境變數仍可使用，向後相容。

### 頁面結構

| 頁面 | 檔案 | 功能 |
|------|------|------|
| Dashboard | `streamlit_app.py` | 系統狀態總覽、已索引數量、健康檢查 |
| 設定 | `pages/1_⚙️_Settings.py` | Provider / API Key / Backend 設定 |
| 索引 | `pages/2_📦_Index.py` | 選擇資料夾、建立/重建索引 |
| 搜尋 | `pages/3_🔍_Search.py` | 搜尋模式選擇（適合回覆 / 契合語意）、自然語言搜尋、梗圖推薦結果 |

## 關鍵運作邏輯

### 1. 索引流程

在「索引」頁面指定梗圖資料夾後，系統會：

1. 遞迴掃描資料夾內的 `.jpg`、`.jpeg`、`.png`、`.webp`。
2. 對每個檔案計算 SHA-256，作為去重與資產識別基礎。
3. 先做 OCR，若 OCR 失敗則降級成空文字，但不會中止整批。
4. 把圖片與 OCR 結果送進 metadata provider，產出統一結構的梗圖描述。
5. 依照 metadata 組合 `embedding_text`，把模板、場景、用途、OCR、情緒、意圖、風格串成向量語料。
6. 為每張梗圖建立兩份 embedding：
   - **語意模式 (semantic)**：使用完整 `embedding_text`，涵蓋所有 metadata。
   - **回覆模式 (reply)**：以 OCR 文字為主（重複強調），適合比對「適合拿來回覆」的梗圖。
7. 呼叫 embedding provider 一次產生兩組向量。
8. 將 canonical metadata 寫入 SQLite，並把兩份向量文件寫入 Chroma 或記憶體向量庫。
8. 若單張圖片失敗，錯誤會記錄在 `index_runs`，其餘圖片繼續處理。

對應模組：

- `src/memetalk/app/indexer.py`
- `src/memetalk/core/models.py`
- `src/memetalk/storage/sqlite_store.py`
- `src/memetalk/storage/vector_store.py`

### 2. 搜尋流程

在「搜尋」頁面輸入查詢後，系統會：

1. 使用者先選擇搜尋模式：
   - **適合回覆 (reply)**：找適合當作回應的梗圖，OCR 文字權重佔 90%。（預設）
   - **契合語意 (semantic)**：找主題相近的梗圖，整體語意匹配。
2. 把使用者查詢分析成結構化欄位：
   - `situation`、`emotions`、`tone`、`reply_intent`、`query_embedding_text`
   - 回覆模式下，`query_embedding_text` 會描述「適合回覆的梗圖會說什麼」而非重述輸入。
3. 用 `query_embedding_text` 產生查詢向量。
4. 從向量庫依所選模式篩選對應的 embedding，取回 top-k 候選圖片（預設 candidate_k=15）。
5. 再用 reranker 依照模式做第二次排序（回覆模式重 OCR 文字、語意模式重整體匹配）。
6. 若 rerank 失敗，會退回純向量排序，且每筆結果都會附 fallback reason。
7. 搜尋結果直接顯示在頁面上，圖片從本機檔案路徑載入。

對應模組：

- `src/memetalk/app/search.py`
- `pages/3_🔍_Search.py`

### 3. 資料儲存方式

MemeTalk 使用雙儲存結構：

- SQLite：保存圖片主資料、metadata、索引紀錄。
- Vector Store：保存 embedding 向量，供相似度搜尋。

目前支援：

- `memory`
- `chroma`

如果使用 `chroma`，請確保安裝相容版：

```bash
pip install -e .[chroma]
```

### 4. Provider 抽象

Provider 層是這個專案的核心抽象。每個能力都可替換：

- OCR：`extract_text`
- Metadata 分析：`analyze_image`
- Embedding：`embed_texts`
- 查詢分析：`analyze_query`
- Rerank：`rerank`

目前 provider backend：

- `openai`
- `lmstudio`
- `mock`
- `local`（MVP 階段保留介面，未完整實作）

對應模組：

- `src/memetalk/core/providers.py`
- `src/memetalk/providers/registry.py`
- `src/memetalk/providers/openai_provider.py`
- `src/memetalk/providers/mock.py`
- `src/memetalk/providers/paddleocr_provider.py`

### 4.1 OCR Runtime 注意事項

PaddleOCR 在 Windows CPU 環境下，目前以 `paddleocr==2.10.0` 搭配 `paddlepaddle==3.1.1` 驗證可正常運作。

若看到以下錯誤：

- `OneDnnContext does not have the input Filter`
- `operator < fused_conv2d > error`

這通常是 Paddle runtime 相容性問題，不是梗圖資料本身壞掉。請先確認你是用專案的 `.venv` 啟動，並將 `paddlepaddle` 降回 `3.1.1`。

### 5. LM Studio 相容邏輯

LM Studio 透過 OpenAI-compatible API 接入，但實作上有幾個專案內已處理的相容細節：

- 可用 `base_url` 切到 `http://127.0.0.1:1234/v1`
- `.webp` 圖片在 LM Studio 路徑下會必要時轉成 PNG data URL
- 結構化 JSON 輸出固定使用低隨機性設定，遇到可恢復的 malformed JSON 會自動 retry
- 如果沒載入 chat / vision / embedding model，錯誤訊息會轉成較可操作的提示

這些處理都在：

- `src/memetalk/providers/openai_provider.py`

## 進階用法

### CLI 索引（替代方式）

除了 Streamlit 介面，也可以用 CLI 建立索引：

```bash
memetalk index build --source D:/path/to/memes
memetalk index build --source D:/path/to/memes --reindex
```

### FastAPI 伺服器（替代方式）

如果需要 HTTP API，可以另外啟動 FastAPI：

```bash
uvicorn memetalk.api.main:app --reload
```

API 端點：

- `GET /api/v1/health`
- `POST /api/v1/search` — 支援 `mode` 欄位（`semantic` | `reply`，預設 `reply`）
- `GET /api/v1/assets/{image_id}`

### 環境變數（替代方式）

設定頁面可以取代手動設定環境變數。如果仍需使用環境變數，系統會讀取以下項目（環境變數優先於 TOML 設定）：

- `MEMETALK_SQLITE_PATH`
- `MEMETALK_VECTOR_BACKEND`
- `MEMETALK_CHROMA_PATH`
- `MEMETALK_CHROMA_COLLECTION`
- `MEMETALK_PROVIDER_BACKEND`
- `MEMETALK_OCR_BACKEND`
- `MEMETALK_OPENAI_BASE_URL`
- `MEMETALK_OPENAI_API_KEY`
- `MEMETALK_OPENAI_CHAT_MODEL`
- `MEMETALK_OPENAI_VISION_MODEL`
- `MEMETALK_OPENAI_EMBEDDING_MODEL`
- `MEMETALK_LMSTUDIO_BASE_URL`
- `MEMETALK_LMSTUDIO_API_KEY`
- `MEMETALK_LMSTUDIO_CHAT_MODEL`
- `MEMETALK_LMSTUDIO_VISION_MODEL`
- `MEMETALK_LMSTUDIO_EMBEDDING_MODEL`
- `MEMETALK_API_BASE_URL`

若只想本機驗證流程，不想碰外部服務，可在設定頁面選擇 `mock` provider 和 `memory` vector backend。

## OpenSpec

此專案以 OpenSpec 作為 source of truth。核心需求定義在：

- `openspec/specs/meme-search-mvp/spec.md`

目前 README 描述的是既有實作與 OpenSpec 對應關係，不是額外規格來源。
