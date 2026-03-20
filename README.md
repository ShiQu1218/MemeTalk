# MemeTalk

MemeTalk 是一個本機單機版的梗圖語意搜尋 MVP。它會把靜態圖片做 OCR、影像語意分析、embedding 建立與索引儲存，最後透過 FastAPI 提供搜尋 API，並附上一個 Streamlit Demo 介面。

## 專案目標

- 讓一批梗圖圖片可被自然語言搜尋。
- 把每張圖片整理成統一 metadata schema。
- 同時保留結構化資料與向量資料，讓搜尋結果可以解釋為什麼推薦這張圖。
- 支援 OpenAI 與 OpenAI-compatible provider，包含 LM Studio。

## 關鍵運作邏輯

### 1. 索引流程

執行 `index build --source <dir> [--reindex]` 時，系統會做以下事情：

1. 遞迴掃描資料夾內的 `.jpg`、`.jpeg`、`.png`、`.webp`。
2. 對每個檔案計算 SHA-256，作為去重與資產識別基礎。
3. 先做 OCR，若 OCR 失敗則降級成空文字，但不會中止整批。
4. 把圖片與 OCR 結果送進 metadata provider，產出統一結構的梗圖描述。
5. 依照 metadata 組合 `embedding_text`，把模板、場景、用途、OCR、情緒、意圖、風格串成向量語料。
6. 呼叫 embedding provider 產生向量。
7. 將 canonical metadata 寫入 SQLite，並把向量文件寫入 Chroma 或記憶體向量庫。
8. 若單張圖片失敗，錯誤會記錄在 `index_runs`，其餘圖片繼續處理。

對應模組：

- `src/memetalk/app/indexer.py`
- `src/memetalk/core/models.py`
- `src/memetalk/storage/sqlite_store.py`
- `src/memetalk/storage/vector_store.py`

### 2. 搜尋流程

呼叫 `POST /api/v1/search` 時，系統會做以下事情：

1. 先把使用者查詢分析成結構化欄位：
   - `situation`
   - `emotions`
   - `tone`
   - `reply_intent`
   - `query_embedding_text`
2. 用 `query_embedding_text` 產生查詢向量。
3. 從向量庫先取回 top-k 候選圖片。
4. 再用 reranker 依照語境、情緒與回覆意圖做第二次排序。
5. 若 rerank 失敗，會退回純向量排序，且每筆結果都會附 fallback reason。
6. API 回傳：
   - `query_analysis`
   - `results`
   - `provider_trace`

對應模組：

- `src/memetalk/app/search.py`
- `src/memetalk/api/main.py`

### 3. 資料儲存方式

MemeTalk 使用雙儲存結構：

- SQLite：保存圖片主資料、metadata、索引紀錄。
- Vector Store：保存 embedding 向量，供相似度搜尋。

目前支援：

- `memory`
- `chroma`

如果使用 `chroma`，請確保安裝的是 Python 3.14 相容版：

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

### 5. LM Studio 相容邏輯

LM Studio 透過 OpenAI-compatible API 接入，但實作上有幾個專案內已處理的相容細節：

- 可用 `base_url` 切到 `http://127.0.0.1:1234/v1`
- `.webp` 圖片在 LM Studio 路徑下會必要時轉成 PNG data URL
- 結構化 JSON 輸出固定使用低隨機性設定，遇到可恢復的 malformed JSON 會自動 retry
- 如果沒載入 chat / vision / embedding model，錯誤訊息會轉成較可操作的提示

這些處理都在：

- `src/memetalk/providers/openai_provider.py`

## 快速開始

1. 安裝專案與需要的 extras。
2. 建索引。
3. 啟動 API。
4. 啟動 Streamlit Demo。

```bash
pip install -e .[dev]
pip install -e .[openai]
pip install -e .[chroma]
memetalk index build --source D:/path/to/memes
uvicorn memetalk.api.main:app --reload
streamlit run streamlit_app.py
```

## 實際操作流程

下面是一套從零開始的實際使用順序，適合第一次把專案跑起來。

### 1. 安裝依賴

```bash
pip install -e .[dev]
pip install -e .[openai]
pip install -e .[chroma]
```

如果你要用 OCR，也要安裝：

```bash
pip install -e .[ocr]
```

### 2. 設定 provider

如果你要用 LM Studio，可先在 PowerShell 設定：

```powershell
$env:MEMETALK_PROVIDER_BACKEND="lmstudio"
$env:MEMETALK_LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
$env:MEMETALK_LMSTUDIO_CHAT_MODEL="your-chat-model-id"
$env:MEMETALK_LMSTUDIO_VISION_MODEL="your-vision-model-id"
$env:MEMETALK_LMSTUDIO_EMBEDDING_MODEL="your-embedding-model-id"
```

如果你只是要先驗證流程是否正常，可改用 mock：

```powershell
$env:MEMETALK_PROVIDER_BACKEND="mock"
$env:MEMETALK_VECTOR_BACKEND="memory"
$env:MEMETALK_OCR_BACKEND="mock"
```

### 3. 準備圖片資料夾

把要索引的梗圖放在同一個資料夾底下，例如：

```text
D:\MemeData
```

支援的圖片格式：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

### 4. 建立索引

第一次建立索引：

```powershell
memetalk index build --source D:\MemeData
```

如果你要整批重建資料，使用：

```powershell
memetalk index build --source D:\MemeData --reindex
```

成功後會看到一段 JSON 結果，裡面包含：

- `processed_count`
- `indexed_count`
- `skipped_count`
- `failed_count`
- `errors`

### 5. 啟動 API

```bash
uvicorn memetalk.api.main:app --reload
```

正常啟動後，API 預設會在：

```text
http://127.0.0.1:8000
```

你可以先用健康檢查確認：

```text
GET http://127.0.0.1:8000/api/v1/health
```

### 6. 啟動 Streamlit Demo

開另一個 terminal 執行：

```bash
streamlit run streamlit_app.py
```

如果沒有 `.streamlit/secrets.toml` 也沒關係，程式會自動退回本機 API 位址。

啟動後通常可在瀏覽器打開：

```text
http://localhost:8501
```

### 7. 實際搜尋

在 Demo 頁面輸入一句自然語言，例如：

- `朋友說快到了但其實根本還沒出門`
- `主管又突然改需求`
- `今天上班好厭世`

送出後，畫面會顯示：

- 查詢分析結果
- top-3 梗圖推薦
- 每張圖的推薦理由
- 情緒標籤與意圖標籤

### 8. 直接打 API 測試

如果你不想先開 Streamlit，也可以直接測 API：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"主管又突然改需求\",\"top_n\":3,\"candidate_k\":8}"
```

### 9. 常見使用順序總結

最常見的操作順序就是：

1. 安裝依賴
2. 設定 provider
3. `memetalk index build --source ...`
4. `uvicorn memetalk.api.main:app --reload`
5. `streamlit run streamlit_app.py`
6. 打開 `http://localhost:8501` 開始搜尋

## 環境變數

系統會讀取以下環境變數：

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

若只想本機驗證流程，不想碰外部服務，可使用：

```bash
MEMETALK_PROVIDER_BACKEND=mock
MEMETALK_VECTOR_BACKEND=memory
```

## LM Studio 設定範例

先安裝 OpenAI client extra：

```bash
pip install -e .[openai]
```

PowerShell 範例：

```powershell
$env:MEMETALK_PROVIDER_BACKEND="lmstudio"
$env:MEMETALK_LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
$env:MEMETALK_LMSTUDIO_CHAT_MODEL="your-chat-model-id"
$env:MEMETALK_LMSTUDIO_VISION_MODEL="your-vision-model-id"
$env:MEMETALK_LMSTUDIO_EMBEDDING_MODEL="your-embedding-model-id"
```

注意：

- `MEMETALK_LMSTUDIO_EMBEDDING_MODEL` 必須是 embedding-capable model。
- 不要假設 chat model 或 vision model 一定支援 `/v1/embeddings`。
- 如果你維持 `MEMETALK_PROVIDER_BACKEND=openai`，也可以只把 `MEMETALK_OPENAI_BASE_URL` 指到 LM Studio。

## API 與 Demo

### FastAPI

- `GET /api/v1/health`
- `POST /api/v1/search`
- `GET /api/v1/assets/{image_id}`

### Streamlit Demo

Demo 會提供：

- 一個自然語言輸入框
- 查詢送出按鈕
- top-3 推薦卡片
- 推薦理由
- 情緒與意圖標籤顯示

如果沒有 `.streamlit/secrets.toml`，Demo 會自動退回 `MEMETALK_API_BASE_URL`，若環境變數也沒有設定，則使用 `http://127.0.0.1:8000`。

## OpenSpec

此專案以 OpenSpec 作為 source of truth。核心需求定義在：

- `openspec/specs/meme-search-mvp/spec.md`

目前 README 描述的是既有實作與 OpenSpec 對應關係，不是額外規格來源。
