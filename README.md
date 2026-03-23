# MemeTalk

MemeTalk 是一個本機單機版的梗圖搜尋系統。它把靜態圖片轉成可檢索的結構化資料與向量資料，支援兩種主要任務：

- `reply`：找一張適合直接拿來回嘴、吐槽、附和、裝可憐的梗圖
- `semantic`：找整體語意、情境、情緒接近的梗圖

目前專案以 OpenSpec 作為 source of truth，規格定義在 [`openspec/specs/meme-search-mvp/spec.md`](openspec/specs/meme-search-mvp/spec.md)。

## 核心原理

MemeTalk 的重點不是只做「單一向量搜尋」，而是把梗圖拆成幾種不同訊號，再在搜尋時重新組合：

1. `OCR 文字`
   - 回覆型梗圖最重要的訊號。
   - 決定這張圖能不能直接拿來回一句話。
2. `模板資訊`
   - 例如 `AnimeReaction`、`anime_reaction` 會正規化成同一組 template family。
3. `場景與用途`
   - 圖片裡發生什麼事，通常拿來表達什麼。
4. `AI 審美描述`
   - `visual_description`：視覺構圖、人物表情、色調風格等審美特徵。
   - `aesthetic_tags`：視覺風格短標籤（例如「對比構圖」「表情誇張」「二段式」）。
   - `usage_scenario`：這張梗圖最適合在什麼對話情境下使用。
5. `情緒 / 意圖 / 風格`
   - 例如傻眼、吐槽、冷幽默、裝可憐。
6. `使用者查詢偏好`
   - 除了原始 query，搜尋頁還能指定偏好的梗圖語氣，例如嘴砲、冷淡、陰陽怪氣。
   - 搜尋也支援上傳參考圖片，系統會先分析圖片內容，再和文字查詢一起組成檢索訊號。

搜尋時不會只走一條路，而是做多路召回：

- `semantic` 向量路徑
- `reply_text` 向量路徑
- SQLite FTS5 keyword / template 路徑

之後再做：

- 候選合併與去重
- deterministic feature scoring
- LLM rerank（可用時）

這樣做的原因很直接：

- 背景語意適合找「主題相近」
- OCR 文字適合找「能直接回覆」
- template / keyword 適合找「那張梗圖」

三者混成單一路徑，精確度通常會掉。

## 架構總覽

```text
Streamlit UI / CLI / FastAPI
        |
        v
   AppContainer
        |
        +-- IndexingService
        |     +-- OCRProvider
        |     +-- MetadataProvider
        |     +-- EmbeddingProvider
        |     +-- SQLiteMemeRepository
        |     +-- VectorStore
        |
        +-- SearchService
        |     +-- QueryAnalyzer
        |     +-- SQLite FTS keyword retrieval
        |     +-- semantic / reply_text vector retrieval
        |     +-- deterministic scoring
        |     +-- Reranker
        |
        +-- EvaluationService
              +-- eval run
              +-- eval tune
```

### 主要模組

- `src/memetalk/app/indexer.py`
  - 建立索引、記錄 warning / error、寫入 SQLite 與向量庫
- `src/memetalk/app/search.py`
  - 多路召回、候選合併、規則打分、rerank、快取
- `src/memetalk/app/evaluation.py`
  - 離線評估與 deterministic scoring profile tuning
- `src/memetalk/storage/sqlite_store.py`
  - canonical metadata、`index_runs`、SQLite FTS5 keyword route
- `src/memetalk/storage/vector_store.py`
  - `memory` / `chroma` 向量存取
- `src/memetalk/providers/`
  - OpenAI-compatible provider profiles、Anthropic Claude、mock、PaddleOCR 等實作

## 系統設計重點

### 1. 統一 metadata schema

每張圖都會整理成同一個 canonical schema，核心欄位包含：

- `ocr_text`
- `ocr_status`
- `template_name`
- `template_canonical_id`
- `template_aliases`
- `scene_description`
- `meme_usage`
- `visual_description`
- `aesthetic_tags`
- `usage_scenario`
- `emotion_tags`
- `intent_tags`
- `style_tags`
- `embedding_text`

這樣做的目的：

- 索引流程可觀測
- 搜尋結果可解釋
- provider 可替換
- 評估與調權有穩定輸入

### 2. 雙向量通道

索引時，每張圖至少會建立兩個向量通道：

- `semantic`
  - 來自完整 `embedding_text`
  - 適合主題、情境、意義相近的搜尋
- `reply_text`
  - 以 OCR 文字與回覆用途為主
  - 適合找一句能直接拿來回的圖

這兩個通道會分開存，並且帶有 `index_version` 邊界：

- provider
- model
- dimension
- channel

所以 embedding 模型或維度變了，不會污染舊 collection。

### 3. Keyword route 不再全表掃描

SQLite repository 現在會維護：

- `meme_assets`
- `index_runs`
- `meme_assets_fts`（FTS5）

`meme_assets_fts` 會索引：

- `keyword_text`
- `template_text`
- `ocr_text`

搜尋時先用 FTS5 做 prefilter，再對小批候選套 lexical scoring，而不是每次把整張表拉回 Python 掃。

### 4. SearchService 有快取

同一個 process 內，重複相同搜尋時會快取：

- query analysis
- query embeddings
- rerank outputs

另外 `reply` 搜尋的 `semantic` / `reply_text` query embedding 也會做 batch 呼叫，減少 provider round-trip。

## 索引流程

建立索引時，系統會：

1. 遞迴掃描 `.jpg`、`.jpeg`、`.png`、`.webp`
2. 計算 SHA-256，避免重複索引
3. （可選）跑獨立 OCR provider（如 PaddleOCR）作為 hint
4. 用 metadata provider 一次 Vision LLM 呼叫完成：
   - OCR 文字辨識（驗證或取代獨立 OCR 的結果）
   - 場景描述、常見用途
   - AI 審美描述（`visual_description`、`aesthetic_tags`、`usage_scenario`）
   - 情緒 / 意圖 / 風格標籤
5. 正規化 template
6. 組出：
   - `embedding_text`（納入視覺描述與使用情境）
   - `reply_text` embedding text
   - `keyword_text`
7. 一次產生 semantic / reply_text embeddings
8. 寫入 SQLite 與 vector store
9. 把單張失敗記進 `index_runs`，不終止整批

合併 OCR 與 metadata 到單一 LLM 呼叫的好處：索引速度約提升一倍，且 AI 能在看著圖片的同時做 OCR，準確度比獨立 OCR 再交給另一個 LLM 更一致。

## 搜尋流程

### Search modes

- `reply`
  - 主任務是找「適合拿來回一句話」的圖
  - OCR keyword / template route 與 `reply_text` 向量路徑優先
  - `semantic` 只當補位
- `semantic`
  - 主任務是找「整體語意接近」的圖
  - `semantic` 向量權重最高

### Query analysis

query 會先被拆成結構化欄位：

- `situation`
- `emotions`
- `tone`
- `reply_intent`
- `preferred_tone`
- `query_terms`
- `template_hints`
- `retrieval_weights`

`preferred_tone` 是額外偏好，不會覆蓋原本情境，而是拿來影響 route text 與 deterministic scoring。

如果有上傳查詢圖片：

- 系統會先用 `metadata provider` 分析該圖片
- 把圖片的 OCR、模板、場景、用途、視覺描述轉成 query analysis 與 route text
- 若同時有文字輸入，會把文字意圖與圖片線索一起合併，而不是二選一

### Multi-route retrieval

搜尋不依賴單一 embedding string，而是走：

- `semantic` vector retrieval
- `reply_text` vector retrieval
- SQLite FTS keyword / template retrieval

之後做：

- merge
- dedupe
- feature packing

### Deterministic scoring

在 LLM rerank 之前，每個候選會先有一個可控的規則分數，特徵包含：

- `semantic_vector`
- `reply_vector`
- `keyword_route`
- `template_route`
- `ocr_overlap`
- `emotion_overlap`
- `intent_match`
- `preferred_tone_match`
- `semantic_text_overlap`
- `reply` 模式下的 OCR mismatch score cap（有 OCR 但文不對題時壓分）

這一層現在已經不是硬編碼常數，而是來自一份 JSON scoring profile。

### LLM rerank

若 reranker 可用：

- `reply` 模式優先看 OCR 台詞與回覆適配度
- `semantic` 模式優先看整體語意與情境匹配

若 rerank 壞掉，會退回 deterministic fallback。

若離線 tuning，則可直接用 `deterministic_only` 路徑評估權重，不依賴 LLM。

## 調權與離線評估

這個 repo 現在支援兩種 evaluation 工作流：

### `eval run`

用固定 query set 跑離線評估，輸出：

- `precision_at_k`
- `mrr`
- `hard_negative_hit_rate`

### `eval tune`

用 evaluation cases 自動搜尋更好的 deterministic scoring profile，目標函數是：

```text
precision_at_k + mrr - hard_negative_hit_rate
```

輸出會寫成 JSON，預設路徑：

- `data/search_scoring_profile.json`

應用程式啟動時會自動載入這份 profile，因此調好的權重會直接進入正式搜尋流程。

## Provider abstraction

MemeTalk 把能力拆成可替換 provider：

- OCR：`extract_text`（可選，作為 metadata 的 hint）
- metadata：`analyze_image(image_path, ocr_hint=None)`（一次完成 OCR + 審美分析 + metadata）
- embeddings：`embed_texts`
- query analysis：`analyze_query`
- rerank：`rerank`

目前支援：

- `openai`
- `lmstudio`
- `ollama`
- `llama_cpp`
- `gemini`
- `claude`
- `mock`
- `local`（保留介面，MVP 未完整實作）

### OpenAI-compatible providers

OpenAI、LM Studio、Ollama、llama.cpp、Gemini 共用一條 OpenAI-compatible 實作，包含：

- chat / vision / embedding
- structured JSON output repair / retry
- LM Studio model-not-loaded 的可操作錯誤提示
- LM Studio `.webp` 轉 PNG data URL 相容處理

各 backend 的目前定位：

- `openai`
  - 預設雲端設定，預設模型為 `gpt-4.1-mini` / `text-embedding-3-small`
- `lmstudio`
  - 走本機 OpenAI-compatible server，適合桌面模型開發與驗證
- `ollama`
  - 預設 base URL 為 `http://localhost:11434/v1`
  - 預設建議模型為 `llama3`、`llava`、`nomic-embed-text`
- `llama_cpp`
  - 預設 base URL 為 `http://localhost:8080/v1`
  - 模型由 llama.cpp server 載入內容決定
- `gemini`
  - 透過 Google 的 OpenAI-compatible endpoint 存取
  - 預設模型為 `gemini-2.0-flash` / `text-embedding-004`

### Claude provider

`claude` backend 使用 Anthropic SDK 處理 chat、vision、query analysis 與 rerank。

- Claude 本身沒有 embedding API
- 因此索引與搜尋用的 embedding 會借用 `openai` 或 `gemini`
- 在 Settings 頁可直接切換 `claude_embedding_provider`

### OCR runtime

PaddleOCR 在 Windows CPU 環境下，目前驗證可用組合是：

- `paddleocr==2.10.0`
- `paddlepaddle==3.1.1`

安裝：

```bash
python -m pip install -e .[ocr]
```

若出現：

- `OneDnnContext does not have the input Filter`
- `operator < fused_conv2d > error`

通常是 Paddle runtime 相容性問題，請退回 `paddlepaddle==3.1.1`。

## 快速開始

### 1. 安裝

完整本機開發安裝（OpenAI-compatible provider + Chroma + PaddleOCR）：

```bash
pip install -e .[dev,openai,chroma,ocr]
```

若要使用 Claude：

```bash
pip install -e .[dev,openai,anthropic,chroma,ocr]
```

若只想快速 smoke test，不碰外部 provider：

```bash
pip install -e .[dev]
```

若只想跑 UI 與一般搜尋，不先安裝 PaddleOCR 也可以；但要做正式索引時，仍建議補上 `.[ocr]`。

### 2. 啟動 Streamlit

```bash
streamlit run streamlit_app.py
```

打開 `http://localhost:8501`。

Windows 也可以直接執行：

```bat
launch.bat
```

`launch.bat` 會自動：

- 偵測 Python
- 建立或重用 `.venv`
- 安裝 `.[openai,chroma,telegram]`
- 依設定決定是否另外啟動 Telegram bot
- 啟動 Streamlit

如果你要用 PaddleOCR 或 Claude，仍需另外補裝對應 extras。

### 3. 基本操作

1. `⚙️ Settings`
   - 選 provider backend、model、vector backend、OCR backend
   - 可設定 Telegram 聊天開關與 `Bot Token`
   - 目前 UI 直接支援 `openai`、`lmstudio`、`ollama`、`llama_cpp`、`gemini`、`claude`、`mock`
   - 儲存預設梗圖資料夾
2. `📦 Index`
   - 以預設梗圖資料夾為起點，或臨時指定另一個資料夾建立索引
3. `🔍 Search`
   - 選擇 `適合回覆` 或 `契合語意`
   - 輸入 query，或上傳參考圖片，也可以兩者一起用
   - 可選填偏好的梗圖語氣
   - Sidebar 可即時調整：顯示結果數量、Rerank 候選池大小、初始檢索數量

## CLI

### 建立索引

```bash
memetalk index build --source D:/path/to/memes
memetalk index build --source D:/path/to/memes --reindex
```

### 跑離線評估

```bash
memetalk eval run --cases D:/path/to/eval_cases.json
```

### 自動調 deterministic scoring profile

```bash
memetalk eval tune --cases D:/path/to/eval_cases.json
memetalk eval tune --cases D:/path/to/eval_cases.json --output data/search_scoring_profile.json
```

### 啟動 FastAPI

```bash
uvicorn memetalk.api.main:app --reload
```

### 啟動 Telegram Bot

先在 `Settings` 頁面保存：

- `啟用 Telegram 聊天功能`
- `Telegram Bot Token`

之後可直接執行：

```bash
memetalk telegram run
```

若使用 `launch.bat`，且設定中已啟用 Telegram 並保存 token，會在啟動 UI 時自動另開視窗啟動 bot。

API endpoints：

- `GET /api/v1/health`
- `POST /api/v1/search`
- `GET /api/v1/assets/{image_id}`

`POST /api/v1/search` 支援：

- 純文字查詢：`query`
- 純圖片查詢：`query_image_base64`
- 混合查詢：`query` + `query_image_base64`

可選欄位還包含：

- `query_image_filename`
- `query_image_media_type`
- `preferred_tone`
- `mode`

## 設定與優先順序

設定來源優先順序：

1. 環境變數
2. `data/memetalk_config.toml`
3. Pydantic 預設值

重要設定包含：

- `MEMETALK_PROVIDER_BACKEND`
- `MEMETALK_OCR_BACKEND`
- `MEMETALK_VECTOR_BACKEND`
- `MEMETALK_OPENAI_API_KEY`
- `MEMETALK_OPENAI_BASE_URL`
- `MEMETALK_LMSTUDIO_BASE_URL`
- `MEMETALK_OLLAMA_BASE_URL`
- `MEMETALK_LLAMA_CPP_BASE_URL`
- `MEMETALK_GEMINI_API_KEY`
- `MEMETALK_CLAUDE_API_KEY`
- `MEMETALK_CLAUDE_EMBEDDING_PROVIDER`
- `MEMETALK_SEARCH_CANDIDATE_K`
- `MEMETALK_SEARCH_TOP_N`
- `MEMETALK_SEARCH_SCORING_PROFILE_PATH`
- `MEMETALK_TELEGRAM_ENABLED`
- `MEMETALK_TELEGRAM_BOT_TOKEN`

預設 scoring profile 路徑是：

- `data/search_scoring_profile.json`

如果檔案不存在，系統會回退到內建預設權重。

## 目前的實務建議

- 想提升 `reply` 精確度：
  - 先補高品質 OCR
  - 再整理 evaluation cases
  - 最後跑 `eval tune`
- 想提升搜尋品質：
  - re-index 以獲得 `visual_description`、`aesthetic_tags`、`usage_scenario` 欄位
  - 搜尋「表情很誇張的那張」之類的視覺描述查詢也能命中
- 想提升搜尋速度：
  - 先用現有的 query cache + batch embeddings + SQLite FTS5
  - 如果 bottleneck 還在，多半是外部 LLM latency
- 調完 scoring profile 後：
  - 不需要重跑索引
  - 重新啟動 app 即可吃到新 profile

## OpenSpec

這個專案的規格 source of truth 在：

- [`openspec/specs/meme-search-mvp/spec.md`](openspec/specs/meme-search-mvp/spec.md)

README 的角色是：

- 解釋原理
- 說明架構
- 提供操作方式

不是額外的規格來源。
