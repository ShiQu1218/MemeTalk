# Meme Search MVP

## Context
- Scope: local single-machine MVP for meme semantic search.
- Primary language: Traditional Chinese metadata and query analysis.
- Out of scope: auth, multi-user sync, cloud deployment, GIF/video ingestion, meme generation.

## Requirements

### REQ-MVP-001 OpenSpec Bootstrap
- The repository MUST keep the MVP requirements under `openspec/specs/meme-search-mvp/spec.md`.
- All production modules MUST implement behavior described in this spec.

### REQ-MVP-002 Unified Meme Metadata
- Every indexed image MUST resolve into one canonical metadata schema containing:
  - `image_id`
  - `file_path`
  - `has_text`
  - `ocr_text`
  - `ocr_status`
  - `ocr_confidence`
  - `ocr_lines`
  - `template_name`
  - `template_canonical_id`
  - `template_aliases`
  - `template_family`
  - `scene_description`
  - `meme_usage`
  - `visual_description`
  - `aesthetic_tags`
  - `usage_scenario`
  - `emotion_tags`
  - `intent_tags`
  - `style_tags`
  - `embedding_text`
- `visual_description` MUST contain an AI-generated aesthetic description of the image's visual composition, expressions, and overall impression, written in Traditional Chinese.
- `aesthetic_tags` MUST contain short tags describing visual style characteristics (for example `對比構圖`, `表情誇張`, `二段式`).
- `usage_scenario` MUST describe the conversational scenario where this meme is most effective, written in Traditional Chinese.
- `embedding_text` MUST concatenate, in order: template information, visual description, scene description, common meme usage, usage scenario, OCR text, emotion tags, intent tags, style tags, aesthetic tags.
- Template normalization MUST map visually similar aliases (for example `AnimeReaction` and `anime_reaction`) to the same `template_canonical_id`.

### REQ-MVP-003 Incremental Indexing Pipeline
- The system MUST provide a CLI command `index build --source <dir> [--reindex]`.
- The indexer MUST recursively scan `.jpg`, `.jpeg`, `.png`, `.webp`.
- The indexer MUST compute SHA-256 per file and skip already indexed files unless `--reindex` is used.
- Incremental skip decisions MUST treat an image as already indexed only when its canonical metadata and all required vector documents for the current embedding index version are present.
- Images left in a partial or failed indexing state, or indexed under an outdated embedding index version, MUST be retried on the next non-`--reindex` build instead of being skipped.
- The metadata provider MUST perform OCR, aesthetic analysis, and metadata extraction in a single Vision LLM call, eliminating the need for a separate OCR step during indexing.
- When a dedicated OCR provider is configured (for example PaddleOCR), the indexer MAY run it before metadata analysis and pass the OCR text as a hint; the metadata provider MUST still accept and process images without a prior OCR result.
- OCR failures MUST degrade to empty text, set `ocr_status` to `failed`, and continue processing.
- OCR empty-but-successful runs MUST set `ocr_status` to `empty`.
- OCR degradation events MUST be recorded in `index_runs` without aborting the batch.
- Metadata or embedding failures for one image MUST be recorded in `index_runs` and MUST NOT abort the batch.
- Canonical metadata MUST be stored in SQLite and vector documents MUST be stored in Chroma-compatible storage.
- ChromaDB HNSW index read failures (corrupted or missing index files) MUST be caught at the vector store layer and MUST return empty results with a logged warning instead of propagating an unhandled exception.
- Vector retrieval route failures during search MUST degrade gracefully per-route (returning zero candidates for the failed route) and MUST be recorded in `search_trace.degraded_routes` instead of aborting the entire search request.
- The indexer MUST store per-channel vector documents with an index schema/version boundary derived from provider/model/dimension/channel so incompatible embedding versions do not share the same vector collection or query surface.
- The indexer MUST maintain a keyword-searchable representation of template aliases, OCR text, and retrieval tags for lexical retrieval.

### REQ-MVP-004 Query Understanding and Vector Retrieval
- The system MUST expose `POST /api/v1/search`.
- Search input MUST support an optional user-provided preferred meme tone hint.
- Search input MUST support text-only, image-only, and mixed text+image queries.
- When an image query is provided, the system MUST analyze the query image through the metadata provider and convert it into retrieval inputs without requiring the image to be pre-indexed first.
- Search input MUST be analyzed into structured query fields: situation, emotion, tone, reply intent, query terms, template hints, retrieval weights, and preferred meme tone.
- For image queries, structured query analysis MUST incorporate image-derived cues including OCR text, template hints, scene description, meme usage, and visual-description signals.
- Search MUST support two search modes via a `mode` field (`semantic` | `reply`, default `reply`):
  - **semantic**: finds memes whose overall meaning is close to the query.
  - **reply**: finds memes suitable as a witty reply to the query; OCR text weight is dominant.
- The indexer MUST produce at least two vector channels per meme:
  - A **semantic** channel built from the full `embedding_text` (template, scene, usage, OCR, tags).
  - A **reply_text** channel built from OCR-focused text that emphasizes the meme's actual wording and reply tone.
- Search MUST use multi-route retrieval:
  - vector retrieval over the semantic channel
  - vector retrieval over the reply_text channel when in `reply` mode
  - keyword/template retrieval over OCR text, template aliases, and retrieval tags
- In `reply` mode, OCR keyword/template retrieval and the `reply_text` channel MUST be the primary routes; the semantic channel MUST be treated as a supplemental backfill route only.
- Search MUST merge, deduplicate, and score candidates from all enabled retrieval routes before reranking.
- Search MUST generate route-specific retrieval inputs from query analysis rather than relying on a single query embedding string as the only retrieval artifact.
- When both text and image query inputs are present, the retrieval inputs MUST merge user text intent with image-derived metadata instead of dropping either source.
- Search SHOULD batch query embedding generation across active retrieval routes within a single search request when those routes use the same embedding provider.
- Search SHOULD reuse in-memory cached query analysis, query embeddings, and rerank outputs for repeated identical searches within the same process when the provider configuration has not changed.
- Search output MUST include `query_analysis`, `results`, `provider_trace`, and a minimal `search_trace` showing candidate-source and degradation information.

### REQ-MVP-005 LLM Reranking and Reasoning
- Retrieved candidates MUST be reranked before final output when a reranker is available.
- Retrieved candidates MUST first receive deterministic feature scores before LLM reranking, including lexical OCR overlap, template matches, tag overlaps, vector scores, and reply-mode OCR gating.
- The deterministic feature scoring profile SHOULD be serializable to JSON so it can be tuned offline and reused at runtime.
- Reranking MUST respect the search mode:
  - **reply** mode: OCR text (`ocr_text`) and reply fitness are the primary ranking signals; candidates with `ocr_status == success` but whose OCR text has low lexical overlap with the query (below `ocr_mismatch_threshold`) MUST have their deterministic score capped at `ocr_mismatch_score_cap`; candidates without OCR text MAY compete freely based on visual expression relevance without a hard score cap.
  - **semantic** mode: overall semantic similarity, scene description, and tag overlap are the primary ranking criteria.
- In `reply` mode, scene/background description MUST be treated as a secondary tie-break signal rather than a primary ranking feature.
- If reranking fails, the system MUST fall back to deterministic candidate ranking and return a non-empty fallback reason for every result.
- Final results MUST include `image_id`, `image_url`, `reason`, `score`, `template_name`, `emotion_tags`, `intent_tags`, and route/debug metadata sufficient to explain degraded ranking.

### REQ-MVP-006 FastAPI Contract
- The API MUST expose `GET /api/v1/health`.
- The API MUST expose an image asset endpoint addressable by `image_id`.
- `POST /api/v1/search` MUST accept JSON search requests containing text input, base64-encoded image input, or both.
- `POST /api/v1/search` MUST reject requests that provide neither text nor image input.
- The API MUST not require Streamlit to access SQLite directly.

### REQ-MVP-007 Streamlit Unified Frontend
- The repository MUST include a multi-page Streamlit app as the primary user interface.
- The app MUST operate in Direct Mode, calling Python services directly without requiring a separate HTTP API server.
- The app MUST be launchable with a single command: `streamlit run streamlit_app.py`.
- The app MUST include the following pages:
  - **Dashboard** (`streamlit_app.py`): system status overview showing current provider, vector backend, indexed meme count, and a health check.
  - **Settings** (`pages/1_⚙️_Settings.py`): provider selection (openai / lmstudio / mock), API key input, base URL, model configuration, vector backend, OCR backend, Telegram chat enable toggle, Telegram bot token input, and a persisted default meme folder path. Settings MUST be persisted to a TOML file (`data/memetalk_config.toml`).
  - **Index** (`pages/2_📦_Index.py`): meme folder path input seeded from the saved default meme folder, optional force-reindex toggle, progress display, and result summary (processed / indexed / skipped / failed counts with error details).
  - **Search** (`pages/3_🔍_Search.py`): search mode selector (適合回覆 / 契合語意), natural-language query input, optional query image uploader with preview, query analysis display, top-N result cards with images loaded from local file paths, recommended reason text, and visible emotion and intent tags.
- The Search page MUST include an optional input for preferred meme tone (for example 嘴砲, 冷淡, 可憐, 陰陽怪氣) and pass that preference into query analysis and reranking.
- The Search page MUST provide sidebar controls that let users adjust at runtime: display result count (`top_n`), rerank pool size, and initial retrieval count (`candidate_k`). Default values MUST come from the persisted configuration.
- The Search page MUST allow searches to run when only a query image is provided, and MUST show a validation error only when both text and image inputs are absent.
- Settings MUST support a priority chain: environment variables > TOML config file > pydantic defaults.
- Settings persistence MUST merge submitted values with the existing persisted settings and MUST NOT clear unrelated stored fields such as `meme_folder` during a partial update.
- The Index page MUST treat its meme folder input as a per-run override only; editing that field in the page MUST NOT automatically rewrite the saved default meme folder path.
- The Streamlit app MUST apply a consistent visual shell across Dashboard, Settings, Index, and Search, including shared page headers, section hierarchy, and readable status presentation on both desktop and narrow layouts.
- The shared visual shell MUST support both Streamlit light and dark themes without per-page overrides, preserving readable contrast for backgrounds, cards, metrics, notices, buttons, form inputs, and expanders in both modes.
- The app MUST NOT require manual environment variable configuration or a `secrets.toml` file to start.

### REQ-MVP-008 Provider Abstraction
- The system MUST define replaceable provider interfaces for:
  - OCR via `extract_text` (optional; may be integrated into the metadata provider)
  - metadata analysis via `analyze_image(image_path, ocr_hint=None)` which MUST return unified metadata including OCR fields, visual description, aesthetic tags, usage scenario, and all traditional metadata fields
  - embeddings via `embed_texts`
- query analysis via `analyze_query(query, mode, preferred_tone=None)`
- reranking via `rerank(query, query_analysis, candidates, top_n, mode)`
- Query-image search flows MUST reuse the metadata provider plus embedding provider path and MUST NOT require a separate provider interface only for uploaded query images.
- OpenAI MUST be the default cloud provider configuration.
- The OpenAI-backed provider path MUST support OpenAI-compatible chat, vision, and embedding endpoints through configurable base URL and model settings.
- The system MUST support the following provider backends: `openai`, `lmstudio`, `ollama`, `llama_cpp`, `gemini`, `claude`, `mock`, `local`.
- Ollama, llama.cpp, and Gemini backends MUST reuse the OpenAI-compatible provider path with appropriate default base URLs and model mappings:
  - Ollama: default base URL `http://localhost:11434/v1`, default models `llama3` (chat), `llava` (vision), `nomic-embed-text` (embedding).
  - llama.cpp: default base URL `http://localhost:8080/v1`, model determined by server-loaded model.
  - Gemini: OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/`, default models `gemini-2.0-flash` (chat/vision), `text-embedding-004` (embedding).
- The Claude backend MUST use the Anthropic SDK for chat, vision, query analysis, and reranking; embedding MUST be delegated to a configurable secondary provider (`openai` or `gemini`) since Anthropic does not offer an embedding API.
- The repository MUST provide a documented local provider configuration for LM Studio without changing application code.
- The repository MUST pin and document a known-good PaddleOCR runtime for Windows CPU environments; the validated MVP combination is `paddleocr==2.10.0` with `paddlepaddle==3.1.1`.
- OpenAI-compatible vision requests used by indexing and query-image search MUST locally normalize supported images into sanitized JPEG or PNG data URLs before upload, including EXIF orientation correction and incompatible color-mode cleanup, instead of forwarding original file bytes unchanged.
- OpenAI-compatible local provider failures MUST surface actionable guidance when required chat, vision, or embedding models are not available.
- OpenAI-compatible local provider image-processing failures MUST surface actionable guidance that the active model may not actually support image input even if the configured vision model id is set.
- PaddleOCR runtime failures caused by known Windows CPU inference incompatibilities MUST surface actionable guidance instead of only returning the raw Paddle error.
- OpenAI-compatible structured outputs MUST retry or repair recoverable malformed JSON before failing a request.
- OpenAI-compatible rerank structured outputs MUST accept a recoverable top-level JSON array response and normalize it into the expected `results` object shape before failing the request.
- Anthropic-backed structured outputs MUST retry or repair recoverable malformed JSON before failing a request.
- Anthropic-backed rerank structured outputs MUST accept a recoverable top-level JSON array response and normalize it into the expected `results` object shape before failing the request.
- The codebase MUST include a local/mock-friendly provider path so tests can run without external services.

### REQ-MVP-009 Evaluation Pipeline
- The repository MUST provide an offline evaluation pipeline over a fixed query set with positives and hard negatives.
- The evaluation pipeline MUST report at least `precision_at_k` and `mrr`.
- The evaluation fixtures MUST cover reply queries, semantic queries, template-driven queries, OCR phrase queries, and reaction/no-text queries.
- The repository MUST provide an offline tuning command `eval tune --cases <file> [--output <file>]` that optimizes the deterministic scoring profile against the evaluation cases and writes the tuned profile to JSON.
- The tuning objective MUST penalize hard-negative hits in addition to improving retrieval relevance metrics.

### REQ-MVP-010 Telegram Chat Integration
- The repository MUST include an optional Telegram long-polling bot integration inside the same codebase.
- Telegram bot runtime settings MUST be configurable through the same `AppSettings` source chain used by the UI (`environment variables > TOML config file > pydantic defaults`), including:
  - `telegram_enabled`
  - `telegram_bot_token`
- The Settings page MUST allow users to persist `telegram_enabled` and `telegram_bot_token` without clearing unrelated settings.
- The CLI MUST expose a Telegram run command that starts the bot only when Telegram chat is enabled and a bot token is configured.
- `launch.bat` MUST start the Telegram bot in a separate process when Telegram chat is enabled and a bot token is configured, without blocking the Streamlit UI startup flow.
- The Telegram bot MUST support `/start`, `/help`, and free-form text messages.
- For free-form text messages, the bot MUST decide between `text`, `meme`, and `both`.
- The bot MUST reuse the active MemeTalk provider/backend settings for routing decisions instead of introducing a second unrelated model configuration surface.
- The bot MUST execute meme retrieval through the in-repo application services directly and MUST NOT require a separately running FastAPI server when used with the Streamlit app's Direct Mode.
- The Telegram bot MUST keep a bounded short-term conversation history per chat in memory and pass that recent context into subsequent routing decisions.
- The Telegram short-term conversation history MAY reset when the bot process restarts.
- The Telegram bot MUST normalize locally loaded meme image bytes into a Telegram-compatible upload payload before calling Telegram when the original file format or mode is not reliably accepted as a photo upload.
- When meme retrieval succeeds, the Telegram bot MUST send only the meme image without caption text and MUST suppress any additional follow-up text for that reply, even when routing decided `both`.
- If routing fails, the bot MUST send a plain-text apology response.
- If meme retrieval fails or returns no result, the bot MUST fall back to plain-text output.
- If a meme image cannot be read, the bot MUST fall back to a plain-text explanation derived from the selected result.

## Acceptance
- The project MUST include automated tests covering schema validation, embedding text composition, provider registry, OCR success/empty/failure branching, template normalization, multi-route retrieval, rerank fallback, index schema/version isolation, evaluation reporting, indexing, image-query search, API response shape, and Telegram settings/runtime integration.
- A minimal dataset of static images MUST be indexable and searchable end-to-end without changing code.
