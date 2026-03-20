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
  - `emotion_tags`
  - `intent_tags`
  - `style_tags`
  - `embedding_text`
- `embedding_text` MUST concatenate, in order: template information, scene description, common meme usage, OCR text, emotion tags, intent tags, style tags.
- Template normalization MUST map visually similar aliases (for example `AnimeReaction` and `anime_reaction`) to the same `template_canonical_id`.

### REQ-MVP-003 Incremental Indexing Pipeline
- The system MUST provide a CLI command `index build --source <dir> [--reindex]`.
- The indexer MUST recursively scan `.jpg`, `.jpeg`, `.png`, `.webp`.
- The indexer MUST compute SHA-256 per file and skip already indexed files unless `--reindex` is used.
- OCR failures MUST degrade to empty text, set `ocr_status` to `failed`, and continue processing.
- OCR empty-but-successful runs MUST set `ocr_status` to `empty`.
- OCR degradation events MUST be recorded in `index_runs` without aborting the batch.
- Metadata or embedding failures for one image MUST be recorded in `index_runs` and MUST NOT abort the batch.
- Canonical metadata MUST be stored in SQLite and vector documents MUST be stored in Chroma-compatible storage.
- The indexer MUST store per-channel vector documents with an index schema/version boundary derived from provider/model/dimension/channel so incompatible embedding versions do not share the same vector collection or query surface.
- The indexer MUST maintain a keyword-searchable representation of template aliases, OCR text, and retrieval tags for lexical retrieval.

### REQ-MVP-004 Query Understanding and Vector Retrieval
- The system MUST expose `POST /api/v1/search`.
- Search input MUST support an optional user-provided preferred meme tone hint.
- Search input MUST be analyzed into structured query fields: situation, emotion, tone, reply intent, query terms, template hints, retrieval weights, and preferred meme tone.
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
- Search SHOULD batch query embedding generation across active retrieval routes within a single search request when those routes use the same embedding provider.
- Search SHOULD reuse in-memory cached query analysis, query embeddings, and rerank outputs for repeated identical searches within the same process when the provider configuration has not changed.
- Search output MUST include `query_analysis`, `results`, `provider_trace`, and a minimal `search_trace` showing candidate-source and degradation information.

### REQ-MVP-005 LLM Reranking and Reasoning
- Retrieved candidates MUST be reranked before final output when a reranker is available.
- Retrieved candidates MUST first receive deterministic feature scores before LLM reranking, including lexical OCR overlap, template matches, tag overlaps, vector scores, and reply-mode OCR gating.
- Reranking MUST respect the search mode:
  - **reply** mode: OCR text (`ocr_text`) and reply fitness are the primary ranking signals; candidates with `ocr_status != success` MUST be explicitly downgraded and MUST NOT compete at the same score tier as candidates with successful OCR unless no better OCR-backed candidates exist.
  - **semantic** mode: overall semantic similarity, scene description, and tag overlap are the primary ranking criteria.
- In `reply` mode, scene/background description MUST be treated as a secondary tie-break signal rather than a primary ranking feature.
- If reranking fails, the system MUST fall back to deterministic candidate ranking and return a non-empty fallback reason for every result.
- Final results MUST include `image_id`, `image_url`, `reason`, `score`, `template_name`, `emotion_tags`, `intent_tags`, and route/debug metadata sufficient to explain degraded ranking.

### REQ-MVP-006 FastAPI Contract
- The API MUST expose `GET /api/v1/health`.
- The API MUST expose an image asset endpoint addressable by `image_id`.
- The API MUST not require Streamlit to access SQLite directly.

### REQ-MVP-007 Streamlit Unified Frontend
- The repository MUST include a multi-page Streamlit app as the primary user interface.
- The app MUST operate in Direct Mode, calling Python services directly without requiring a separate HTTP API server.
- The app MUST be launchable with a single command: `streamlit run streamlit_app.py`.
- The app MUST include the following pages:
  - **Dashboard** (`streamlit_app.py`): system status overview showing current provider, vector backend, indexed meme count, and a health check.
  - **Settings** (`pages/1_⚙️_Settings.py`): provider selection (openai / lmstudio / mock), API key input, base URL, model configuration, vector backend, and OCR backend. Settings MUST be persisted to a TOML file (`data/memetalk_config.toml`).
  - **Index** (`pages/2_📦_Index.py`): meme folder path input, optional force-reindex toggle, progress display, and result summary (processed / indexed / skipped / failed counts with error details).
  - **Search** (`pages/3_🔍_Search.py`): search mode selector (適合回覆 / 契合語意), natural-language query input, query analysis display, top-N result cards with images loaded from local file paths, recommended reason text, and visible emotion and intent tags.
- The Search page MUST include an optional input for preferred meme tone (for example 嘴砲, 冷淡, 可憐, 陰陽怪氣) and pass that preference into query analysis and reranking.
- Settings MUST support a priority chain: environment variables > TOML config file > pydantic defaults.
- The app MUST NOT require manual environment variable configuration or a `secrets.toml` file to start.

### REQ-MVP-008 Provider Abstraction
- The system MUST define replaceable provider interfaces for:
  - OCR via `extract_text`
  - metadata analysis via `analyze_image`
  - embeddings via `embed_texts`
  - query analysis via `analyze_query(query, mode, preferred_tone=None)`
  - reranking via `rerank(query, query_analysis, candidates, top_n, mode)`
- OpenAI MUST be the default cloud provider configuration.
- The OpenAI-backed provider path MUST support OpenAI-compatible chat, vision, and embedding endpoints through configurable base URL and model settings.
- The repository MUST provide a documented local provider configuration for LM Studio without changing application code.
- The repository MUST pin and document a known-good PaddleOCR runtime for Windows CPU environments; the validated MVP combination is `paddleocr==2.10.0` with `paddlepaddle==3.1.1`.
- OpenAI-compatible local provider failures MUST surface actionable guidance when required chat, vision, or embedding models are not available.
- PaddleOCR runtime failures caused by known Windows CPU inference incompatibilities MUST surface actionable guidance instead of only returning the raw Paddle error.
- OpenAI-compatible structured outputs MUST retry or repair recoverable malformed JSON before failing a request.
- The codebase MUST include a local/mock-friendly provider path so tests can run without external services.

### REQ-MVP-009 Evaluation Pipeline
- The repository MUST provide an offline evaluation pipeline over a fixed query set with positives and hard negatives.
- The evaluation pipeline MUST report at least `precision_at_k` and `mrr`.
- The evaluation fixtures MUST cover reply queries, semantic queries, template-driven queries, OCR phrase queries, and reaction/no-text queries.

## Acceptance
- The project MUST include automated tests covering schema validation, embedding text composition, provider registry, OCR success/empty/failure branching, template normalization, multi-route retrieval, rerank fallback, index schema/version isolation, evaluation reporting, indexing, and API response shape.
- A minimal dataset of static images MUST be indexable and searchable end-to-end without changing code.
