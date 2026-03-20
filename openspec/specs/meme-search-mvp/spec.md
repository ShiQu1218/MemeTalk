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
  - `template_name`
  - `scene_description`
  - `meme_usage`
  - `emotion_tags`
  - `intent_tags`
  - `style_tags`
  - `embedding_text`
- `embedding_text` MUST concatenate, in order: template information, scene description, common meme usage, OCR text, emotion tags, intent tags, style tags.

### REQ-MVP-003 Incremental Indexing Pipeline
- The system MUST provide a CLI command `index build --source <dir> [--reindex]`.
- The indexer MUST recursively scan `.jpg`, `.jpeg`, `.png`, `.webp`.
- The indexer MUST compute SHA-256 per file and skip already indexed files unless `--reindex` is used.
- OCR failures MUST degrade to empty text and continue processing.
- Metadata or embedding failures for one image MUST be recorded in `index_runs` and MUST NOT abort the batch.
- Canonical metadata MUST be stored in SQLite and vector documents MUST be stored in Chroma-compatible storage.

### REQ-MVP-004 Query Understanding and Vector Retrieval
- The system MUST expose `POST /api/v1/search`.
- Search input MUST be analyzed into structured query fields: situation, emotion, tone, reply intent.
- Search MUST generate a query embedding from structured query text, retrieve vector top-k candidates, and produce a top-n response.
- Search output MUST include `query_analysis`, `results`, and `provider_trace`.

### REQ-MVP-005 LLM Reranking and Reasoning
- Retrieved candidates MUST be reranked before final output when a reranker is available.
- If reranking fails, the system MUST fall back to vector ranking and return a non-empty fallback reason for every result.
- Final results MUST include `image_id`, `image_url`, `reason`, `score`, `template_name`, `emotion_tags`, and `intent_tags`.

### REQ-MVP-006 FastAPI Contract
- The API MUST expose `GET /api/v1/health`.
- The API MUST expose an image asset endpoint addressable by `image_id`.
- The API MUST not require Streamlit to access SQLite directly.

### REQ-MVP-007 Streamlit Demo UI
- The repository MUST include a Streamlit app for the MVP demo.
- The demo UI MUST provide:
  - one natural-language query input
  - a submit action
  - top-3 result cards
  - recommended reason text
  - visible emotion and intent tags

### REQ-MVP-008 Provider Abstraction
- The system MUST define replaceable provider interfaces for:
  - OCR via `extract_text`
  - metadata analysis via `analyze_image`
  - embeddings via `embed_texts`
  - query analysis via `analyze_query`
  - reranking via `rerank`
- OpenAI MUST be the default cloud provider configuration.
- The OpenAI-backed provider path MUST support OpenAI-compatible chat, vision, and embedding endpoints through configurable base URL and model settings.
- The repository MUST provide a documented local provider configuration for LM Studio without changing application code.
- The codebase MUST include a local/mock-friendly provider path so tests can run without external services.

## Acceptance
- The project MUST include automated tests covering schema validation, embedding text composition, provider registry, OCR/non-OCR branching, rerank fallback, indexing, and API response shape.
- A minimal dataset of static images MUST be indexable and searchable end-to-end without changing code.
