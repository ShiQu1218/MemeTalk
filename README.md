# MemeTalk

MemeTalk is a local MVP for semantic meme retrieval. It indexes static meme images into structured metadata, stores canonical records in SQLite, mirrors embeddings into a vector store, and exposes search through FastAPI plus a simple Streamlit demo.

## Quickstart

1. Install the package and optional provider extras that match your environment.
2. Index a meme directory.
3. Run the API.
4. Run the Streamlit demo.

```bash
pip install -e .[dev]
memetalk index build --source D:/path/to/memes
uvicorn memetalk.api.main:app --reload
streamlit run streamlit_app.py
```

## Environment

The application reads these environment variables:

- `MEMETALK_SQLITE_PATH`
- `MEMETALK_VECTOR_BACKEND`
- `MEMETALK_CHROMA_PATH`
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

Use `MEMETALK_PROVIDER_BACKEND=mock` and `MEMETALK_VECTOR_BACKEND=memory` for local testing without external services.

## LM Studio

LM Studio can be used through the OpenAI-compatible provider path. Install the OpenAI client extra first:

```bash
pip install -e .[openai]
```

Then point the provider backend at LM Studio's local server and set the loaded model IDs:

```bash
$env:MEMETALK_PROVIDER_BACKEND="lmstudio"
$env:MEMETALK_LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
$env:MEMETALK_LMSTUDIO_CHAT_MODEL="your-chat-model-id"
$env:MEMETALK_LMSTUDIO_VISION_MODEL="your-vision-model-id"
$env:MEMETALK_LMSTUDIO_EMBEDDING_MODEL="your-embedding-model-id"
```

If you prefer to keep `MEMETALK_PROVIDER_BACKEND=openai`, you can also point `MEMETALK_OPENAI_BASE_URL` at an OpenAI-compatible endpoint such as LM Studio.
