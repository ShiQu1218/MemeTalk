from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memetalk.core.models import IndexRunSummary, MemeAsset, MemeMetadata


class SQLiteMemeRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meme_assets (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_sha256 TEXT NOT NULL UNIQUE,
                    has_text INTEGER NOT NULL,
                    ocr_text TEXT NOT NULL,
                    template_name TEXT,
                    scene_description TEXT NOT NULL,
                    meme_usage TEXT NOT NULL,
                    emotion_tags TEXT NOT NULL,
                    intent_tags TEXT NOT NULL,
                    style_tags TEXT NOT NULL,
                    embedding_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS index_runs (
                    run_id TEXT PRIMARY KEY,
                    source_dir TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    reindex INTEGER NOT NULL,
                    processed_count INTEGER NOT NULL,
                    indexed_count INTEGER NOT NULL,
                    skipped_count INTEGER NOT NULL,
                    failed_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_items TEXT NOT NULL
                )
                """
            )

    def upsert_asset(self, asset: MemeAsset) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meme_assets (
                    image_id, file_path, file_sha256, has_text, ocr_text, template_name,
                    scene_description, meme_usage, emotion_tags, intent_tags, style_tags,
                    embedding_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_sha256=excluded.file_sha256,
                    has_text=excluded.has_text,
                    ocr_text=excluded.ocr_text,
                    template_name=excluded.template_name,
                    scene_description=excluded.scene_description,
                    meme_usage=excluded.meme_usage,
                    emotion_tags=excluded.emotion_tags,
                    intent_tags=excluded.intent_tags,
                    style_tags=excluded.style_tags,
                    embedding_text=excluded.embedding_text,
                    updated_at=excluded.updated_at
                """,
                (
                    asset.image_id,
                    asset.file_path,
                    asset.file_sha256,
                    int(asset.metadata.has_text),
                    asset.metadata.ocr_text,
                    asset.metadata.template_name,
                    asset.metadata.scene_description,
                    asset.metadata.meme_usage,
                    json.dumps(asset.metadata.emotion_tags, ensure_ascii=False),
                    json.dumps(asset.metadata.intent_tags, ensure_ascii=False),
                    json.dumps(asset.metadata.style_tags, ensure_ascii=False),
                    asset.metadata.embedding_text,
                    asset.created_at.isoformat(),
                    asset.updated_at.isoformat(),
                ),
            )

    def get_asset_by_sha256(self, file_sha256: str) -> MemeAsset | None:
        row = self._fetch_one("SELECT * FROM meme_assets WHERE file_sha256 = ?", (file_sha256,))
        return self._row_to_asset(row) if row else None

    def get_asset_by_id(self, image_id: str) -> MemeAsset | None:
        row = self._fetch_one("SELECT * FROM meme_assets WHERE image_id = ?", (image_id,))
        return self._row_to_asset(row) if row else None

    def save_index_run(self, summary: IndexRunSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_runs (
                    run_id, source_dir, started_at, completed_at, reindex,
                    processed_count, indexed_count, skipped_count, failed_count, status, error_items
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    processed_count=excluded.processed_count,
                    indexed_count=excluded.indexed_count,
                    skipped_count=excluded.skipped_count,
                    failed_count=excluded.failed_count,
                    status=excluded.status,
                    error_items=excluded.error_items
                """,
                (
                    summary.run_id,
                    summary.source_dir,
                    summary.started_at.isoformat(),
                    summary.completed_at.isoformat() if summary.completed_at else None,
                    int(summary.reindex),
                    summary.processed_count,
                    summary.indexed_count,
                    summary.skipped_count,
                    summary.failed_count,
                    summary.status,
                    json.dumps([error.model_dump() for error in summary.errors], ensure_ascii=False),
                ),
            )

    def count_assets(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM meme_assets").fetchone()
            return row[0] if row else 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _fetch_one(self, query: str, params: tuple) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(query, params).fetchone()

    def _row_to_asset(self, row: sqlite3.Row) -> MemeAsset:
        metadata = MemeMetadata(
            has_text=bool(row["has_text"]),
            ocr_text=row["ocr_text"],
            template_name=row["template_name"],
            scene_description=row["scene_description"],
            meme_usage=row["meme_usage"],
            emotion_tags=json.loads(row["emotion_tags"]),
            intent_tags=json.loads(row["intent_tags"]),
            style_tags=json.loads(row["style_tags"]),
            embedding_text=row["embedding_text"],
        )
        return MemeAsset(
            image_id=row["image_id"],
            file_path=row["file_path"],
            file_sha256=row["file_sha256"],
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
