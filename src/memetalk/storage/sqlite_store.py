from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memetalk.core.models import (
    IndexRunSummary,
    MemeAsset,
    MemeMetadata,
    OCRStatus,
    SearchMatch,
)
from memetalk.core.retrieval import build_keyword_text, lexical_overlap_score

_FTS_TABLE_NAME = "meme_assets_fts"
_FTS_PREFILTER_MULTIPLIER = 8
_FTS_PREFILTER_MIN = 24

_MEME_ASSET_COLUMNS = {
    "image_id": "TEXT PRIMARY KEY",
    "file_path": "TEXT NOT NULL",
    "file_sha256": "TEXT NOT NULL UNIQUE",
    "has_text": "INTEGER NOT NULL",
    "ocr_text": "TEXT NOT NULL",
    "ocr_status": "TEXT NOT NULL DEFAULT 'empty'",
    "ocr_confidence": "REAL",
    "ocr_lines": "TEXT NOT NULL DEFAULT '[]'",
    "template_name": "TEXT",
    "template_canonical_id": "TEXT NOT NULL DEFAULT ''",
    "template_aliases": "TEXT NOT NULL DEFAULT '[]'",
    "template_family": "TEXT NOT NULL DEFAULT ''",
    "scene_description": "TEXT NOT NULL",
    "meme_usage": "TEXT NOT NULL",
    "visual_description": "TEXT NOT NULL DEFAULT ''",
    "aesthetic_tags": "TEXT NOT NULL DEFAULT '[]'",
    "usage_scenario": "TEXT NOT NULL DEFAULT ''",
    "emotion_tags": "TEXT NOT NULL",
    "intent_tags": "TEXT NOT NULL",
    "style_tags": "TEXT NOT NULL",
    "embedding_text": "TEXT NOT NULL",
    "keyword_text": "TEXT NOT NULL DEFAULT ''",
    "created_at": "TEXT NOT NULL",
    "updated_at": "TEXT NOT NULL",
}

_INDEX_RUN_COLUMNS = {
    "run_id": "TEXT PRIMARY KEY",
    "source_dir": "TEXT NOT NULL",
    "started_at": "TEXT NOT NULL",
    "completed_at": "TEXT",
    "reindex": "INTEGER NOT NULL",
    "processed_count": "INTEGER NOT NULL",
    "indexed_count": "INTEGER NOT NULL",
    "skipped_count": "INTEGER NOT NULL",
    "failed_count": "INTEGER NOT NULL",
    "warning_count": "INTEGER NOT NULL DEFAULT 0",
    "status": "TEXT NOT NULL",
    "error_items": "TEXT NOT NULL",
    "warning_items": "TEXT NOT NULL DEFAULT '[]'",
}


class SQLiteMemeRepository:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        self._fts_enabled = False

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
                    ocr_status TEXT NOT NULL DEFAULT 'empty',
                    ocr_confidence REAL,
                    ocr_lines TEXT NOT NULL DEFAULT '[]',
                    template_name TEXT,
                    template_canonical_id TEXT NOT NULL DEFAULT '',
                    template_aliases TEXT NOT NULL DEFAULT '[]',
                    template_family TEXT NOT NULL DEFAULT '',
                    scene_description TEXT NOT NULL,
                    meme_usage TEXT NOT NULL,
                    visual_description TEXT NOT NULL DEFAULT '',
                    aesthetic_tags TEXT NOT NULL DEFAULT '[]',
                    usage_scenario TEXT NOT NULL DEFAULT '',
                    emotion_tags TEXT NOT NULL,
                    intent_tags TEXT NOT NULL,
                    style_tags TEXT NOT NULL,
                    embedding_text TEXT NOT NULL,
                    keyword_text TEXT NOT NULL DEFAULT '',
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
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_items TEXT NOT NULL,
                    warning_items TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            self._ensure_columns(conn, "meme_assets", _MEME_ASSET_COLUMNS)
            self._ensure_columns(conn, "index_runs", _INDEX_RUN_COLUMNS)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_meme_assets_sha256 ON meme_assets(file_sha256)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_meme_assets_template_canonical_id ON meme_assets(template_canonical_id)"
            )
            self._initialize_fts(conn)

    def upsert_asset(self, asset: MemeAsset) -> None:
        keyword_text = build_keyword_text(asset.metadata)
        template_text = self._build_template_text_from_metadata(asset.metadata)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meme_assets (
                    image_id, file_path, file_sha256, has_text, ocr_text, ocr_status, ocr_confidence,
                    ocr_lines, template_name, template_canonical_id, template_aliases, template_family,
                    scene_description, meme_usage, visual_description, aesthetic_tags, usage_scenario,
                    emotion_tags, intent_tags, style_tags,
                    embedding_text, keyword_text, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    file_sha256=excluded.file_sha256,
                    has_text=excluded.has_text,
                    ocr_text=excluded.ocr_text,
                    ocr_status=excluded.ocr_status,
                    ocr_confidence=excluded.ocr_confidence,
                    ocr_lines=excluded.ocr_lines,
                    template_name=excluded.template_name,
                    template_canonical_id=excluded.template_canonical_id,
                    template_aliases=excluded.template_aliases,
                    template_family=excluded.template_family,
                    scene_description=excluded.scene_description,
                    meme_usage=excluded.meme_usage,
                    visual_description=excluded.visual_description,
                    aesthetic_tags=excluded.aesthetic_tags,
                    usage_scenario=excluded.usage_scenario,
                    emotion_tags=excluded.emotion_tags,
                    intent_tags=excluded.intent_tags,
                    style_tags=excluded.style_tags,
                    embedding_text=excluded.embedding_text,
                    keyword_text=excluded.keyword_text,
                    updated_at=excluded.updated_at
                """,
                (
                    asset.image_id,
                    asset.file_path,
                    asset.file_sha256,
                    int(asset.metadata.has_text),
                    asset.metadata.ocr_text,
                    asset.metadata.ocr_status.value,
                    asset.metadata.ocr_confidence,
                    json.dumps(asset.metadata.ocr_lines, ensure_ascii=False),
                    asset.metadata.template_name,
                    asset.metadata.template_canonical_id,
                    json.dumps(asset.metadata.template_aliases, ensure_ascii=False),
                    asset.metadata.template_family,
                    asset.metadata.scene_description,
                    asset.metadata.meme_usage,
                    asset.metadata.visual_description,
                    json.dumps(asset.metadata.aesthetic_tags, ensure_ascii=False),
                    asset.metadata.usage_scenario,
                    json.dumps(asset.metadata.emotion_tags, ensure_ascii=False),
                    json.dumps(asset.metadata.intent_tags, ensure_ascii=False),
                    json.dumps(asset.metadata.style_tags, ensure_ascii=False),
                    asset.metadata.embedding_text,
                    keyword_text,
                    asset.created_at.isoformat(),
                    asset.updated_at.isoformat(),
                ),
            )
            self._upsert_fts_row(
                conn,
                image_id=asset.image_id,
                keyword_text=keyword_text,
                template_text=template_text,
                ocr_text=asset.metadata.ocr_text,
            )

    def get_asset_by_sha256(self, file_sha256: str) -> MemeAsset | None:
        row = self._fetch_one("SELECT * FROM meme_assets WHERE file_sha256 = ?", (file_sha256,))
        return self._row_to_asset(row) if row else None

    def get_asset_by_id(self, image_id: str) -> MemeAsset | None:
        row = self._fetch_one("SELECT * FROM meme_assets WHERE image_id = ?", (image_id,))
        return self._row_to_asset(row) if row else None

    def get_assets_by_ids(self, image_ids: list[str]) -> dict[str, MemeAsset]:
        if not image_ids:
            return {}
        placeholders = ", ".join("?" for _ in image_ids)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM meme_assets WHERE image_id IN ({placeholders})", tuple(image_ids)).fetchall()
        return {row["image_id"]: self._row_to_asset(row) for row in rows}

    def search_keyword_candidates(self, query_terms: list[str], template_hints: list[str], top_k: int) -> list[SearchMatch]:
        if not query_terms and not template_hints:
            return []
        rows = self._search_keyword_rows(query_terms, template_hints, top_k)
        matches: list[SearchMatch] = []
        for row in rows:
            keyword_score = lexical_overlap_score(query_terms, row["keyword_text"] or "")
            template_blob = self._build_template_blob(row)
            template_score = lexical_overlap_score(template_hints, template_blob)
            ocr_score = lexical_overlap_score(query_terms, row["ocr_text"] or "")
            score = max(keyword_score, template_score) + (ocr_score * 0.25)
            if score <= 0:
                continue
            matches.append(
                SearchMatch(
                    image_id=row["image_id"],
                    score=score,
                    metadata={
                        "channel": "keyword",
                        "keyword_score": keyword_score,
                        "template_route_score": template_score,
                        "ocr_route_score": ocr_score,
                    },
                    channel="keyword",
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:top_k]

    def save_index_run(self, summary: IndexRunSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO index_runs (
                    run_id, source_dir, started_at, completed_at, reindex,
                    processed_count, indexed_count, skipped_count, failed_count, warning_count,
                    status, error_items, warning_items
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    processed_count=excluded.processed_count,
                    indexed_count=excluded.indexed_count,
                    skipped_count=excluded.skipped_count,
                    failed_count=excluded.failed_count,
                    warning_count=excluded.warning_count,
                    status=excluded.status,
                    error_items=excluded.error_items,
                    warning_items=excluded.warning_items
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
                    summary.warning_count,
                    summary.status,
                    json.dumps([error.model_dump() for error in summary.errors], ensure_ascii=False),
                    json.dumps([warning.model_dump() for warning in summary.warnings], ensure_ascii=False),
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

    def _ensure_columns(self, conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        for column_name, definition in columns.items():
            if column_name in existing:
                continue
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _initialize_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE_NAME}
                USING fts5(
                    image_id UNINDEXED,
                    keyword_text,
                    template_text,
                    ocr_text
                )
                """
            )
        except sqlite3.OperationalError:
            self._fts_enabled = False
            return
        self._fts_enabled = True
        asset_count = conn.execute("SELECT COUNT(*) FROM meme_assets").fetchone()[0]
        fts_count = conn.execute(f"SELECT COUNT(*) FROM {_FTS_TABLE_NAME}").fetchone()[0]
        if asset_count != fts_count:
            self._rebuild_fts_index(conn)

    def _rebuild_fts_index(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"DELETE FROM {_FTS_TABLE_NAME}")
        rows = conn.execute(
            """
            SELECT image_id, keyword_text, template_name, template_canonical_id, template_aliases, ocr_text
            FROM meme_assets
            """
        ).fetchall()
        if not rows:
            return
        conn.executemany(
            f"""
            INSERT INTO {_FTS_TABLE_NAME} (image_id, keyword_text, template_text, ocr_text)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    row["image_id"],
                    row["keyword_text"] or "",
                    self._build_template_blob(row),
                    row["ocr_text"] or "",
                )
                for row in rows
            ],
        )

    def _upsert_fts_row(
        self,
        conn: sqlite3.Connection,
        image_id: str,
        keyword_text: str,
        template_text: str,
        ocr_text: str,
    ) -> None:
        if not self._fts_enabled:
            return
        conn.execute(f"DELETE FROM {_FTS_TABLE_NAME} WHERE image_id = ?", (image_id,))
        conn.execute(
            f"""
            INSERT INTO {_FTS_TABLE_NAME} (image_id, keyword_text, template_text, ocr_text)
            VALUES (?, ?, ?, ?)
            """,
            (image_id, keyword_text, template_text, ocr_text),
        )

    def _search_keyword_rows(self, query_terms: list[str], template_hints: list[str], top_k: int) -> list[sqlite3.Row]:
        if self._fts_enabled:
            fts_rows = self._search_keyword_rows_via_fts(query_terms, template_hints, top_k)
            if fts_rows:
                return fts_rows
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT image_id, keyword_text, template_name, template_canonical_id, template_aliases, ocr_text
                FROM meme_assets
                """
            ).fetchall()

    def _search_keyword_rows_via_fts(
        self,
        query_terms: list[str],
        template_hints: list[str],
        top_k: int,
    ) -> list[sqlite3.Row]:
        match_query = self._build_fts_match_query(query_terms, template_hints)
        if not match_query:
            return []
        with self._connect() as conn:
            try:
                return conn.execute(
                    f"""
                    SELECT image_id, keyword_text, template_text, ocr_text
                    FROM {_FTS_TABLE_NAME}
                    WHERE {_FTS_TABLE_NAME} MATCH ?
                    ORDER BY bm25({_FTS_TABLE_NAME}, 1.0, 2.5, 2.0)
                    LIMIT ?
                    """,
                    (match_query, max(top_k * _FTS_PREFILTER_MULTIPLIER, _FTS_PREFILTER_MIN)),
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    def _build_fts_match_query(self, query_terms: list[str], template_hints: list[str]) -> str:
        clauses: list[str] = []
        for value in [*template_hints, *query_terms]:
            normalized = " ".join(str(value or "").replace('"', " ").split())
            if not normalized:
                continue
            clauses.append(f'"{normalized}"')
        return " OR ".join(dict.fromkeys(clauses))

    def _build_template_text_from_metadata(self, metadata: MemeMetadata) -> str:
        return " ".join(
            part
            for part in [
                metadata.template_name or "",
                metadata.template_canonical_id,
                " ".join(metadata.template_aliases),
                metadata.template_family,
            ]
            if part
        )

    def _build_template_blob(self, row: sqlite3.Row) -> str:
        if "template_text" in row.keys() and row["template_text"]:
            return row["template_text"]
        return " ".join(
            part
            for part in [
                row["template_name"] or "",
                row["template_canonical_id"] or "",
                " ".join(json.loads(row["template_aliases"] or "[]")),
            ]
            if part
        )

    def _row_to_asset(self, row: sqlite3.Row) -> MemeAsset:
        metadata = MemeMetadata(
            has_text=bool(row["has_text"]),
            ocr_text=row["ocr_text"],
            ocr_status=OCRStatus(row["ocr_status"] or OCRStatus.EMPTY.value),
            ocr_confidence=row["ocr_confidence"],
            ocr_lines=json.loads(row["ocr_lines"] or "[]"),
            template_name=row["template_name"],
            template_canonical_id=row["template_canonical_id"] or "",
            template_aliases=json.loads(row["template_aliases"] or "[]"),
            template_family=row["template_family"] or "",
            scene_description=row["scene_description"],
            meme_usage=row["meme_usage"],
            visual_description=row["visual_description"] or "",
            aesthetic_tags=json.loads(row["aesthetic_tags"] or "[]"),
            usage_scenario=row["usage_scenario"] or "",
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
