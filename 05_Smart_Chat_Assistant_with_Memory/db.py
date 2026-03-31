"""
db.py – PostgreSQL + pgvector persistence layer for chat messages.
"""

import psycopg
from pgvector.psycopg import register_vector


# ── Schema ─────────────────────────────────────────────────────────────────────
create_table_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    role        TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    embedding   VECTOR({dim}),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    summary     TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);
"""

def init_db(pg_path: str, dim: int):
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(create_table_SQL.format(dim=dim))
        connection.commit()


# ── Write ──────────────────────────────────────────────────────────────────────
insert_message_SQL = """
INSERT INTO chat_messages (session_id, role, content, embedding)
VALUES (%s, %s, %s, %s)
"""

def insert_message(pg_path: str, session_id: str, role: str, content: str, embedding=None):
    vec = np.array(embedding) if embedding is not None else None   # ← cast here too
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(insert_message_SQL, (session_id, role, content, vec))
        connection.commit()


# ── Read ───────────────────────────────────────────────────────────────────────
load_history_SQL = """
SELECT role, content, created_at
FROM   chat_messages
WHERE  session_id = %s
ORDER  BY created_at ASC, id ASC
"""

def load_history(pg_path: str, session_id: str):
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(load_history_SQL, (session_id,))
            rows = cursor.fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]


load_history_paginated_SQL = """
SELECT role, content, created_at
FROM   chat_messages
WHERE  session_id = %s
ORDER  BY created_at ASC, id ASC
LIMIT  %s OFFSET %s
"""

def load_history_paginated(pg_path: str, session_id: str, limit: int = 50, offset: int = 0):
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(load_history_paginated_SQL, (session_id, limit, offset))
            rows = cursor.fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]


# ── Semantic search ────────────────────────────────────────────────────────────
import numpy as np

search_similar_SQL = """
SELECT role, content, created_at,
       1 - (embedding <=> %s::vector) AS score
FROM   chat_messages
WHERE  session_id = %s
  AND  embedding IS NOT NULL
ORDER  BY embedding <=> %s::vector
LIMIT  %s
"""

def search_similar(pg_path: str, session_id: str, query_embedding, k: int = 5):
    vec = np.array(query_embedding)        # ← cast to numpy array
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(search_similar_SQL, (vec, session_id, vec, k))
            rows = cursor.fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2], "score": r[3]} for r in rows]

# ── Delete ─────────────────────────────────────────────────────────────────────
delete_session_SQL = """
DELETE FROM chat_messages WHERE session_id = %s
"""

def delete_session(pg_path: str, session_id: str):
    with psycopg.connect(pg_path) as connection:
        with connection.cursor() as cursor:
            cursor.execute(delete_session_SQL, (session_id,))
        connection.commit()


# ── Stats ──────────────────────────────────────────────────────────────────────
session_stats_SQL = """
SELECT role, COUNT(*) AS count
FROM   chat_messages
WHERE  session_id = %s
GROUP  BY role
"""

def session_stats(pg_path: str, session_id: str):
    with psycopg.connect(pg_path) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(session_stats_SQL, (session_id,))
            rows = cursor.fetchall()
    return {r[0]: r[1] for r in rows}

save_summary_SQL = """
INSERT INTO chat_sessions (session_id, summary, updated_at)
VALUES (%s, %s, NOW())
ON CONFLICT (session_id)
DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW()
"""

def save_summary(pg_path: str, session_id: str, summary: str):
    with psycopg.connect(pg_path) as connection:
        with connection.cursor() as cursor:
            cursor.execute(save_summary_SQL, (session_id, summary))
        connection.commit()


load_summary_SQL = """
SELECT summary FROM chat_sessions WHERE session_id = %s
"""

def load_summary(pg_path: str, session_id: str) -> str:
    with psycopg.connect(pg_path) as connection:
        with connection.cursor() as cursor:
            cursor.execute(load_summary_SQL, (session_id,))
            row = cursor.fetchone()
    return row[0] if row else ""