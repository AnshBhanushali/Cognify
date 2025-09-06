import json
import os
import sqlite3
from typing import List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "store.sqlite")

def _ensure_schema():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id TEXT PRIMARY KEY,
        filename TEXT,
        timestamp TEXT,
        embedding_json TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS audios (
        id TEXT PRIMARY KEY,
        filename TEXT,
        timestamp TEXT,
        duration REAL,
        transcript TEXT,
        avg_conf REAL,
        is_broken INTEGER,
        broken_json TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_image_record(id_: str, filename: str, timestamp: str, embedding: List[float]) -> None:
    _ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("REPLACE INTO images (id, filename, timestamp, embedding_json) VALUES (?, ?, ?, ?)",
                (id_, filename, timestamp, json.dumps(embedding)))
    conn.commit()
    conn.close()

def save_audio_record(
    id_: str, filename: str, timestamp: str, duration: float, transcript: str,
    avg_conf: float, is_broken: bool, broken_words: List[str]
) -> None:
    _ensure_schema()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO audios (id, filename, timestamp, duration, transcript, avg_conf, is_broken, broken_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (id_, filename, timestamp, duration, transcript, avg_conf, int(is_broken), json.dumps(broken_words))
    )
    conn.commit()
    conn.close()
