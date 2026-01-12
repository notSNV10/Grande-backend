"""
Lightweight DB helper module.
- Centralizes MySQL connection handling.
- Intended for simple queries for analytics/demo (no ORM).
"""
import logging
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection():
    """Context manager to open/close a DB connection."""
    settings = get_settings()
    conn = None
    try:
        conn = mysql.connector.connect(
            host=settings.db_host,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            port=settings.db_port,
        )
        yield conn
    except Error as exc:
        logger.error("DB connection error: %s", exc)
        raise
    finally:
        if conn:
            conn.close()


def fetch_all(query: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
    """Run SELECT and return rows as dicts."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or [])
        rows = cursor.fetchall()
        cursor.close()
        return rows


def fetch_one(query: str, params: Optional[Iterable[Any]] = None) -> Optional[Dict[str, Any]]:
    """Run SELECT and return single row as dict."""
    with get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or [])
        row = cursor.fetchone()
        cursor.close()
        return row


def execute(query: str, params: Optional[Iterable[Any]] = None) -> int:
    """Run INSERT/UPDATE/DELETE and return affected rows."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or [])
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        return affected

