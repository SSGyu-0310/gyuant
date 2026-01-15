import os

import pytest

from utils.db import close_db_connection, get_db_connection


def test_postgres_connection() -> None:
    if os.getenv("RUN_DB_TESTS") != "1":
        pytest.skip("Set RUN_DB_TESTS=1 to enable Postgres connection test.")
    if os.getenv("USE_POSTGRES", "true").lower() != "true":
        pytest.skip("USE_POSTGRES is disabled.")

    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT 1")
        row = cur.fetchone()
    close_db_connection()
    assert row and row[0] == 1
