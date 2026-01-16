import sqlite3
from pathlib import Path
conn = sqlite3.connect(Path('us_market/gyuant.db'))
rows = conn.execute( SELECT name FROM sqlite_master WHERE type=table).fetchall()
print(rows)
conn.close()
