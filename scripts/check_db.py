from utils.db import get_engine
import pandas as pd

engine = get_engine()

# Check available tables in market schema
query = """
SELECT table_schema, table_name 
FROM information_schema.tables 
WHERE table_schema IN ('market', 'factors', 'backtest')
ORDER BY table_schema, table_name
"""
df = pd.read_sql(query, engine)
print("=== Available Tables ===")
print(df.to_string())

# Check market_smart_money tables
query2 = """
SELECT MIN(analysis_date) as min_date, MAX(analysis_date) as max_date, COUNT(*) as cnt 
FROM market_smart_money_runs
"""
try:
    df2 = pd.read_sql(query2, engine)
    print("\n=== market_smart_money_runs ===")
    print(df2.to_string())
except Exception as e:
    print(f"market_smart_money_runs not found: {e}")

# Check market_smart_money_picks
query3 = """
SELECT COUNT(*) as cnt FROM market_smart_money_picks
"""
try:
    df3 = pd.read_sql(query3, engine)
    print("\n=== market_smart_money_picks ===")
    print(df3.to_string())
except Exception as e:
    print(f"market_smart_money_picks not found: {e}")
