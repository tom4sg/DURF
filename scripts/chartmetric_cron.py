import sqlite3
import pandas as pd

conn = sqlite3.connect('chartmetric_data.db')
df = pd.read_sql_query("SELECT * FROM artist_metrics", conn)
print(df)