#%%

import boto3
import pandas as pd
import time
import s3fs

#%% 

athena = boto3.client("athena", region_name="us-east-2")
# %%

query = """
SELECT DISTINCT TRIM(SPLIT("artist and title", ' - ')[1]) AS artist
FROM kworb_clean
"""
# %%

response = athena.start_query_execution(
    QueryString=query,
    QueryExecutionContext={"Database": "chart_data"},
    ResultConfiguration={"OutputLocation": "s3://tomasgutierrez-athena-results/athena-results/"}
)
# %%

bucket = "tomasgutierrez-athena-results"
prefix = "athena-results"

query_execution_id = response["QueryExecutionId"]
status = "RUNNING"
while status in ["RUNNING", "QUEUED"]:
    response = athena.get_query_execution(QueryExecutionId=query_execution_id)
    status = response["QueryExecution"]["Status"]["State"]
    time.sleep(1)

result = athena.get_query_results(QueryExecutionId=query_execution_id)
result_s3 = f"s3://{bucket}/{prefix}/{query_execution_id}.csv"

df = pd.read_csv(result_s3)
artist_list = df['artist'].dropna().tolist()
# %%
