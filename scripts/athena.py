#%%

import boto3
import pandas as pd
import time

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

#%%
print(response)

# %%

bucket = "tomasgutierrez-athena-results"
prefix = "athena-results"

query_execution_id = response["QueryExecutionId"]
status = "RUNNING"
while status in ["RUNNING", "QUEUED"]:
    response = athena.get_query_execution(QueryExecutionId=query_execution_id)
    status = response["QueryExecution"]["Status"]["State"]
    time.sleep(1)

if state == "SUCCEEDED":
    result_response = athena.get_query_results(QueryExecutionId=query_execution_id)
    for row in result_response['ResultSet']['Rows'][1:]:  # skip header
        print(row['Data'][0]['VarCharValue'])
else:
    print(f"Query failed with state: {state}")

#%%

df = pd.read_csv(result_s3)
artist_list = df['artist'].dropna().tolist()

# %%
