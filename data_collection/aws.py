#%%

import boto3
import pandas as pd
import time
import logging
import re
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

#%% 
boto3.set_stream_logger('botocore')
athena = boto3.client("athena", region_name="us-east-2")
# %%

try:
    session = boto3.Session()
    credentials = session.get_credentials().get_frozen_credentials()
    print("Access Key:", credentials.access_key)
    print("Secret Key:", "(hidden for safety)")
    print("Region:", session.region_name)
except (NoCredentialsError, PartialCredentialsError) as e:
    print("Credentials error:", str(e))

#%%
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

query_execution_id = response["QueryExecutionId"]
status = "RUNNING"
while status in ["RUNNING", "QUEUED"]:
    response = athena.get_query_execution(QueryExecutionId=query_execution_id)
    status = response["QueryExecution"]["Status"]["State"]
    time.sleep(1)

if status == "SUCCEEDED":
    result_response = athena.get_query_results(QueryExecutionId=query_execution_id)
    for row in result_response['ResultSet']['Rows'][1:]:  # skip header row
        print(row['Data'][0]['VarCharValue'])
else:
    print(f"Query failed with status: {status}")

#%%
bucket = "tomasgutierrez-athena-results"
prefix = "athena-results"
result_s3 = f"s3://{bucket}/{prefix}/{query_execution_id}.csv"
df = pd.read_csv(result_s3)
artist_list = df['artist'].dropna().tolist()

# %%
print(sorted(artist_list))

# %%
filtered_data = [item for item in artist_list if re.search("Alex Warren", item)]
print(filtered_data)

# %%
