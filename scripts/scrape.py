import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import s3fs
import os
import pyarrow
import time, random

# Let's test out loading the all the apple music chart endpoints from the Kworb Archives

apple_charts_link = "https://kworb.net/apple_songs/archive/"
response = requests.get(apple_charts_link)

if response.status_code != 200:
    print(f"Failed to retrieve Kworb, status code: {response.status_code}")
    exit(1)

soup = BeautifulSoup(response.text, 'html.parser')

chart_links = []
for link in soup.find_all('a')[1:]:

    # .get('href') retrieves what is inside <a href="20250528.html">20250528.html</a>
    chart_links.append(link.get('href'))

# Let's write a scraping function

def scrape_kworb_chart(url, chart_endpoints, chart_number):
    endpoint = chart_endpoints[chart_number]
    response = requests.get(url + endpoint)
    individual_chart = BeautifulSoup(response.text, 'html.parser')
    individual_chart_table = individual_chart.find('table', class_='sortable')
    df_list = pd.read_html(str(individual_chart_table))
    df = df_list[0]
    return df

# Let's initialize some variables for the S3 Bucket

s3_bucket_name = os.getenv("S3_BUCKET_NAME")
s3_bucket_prefix = os.getenv("S3_BUCKET_PREFIX")
fs = s3fs.S3FileSystem()

print(chart_links[1862])
# 20220803.html


for i in range(1863, len(chart_links)):

    # Let's get endpoint so we can store the charts associated date
    endpoint = chart_links[i]
    date_str = endpoint.replace(".html", "")
    chart_date = datetime.strptime(date_str, "%Y%m%d").date() # chart_date contains the current runs formatted date eg. 2025-05-28

    # Let's sleep for a random amount before scraping the next chart
    time.sleep(random.uniform(1, 3))

    # Let's get the chart
    chart = scrape_kworb_chart(apple_charts_link, chart_links, i)
    print(f"    scraped {len(chart)} rows for {chart_date}", flush=True)

    # Let's add the chart date to a new column in the chart
    chart["chart_date"] = chart_date

    # Let's build a unique s3 key per date
    key = (
        f"{s3_bucket_prefix}/"
        f"chart_date={chart_date}/"           # partition folder
        f"apple_songs_{chart_date}.parquet"    # filename
    )

    # write Parquet to S3
    chart.to_parquet(
        f"s3://{s3_bucket_name}/{key}",
        engine="pyarrow",
        index=False,
        compression="snappy",
        filesystem=fs
    )
    print(f"    Wrote to s3://{s3_bucket_name}/{key}\n", flush=True)

print("Finished scraping and writing to S3 Bucket", flush=True)