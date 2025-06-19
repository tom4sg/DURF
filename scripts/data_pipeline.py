from requests import get, post
from sys import exit
from pprint import pprint
import os
from dotenv import load_dotenv

load_dotenv()

# Read environment variables
HOST = os.getenv("CHARTMETRIC_API_URL")
REFRESH_TOKEN = os.getenv("CHARTMETRIC_REFRESH_TOKEN")

# Get access token
res = post(f'{HOST}/api/token', json={"refreshtoken": REFRESH_TOKEN})
if res.status_code != 200:
    print(f'ERROR: received a {res.status_code} instead of 200 from /api/token')
    exit(1)

access_token = res.json()['token']

# Define GET function using access token
def Get(uri):
    return get(f'{HOST}{uri}', headers={'Authorization': f'Bearer {access_token}'})

# Example query: get BTS data
res = Get('/api/artist/206557/where-people-listen')
if res.status_code != 200:
    print(f'ERROR: received a {res.status_code} instead of 200 from /api/artist/:id')
    exit(1)

pprint(res.json())