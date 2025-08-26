#%%

import os
from requests import get, post
from pprint import pprint

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

res = post(
    "https://accounts.spotify.com/api/token",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data={"grant_type": "client_credentials"},
    auth=(client_id, client_secret),
)

# %%

access_token = res.json()["access_token"]
headers = {
    "Authorization": f"Bearer {access_token}"
}

# %%

res = get("https://api.spotify.com/v1/artists/0TnOYISbd1XYRBk9myaseg", headers=headers)
pprint(res.json())

# %%

