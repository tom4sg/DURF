#%%
import pandas as pd
import requests
import os
from pprint import pprint

#%%

# The Below can be used for Instagram Vault API Calls

base_url = "https://matrix.sbapis.com/b"
endpoint = "/instagram/statistics?query=nba_youngboy&history=vault&allow-stale=false"


headers = {
    "clientid": os.getenv("SOCIAL_BLADE_CLIENT_ID"),
    "token": os.getenv("SOCIAL_BLADE_API_TOKEN")
}

response = requests.get(base_url + endpoint, headers=headers)

data = response.json()
# %%

df = pd.DataFrame(data["data"]["daily"])
df.to_csv("instagram_vault_data.csv", index=False)
# %%

# The Below can be used for TikTok Vault API Calls

base_url = "https://matrix.sbapis.com/b"
endpoint = "/tiktok/statistics?query=SocialBlade.com&history=vault&allow-stale=false"


headers = {
    "clientid": os.getenv("SOCIAL_BLADE_CLIENT_ID"),
    "token": os.getenv("SOCIAL_BLADE_API_TOKEN")
}

response = requests.get(base_url + endpoint, headers=headers)

data = response.json()
# %%

df = pd.DataFrame(data["data"]["daily"])
df.to_csv("tiktok_vault_data.csv", index=False)
# %%

# The Below can be used for YouTube Vault API Calls

base_url = "https://matrix.sbapis.com/b"
endpoint = "/youtube/statistics?query=@Alemán&history=archive&allow-stale=false"


headers = {
    "clientid": os.getenv("SOCIAL_BLADE_CLIENT_ID"),
    "token": os.getenv("SOCIAL_BLADE_API_TOKEN")
}

response = requests.get(base_url + endpoint, headers=headers)

data = response.json()
pprint(data)
# %%

df = pd.DataFrame(data["data"]["daily"])
df.to_csv("youtube_vault_data.csv", index=False)

# %%
