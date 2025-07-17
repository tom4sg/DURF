#%%
import pandas as pd
import requests
import os
from pprint import pprint

base_url = "https://matrix.sbapis.com/b"
endpoint = "/instagram/statistics?query=SocialBlade&history=archive&allow-stale=false"


headers = {
    "clientid": os.getenv("SOCIAL_BLADE_CLIENT_ID"),
    "token": os.getenv("SOCIAL_BLADE_API_TOKEN")
}

response = requests.get(base_url + endpoint, headers=headers)

data = response.json()

#%%


df = pd.DataFrame(data["data"])
print(df.head())

# %%

pprint(data["data"]["daily"])