#%%
from requests import get
import os
from pprint import pprint

base_url = " http://ws.audioscrobbler.com/2.0/"
api_key = os.getenv("LASTFM_API_KEY")
method = "artist.getTopTracks"

#%%

res = get(f"{base_url}?method={method}&api_key={api_key}&format=json")
pprint(res.json())

#%%

artist = "Sally Shapiro"
method = "track.getTags"
track = "I'll be by your side"

# %%
res = get(f"{base_url}?method={method}&api_key={api_key}&artist={artist}&track={track}&format=json")
pprint(res.json())

# %%
