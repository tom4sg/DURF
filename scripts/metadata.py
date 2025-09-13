# %%

import os
import applemusicpy
import pandas as pd
from tqdm import tqdm

# %%

p8_path = os.getenv("P8_PATH")
with open(p8_path, "r") as f:
    private_key = f.read()

# %%

secret_key = private_key
key_id = os.getenv("KEY_ID")
team_id = os.getenv("TEAM_ID")

#%%

am = applemusicpy.AppleMusic(secret_key=secret_key, key_id=key_id, team_id=team_id)
results = am.search('travis scott', types=['songs'], limit=10)
for item in results['results']['songs']['data']:
    print(item['attributes']['name'])

#%%

mappings = [
    "JIN", "Jin",
    "4*TOWN (From Disney", "4*TOWN (From Disney And Pixar's Turning Red)",
    "mgk", "Machine Gun Kelly",
    "¥$: Kanye West", "Kanye West",
    "$uicideBoy$", "$uicideboy$",
    "Charli XCX", "Charli xcx",
    "Yahritza y Su Esencia", "Yahritza Y Su Esencia",
    "Tyler, the Creator", "Tyler, The Creator",
    "twenty one pilots", "Twenty One Pilots",
    "jessie murph", "Jessie Murph",
    "BLEU", "Yung Bleu",
    "HUNTRX", "HUNTR/X",
    "HUNTR", "HUNTR/X",
    "Twice", "TWICE",
    "Pharrell", "Pharrell Williams",
    "BossMan DLow", "BossMan Dlow",
    "Richy Mitch", "Richy Mitch And The Coal Miners",
    "Jennie", "JENNIE",
    "Mariah The Scientist", "Mariah the Scientist"
]

# %%
from rapidfuzz import fuzz

def get_song_id(song_name, artist_name, durf_id, limit=25, storefront="us"):
    try:
        durf_id = durf_id.split(" — ")[1]
        term = f"{durf_id}"
        results = am.search(term, types=["songs"], limit=limit, storefront=storefront)

        for song in results.get("results", {}).get("songs", {}).get("data", []):
            title = song["attributes"]["name"].lower()
            artist = song["attributes"]["artistName"].lower()
            if ((fuzz.ratio(song_name.lower(), title) >= 65 or
                 song_name.lower() in title.lower()) and 
                 (fuzz.ratio(artist_name.lower(), artist) >= 65 or
                 artist_name.lower() in artist.lower())
                 or artist_name in mappings):
                return song["id"]
        return None
    except Exception as e:
        print(f"Error for {song_name} - {artist_name}: {e}")
        return None

def get_song_metadata(song_id, storefront="us"):
    try:
        song = am.song(song_id, storefront=storefront)
        attrs = song["data"][0]["attributes"]

        # define all the fields you care about
        fields = [
            "albumName", "artistName", "artistUrl", "artwork", "audioVariants",
            "composerName", "contentRating", "discNumber", "durationInMillis",
            "editorialNotes", "genreNames", "hasLyrics", "isAppleDigitalMaster",
            "isrc", "name", "playParams", "previews", "releaseDate",
            "trackNumber", "url"
        ]

        # build dict safely (use None if not present)
        metadata = {field: attrs.get(field, None) for field in fields}

        return metadata

    except Exception as e:
        print(f"Error fetching metadata for {song_id}: {e}")
        return {field: None for field in fields}

# %%
import time

FIELDS = [
    "albumName","artistName","artistUrl","artwork","audioVariants",
    "composerName","contentRating","discNumber","durationInMillis",
    "editorialNotes","genreNames","hasLyrics","isAppleDigitalMaster",
    "isrc","name","playParams","previews","releaseDate","trackNumber","url"
]

#%%
import numpy as np


# 1) Load previous output and ensure columns exist
df = pd.read_csv("data/processed_data/metadata.csv")

for col in (["apple_song_id"] + FIELDS):
    if col not in df.columns:
        df[col] = np.nan  # ensure presence

# 2) Build mask of rows to retry:
#   - either no song id
#   - or all metadata fields are null
mask_no_id = df["apple_song_id"].isna()
mask_all_null_meta = df[FIELDS].isnull().all(axis=1)
retry_mask = mask_no_id | mask_all_null_meta

null_rows = df.loc[retry_mask].copy()
print("Rows with missing ID or metadata:", len(null_rows))

# 3) Retry only those rows; keep original index to merge back
records_retry = []
save_every = 50
out_partial = "data/processed_data/unique_emerging_songs_apple_music_retry_partial.csv"

for k, (idx, row) in enumerate(tqdm(null_rows.iterrows(), total=len(null_rows)), start=1):
    title = row["title"]
    artist = row["main_artist"]
    durf_id = row["song_id"]

    # 3a) Resolve/refresh song_id if missing
    song_id = row.get("apple_song_id")
    if pd.isna(song_id) or not song_id:
        song_id = get_song_id(title, artist, durf_id)

    # 3b) Fetch metadata (fill all fields with None if still missing)
    meta = get_song_metadata(song_id) if song_id else {f: None for f in FIELDS}

    # 3c) Store result with original index for precise updating
    record = {"_idx": idx, "apple_song_id": song_id, **meta}
    records_retry.append(record)

    # Gentle pacing (optional)
    time.sleep(0.15)

    # Periodic checkpoint
    if k % save_every == 0:
        pd.DataFrame(records_retry).to_csv(out_partial, index=False)

df_retry = pd.DataFrame(records_retry).set_index("_idx")

# 4) Update the original df in place using index alignment
cols_to_update = ["apple_song_id"] + FIELDS
df.loc[df_retry.index, cols_to_update] = df_retry[cols_to_update]

#%%

# 5) Save final
df.to_csv("data/processed_data/metadata_updated.csv", index=False)
print("Updated rows:", len(df_retry))

# %%
am.search("Bad BDos Mil 16", types=["songs"], limit=10)
# %%
results = am.search("Love Me 4 Me SOS Deluxe - SZA", types=["songs"], limit=25)
for item in results['results']['songs']['data']:
    if (fuzz.ratio(item['attributes']['name'].lower(), "Love Me 4 Me".lower()) >= 70 
        and fuzz.ratio(item['attributes']['artistName'].lower(), "SZA".lower()) >= 70):
        print(item['attributes']['name'])
        print(item['attributes']['artistName'])
    else:
        print("Not found, here we had: ", item['attributes']['name'], item['attributes']['artistName'])
#%%
results = am.search("emo girl machine gun kelly", types=["songs"], limit=25)
for item in results['results']['songs']['data']:
    print("Song: ", item)
    
#%%
from rapidfuzz import fuzz

similarity = fuzz.ratio("Gunna", "Gunna & Future")
print(similarity)



"""
Songs that can be answered with .search()

Pushin P
wgft
gp
M.T.B.T.T.F.
"""
# %%
'Gunna'.lower() in 'Gunna & Future'.lower()
# %%
"Broadway Girls — Lil Durk Featuring Morgan Wallen".split(" — ")[1]
# %%