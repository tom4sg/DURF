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

# %%

def get_song_id(song_name, artist_name, limit=5, storefront="us"):
    try:
        term = f"{song_name} {artist_name}"
        results = am.search(term, types=["songs"], limit=limit, storefront=storefront)

        for song in results.get("results", {}).get("songs", {}).get("data", []):
            title = song["attributes"]["name"].lower()
            artist = song["attributes"]["artistName"].lower()
            if song_name.lower() in title and artist_name.lower() in artist:
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

# 1) Load previous output and ensure columns exist
df = pd.read_csv("data/processed_data/unique_emerging_songs_apple_music.csv")

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

    # 3a) Resolve/refresh song_id if missing
    song_id = row.get("apple_song_id")
    if pd.isna(song_id) or not song_id:
        song_id = get_song_id(title, artist)

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

# 5) Save final
df.to_csv("data/processed_data/unique_emerging_songs_apple_music_updated.csv", index=False)
print("Updated rows:", len(df_retry))

# %%
