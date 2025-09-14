"""

I should do power analysis before train test split!

Ok, we have to consider a few things in the feature engineering process:

Some of our unique artists have several songs, we need to randomly select one per artist

We only want to use data prior to, or on a particular song's release date. 

We must remember that there are one-off null values in Social Data, which we need to account for (Maybe Imputation?).

We want to create features inspired by the following:

- Artist Follower Count (On Release Date)
- Compound Weekly Growth Rate of Artist Followers (the 4 weeks prior to release)
- Relevancy-to-Release Score - How relevant is the content of their posting to the release? (via NLP on comments, captions, hashtags)

- post rate
- cwgr

Potential:
- Primary Genre Tag (To account for the imbalance of solo releases)
- First week position on Hot 100
- Song Duration

Feature Scaling:
It might be advantageous to log transform for CWGR.
Maybe Min Max normalize or z-score. 
"""
#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

emerging_songs = pd.read_csv("data/processed_data/emerging_songs.csv")
solo_songs = emerging_songs[emerging_songs["main_artist"] == emerging_songs["performers"]].copy()
solo_songs = solo_songs.drop_duplicates(subset="song_id")
solo_songs.groupby("main_artist")["song_id"].nunique()

#%%

sampled_solo_songs = (solo_songs
           .groupby("main_artist", group_keys=False)
           .apply(lambda g: g.sample(1, random_state=42))
           .reset_index(drop=True))

song_peaks = (emerging_songs
           .groupby("song_id", as_index=False)["current_week"]
           .min()
           .rename(columns={"current_week": "peak_pos"}))

lifespans = (emerging_songs
           .groupby("song_id", as_index=False)["wks_on_chart"]
           .max()
           .rename(columns={"wks_on_chart": "lifespan"}))

#%%

"""
Here are features we can add from the emerging_songs dataframe:
- song_id
- title
- artist
- release_date
- entry_week_date
- entry_week_pos
- peak_pos
- lifespan

Let's add the first 3...
"""

#%%

feature_df = pd.DataFrame({
    "song_id": sampled_solo_songs["song_id"],
    "title": sampled_solo_songs["title"],
    "artist": sampled_solo_songs["main_artist"],
    "entry_week_date": sampled_solo_songs["chart_week"],
    "entry_week_pos": sampled_solo_songs["current_week"],
})

feature_df

#%%

"""
For release date, we can get this from metadata.csv based on song_id.
"""

#%%

import ast

#%%

metadata = pd.read_csv("data/processed_data/metadata.csv")
metadata["genreNames"] = metadata["genreNames"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
metadata.drop(columns=["Unnamed: 0"], inplace=True)
metadata

#%%

feature_df = feature_df.merge(
    metadata[["song_id", "releaseDate", "genreNames", "durationInMillis"]]
      .rename(columns={"releaseDate": "release_date", "genreNames": "genres", "durationInMillis": "song_length"}),
    on="song_id",
    how="left"
)

# remove the string "Music" genres column
feature_df["genres"] = feature_df["genres"].apply(
    lambda g: [x for x in g if x != "Music"]
)

feature_df["song_length"] = feature_df["song_length"] / 60000

feature_df = feature_df.merge(song_peaks, on="song_id", how="left")
feature_df = feature_df.merge(lifespans, on="song_id", how="left")

# Let's reorder the columns to what I described in the comment

cols = ["song_id", "title", "artist", "genres",
        "song_length", "release_date", "entry_week_date", 
        "entry_week_pos", "peak_pos", "lifespan"] 
feature_df = feature_df[cols]
feature_df

#%%

"""
Ok, now, we do / compute the following before the train test split:

1. Impute time-series for YouTube Views with pandas linear interpolation
2. Growth Rate features
  - Compute weekly growth rate for Followers / Subscribers (4 weeks prior release)
  - Compute weekly growth rate for Likes / Views (4 weeks prior release)
3. Release Date features
  - Parse total Followers / Subscribers (On release date)
  - Parse total Likes / Views (On release date)
4. Posting Features
  - Compute average daily posts by Artist (4 weeks prior release)
"""

#%%

instagram = pd.read_csv("data/raw_data/social_archives/instagram_archive.csv")
instagram["date"] = pd.to_datetime(instagram["date"])
instagram.drop(columns=["Unnamed: 0"], inplace=True)

tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok.drop(columns=["Unnamed: 0"], inplace=True)

youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
youtube["date"] = pd.to_datetime(youtube["date"])
youtube.drop(columns=["Unnamed: 0"], inplace=True)

#%%

"""
Let's plot a null heatmap to see where artists are missing data in each social
platform. 

for Youtube, it seems to be specifically when SocialBlade API changed their API.
"""

#%%

youtube = youtube.merge(
    feature_df[["artist", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

#%%

youtube["date"] = pd.to_datetime(youtube["date"])
youtube["release_date"] = pd.to_datetime(youtube["release_date"])

mask = (
    (youtube["date"] >= youtube["release_date"] - pd.Timedelta(days=28)) &
    (youtube["date"] <= youtube["release_date"])
)

yt_pre4_df = youtube.loc[mask, ["artist","platform","date","release_date","subs","views"]].copy()

yt_pre4_df = yt_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)
yt_pre4_df[yt_pre4_df["artist"] == "Sleep Token"]

#%%
"""
Let's plot ALL Artist youtube data prior to release.
"""

for (artist, platform), g in yt_pre4_df.groupby(["artist", "platform"]):
    g = g.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(8,4))
    line_subs, = ax1.plot(g["date"], g["subs"], marker="o", label="Subscribers")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Subscribers")

    ax2 = ax1.twinx()
    line_views, = ax2.plot(g["date"], g["views"], marker=".", linestyle="--", label="Views")
    ax2.set_ylabel("Views")

    rd = g["release_date"].iloc[0]
    ax1.axvline(rd, linestyle=":", linewidth=1)

    fig.autofmt_xdate()
    handles = [line_subs, line_views]
    labels  = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"graphs/yt_original/youtube_pre4_{artist}_{platform}.png", dpi=150)
    plt.close(fig)

#%%

grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
views_has_pos = yt_pre4_df.groupby(grp)["views"].transform("max") > 0
subs_has_pos  = yt_pre4_df.groupby(grp)["subs"].transform("max")  > 0

zero_views_missing = (yt_pre4_df["views"] == 0) & views_has_pos
zero_subs_missing  = (yt_pre4_df["subs"]  == 0) & subs_has_pos

yt_pre4_df["views_zero_proxy_missing"] = zero_views_missing
yt_pre4_df["subs_zero_proxy_missing"]  = zero_subs_missing
yt_pre4_df

#%%

yt_pre4_df[yt_pre4_df["views_zero_proxy_missing"] == True]

#%%

"""
Artists missing views data:

Chuckyy (Impute)
Eric Church (Impute)
Karol G (Impute)
Mariah the Scientist (Impute)
Sleep Token (Impute)
YG Marley (Don't Impute)
---

It's important to note, SocialBlade API went down for a few days in April 2025. 

Those above with (Impute) are because of this, YG Marley's page was at 0 views 
4 weeks before release. 
"""

#%%

yt_pre4_df[yt_pre4_df["subs_zero_proxy_missing"] == True]
# None!

#%%
"""
Ok, let's turn the 0.0 values to NaNs and then impute them with linear interpolation.
"""

#%%

import numpy as np

#%%

target_artists = ["Chuckyy", "Eric Church", "Karol G", "Mariah the Scientist", "Sleep Token"]

mask = (yt_pre4_df["artist"].isin(target_artists) & (yt_pre4_df["views"] == 0.0))

yt_pre4_df.loc[mask, "views"] = np.nan

yt_pre4_df.loc[yt_pre4_df["artist"].isin(target_artists), "views"] = (
    yt_pre4_df[yt_pre4_df["artist"].isin(target_artists)]
    .groupby("artist")["views"]
    .transform(lambda s: s.interpolate())
)

#%%

yt_pre4_df[yt_pre4_df["artist"] == "Sleep Token"]["views"]

#%%

df_filtered = yt_pre4_df[yt_pre4_df["artist"].isin(target_artists)]

for (artist, platform), g in df_filtered.groupby(["artist", "platform"]):
    g = g.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(8,4))
    line_subs, = ax1.plot(g["date"], g["subs"], marker="o", label="Subscribers")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Subscribers")

    ax2 = ax1.twinx()
    line_views, = ax2.plot(g["date"], g["views"], marker=".", linestyle="--", label="Views")
    ax2.set_ylabel("Views")

    rd = g["release_date"].iloc[0]
    ax1.axvline(rd, linestyle=":", linewidth=1)

    fig.autofmt_xdate()
    handles = [line_subs, line_views]
    labels  = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"graphs/yt_imputed/youtube_pre4_{artist}_{platform}.png", dpi=150)
    plt.close(fig)

#%%

"""
Ok, Let's check for IG. 
"""
#%%

instagram = instagram.merge(
    feature_df[["artist", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

#%%

instagram["date"] = pd.to_datetime(instagram["date"])
instagram["release_date"] = pd.to_datetime(instagram["release_date"])

mask = (
    (instagram["date"] >= instagram["release_date"] - pd.Timedelta(days=28)) &
    (instagram["date"] <= instagram["release_date"])
)

ig_pre4_df = instagram.loc[mask, ["artist","platform","date","release_date","followers","following", "media"]].copy()

ig_pre4_df = ig_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)
ig_pre4_df

#%%

grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = ig_pre4_df.groupby(grp)["followers"].transform("max") > 0
media_has_pos  = ig_pre4_df.groupby(grp)["media"].transform("max")  > 0

zero_followers_missing = (ig_pre4_df["followers"] == 0) & followers_has_pos
zero_media_missing  = (ig_pre4_df["media"]  == 0) & media_has_pos

ig_pre4_df["followers_zero_proxy_missing"] = zero_followers_missing
ig_pre4_df["media_zero_proxy_missing"]  = zero_media_missing
ig_pre4_df

#%%

ig_pre4_df[ig_pre4_df["followers_zero_proxy_missing"] == True]
# None!
#%%

ig_pre4_df[ig_pre4_df["media_zero_proxy_missing"] == True]

#%%
"""
Bad Bunny
Charlie Puth
Dua Lipa
Justin Timberlake
Ken Carson
Key Glock
Lewis Capaldi
Lil Baby
Roddy Ricch
Young Thug
---

The thing is here though, this might not be an error. Maybe the artist just decided to 
delete posts for a day / archive them - but, we can check the plots to see if this is viable. 

It's actually common for artists to take down all their posts. 

Also, none occured during the SocialBlade API Outage...
"""

#%%

for (artist, platform), g in ig_pre4_df.groupby(["artist", "platform"]):
    g = g.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(8,4))
    line_followers, = ax1.plot(g["date"], g["followers"], marker="o", label="Followers")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Followers")

    ax2 = ax1.twinx()
    line_media, = ax2.plot(g["date"], g["media"], marker=".", linestyle="--", label="Media")
    ax2.set_ylabel("Media")

    rd = g["release_date"].iloc[0]
    ax1.axvline(rd, linestyle=":", linewidth=1)

    fig.autofmt_xdate()
    handles = [line_followers, line_media]
    labels  = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"graphs/ig/instagram_pre4_{artist}_{platform}.png", dpi=150)
    plt.close(fig)

#%%
"""
Same for Instagram. Let's check for TikTok.
"""
#%%

tiktok = tiktok.merge(
    feature_df[["artist", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

#%%

tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok["release_date"] = pd.to_datetime(tiktok["release_date"])

mask = (
    (tiktok["date"] >= tiktok["release_date"] - pd.Timedelta(days=28)) &
    (tiktok["date"] <= tiktok["release_date"])
)

tt_pre4_df = tiktok.loc[mask, ["artist","platform","date","release_date","followers","following", "uploads", "likes"]].copy()

tt_pre4_df = tt_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)
tt_pre4_df

#%%

grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = tt_pre4_df.groupby(grp)["followers"].transform("max") > 0
uploads_has_pos  = tt_pre4_df.groupby(grp)["uploads"].transform("max")  > 0
likes_has_pos  = tt_pre4_df.groupby(grp)["likes"].transform("max")  > 0

zero_followers_missing = (tt_pre4_df["followers"] == 0) & followers_has_pos
zero_media_missing  = (tt_pre4_df["uploads"]  == 0) & uploads_has_pos
zero_likes_missing  = (tt_pre4_df["likes"]  == 0) & likes_has_pos

tt_pre4_df["followers_zero_proxy_missing"] = zero_followers_missing
tt_pre4_df["uploads_zero_proxy_missing"]  = zero_media_missing
tt_pre4_df["likes_zero_proxy_missing"]  = zero_likes_missing
tt_pre4_df

#%%

tt_pre4_df[tt_pre4_df["followers_zero_proxy_missing"] == True]

"""
Jack Harlow (Impute)
Jonas Brothers (Impute)
Labrinth (Impute)
Teddy Swims (Impute)
Tim McGraw (Impute)

You can tell visually from their graphs, that these are all errors.
"""
#%%

tt_pre4_df[tt_pre4_df["uploads_zero_proxy_missing"] == True]

"""
Doja Cat (No Need to Impute)
Dua Lipa (No Need to Impute)
Linkin Park (No Need to Impute)
Roddy Ricch (No Need to Impute)
The Chainsmokers (No Need to Impute)
Tito Double P (No Need to Impute)

It os common for artists to archive and unarchive posts in quantities. 

No need to impute or be concerned. 
"""

#%%

tt_pre4_df[tt_pre4_df["likes_zero_proxy_missing"] == True]
# None!

#%%

for (artist, platform), g in tt_pre4_df.groupby(["artist", "platform"]):
    g = g.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(8,4))
    line_followers, = ax1.plot(g["date"], g["followers"], marker="o", label="Followers")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Followers")

    ax2 = ax1.twinx()
    line_uploads, = ax2.plot(g["date"], g["uploads"], marker=".", linestyle="--", label="Uploads")
    ax2.set_ylabel("Uploads")

    rd = g["release_date"].iloc[0]
    ax1.axvline(rd, linestyle=":", linewidth=1)

    fig.autofmt_xdate()
    handles = [line_followers, line_uploads]
    labels  = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"graphs/tt/tiktok_pre4_{artist}_{platform}.png", dpi=150)
    plt.close(fig)
# %%
