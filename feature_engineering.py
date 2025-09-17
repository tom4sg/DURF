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

Here are features we can add from the emerging_songs dataframe:
- song_id
- title
- artist
- release_date
- entry_week_date
- entry_week_pos
- peak_pos
- lifespan
"""
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast
from utils import expand_to_full_window, plot_two_metrics

#%%

emerging_songs = pd.read_csv("data/processed_data/emerging_songs.csv")
solo_songs = emerging_songs[emerging_songs["main_artist"] == emerging_songs["performers"]].copy()
solo_songs = solo_songs.drop_duplicates(subset="song_id")
solo_songs.groupby("main_artist")["song_id"].nunique()

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

feature_df = pd.DataFrame({
    "song_id": sampled_solo_songs["song_id"],
    "title": sampled_solo_songs["title"],
    "artist": sampled_solo_songs["main_artist"],
    "entry_week_date": sampled_solo_songs["chart_week"],
    "entry_week_pos": sampled_solo_songs["current_week"],
})

metadata = pd.read_csv("data/processed_data/metadata.csv")
metadata["genreNames"] = metadata["genreNames"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
metadata.drop(columns=["Unnamed: 0"], inplace=True)

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

youtube["date"] = pd.to_datetime(youtube["date"])
youtube["release_date"] = pd.to_datetime(youtube["release_date"])

mask = (
    (youtube["date"] >= youtube["release_date"] - pd.Timedelta(days=28)) &
    (youtube["date"] <= youtube["release_date"])
)

yt_pre4_df = youtube.loc[mask, ["artist","platform","date","release_date","subs","views"]].copy()
yt_pre4_df = yt_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)

yt_pre4_clean = expand_to_full_window(yt_pre4_df, ("artist","platform"), "date", "release_date", 28)

#%%
"""
Let's plot ALL Artist youtube data prior to release.
"""
plot_two_metrics(yt_pre4_clean, metric1="subs", metric2="views",
                 outdir="graphs/yt_original",
                 filename_pattern="yt_{artist}_{platform}.png")

#%%

grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
views_has_pos = yt_pre4_clean.groupby(grp)["views"].transform("max") > 0
subs_has_pos  = yt_pre4_clean.groupby(grp)["subs"].transform("max")  > 0

zero_views_missing = (yt_pre4_clean["views"] == 0) & views_has_pos
zero_subs_missing  = (yt_pre4_clean["subs"]  == 0) & subs_has_pos

yt_pre4_clean["views_zero_proxy_missing"] = zero_views_missing
yt_pre4_clean["subs_zero_proxy_missing"]  = zero_subs_missing

#%%

yt_pre4_clean[yt_pre4_clean["views_zero_proxy_missing"] == True]

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

yt_pre4_clean[yt_pre4_clean["subs_zero_proxy_missing"] == True]
# None!

#%%
"""
Ok, let's turn the 0.0 values to NaNs and then impute them with linear interpolation.
"""

#%%

target_artists = ["Chuckyy", "Eric Church", "Karol G", "Mariah the Scientist", "Sleep Token"]
mask = (yt_pre4_clean["artist"].isin(target_artists) & (yt_pre4_clean["views"] == 0.0))
yt_pre4_clean.loc[mask, "views"] = np.nan

yt_pre4_clean["views"] = (
    yt_pre4_clean
    .groupby("artist")["views"]
    .transform(lambda s: s.interpolate())
)

yt_pre4_clean["subs"] = (
    yt_pre4_clean
    .groupby("artist")["subs"]
    .transform(lambda s: s.interpolate())
)

yt_pre4_clean[yt_pre4_clean["artist"] == "Sleep Token"]["views"]

#%%

plot_two_metrics(yt_pre4_clean, metric1="subs", metric2="views",
                artists=["Chuckyy", "Eric Church", "Karol G", 
                "Mariah the Scientist", "Sleep Token"],
                 outdir="graphs/yt_imputed",
                 filename_pattern="yt_{artist}_{platform}.png")

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

instagram["date"] = pd.to_datetime(instagram["date"])
instagram["release_date"] = pd.to_datetime(instagram["release_date"])

mask = (
    (instagram["date"] >= instagram["release_date"] - pd.Timedelta(days=28)) &
    (instagram["date"] <= instagram["release_date"])
)

ig_pre4_df = instagram.loc[mask, ["artist","platform","date","release_date","followers","following", "media"]].copy()
ig_pre4_df = ig_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)

ig_pre4_clean = expand_to_full_window(ig_pre4_df, ("artist","platform"), "date", "release_date", 28)

#%%
grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = ig_pre4_clean.groupby(grp)["followers"].transform("max") > 0
media_has_pos  = ig_pre4_clean.groupby(grp)["media"].transform("max")  > 0

zero_followers_missing = (ig_pre4_clean["followers"] == 0) & followers_has_pos
zero_media_missing  = (ig_pre4_clean["media"]  == 0) & media_has_pos

ig_pre4_clean["followers_zero_proxy_missing"] = zero_followers_missing
ig_pre4_clean["media_zero_proxy_missing"]  = zero_media_missing

#%%

ig_pre4_clean[ig_pre4_clean["followers_zero_proxy_missing"] == True]
# None!

#%%

ig_pre4_clean[ig_pre4_clean["media_zero_proxy_missing"] == True]

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

plot_two_metrics(ig_pre4_clean, metric1="followers", metric2="media",
                 outdir="graphs/ig_original",
                 filename_pattern="ig_{artist}_{platform}.png")

#%%
"""
There are instances after using expand to full window where we need to beware of NaNs
and impute those, rather than just 0s

Let's impute, but skip the protocol of making 0s NaNs
"""

ig_pre4_clean["followers"] = (
    ig_pre4_clean
    .groupby("artist")["followers"]
    .transform(lambda s: s.interpolate())
)

ig_pre4_clean["media"] = (
    ig_pre4_clean
    .groupby("artist")["media"]
    .transform(lambda s: s.interpolate())
)

#%%

plot_two_metrics(ig_pre4_clean, metric1="followers", metric2="media",
                 outdir="graphs/ig_imputed",
                 filename_pattern="ig_{artist}_{platform}.png")

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

tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok["release_date"] = pd.to_datetime(tiktok["release_date"])

mask = (
    (tiktok["date"] >= tiktok["release_date"] - pd.Timedelta(days=28)) &
    (tiktok["date"] <= tiktok["release_date"])
)

tt_pre4_df = tiktok.loc[mask, ["artist","platform","date","release_date","followers","following", "uploads", "likes"]].copy()
tt_pre4_df = tt_pre4_df.sort_values(["artist","date"]).reset_index(drop=True)

tt_pre4_clean = expand_to_full_window(tt_pre4_df, ("artist","platform"), "date", "release_date", 28)

#%%

grp = ["artist", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = tt_pre4_clean.groupby(grp)["followers"].transform("max") > 0
uploads_has_pos  = tt_pre4_clean.groupby(grp)["uploads"].transform("max")  > 0
likes_has_pos  = tt_pre4_clean.groupby(grp)["likes"].transform("max")  > 0

zero_followers_missing = (tt_pre4_clean["followers"] == 0) & followers_has_pos
zero_media_missing  = (tt_pre4_clean["uploads"]  == 0) & uploads_has_pos
zero_likes_missing  = (tt_pre4_clean["likes"]  == 0) & likes_has_pos

tt_pre4_clean["followers_zero_proxy_missing"] = zero_followers_missing
tt_pre4_clean["uploads_zero_proxy_missing"]  = zero_media_missing
tt_pre4_clean["likes_zero_proxy_missing"]  = zero_likes_missing

#%%

tt_pre4_clean[tt_pre4_clean["followers_zero_proxy_missing"] == True]

"""
Jack Harlow (Impute)
Jonas Brothers (Impute)
Labrinth (Impute)
Teddy Swims (Impute)
Tim McGraw (Impute)

You can tell visually from their graphs, that these are all errors.
"""
#%%

tt_pre4_clean[tt_pre4_clean["uploads_zero_proxy_missing"] == True]

"""
Doja Cat (No Need to Impute)
Dua Lipa (No Need to Impute)
Linkin Park (No Need to Impute)
Roddy Ricch (No Need to Impute)
The Chainsmokers (No Need to Impute)
Tito Double P (No Need to Impute)

It is common for artists to archive and unarchive posts in quantities. 

No need to impute or be concerned. 
"""

#%%

tt_pre4_clean[tt_pre4_clean["likes_zero_proxy_missing"] == True]
# None!

#%%

plot_two_metrics(tt_pre4_clean, metric1="followers", metric2="likes",
                 outdir="graphs/tt_original",
                 filename_pattern="tt_{artist}_{platform}.png")

#%%
# Impute Tiktok Followers for these artists

target_artists = ["Jack Harlow", "Jonas Brothers", "Labrinth", "Teddy Swims", "Tim McGraw"]
mask = (tt_pre4_clean["artist"].isin(target_artists) & (tt_pre4_clean["followers"] == 0.0))
tt_pre4_clean.loc[mask, "followers"] = np.nan

tt_pre4_clean["followers"] = (
    tt_pre4_clean
    .groupby("artist")["followers"]
    .transform(lambda s: s.interpolate())
)

tt_pre4_clean["uploads"] = (
    tt_pre4_clean
    .groupby("artist")["uploads"]
    .transform(lambda s: s.interpolate())
)

tt_pre4_clean["likes"] = (
    tt_pre4_clean
    .groupby("artist")["likes"]
    .transform(lambda s: s.interpolate())
)

tt_pre4_clean[tt_pre4_clean["artist"] == "Jack Harlow"]["followers"]

#%%

plot_two_metrics(tt_pre4_clean, metric1="followers", metric2="likes",
                 outdir="graphs/tt_imputed",
                 filename_pattern="tt_{artist}_{platform}.png")

# %%
"""
Ok, we've linearly interpolated missing time-series data for artists!

Now we can begin calculating features around social data...

Release Date features:
- Parse total Followers / Subscribers (On release date)
- Parse total Likes / Views (On release date)

Posting Features:
- Compute average daily posts by Artist (4 weeks prior release)

Growth Rate features:
- Compute weekly growth rate for Followers / Subscribers (4 weeks prior release)
- Compute weekly growth rate for Likes / Views (4 weeks prior release)
"""

#%%
# We can start by making a new column in each of our pre4_df's for release date values

release_rows = yt_pre4_clean[yt_pre4_clean["date"] == yt_pre4_clean["release_date"]][
    ["artist", "release_date", "subs", "views"]
].rename(columns={"subs": "yt_subs_release_date", "views": "yt_views_release_date"})

feature_df["release_date"] = pd.to_datetime(feature_df["release_date"])
release_rows["release_date"] = pd.to_datetime(release_rows["release_date"])

feature_df = feature_df.merge(
    release_rows,
    on=["artist", "release_date"],
    how="left"
)

yt_pre4_clean.drop(columns=["views_zero_proxy_missing", "subs_zero_proxy_missing"], inplace=True)

# TT: followers, uploads, likes
release_rows = tt_pre4_clean[tt_pre4_clean["date"] == tt_pre4_clean["release_date"]][
    ["artist", "release_date", "followers", "uploads", "likes"]
].rename(columns={"followers": "tt_followers_release_date", "uploads": "tt_uploads_release_date", "likes": "tt_likes_release_date"})

feature_df = feature_df.merge(
    release_rows,
    on=["artist", "release_date"],
    how="left"
)

tt_pre4_clean.drop(columns=["followers_zero_proxy_missing", "uploads_zero_proxy_missing", "likes_zero_proxy_missing"], inplace=True)

# IG: followers, media
release_rows = ig_pre4_clean[ig_pre4_clean["date"] == ig_pre4_clean["release_date"]][
    ["artist", "release_date", "followers", "media"]
].rename(columns={"followers": "ig_followers_release_date", "media": "ig_media_release_date"})

feature_df = feature_df.merge(
    release_rows,
    on=["artist", "release_date"],
    how="left"
)

ig_pre4_clean.drop(columns=["followers_zero_proxy_missing", "media_zero_proxy_missing"], inplace=True)
feature_df

#%%

ig_pre4_clean = ig_pre4_clean.copy()
ig_pre4_clean['date'] = pd.to_datetime(ig_pre4_clean['date'], errors='coerce')
ig_pre4_clean = ig_pre4_clean.sort_values(['artist','date'])

# For growth rate, let's calculate geometric mean - 1 (CAGR style)
ig_pre4_clean['ig_media_cgr_4w'] = (
    ig_pre4_clean
      .groupby('artist')['media']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

ig_pre4_clean['ig_followers_cgr_4w'] = (
    ig_pre4_clean
      .groupby('artist')['followers']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

tt_pre4_clean = tt_pre4_clean.copy()
tt_pre4_clean['date'] = pd.to_datetime(tt_pre4_clean['date'], errors='coerce')
tt_pre4_clean = tt_pre4_clean.sort_values(['artist','date'])


tt_pre4_clean['tt_followers_cgr_4w'] = (
    tt_pre4_clean
      .groupby('artist')['followers']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

tt_pre4_clean['tt_uploads_cgr_4w'] = (
    tt_pre4_clean
      .groupby('artist')['uploads']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

tt_pre4_clean['tt_likes_cgr_4w'] = (
    tt_pre4_clean
      .groupby('artist')['likes']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

yt_pre4_clean = yt_pre4_clean.copy()
yt_pre4_clean['date'] = pd.to_datetime(yt_pre4_clean['date'], errors='coerce')
yt_pre4_clean = yt_pre4_clean.sort_values(['artist','date'])

yt_pre4_clean['yt_subs_cgr_4w'] = (
    yt_pre4_clean
      .groupby('artist')['subs']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

yt_pre4_clean['yt_views_cgr_4w'] = (
    yt_pre4_clean
      .groupby('artist')['views']
      .transform(lambda s: ((s / s.shift(28))**(1/28) - 1) * 100)
)

#%%

# Instagram
ig_feat = (
    ig_pre4_clean
      .loc[ig_pre4_clean['date'] == ig_pre4_clean['release_date'],
           ['artist','release_date','ig_media_cgr_4w','ig_followers_cgr_4w']]
      .drop_duplicates(['artist','release_date'])
)

# TikTok
tt_feat = (
    tt_pre4_clean
      .loc[tt_pre4_clean['date'] == tt_pre4_clean['release_date'],
           ['artist','release_date','tt_followers_cgr_4w','tt_uploads_cgr_4w','tt_likes_cgr_4w']]
      .drop_duplicates(['artist','release_date'])
)

# YouTube
yt_feat = (
    yt_pre4_clean
      .loc[yt_pre4_clean['date'] == yt_pre4_clean['release_date'],
           ['artist','release_date','yt_subs_cgr_4w','yt_views_cgr_4w']]
      .drop_duplicates(['artist','release_date'])
)

# Now merge on BOTH keys
feature_df = (
    feature_df
      .merge(ig_feat, on=['artist','release_date'], how='left', validate='many_to_one')
      .merge(tt_feat, on=['artist','release_date'], how='left', validate='many_to_one')
      .merge(yt_feat, on=['artist','release_date'], how='left', validate='many_to_one')
)
feature_df

#%%

feature_df.to_csv("data/processed_data/feature_df.csv", index=False)
#%%

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

mask = (feature_df["yt_views_cgr_4w"] < 100_000) & (feature_df["yt_views_cgr_4w"] > -1)
filtered = feature_df.loc[mask, "yt_views_cgr_4w"]

plt.figure(figsize=(8,5))
plt.hist(filtered * 100, bins=50)
plt.xlabel("YouTube Views 4-Week Growth Rate")
plt.ylabel("Frequency")
plt.title("Distribution of YouTube Views 4-Week Growth Rate")
plt.tight_layout()
plt.show()

#%%

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

# def billions(x, pos):
#     return f'{x/1e9:.1f}B'

# def millions(x, pos):
#     return f'{x/1e6:.1f}M'

#%%

# Instagram

# ig_followers_at_release = ig_pre4_clean.groupby('artist')['followers_release_date'].first()
# ig_media_at_release = ig_pre4_clean.groupby('artist')['media_release_date'].first()

# plt.figure(figsize=(8,5))
# plt.hist(ig_followers_at_release, bins=50)
# plt.xlabel("Followers (Millions)")
# plt.ylabel("Artist Count")
# plt.title("Distribution of Instagram Followers (release-date)")

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FuncFormatter(millions))
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(ig_media_at_release, bins=50)
# plt.xlabel("All-time Uploads")
# plt.ylabel("Artist Count")
# plt.title("Distribution of all-time Instagram uploads (release-date)")

# ax = plt.gca()
# plt.tight_layout()
# plt.show()

# #%%

# # TikTok

# tt_followers_at_release = tt_pre4_clean.groupby('artist')['followers_release_date'].first()
# tt_uploads_at_release = tt_pre4_clean.groupby('artist')['uploads_release_date'].first()
# tt_likes_at_release = tt_pre4_clean.groupby('artist')['likes_release_date'].first()

# plt.figure(figsize=(8,5))
# plt.hist(tt_followers_at_release, bins=50)
# plt.xlabel("Followers (Millions)")
# plt.ylabel("Artist Count")
# plt.title("Distribution of TikTok Followers (release-date)")

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FuncFormatter(millions))
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(tt_uploads_at_release, bins=50)
# plt.xlabel("All-time Uploads")
# plt.ylabel("Artist Count")
# plt.title("Distribution of all-time TikTok uploads (release-date)")

# ax = plt.gca()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(tt_likes_at_release, bins=50)
# plt.xlabel("All-time Likes (Billions)")
# plt.ylabel("Artist Count")
# plt.title("Distribution of all-time TikTok Likes (release-date)")

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FuncFormatter(billions))
# plt.tight_layout()
# plt.show()

# #%%
# # YouTube

# yt_views_at_release = yt_pre4_clean.groupby('artist')['views_release_date'].first()
# yt_subs_at_release = yt_pre4_clean.groupby('artist')['subs_release_date'].first()

# plt.figure(figsize=(8,5))
# plt.hist(yt_views_at_release, bins=50)
# plt.xlabel("All-Time Views on release date (Billions)")
# plt.ylabel("Count")
# plt.title("Distribution all-time YouTube Views (release-date)")

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FuncFormatter(billions))
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(yt_subs_at_release, bins=50)
# plt.xlabel("Subscribers on release date (Millions)")
# plt.ylabel("Count")
# plt.title("Distribution of YouTube Subscribers (release-date)")

# ax = plt.gca()
# ax.xaxis.set_major_formatter(FuncFormatter(millions))
# plt.tight_layout()
# plt.show()

"""
Statistical Analysis with Time-series data
"""


# %%
