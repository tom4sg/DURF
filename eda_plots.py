#%%

"""
From Data Processing, we have 485 unique main_artists from all songs and 367 unique artists from solo songs.

And, I have collected the following: 

- Metadata: for the 2272 emerging songs (Apple Music API)
- Social media data: for the 485 artists (SocialBlade API)

But, we will only study songs from the 367 artists with solo songs. 

------------------------------------------------------------------------------------------------

Let's plot the distribution of the wks_on_chart field for all songs, solo songs, and collab songs.
"""
#%%

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast


#%%

emerging_songs_df = pd.read_csv("data/processed_data/emerging_songs.csv")

collab_rows = emerging_songs_df[
    emerging_songs_df["main_artist"] != emerging_songs_df["performers"]
]

solo_rows = emerging_songs_df[
    emerging_songs_df["main_artist"] == emerging_songs_df["performers"]
]

num_unique_collab_songs = collab_rows["song_id"].nunique()
num_unique_solo_songs = solo_rows["song_id"].nunique()
num_unique_solo_artists = solo_rows["main_artist"].nunique()

print(f"Unique songs (solo releases): {num_unique_solo_songs}")
print(f"Unique songs (collab releases): {num_unique_collab_songs}")
print(f"Unique solo artists: {num_unique_solo_artists}")
print(f"Unique performers: {emerging_songs_df['performers'].nunique()}")
print(f"Unique main artists: {emerging_songs_df['main_artist'].nunique()}")

#%%

weeks_on_chart = emerging_songs_df.copy()
weeks_on_chart.sort_values(by="wks_on_chart", ascending=False)
weeks_on_chart = weeks_on_chart.groupby("song_id")["wks_on_chart"].max().reset_index()

weeks_on_chart_no_collabs = solo_rows.copy()
weeks_on_chart_no_collabs.sort_values(by="wks_on_chart", ascending=False)
weeks_on_chart_no_collabs = weeks_on_chart_no_collabs.groupby("song_id")["wks_on_chart"].max().reset_index()

weeks_on_chart_only_collabs = collab_rows.copy()
weeks_on_chart_only_collabs.sort_values(by="wks_on_chart", ascending=False)
weeks_on_chart_only_collabs = weeks_on_chart_only_collabs.groupby("song_id")["wks_on_chart"].max().reset_index()

#%%

# weeks on chart plot across all songs, solo songs, and collab songs
plt.figure(figsize=(12,6))
sns.histplot(weeks_on_chart["wks_on_chart"], bins=30, kde=True, edgecolor="black")
plt.title("How Long Songs Stay on the Billboard Hot 100 (2022–2025)")
plt.xlabel("Weeks on Chart")
plt.ylabel("Frequency")
plt.show()

#%%

# weeks on chart plot across all songs, solo songs, and collab songs
plt.figure(figsize=(12,6))
sns.histplot(weeks_on_chart["wks_on_chart"], bins=30, edgecolor="black")
sns.histplot(weeks_on_chart_no_collabs["wks_on_chart"], bins=30, color="orange", edgecolor="black")
sns.histplot(weeks_on_chart_only_collabs["wks_on_chart"], bins=30, color="green", edgecolor="black")
plt.title("How Long Songs Stay on the Billboard Hot 100 (2022–2025)")
plt.xlabel("Weeks on Chart")
plt.ylabel("Frequency")
plt.legend(["All Songs", "One Artist", "Multiple Artists"])
plt.show()

#%%

import numpy as np
from matplotlib.lines import Line2D

#%%

"""
Let's try a spaghetti plot to plot lifespans as per Chon 2006
for emerging songs between 2022 - 2025
"""

#%%

df = emerging_songs_df.copy()
df = df.sort_values(["song_id", "chart_week"])
df["week_idx"] = df.groupby("song_id").cumcount()

lifespan = (df.groupby("song_id")["wks_on_chart"]
               .max()
               .rename("lifespan"))

bins = [0, 2, 8, 16, 32, 64, np.inf]
labels = ["1–2", "3–8", "9–16", "17–32", "33–64", "65+"]

lifespan_bins = pd.cut(lifespan, bins=bins, labels=labels, include_lowest=True, right=True)
song_bins = lifespan.to_frame().assign(lifespan_bin=lifespan_bins)

df = df.merge(song_bins, on="song_id", how="left")

bin_colors = {
    "1–2":   "#6b7280",  # gray
    "3–8":   "#1f77b4",  # blue
    "9–16":  "#2ca02c",  # green
    "17–32": "#ff7f0e",  # orange
    "33–64": "#9467bd",  # purple
    "65+":   "#d62728",  # red
}

# Aggregate by (lifespan_bin, week_idx)
agg_mean = (df
    .groupby(['lifespan_bin','week_idx'])['current_week']
    .mean()
    .reset_index(name='rank_mean'))

# IQR band for each bin/week
q25 = (df.groupby(['lifespan_bin','week_idx'])['current_week']
         .quantile(0.25).reset_index(name='q25'))
q75 = (df.groupby(['lifespan_bin','week_idx'])['current_week']
         .quantile(0.75).reset_index(name='q75'))
agg = agg_mean.merge(q25, on=['lifespan_bin','week_idx']).merge(q75, on=['lifespan_bin','week_idx'])

plt.figure(figsize=(12,6))

for label in labels:
    sub = agg[agg['lifespan_bin'] == label].sort_values('week_idx')
    if sub.empty: 
        continue

    # IQR band
    plt.fill_between(sub['week_idx'], sub['q25'], sub['q75'],
                     color=bin_colors[str(label)], alpha=0.12, linewidth=0)

    # Mean trajectory
    plt.plot(sub['week_idx'], sub['rank_mean'],
             color=bin_colors[str(label)], linewidth=2, label=label)

plt.gca().invert_yaxis()
plt.xlim(0, df["week_idx"].max())
plt.ylim(100.5, 0.5)
plt.xlabel("Weeks since debut")
plt.ylabel("Billboard Hot 100 Rank")
plt.title("Hot 100 Song Average Lifecycles (aligned at debut week)")
# plt.axhline(y=50, color="black", linestyle="--", label="50 weeks")
handles = [Line2D([0],[0], color=bin_colors[l], lw=3, label=l) for l in labels]
plt.legend(handles=handles, title="Total weeks on chart (Lifespan)", frameon=False, ncol=3)
plt.tight_layout()
plt.show()

#%%

df = emerging_songs_df.copy()
df = df.sort_values(["song_id", "chart_week"])
df["week_idx"] = df.groupby("song_id").cumcount()

lifespan = (df.groupby("song_id")["wks_on_chart"]
               .max()
               .rename("lifespan"))

bins = [0, 2, 8, 16, 32, 64, np.inf]
labels = ["1–2", "3–8", "9–16", "17–32", "33–64", "65+"]

lifespan_bins = pd.cut(lifespan, bins=bins, labels=labels, include_lowest=True, right=True)
song_bins = lifespan.to_frame().assign(lifespan_bin=lifespan_bins)

df = df.merge(song_bins, on="song_id", how="left")

bin_colors = {
    "1–2":   "#6b7280",  # gray
    "3–8":   "#1f77b4",  # blue
    "9–16":  "#2ca02c",  # green
    "17–32": "#ff7f0e",  # orange
    "33–64": "#9467bd",  # purple
    "65+":   "#d62728",  # red
}

plt.figure(figsize=(12,6))

for label in labels:
    sub = df[df["lifespan_bin"] == label]
    for sid, g in sub.groupby("song_id"):
        plt.plot(g["week_idx"], g["current_week"],
                 linewidth=0.28, alpha=0.5, color=bin_colors[str(label)], rasterized=True)

plt.gca().invert_yaxis()
plt.xlim(0, df["week_idx"].max())
plt.ylim(100.5, 0.5)
plt.xlabel("Weeks since debut")
plt.ylabel("Billboard Hot 100 Rank")
plt.title("Hot 100 Song lifecycles (aligned at debut week)")
plt.axhline(y=50, color="black", linestyle="--", label="50 weeks")
handles = [Line2D([0],[0], color=bin_colors[l], lw=3, label=l) for l in labels]
plt.legend(handles=handles, title="Total weeks on chart (Lifespan)", frameon=False, ncol=3)
plt.tight_layout()
plt.show()

#%%
"""
Notice two things:

- In the histogram: The bump at 20 weeks.
- In the spaghetti plot: The cutoff of songs below Pos. 50 after 20 weeks

I didn't know this until now, but Look at these rules!
https://www.billboard.com/billboard-charts-legend/

"Descending songs are removed from the Billboard Hot 100 and Radio Songs simultaneously after 20 weeks on 
the Billboard Hot 100 and if ranking below No. 50, or after 52 weeks if below No. 25." 

So, if this weren't the case, the gap between week 1 and week 20 would be filled
with a smooth exponential type decay. Records falling from 50 would fall to fill these
spots in the 70s 80s etc.

And of course, we would see the downward trajectory of songs after 20 weeks and on
in the spaghetti plot.

------------------------------------------------------------------------------------------------
Ok, so now we will load in our metadata for our songs and see how removing collaborations effects all of our
potential independent variables (ie. metadata and socials)

For reference, you can find the metadata I collected for each song
in the documentation for the song.Attributes object within the
 Apple Music API below:
https://developer.apple.com/documentation/applemusicapi/songs/attributes-data.dictionary
"""

#%%

# Load our collected metadata
metadata = pd.read_csv("data/processed_data/metadata.csv").reset_index(drop=True)
metadata["genreNames"] = metadata["genreNames"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
metadata.drop(columns=["Unnamed: 0"], inplace=True)
metadata

#%%

metadata_no_collabs = metadata[metadata["performers"] == metadata["main_artist"]].copy()
metadata_no_collabs

#%%

metadata_only_collabs = metadata[metadata["performers"] != metadata["main_artist"]].copy()
metadata_only_collabs

#%%

# Let's see how genre distribution was affected by removing collaborations.

#%%
genres_exploded = metadata.explode("genreNames")
genres_exploded = genres_exploded[genres_exploded["genreNames"] != "Music"]

genres_exploded_no_collabs = metadata_no_collabs.explode("genreNames")
genres_exploded_no_collabs = genres_exploded_no_collabs[genres_exploded_no_collabs["genreNames"] != "Music"]

genre_exploded_only_collabs = metadata_only_collabs.explode("genreNames")
genre_exploded_only_collabs = genre_exploded_only_collabs[genre_exploded_only_collabs["genreNames"] != "Music"]

#%%

genre_counts = genres_exploded["genreNames"].value_counts()
genre_counts_no_collabs = genres_exploded_no_collabs["genreNames"].value_counts()
genre_counts_only_collabs = genre_exploded_only_collabs["genreNames"].value_counts()

top_genres = genre_counts.head(30)
top_genres_no_collabs = genre_counts_no_collabs.head(30)
top_genres_only_collabs = genre_counts_only_collabs.head(30)

#%%

order = top_genres.index

plt.figure(figsize=(12,6))
top_genres.loc[order].plot(kind="bar")
top_genres_only_collabs.reindex(order).plot(kind="bar", color="orange")
plt.title("Distribution of Genre Tags for Billboard Hot 100 Songs (2022-2025)")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Songs", fontsize=12)
plt.legend(["One Artist", "Multiple Artists"])
plt.show()

#%%

"""
Make sense, see here:
https://www.economist.com/business/2018/02/03/in-popular-music-collaborations-rock

Hip-Hop, R&B, and adjacent see a dip in the Solo Artist graph, 
as these are genres known for collaborations.
"""

#%%

top_genres = genre_counts.head(30)
plt.figure(figsize=(12,6))
top_genres.plot(kind="bar")
plt.title("Distribution of Genre Tags (2022-2025)")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Songs", fontsize=12)
plt.show()

#%%

# Let's see duration of songs

#%%

plt.figure(figsize=(12,6))
plt.hist(metadata["durationInMillis"] / 60000, bins=30, label="All Songs")
plt.title("Distribution of Song Durations on Billboard Hot 100 (2022–2025)")
plt.xlabel("Song Duration (minutes)")
plt.ylabel("Count")
plt.axvline(x=(metadata["durationInMillis"] / 60000).mean(), color="black", linestyle="--")
plt.legend(["All Songs", "x̄=" + str(round((metadata["durationInMillis"] / 60000).mean(), 2))])
plt.show()

#%%

print(metadata["durationInMillis"].describe())
print(metadata_no_collabs["durationInMillis"].describe())
print(metadata_only_collabs["durationInMillis"].describe())

#%%

# Possibly do some significance testing here.

#%%

"""
I have also collected social media data for these artists 485 artists. Note that some don't have socials
But, we can work with the ones that do...

Log-followers at release (scale)
30 to 60 to 60 day compound weekly growth rate before release (momentum)
"""

#%%

import pandas as pd

#%%

tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok.drop(columns=["Unnamed: 0"], inplace=True)
tiktok

#%%

artist_name = "JENNIE"

artist_tiktok = tiktok[tiktok["artist_id"] == artist_name].sort_values("date")

normed = artist_tiktok[["followers","uploads","likes"]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
normed["date"] = artist_tiktok["date"]

plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="followers", data=normed, label="Followers")
sns.lineplot(x="date", y="uploads", data=normed, label="Uploads")
sns.lineplot(x="date", y="likes", data=normed, label="Likes")

plt.title(f"{artist_name} TikTok Metrics (Normalized)")
plt.ylabel("Normalized (0–1)")
plt.show()

# %%

youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
youtube["date"] = pd.to_datetime(youtube["date"])
youtube.drop(columns=["Unnamed: 0"], inplace=True)
youtube

#%%
artist_name = "Taylor Swift"

artist_youtube = youtube[youtube["artist_id"] == artist_name].sort_values("date")

normed = artist_youtube[["subs","views"]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
normed["date"] = artist_youtube["date"]
normed

#%%

plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="subs", data=normed, label="subscribers")
sns.lineplot(x="date", y="views", data=normed, label="Views")

plt.title(f"{artist_name} TikTok Metrics (Normalized)")
plt.ylabel("Normalized (0–1)")
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns

features = pd.read_csv('data/processed_data/feature_df.csv')

# %%

plt.scatter(features["ig_followers_cgr_4w"],features["lifespan"])
plt.show()

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

# %%

import pandas as pd
feature_df = pd.read_csv('data/processed_data/feature_df.csv')
instagram = pd.read_csv('data/raw_data/social_archives/instagram_archive.csv')
instagram = instagram.drop(columns=['Unnamed: 0'])
instagram["date"] = pd.to_datetime(instagram["date"])
feature_df["release_date"] = pd.to_datetime(feature_df["release_date"])
ig_nonzero = instagram[instagram['followers'] > 0]
earliest_per_artist = ig_nonzero.groupby("artist_id")["date"].min()
latest_per_artist = ig_nonzero.groupby("artist_id")["date"].max()

ig_coverage = feature_df.merge(
    earliest_per_artist.rename('ig_earliest'),
    left_on='artist', right_index=True, how='left'
).merge(
    latest_per_artist.rename('ig_latest'),
    left_on='artist', right_index=True, how='left'
)

ig_coverage['days_before_release'] = (ig_coverage['release_date'] - ig_coverage['ig_earliest']).dt.days
ig_coverage['days_after_release']  = (ig_coverage['ig_latest'] - ig_coverage['release_date']).dt.days

tiktok = pd.read_csv('data/raw_data/social_archives/tiktok_archive.csv')
tiktok = tiktok.drop(columns=['Unnamed: 0'])
tiktok["date"] = pd.to_datetime(tiktok["date"])
feature_df["release_date"] = pd.to_datetime(feature_df["release_date"])
tt_nonzero = tiktok[tiktok['followers'] > 0]
earliest_per_artist = tt_nonzero.groupby("artist_id")["date"].min()
latest_per_artist = tt_nonzero.groupby("artist_id")["date"].max()
tt_coverage = feature_df.merge(
    earliest_per_artist.rename('tt_earliest'),
    left_on='artist', right_index=True, how='left'
).merge(
    latest_per_artist.rename('tt_latest'),
    left_on='artist', right_index=True, how='left'
)

tt_coverage['days_before_release'] = (tt_coverage['release_date'] - tt_coverage['tt_earliest']).dt.days
tt_coverage['days_after_release']  = (tt_coverage['tt_latest'] - tt_coverage['release_date']).dt.days

youtube = pd.read_csv('data/raw_data/social_archives/youtube_archive.csv')
youtube = youtube.drop(columns=['Unnamed: 0'])
youtube["date"] = pd.to_datetime(youtube["date"])
feature_df["release_date"] = pd.to_datetime(feature_df["release_date"])
yt_nonzero = youtube[youtube['subs'] > 0]
earliest_per_artist = yt_nonzero.groupby("artist_id")["date"].min()
latest_per_artist = yt_nonzero.groupby("artist_id")["date"].max()

yt_coverage = feature_df.merge(
    earliest_per_artist.rename('yt_earliest'),
    left_on='artist', right_index=True, how='left'
).merge(
    latest_per_artist.rename('yt_latest'),
    left_on='artist', right_index=True, how='left'
)

yt_coverage['days_before_release'] = (yt_coverage['release_date'] - yt_coverage['yt_earliest']).dt.days
yt_coverage['days_after_release']  = (yt_coverage['yt_latest'] - yt_coverage['release_date']).dt.days

#%%

import numpy as np
import matplotlib.pyplot as plt

months_thresholds = np.arange(1, 13)
days_thresholds = months_thresholds * 30

# assume you have three DataFrames: ig_coverage, yt_coverage, tt_coverage
pre_counts_ig  = [(ig_coverage['days_before_release'] > d).sum() for d in days_thresholds]
post_counts_ig = [(ig_coverage['days_after_release']  > d).sum() for d in days_thresholds]

pre_counts_yt  = [(yt_coverage['days_before_release'] > d).sum() for d in days_thresholds]
post_counts_yt = [(yt_coverage['days_after_release']  > d).sum() for d in days_thresholds]

pre_counts_tt  = [(tt_coverage['days_before_release'] > d).sum() for d in days_thresholds]
post_counts_tt = [(tt_coverage['days_after_release']  > d).sum() for d in days_thresholds]

plt.figure(figsize=(10,6))

# Instagram (violet-pink)
plt.plot(months_thresholds, pre_counts_ig, color='#8A3AB9', label='Instagram Pre-release')
plt.plot(months_thresholds, post_counts_ig, color='#8A3AB9', linestyle='--', label='Instagram Post-release')

# YouTube (red)
plt.plot(months_thresholds, pre_counts_yt, color='#FF0000', label='YouTube Pre-release')
plt.plot(months_thresholds, post_counts_yt, color='#FF0000', linestyle='--', label='YouTube Post-release')

# TikTok (muted teal)
plt.plot(months_thresholds, pre_counts_tt, color='#4DB6AC', label='TikTok Pre-release')
plt.plot(months_thresholds, post_counts_tt, color='#4DB6AC', linestyle='--', label='TikTok Post-release')

plt.xlabel('Months of data (threshold)')
plt.ylabel('Number of artists with more than threshold')
plt.title('Artists exceeding X months of social data coverage')
plt.grid(True)
plt.legend()
plt.show()

# %%

#%%

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

# def billions(x, pos):
#     return f'{x/1e9:.1f}B'

# def millions(x, pos):
#     return f'{x/1e6:.1f}M'

# #%%

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

# # #%%

# # # TikTok

# # tt_followers_at_release = tt_pre4_clean.groupby('artist')['followers_release_date'].first()
# # tt_uploads_at_release = tt_pre4_clean.groupby('artist')['uploads_release_date'].first()
# # tt_likes_at_release = tt_pre4_clean.groupby('artist')['likes_release_date'].first()

# # plt.figure(figsize=(8,5))
# # plt.hist(tt_followers_at_release, bins=50)
# # plt.xlabel("Followers (Millions)")
# # plt.ylabel("Artist Count")
# # plt.title("Distribution of TikTok Followers (release-date)")

# # ax = plt.gca()
# # ax.xaxis.set_major_formatter(FuncFormatter(millions))
# # plt.tight_layout()
# # plt.show()

# # plt.figure(figsize=(8,5))
# # plt.hist(tt_uploads_at_release, bins=50)
# # plt.xlabel("All-time Uploads")
# # plt.ylabel("Artist Count")
# # plt.title("Distribution of all-time TikTok uploads (release-date)")

# # ax = plt.gca()
# # plt.tight_layout()
# # plt.show()

# # plt.figure(figsize=(8,5))
# # plt.hist(tt_likes_at_release, bins=50)
# # plt.xlabel("All-time Likes (Billions)")
# # plt.ylabel("Artist Count")
# # plt.title("Distribution of all-time TikTok Likes (release-date)")

# # ax = plt.gca()
# # ax.xaxis.set_major_formatter(FuncFormatter(billions))
# # plt.tight_layout()
# # plt.show()

# # #%%
# # # YouTube

# # yt_views_at_release = yt_pre4_clean.groupby('artist')['views_release_date'].first()
# # yt_subs_at_release = yt_pre4_clean.groupby('artist')['subs_release_date'].first()

# # plt.figure(figsize=(8,5))
# # plt.hist(yt_views_at_release, bins=50)
# # plt.xlabel("All-Time Views on release date (Billions)")
# # plt.ylabel("Count")
# # plt.title("Distribution all-time YouTube Views (release-date)")

# # ax = plt.gca()
# # ax.xaxis.set_major_formatter(FuncFormatter(billions))
# # plt.tight_layout()
# # plt.show()

# # plt.figure(figsize=(8,5))
# # plt.hist(yt_subs_at_release, bins=50)
# # plt.xlabel("Subscribers on release date (Millions)")
# # plt.ylabel("Count")
# # plt.title("Distribution of YouTube Subscribers (release-date)")

# # ax = plt.gca()
# # ax.xaxis.set_major_formatter(FuncFormatter(millions))
# # plt.tight_layout()
# # plt.show()