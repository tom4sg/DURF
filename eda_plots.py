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
