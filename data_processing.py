"""
In our initial meeting with me and Fin, we discussed my DURF proposal, 
which was interested in predicting artist streaming consumption and recurring listenership (ie. how much one listener listens to an artist's songs) based on 
the artists' social media engagement.

So the initial plan was to:
- Frame it as a classification problem of the streaming trend upon release:
    - Slow rise slow fall, Slow rise fast fall, fast rise fast fall, fast rise slow fall
- get artist level streaming data for ~1000 artists
- collect metadata for those artists ie. Genre, release date, top 3 tracks
- find a feature engineering solution for social engagement data (Instagram, TikTok, YouTube, UGC)
- Using something like Random forest, xgboost, support vector machine. 

------------------------------------------------------------------------------------------------

So, I first set out to collect the Streaming data for artists. 

Luminate API:
- I spoke to a Luminate Representative about getting access to the Luminate API, a product under Nielsen. 
- This is what we used to collect streaming data when I worked at Warner. 
- They basically told me that it would be $4,000 a month minimum with a minimum 1 year contract. 

Chartmetric Developer API:
- So I tried to get access to a different API, the Chartmetric Developer API. 
- This had pretty underdeveloped Spotify play count data, and would only let you look back 1 month. 

But, I need archival streaming data!

------------------------------------------------------------------------------------------------

So I ended up reading up on this research field called "Hit Song Science", which has been around for 2 decades. 

a field in which people try to detect successful songs based on intrinsic and extrinsic features. 

intrinsic being this relating to the songs itself, its audio features, lyrics, etc.
extrinsic being how consumers interact with the song, the culture context that surrounds it etc.

The majority of research frames it as a classification problem, ie. whether a song is a hit or non-hit. 

Some define a hit as a song that has reached any spot on any billboard chart...
others define it by reaching a particular spot on the Billboard Hot 100...

But what was made clear to me specifically via a 2023 survey of HSS (Seufitelli 2023) was that
Billboard data was the most commonly used dataset for measuring success.

I came accross a handful of research papers you set out to predict the number of weeks a song or album spends on a billboard chart.
And this has become my main interest.

Chon 2006 did this with Albums on Top Jazz chart, and defined the following two terms:
- Lifecyle: The trajectory of the album’s weekly positions from the very first week to the very last week on the charts (which can include re-entries)
- Lifespan: The span of the lifecycle in weeks

# Ref: reference_papers/hit-song-science/chon-2006-predicting-success-from-music-sales-data-a-statistical-and-adaptive-approach.pdf

But, she does this with nearest neightbours, representing the first few weeks of the lifecycle as a vector, and 
comparing against buckets of lifespans on charts. 

------------------------------------------------------------------------------------------------

So, I have decided that I want to predice the number of weeks a song spends on the Billboard Hot 100, in the same vein as Chon 2006, but
using a feature representing the artists social media engagement prior to the song's release date. 
"""

#%%
import pandas as pd
import os

#%%

years = ["2022", "2023", "2024", "2025"]

total_hot_100 = pd.DataFrame()

for year in years:

    file_count = 0
    
    for file in os.listdir(f"data/raw_data/hot_100/{year}"):

        total_hot_100 = pd.concat([total_hot_100, pd.read_csv(f"data/raw_data/hot_100/{year}/{file}")])
        file_count += 1
        
    print(f"Added {file_count} files to hot_100 from {year}")

total_hot_100

#%%

"""
First thing to note here:
Song names are not unique identifiers. There are songs within this dataset from different artists that have the same name. 
Let's add a song_id field, which is (title) - (performers). 
"""

#%%

total_hot_100 = total_hot_100.sort_values(by=["chart_week", "current_week"], ascending=[True, True])
total_hot_100["song_id"] = total_hot_100["title"].str.strip() + " — " + total_hot_100["performer"].str.strip()
total_hot_100.rename(columns={"performer": "performers"}, inplace=True)
print(f"Unique songs: {total_hot_100['song_id'].nunique()}")
print(f"Unique performers: {total_hot_100['performers'].nunique()}")

#%%

"""
Here we have all the Billboard Hot 100 weekly charts from 2022 to 2025.
Within this, we have 2479 unique songs. 

This is great, but there are a few issues that we need to address:

1. There are songs whose lifecycle started before 2022, ie. their entry week to the billboard hot 100
was before our recorded period. 

2. By looking at the data, the "performers" field is a bit misleading, as in a collaboration, 
a string of several artist names are considered as a single performer. 

3. (Which I still haven't fully addressed) There are songs which are associated with movies, 
not necessarily with an artist ie. We Don't Talk About Bruno. 

------------------------------------------------------------------------------------------------

Since the goal is to use social data prior to a song's release date, we need to remove all songs 
whose entry week to the billboard hot 100 was before our recorded period. 

------------------------------------------------------------------------------------------------

If we really want to predict an song lifespan with Artist's social media engagement, these songs 
associated with movies will have strong spurious correlations.

Movie success leads to an artist's social media engagement, and the song success at the same time. 

Beware of these artists:
Jack Black, Ryan Gosling, Huntrx, citizens of halloween, encanto, 4town, Wicked, 
Saja Boys, Rumi
------------------------------------------------------------------------------------------------

So, let's keep the songs_ids whose lifecycle started within 2022 - 2025. 
(ie. All I Want For Christmas Is You by Mariah Carey)
"""

#%%

emerging_song_ids = total_hot_100.loc[total_hot_100["wks_on_chart"] == 1, "song_id"].unique()
print(f"Unique emerging songs: {len(emerging_song_ids)}")

# %%

emerging_songs_df = total_hot_100[total_hot_100["song_id"].isin(emerging_song_ids)].copy()
print(f"Unique emerging songs: {len(emerging_songs_df['song_id'].unique())}")

# %%

"""
Ok, now we have the Billboard Hot 100 chart data for songs whose lifecycle starts
within 2022 - 2025.

------------------------------------------------------------------------------------------------

The next bottleneck is deciding how to deal with the "performers" field, and 
songs with several artists on them. 

Initially, for the sake of ease and cost-friendliness, I wrote some code to
parse a "main_artist" field from the "performers" field. This would retrieve the 
first artist listed, assuming the first artist is the most relevant to the release. 

The first artist would be the one we collect social media data for...

But, is this really justified? I'm not sure if we can say that only one artist socials
are predictive of a song's success, when there are certainly scenarios in which a song's
success is do to the prominence of a featuring artist, and not the main artist.

regardless, I will still parse this "main_artist" field, as it will help us differentiate
between solo and collaborative songs.

------------------------------------------------------------------------------------------------

The following are the most common delimiters for songs with multiple artists:

- " Featuring "
- " And "
- " / "
- " X "
- " & "
- ", "
- " With "

Although this is not a universal truth, and again, for the sake of efficiency and cost-friendliness, we will consider the artist that appears first in the listed performers as the Main Artist, and the one for which we will retrieve Social Media Profile Data.
There are of course some caveats to this approach, as in certain situations, featuring artists could bring more attention to a song than the main artist on a given track. In future studies, and with more capital, I’d love to address this.
Some interesting artist names that might include delimiters like these, such as "Tyler, The Creator", "Dan + Shay", "Florence + The Machine". 
We can treat these cases seperately.

------------------------------------------------------------------------------------------------

Also, notice that some artists release songs under different names!!
ie. Machine Gun Kelly vs. mgk
We need to normalize these...
"""

#%%

import re

#%%

# Here are a the exceptions that I have compiled. 
artist_exceptions = {
    "tyler, the creator",
    "Brooks & Dunn",
    "Yahritza y Su Esencia",
    "Dan + Shay",
    "Florence + The Machine",
    "Lil Nas X",
    "Richy Mitch And The Coal Miners"
    "HUNTR/X",
}

# find the last - in song_id, which delimits song title from song artist
title_artist_split = re.compile(r"\s[—-]\s(?!.*\s[—-]\s)")

# Find the common delimiters for songs with multiple artists
delims = re.compile(
    r"""
    (?:                      
        ,\s*                                  # comma (no leading space required)
      | \s+(?:featuring|presents|with|and)\s+ # word delimiters with spaces
      | \s+y\s+                               # ' y ' with spaces
      | \s+x\s+                               # ' x ' with spaces
      | \s*&\s*                               # & with spaces
      | \s*/\s*                               # / with spaces
      | \s*\+\s*                              # + with spaces
    )
    """,
    re.IGNORECASE | re.VERBOSE
)

# find instances where the main artist is among the exceptions
exceptions_as_main = re.compile(
    r"^(?:%s)\b" % "|".join(re.escape(x) for x in artist_exceptions),
    flags=re.IGNORECASE
)


def extract_artists(song_id: str) -> str:
    """Extract the artist(s) portion from a song_id."""
    
    # use title_artist_split
    parts = title_artist_split.split(song_id.strip())
    if len(parts) >= 2:
        return parts[-1].strip()
    
    return song_id.strip()


def compute_main_artist_from_artists(artists_str: str) -> str:
    """Given the artist(s) string, return the main artist."""
    
    s = (artists_str or "").strip()
    if not s:
        return ""
    
    # Let's check if main artist is an exception
    m = exceptions_as_main.match(s)
    if m:
        return m.group(0).strip()

    # Split on first delimiter found and return what's before
    main_artist = delims.split(s, maxsplit=1)[0].strip()

    return main_artist


def compute_main_artist_from_song_id(song_id: str) -> str:
    return compute_main_artist_from_artists(extract_artists(song_id))

# %%

# Let's apply the code above to parse the main artists from each release, and calculate unique performers etc.
emerging_songs_df["main_artist"] = emerging_songs_df["song_id"].apply(compute_main_artist_from_song_id)

# Here are some instances of artists with two names that we need to normalize
emerging_songs_df.replace({
    "JIN": "Jin",
    "4*TOWN (From Disney": "4*TOWN (From Disney And Pixar's Turning Red)",
    "mgk": "Machine Gun Kelly",
    "¥$: Kanye West": "Kanye West",
    "$uicideBoy$": "$uicideboy$",
    "Charli XCX": "Charli xcx",
    "Yahritza y Su Esencia": "Yahritza Y Su Esencia",
    "Tyler, the Creator": "Tyler, The Creator",
    "twenty one pilots": "Twenty One Pilots",
    "jessie murph": "Jessie Murph",
    "BLEU": "Yung Bleu",
    "HUNTRX": "HUNTR/X",
    "HUNTR": "HUNTR/X",
    "Twice": "TWICE",
    "Pharrell": "Pharrell Williams",
    "BossMan DLow": "BossMan Dlow",
    "Richy Mitch": "Richy Mitch And The Coal Miners",
    "Jennie": "JENNIE",
    "Mariah The Scientist": "Mariah the Scientist"
}, inplace=True)

collab_rows = emerging_songs_df[
    emerging_songs_df["main_artist"] != emerging_songs_df["performers"]
]

solo_rows = emerging_songs_df[
    emerging_songs_df["main_artist"] == emerging_songs_df["performers"]
]

num_unique_collab_songs = collab_rows["song_id"].nunique()
num_unique_solo_songs = solo_rows["song_id"].nunique()
num_unique_solo_artists = solo_rows["main_artist"].nunique()

# %%

print(f"Unique songs (solo releases): {num_unique_solo_songs}")
print(f"Unique songs (collab releases): {num_unique_collab_songs}")
print(f"Unique solo artists: {num_unique_solo_artists}")
print(f"Unique performers: {emerging_songs_df['performers'].nunique()}")
print(f"Unique main artists: {emerging_songs_df['main_artist'].nunique()}")

#%%

"""
Now we have 485 unique main_artists from all songs and 367 unique artists from solo songs.

And, I have collected the following: 

- Metadata: for the 2272 emerging songs (Apple Music API)
- Social media data: for the 485 artists (SocialBlade API)

But, we will most likely only study songs from the 367 artists with solo songs. 

------------------------------------------------------------------------------------------------

Let's plot the distribution of the wks_on_chart field for all songs, solo songs, and collab songs.
"""

#%%

import seaborn as sns
import matplotlib.pyplot as plt
import ast

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
plt.title("Hot 100 Song lifecyles (aligned at debut week)")
# plt.axhline(y=50, color="black", linestyle="--", label="50 weeks")
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

plt.figure(figsize=(12,6))
top_genres.plot(kind="bar")
top_genres_no_collabs.plot(kind="bar", color="orange")
top_genres_only_collabs.plot(kind="bar", color="green")
plt.title("Distribution of Genre Tags for Billboard Hot 100 Songs (2022-2025)")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Songs", fontsize=12)
plt.legend(["All Songs", "One Artist", "Multiple Artists"])
plt.show()

#%%

"""
Make sense, see here:
https://www.economist.com/business/2018/02/03/in-popular-music-collaborations-rock

Hip-Hop, R&B, and adjacent see a dip in the Solo Artist graph, 
as these are genres known for collaborations.
"""

#%%

# Let's see how duration of songs was affected by removing collaborations.

#%%

plt.figure(figsize=(12,6))
plt.hist(metadata["durationInMillis"] / 60000, bins=30, alpha=0.6, label="All Songs")
plt.hist(metadata_no_collabs["durationInMillis"] / 60000, bins=30, alpha=0.6, color="orange", label="One Artist")
plt.hist(metadata_only_collabs["durationInMillis"] / 60000, bins=30, alpha=0.6, color="green", label="Multiple Artists")
plt.title("Distribution of Song Durations on Billboard Hot 100 (2022–2025)")
plt.xlabel("Song Duration (minutes)")
plt.ylabel("Count")
plt.legend(["All Songs", "One Artist", "Multiple Artists"])
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
import matplotlib.pyplot as plt
import seaborn as sns

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
# %%
