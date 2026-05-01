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

The majority of research frames it as a binary classification problem, ie. whether a song is a hit or non-hit. 

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
    
    for file in os.listdir(f"data/raw_data_collection_08_23_25/hot_100/{year}"):

        total_hot_100 = pd.concat([total_hot_100, pd.read_csv(f"data/raw_data_collection_08_23_25/hot_100/{year}/{file}")])
        file_count += 1
        
    print(f"Added {file_count} files to hot_100 from {year}")

# total_hot_100.to_csv("data/processed_data/total_hot_100.csv", index=False)

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

#%%

emerging_songs = total_hot_100[total_hot_100["song_id"].isin(emerging_song_ids)].copy()
print(f"Unique emerging songs: {len(emerging_songs['song_id'].unique())}")

#%%

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

#%%

# Let's apply the code above to parse the main artists from each release, and calculate unique performers etc.
emerging_songs["main_artist"] = emerging_songs["song_id"].apply(compute_main_artist_from_song_id)

# Here are some instances of artists with two names that we need to normalize
emerging_songs.replace({
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

collab_rows = emerging_songs[
    emerging_songs["main_artist"] != emerging_songs["performers"]
]

solo_rows = emerging_songs[
    emerging_songs["main_artist"] == emerging_songs["performers"]
]

num_unique_collab_songs = collab_rows["song_id"].nunique()
num_unique_solo_songs = solo_rows["song_id"].nunique()
num_unique_solo_artists = solo_rows["main_artist"].nunique()

# emerging_songs_df.to_csv("data/processed_data/emerging_songs.csv", index=False)

#%%

print(f"Unique songs (solo releases): {num_unique_solo_songs}")
print(f"Unique songs (collab releases): {num_unique_collab_songs}")
print(f"Unique solo artists: {num_unique_solo_artists}")
print(f"Unique performers: {emerging_songs['performers'].nunique()}")
print(f"Unique main artists: {emerging_songs['main_artist'].nunique()}")

#%%

"""
There are a few potential considerations before data splitting and feature engineering:

Some of our unique artists have several songs - should we randomly select one per artist?

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

Here are some fields to add to a new df prior to spltting:
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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast
from utils import expand_to_full_window, plot_two_metrics_song, plot_two_metrics_artist

#%%

# emerging_songs = pd.read_csv("data/processed_data/emerging_songs.csv")

solo_songs = emerging_songs[emerging_songs["main_artist"] == emerging_songs["performers"]].copy()
solo_songs = solo_songs.drop_duplicates(subset="song_id")

metadata = pd.read_csv("data/processed_data/metadata.csv")
metadata["genreNames"] = metadata["genreNames"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
metadata.drop(columns=["Unnamed: 0"], inplace=True)

#%%

# sampled_solo_songs = (solo_songs
#            .groupby("main_artist", group_keys=False)
#            .apply(lambda g: g.sample(1, random_state=42))
#            .reset_index(drop=True))

song_peaks = (emerging_songs
           .groupby("song_id", as_index=False)["current_week"]
           .min()
           .rename(columns={"current_week": "peak_pos"}))

lifespans = (emerging_songs
           .groupby("song_id", as_index=False)["wks_on_chart"]
           .max()
           .rename(columns={"wks_on_chart": "lifespan"}))

songs = pd.DataFrame({
    "song_id": solo_songs["song_id"],
    "title": solo_songs["title"],
    "artist": solo_songs["main_artist"],
    "entry_week_date": solo_songs["chart_week"],
    "entry_week_pos": solo_songs["current_week"],
})

# songs = pd.DataFrame({
#     "song_id": sampled_solo_songs["song_id"],
#     "title": sampled_solo_songs["title"],
#     "artist": sampled_solo_songs["main_artist"],
#     "entry_week_date": sampled_solo_songs["chart_week"],
#     "entry_week_pos": sampled_solo_songs["current_week"],
# })

songs = songs.merge(
    metadata[["song_id", "releaseDate", "genreNames", "durationInMillis"]]
      .rename(columns={"releaseDate": "release_date", "genreNames": "genres", "durationInMillis": "song_length"}),
    on="song_id",
    how="left"
)

# remove the string "Music" genres column
songs["genres"] = songs["genres"].apply(
    lambda g: [x for x in g if x != "Music"]
)

songs["song_length"] = songs["song_length"] / 60000

songs = songs.merge(song_peaks, on="song_id", how="left")
songs = songs.merge(lifespans, on="song_id", how="left")

#%%

# bins = [0, 2, 8, 16, 32, 64, songs['lifespan'].max()]
# labels = ["1–2", "3–8", "9–16", "17–32", "33–64", "65+"]
# songs['lifespan_bin'] = pd.cut(songs['lifespan'], bins=bins, labels=labels, include_lowest=True, right=True)

#%%

cols = ["song_id", "title", "artist", "genres",
        "song_length", "release_date", "entry_week_date", 
        "entry_week_pos", "peak_pos", "lifespan"] # add lifespan_bin if following chon

songs = songs[cols]
songs

#%%

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
genre_df = pd.DataFrame(
    mlb.fit_transform(songs['genres']),
    columns=mlb.classes_,
    index=songs.index
)

songs = pd.concat([songs.drop(columns='genres'), genre_df], axis=1)

#%%

"""
Ok, now, we do / compute the following before the train test split
1. Linearly interpolate youtube time-series when almost certainly missing
"""

#%%

instagram = pd.read_csv("data/raw_data_collection_08_23_25/social_archives/instagram_archive.csv")
instagram["date"] = pd.to_datetime(instagram["date"])
instagram.drop(columns=["Unnamed: 0"], inplace=True)

tiktok = pd.read_csv("data/raw_data_collection_08_23_25/social_archives/tiktok_archive.csv")
tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok.drop(columns=["Unnamed: 0"], inplace=True)

youtube = pd.read_csv("data/raw_data_collection_08_23_25/social_archives/youtube_archive.csv")
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
    songs[["artist", "song_id", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

youtube["date"] = pd.to_datetime(youtube["date"])
youtube["release_date"] = pd.to_datetime(youtube["release_date"])

mask = (
    (youtube["date"] >= youtube["release_date"] - pd.Timedelta(days=365)) &
    (youtube["date"] <= youtube["release_date"])
)

yt_pre12_df = (
    youtube.loc[mask, ["artist","song_id","platform","date","release_date","subs","views"]]
    .sort_values(["artist","song_id","date"])
    .reset_index(drop=True)
)

yt_pre12_clean = expand_to_full_window(yt_pre12_df, ("artist","song_id","platform"), "date", "release_date", 365)

#%%
"""
Let's plot ALL Artist youtube data prior to release.
"""
plot_two_metrics_song(
    yt_pre12_clean,
    metric1="subs",
    metric2="views",
    outdir="graphs/yt_12_month_raw",
    filename_pattern="yt_{song}_{platform}.png"
)

#%%

df = yt_pre12_clean.sort_values(["song_id", "platform", "date"]).copy()
grp = ["song_id", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
views_has_pos = df.groupby(grp)["views"].transform("max") > 0
subs_has_pos  = df.groupby(grp)["subs"].transform("max")  > 0

zero_views_missing = (df["views"] == 0) & views_has_pos
zero_subs_missing  = (df["subs"]  == 0) & subs_has_pos

df["views_zero_proxy_missing"] = zero_views_missing
df["subs_zero_proxy_missing"]  = zero_subs_missing

#%%

target = df[df["views_zero_proxy_missing"] == True]['artist'].unique()

#%%

"""
array(['sombr', 'Jessie Murph', 'Morgan Wallen', 'Justin Bieber',
       'Megan Moroney', 'Max McNown', 'Luke Combs', 'Lil Wayne',
       'Mariah the Scientist', '$uicideboy$', 'Sleep Token',
       'Tyler, The Creator', 'Travis Scott', 'Miley Cyrus', 'Alex Warren',
       'Don Toliver', 'Addison Rae', 'Kehlani', 'Dareyes de La Sierra',
       'KATSEYE', 'Eric Church', 'Chris Brown', 'Lady Gaga', 'ATEEZ',
       'BLACKPINK', 'Tate McRae', 'Gunna', 'Karol G', 'Paul Russell',
       'ILLIT', 'Sabrina Carpenter', 'Fuerza Regida', 'Ivan Cornejo',
       'Benson Boone', 'Chuckyy', 'Lil Tecca', 'JT', 'Ed Sheeran',
       'Cardi B', 'YG Marley', 'Zach Bryan', 'Latto', 'Lainey Wilson',
       'Chappell Roan', 'Gelo', 'GloRilla', 'Mariah Carey',
       'BabyChiefDoit', 'Drake', 'Lorde', 'Fridayy'], dtype=object)

It's important to note, SocialBlade API went down for a few days in April 2025. 

Those above with (Impute) are because of this, YG Marley's page was at 0 views 
4 weeks before release. 
"""

#%%

df[df["subs_zero_proxy_missing"] == True]['artist'].unique()
# None

#%%
"""
Ok, let's turn the 0.0 values to NaNs and then impute them with linear interpolation.
"""

#%%

df.loc[zero_views_missing, "views"] = np.nan

grp_interp = ["song_id", "platform"]
df["views"] = df.groupby(grp_interp)["views"].transform(lambda s: s.ffill())
df["subs"]  = df.groupby(grp_interp)["subs"].transform(lambda s: s.ffill())

yt_pre12_clean = df

#%%

plot_two_metrics_song(yt_pre12_clean, metric1="subs", metric2="views",
                artists=target,
                 outdir="graphs/yt_12_month_imputed",
                 filename_pattern="yt_{song}_{platform}.png")

#%%
"""

Interpolation looks good based on the graphs,
one thing to note though:
Drake
Morgan Wallen
Benson Boone

were all victim to what seems like a data quality issue on SocialBlade's part

Rather than the 0.0 filling for missed values, in 2023 march - may, 

there are consistent dips among the three of their views, basically carved out in th same spots

I anticipate these are things we should interpolate as this must be a scraping error on socialblades behalf
"""
#%%

df["song_id"].nunique()

#%%
"""
Ok, Let's check for IG. 
"""
#%%

instagram = instagram.merge(
    songs[["artist", "song_id", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

instagram["date"] = pd.to_datetime(instagram["date"])
instagram["release_date"] = pd.to_datetime(instagram["release_date"])

mask = (
    (instagram["date"] >= instagram["release_date"] - pd.Timedelta(days=365)) &
    (instagram["date"] <= instagram["release_date"])
)

ig_pre12_df = (
    instagram.loc[mask, ["artist","song_id","platform","date","release_date","followers","following", "media"]].copy()
    .sort_values(["artist","song_id","date"])
    .reset_index(drop=True)
)

ig_pre12_clean = expand_to_full_window(ig_pre12_df, ("artist","song_id","platform"), "date", "release_date", 365)

#%%
"""
Let's plot ALL Artist Instagram data prior to release.
"""
plot_two_metrics_song(
    ig_pre12_clean,
    metric1="followers",
    metric2="media",
    outdir="graphs/ig_12_month_raw",
    filename_pattern="ig_{song}_{platform}.png"
)

#%%

df = ig_pre12_clean.sort_values(["song_id", "platform", "date"]).copy()
grp = ["song_id", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = df.groupby(grp)["followers"].transform("max") > 0
media_has_pos  = df.groupby(grp)["media"].transform("max")  > 0

zero_followers_missing = (df["followers"] == 0) & followers_has_pos
zero_media_missing  = (df["media"]  == 0) & media_has_pos

df["followers_zero_proxy_missing"] = zero_followers_missing
df["media_zero_proxy_missing"]  = zero_media_missing

#%%

df[df["followers_zero_proxy_missing"] == True]['artist'].unique()

#%%
"""
array(['Lil Durk', 'Polo G', 'Moneybagg Yo', 'j-hope'

There are instances after using expand to full window where we need to beware of NaNs
and impute those, rather than just 0s

Let's impute, but skip the protocol of making 0s NaNs
"""

#%%
df.loc[zero_followers_missing, "followers"] = np.nan

grp_interp = ["song_id", "platform"]
df["followers"] = df.groupby(grp_interp)["followers"].transform(lambda s: s.ffill())
df["media"] = df.groupby(grp_interp)["media"].transform(lambda s: s.ffill())

ig_pre12_clean = df

#%%

plot_two_metrics_song(
    ig_pre12_clean,
    metric1="followers",
    metric2="media",
    outdir="graphs/ig_12_month_imputed",
    filename_pattern="ig_{song}_{platform}.png"
)

#%%
"""
Same for Instagram. Let's check for TikTok.
"""
#%%

tiktok = tiktok.merge(
    songs[["artist", "song_id", "release_date"]],
    left_on="artist_id",
    right_on="artist",
    how="left"
)

tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok["release_date"] = pd.to_datetime(tiktok["release_date"])

mask = (
    (tiktok["date"] >= tiktok["release_date"] - pd.Timedelta(days=365)) &
    (tiktok["date"] <= tiktok["release_date"])
)

tt_pre12_df = (
    tiktok.loc[mask, ["artist","song_id","platform","date","release_date","followers","following","uploads","likes"]].copy()
    .sort_values(["artist","song_id","date"])
    .reset_index(drop=True)
)

tt_pre12_clean = expand_to_full_window(tt_pre12_df, ("artist","song_id","platform"), "date", "release_date", 365)

#%%

plot_two_metrics_song(
    tt_pre12_clean,
    metric1="followers",
    metric2="likes",
    outdir="graphs/tt_12_month_raw",
    filename_pattern="tt_{song}_{platform}.png"
)

#%%

df = tt_pre12_clean.sort_values(["song_id", "platform", "date"]).copy()
grp = ["song_id", "platform"]

# 1) Mark zeros that are almost surely "missing" (group has some positive values)
followers_has_pos = df.groupby(grp)["followers"].transform("max") > 0
uploads_has_pos  = df.groupby(grp)["uploads"].transform("max")  > 0
likes_has_pos  = df.groupby(grp)["likes"].transform("max")  > 0

zero_followers_missing = (df["followers"] == 0) & followers_has_pos
zero_media_missing  = (df["uploads"]  == 0) & uploads_has_pos
zero_likes_missing  = (df["likes"]  == 0) & likes_has_pos

df["followers_zero_proxy_missing"] = zero_followers_missing
df["uploads_zero_proxy_missing"]  = zero_media_missing
df["likes_zero_proxy_missing"]  = zero_likes_missing

#%%

df[df["followers_zero_proxy_missing"] == True]['artist'].unique()
"""
array(['Sabrina Carpenter', 'Lizzo', 'Tate McRae', 'Megan Moroney',
       'Lil Yachty', 'Lady Gaga', 'Doja Cat', 'Luke Combs', '21 Savage',
       'Olivia Rodrigo', 'Beyonce', 'Rod Wave', 'The Kid LAROI',
       'Bad Bunny', 'Gunna', 'Benson Boone', 'Lil Wayne', 'Latto',
       'Megan Thee Stallion', 'Billie Eilish', 'Dove Cameron',
       'Kane Brown', 'Lil Uzi Vert', 'Stray Kids', 'Jack Harlow',
       'Luke Bryan', 'LE SSERAFIM', 'Meghan Trainor', 'David Kushner',
       'Rauw Alejandro', 'Ice Spice', 'Doechii', 'Lil Baby',
       'Travis Scott', 'Hozier', 'Cardi B', 'Alex Warren', 'Don Toliver',
       'd4vd', 'Lewis Capaldi', 'NLE Choppa', 'Young Thug',
       'Russell Dickerson', 'Dua Lipa', 'ATEEZ', 'Lauren Spencer-Smith',
       'Teddy Swims', 'Selena Gomez', 'Muni Long', 'Leon Thomas', 'Russ',
       'Tinashe', 'Fuerza Regida', 'Labrinth', 'The Marias', 'Tyla',
       'Chris Brown', 'BTS', 'Laufey', 'Karol G', 'Tim McGraw', 'TWICE',
       'Kenny Chesney', 'Chloe', 'Giveon', 'Lil Durk', 'Jonas Brothers',
       'Tucker Wetmore', 'Noah Kahan', 'Forrest Frank'], dtype=object)
"""
#%%

df[df["likes_zero_proxy_missing"] == True]['artist'].unique()
"""
array(['Morgan Wallen', 'Lady Gaga', 'Doja Cat', 'Beyonce',
       'Chris Janson', 'Karol G', 'Bad Bunny', 'sombr', 'Don Toliver',
       'Jordan Davis', 'Kane Brown', 'Luke Bryan', 'Juice WRLD',
       'Sabrina Carpenter', 'Russell Dickerson', 'Megan Thee Stallion',
       'Cynthia Erivo', 'Lil Nas X', 'Stray Kids', 'Paul Russell',
       'Chase Matthew', 'Jack Harlow', 'Muni Long', 'TWICE', 'Russ',
       'P!nk', 'Megan Moroney', 'Blink-182', 'Jack Black', 'Corey Kent',
       'Sia', 'Jonas Brothers'], dtype=object)
"""

#%%
# Impute Tiktok Followers for these artists

df.loc[zero_followers_missing, "followers"] = np.nan
df.loc[zero_likes_missing, "likes"] = np.nan

grp_interp = ["song_id", "platform"]
df["followers"] = df.groupby(grp_interp)["followers"].transform(lambda s: s.ffill())
df["likes"]  = df.groupby(grp_interp)["likes"].transform(lambda s: s.ffill())
df["uploads"]  = df.groupby(grp_interp)["uploads"].transform(lambda s: s.ffill())

tt_pre12_clean = df

#%%

plot_two_metrics_song(
    tt_pre12_clean,
    metric1="followers",
    metric2="likes",
    outdir="graphs/tt_12_month_imputed",
    filename_pattern="tt_{song}_{platform}.png"
)

#%%
"""
Ok, we've linearly interpolated missing time-series data for artists!
"""
#%%

yt_pre12_clean.drop(columns=["views_zero_proxy_missing", "subs_zero_proxy_missing"], inplace=True)
tt_pre12_clean.drop(columns=["followers_zero_proxy_missing", "uploads_zero_proxy_missing", "likes_zero_proxy_missing"], inplace=True)
ig_pre12_clean.drop(columns=["followers_zero_proxy_missing", "media_zero_proxy_missing"], inplace=True)

#%%

ig_pre12_clean.to_csv("data/processed_data/12m_ig_pre_release.csv", index=False)
tt_pre12_clean.to_csv("data/processed_data/12m_tt_pre_release.csv", index=False)
yt_pre12_clean.to_csv("data/processed_data/12m_yt_pre_release.csv", index=False)
songs.to_csv("data/processed_data/songs.csv", index=False)

#%%

# 2-week post-release social data extract
# Captures the social engagement spike in the 14 days immediately after release.
# Songs released within 14 days of the scrape date will have partial windows — that's expected.

POST_RELEASE_DAYS = 14

ig_post2w_mask = (
    (instagram["date"] > instagram["release_date"]) &
    (instagram["date"] <= instagram["release_date"] + pd.Timedelta(days=POST_RELEASE_DAYS))
)
ig_post2w = (
    instagram.loc[ig_post2w_mask, ["artist", "song_id", "platform", "date", "release_date", "followers", "following", "media"]]
    .sort_values(["artist", "song_id", "date"])
    .reset_index(drop=True)
)

tt_post2w_mask = (
    (tiktok["date"] > tiktok["release_date"]) &
    (tiktok["date"] <= tiktok["release_date"] + pd.Timedelta(days=POST_RELEASE_DAYS))
)
tt_post2w = (
    tiktok.loc[tt_post2w_mask, ["artist", "song_id", "platform", "date", "release_date", "followers", "following", "uploads", "likes"]]
    .sort_values(["artist", "song_id", "date"])
    .reset_index(drop=True)
)

yt_post2w_mask = (
    (youtube["date"] > youtube["release_date"]) &
    (youtube["date"] <= youtube["release_date"] + pd.Timedelta(days=POST_RELEASE_DAYS))
)
yt_post2w = (
    youtube.loc[yt_post2w_mask, ["artist", "song_id", "platform", "date", "release_date", "subs", "views"]]
    .sort_values(["artist", "song_id", "date"])
    .reset_index(drop=True)
)

ig_post2w.to_csv("data/processed_data/2w_ig_post_release.csv", index=False)
tt_post2w.to_csv("data/processed_data/2w_tt_post_release.csv", index=False)
yt_post2w.to_csv("data/processed_data/2w_yt_post_release.csv", index=False)

#%%

import pandas as pd
import matplotlib.pyplot as plt

ig_pre12_clean = pd.read_csv('data/processed_data/12m_ig_pre_release.csv')
tt_pre12_clean = pd.read_csv('data/processed_data/12m_tt_pre_release.csv')
yt_pre12_clean = pd.read_csv('data/processed_data/12m_yt_pre_release.csv')

ig_post2w = pd.read_csv("data/processed_data/2w_ig_post_release.csv")
tt_post2w = pd.read_csv("data/processed_data/2w_tt_post_release.csv")
yt_post2w = pd.read_csv("data/processed_data/2w_yt_post_release.csv")

#%%

'''
TIKTOK
---
From 1 -> 10,000
TT changes by 1

From 10,000 -> 1,000,000
TT changes by 100

From 1,000,000 -> 10,000,000
TT changes by 10K or (25K sometimes)

From 10,000,000 -> 100,000,000
YT changes by 100K

YOUTUBE SUBS

From 1 -> 1,000
YT changes by 1

From 1,000 -> 10,000
YT changes by 10

From 10,000 -> 100,000
YT changes by 100

From 100,000 -> 1,000,000
YT changes by 1K

From 1,000,000 -> 10,000,000
YT changes by 10K

From 10,000,000 -> 100,000,000
YT changes by 100K


NOTICE:

YT Views tends to have the absurd drops

These values actually aren't rounded, but there is clearly issue with how often
the view values are being scraped, and because of this, there are still places that must be imputed
'''

#%%

# ROUNDING_RULES = {
#     "yt_subs": [
#         (1_000,        1),
#         (10_000,       10),
#         (100_000,      100),
#         (1_000_000,    1_000),
#         (10_000_000,   10_000),
#         (100_000_000,  100_000),
#         (1_000_000_000, None),
#         (10_000_000_000, None),
#         (100_000_000_000, None),
#     ],
#     "yt_views": [
#         (1_000,        1),
#         (10_000,       1),
#         (100_000,      1),
#         (1_000_000,    1),
#         (10_000_000,   1),
#         (100_000_000,  1),
#         (1_000_000_000, None),
#         (10_000_000_000, None),
#         (100_000_000_000, None),
#     ],
#     "tt_followers": [
#         (1_000,        1),
#         (10_000,       1),
#         (100_000,      100),
#         (1_000_000,    100),
#         (10_000_000,   100_000),
#         (100_000_000,  100_000),
#         (1_000_000_000, None),
#         (10_000_000_000, None),
#         (100_000_000_000, None),
#     ],
#     "tt_likes": [
#         (1_000,        1),
#         (10_000,       1),
#         (100_000,      100),
#         (1_000_000,    100),
#         (10_000_000,   100_000),
#         (100_000_000,  100_000),
#         (1_000_000_000, 100_000),
#         (10_000_000_000, 100_000_000),
#         (100_000_000_000, None),
#     ],
#     "ig_followers": [
#         (1_000,        1),
#         (10_000,       1),
#         (100_000,      1),
#         (1_000_000,    1),
#         (10_000_000,   1),
#         (100_000_000,  1),
#         (1_000_000_000, 1),
#         (10_000_000_000, None),
#         (100_000_000_000, None),
#     ],
# }

# def display_value(x, field):

#     rules = ROUNDING_RULES[field]
#     for upper, unit in rules:
#         if x < upper:
#             if unit is None:
#                 return x
#             return (x // unit) * unit
#     return x

#%%

def keep_level_crossings(series, field):

    d = np.array(series)
    n = len(d)

    keep = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        j = i

        # iterate until we hit a point which is different from i
        while j + 1 < n and d[j+1] == d[i]:
            j += 1

        prev = d[i-1] if i > 0 else None
        curr = d[i]
        next_ = d[j+1] if j+1 < n else None

        # Case 1: monotonically increasing (AKA if previous level is less and next level is more)
        if prev is not None and prev < curr and (next_ is None or next_ > curr):
            keep[i] = True   # left

        # Case 2: monotonically decreasing (AKA if previous level is more and next level is less)
        if prev is not None and prev > curr and (next_ is None or next_ < curr):
            keep[j] = True   # right

        # Case 3: up, flat, down 
        if prev is not None and prev < curr and next_ is not None and next_ < curr:
            keep[i] = True   # left
            keep[j] = True   # right

        # Case 4: down, flat, up
        if prev is not None and prev > curr and next_ is not None and next_ > curr:
            keep[i] = True   # left
            keep[j] = True   # right
            
            # if we want middle value instead
            # keep[(i + j) / 2] = True

            keep[i + 1] = True   # left
            keep[j - 1] = True   # right

        # If no prev (start)
        if prev is None:
            keep[i] = True

        # If no next (end)
        if next_ is None:
            keep[j] = True

        i = j + 1

    return pd.Series(keep, index=series.index)

def apply_keep_flag(group, col, model_key, out_col):
    g = group.sort_values("date").copy()
    g[out_col] = keep_level_crossings(g[col], model_key)
    return g

#%%

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#%%

import numpy as np

yt_pre12_clean = (
    yt_pre12_clean
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "subs", "yt_subs", "subs_keep"))
)

yt_pre12_clean = (
    yt_pre12_clean
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "views", "yt_views", "views_keep"))
)

tt_pre12_clean = (
    tt_pre12_clean
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "followers", "tt_followers", "followers_keep"))
)

tt_pre12_clean = (
    tt_pre12_clean
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "likes", "tt_likes", "likes_keep"))
)

#%%

tt_post2w = (
    tt_post2w
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "followers", "tt_followers", "followers_keep"))
)

tt_post2w = (
    tt_post2w
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "likes", "tt_likes", "likes_keep"))
)

yt_post2w = (
    yt_post2w
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "subs", "yt_subs", "subs_keep"))
)

yt_post2w = (
    yt_post2w
    .groupby("song_id", group_keys=False)
    .apply(lambda g: apply_keep_flag(g, "views", "yt_views", "views_keep"))
)

#%%

# YOUTUBE VIEWS

'''
Biggest deviations:

Miami — Morgan Wallen
I'm A Little Crazy — Morgan Wallen
Falling Apart — Morgan Wallen
'''

song = yt_pre12_clean.song_id.sample().iloc[0]

d = yt_pre12_clean[yt_pre12_clean["song_id"] == song].sort_values("date")
d["date"] = pd.to_datetime(d["date"])

import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(d["date"], d["subs"], linewidth=1)
axes[0].scatter(d[d["subs_keep"]]["date"], d[d["subs_keep"]]["subs"], linewidth=2, alpha=0.2)
axes[0].set_title(f"Youtube Subscribers Raw ({song})")

axes[1].plot(d["date"], d["subs_interp"], linewidth=1)
axes[1].scatter(d[d["subs_keep"]]["date"], d[d["subs_keep"]]["subs"], linewidth=2, alpha=0.2)
axes[1].set_title(f"Youtube Subscribers Rounded ({song})")

axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

#%%

# TIKTOK FOLLOWERS

song = tt_pre12_clean.song_id.sample().iloc[0]

d = tt_pre12_clean[tt_pre12_clean["song_id"] == song].sort_values("date")
d["date"] = pd.to_datetime(d["date"])

import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(d["date"], d["followers"], linewidth=1)
axes[0].scatter(d[d["followers_keep"]]["date"], d[d["followers_keep"]]["followers"], linewidth=2, alpha=0.2)
axes[0].set_title(f"TikTok Followers Raw ({song})")

axes[1].plot(d["date"], d["followers_interp"], linewidth=1)
axes[1].scatter(d[d["followers_keep"]]["date"], d[d["followers_keep"]]["followers"], linewidth=2, alpha=0.2)
axes[1].set_title(f"TikTok Followers Rounded ({song})")

axes[1].xaxis.set_major_locator(mdates.MonthLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

#%%

# TIKTOK FOLLOWERS

'''
Biggest deviations:

Mr Electric Blue — Benson Boone
'''

song = tt_pre12_clean.song_id.sample().iloc[0]

d = tt_pre12_clean[tt_pre12_clean["song_id"] == song].sort_values("date")
d["date"] = pd.to_datetime(d["date"])

import matplotlib.dates as mdates

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(d["date"], d["followers"], linewidth=1)
axes[0].scatter(d[d["followers_keep"]]["date"], d[d["followers_keep"]]["followers"], linewidth=2, alpha=0.2)
axes[0].set_title(f"TikTok Followers ({song})")

axes[1].plot(d["date"], d["followers_interp"], linewidth=1)
axes[1].scatter(d[d["followers_keep"]]["date"], d[d["followers_keep"]]["followers_interp"], linewidth=2, alpha=0.2)
axes[1].set_title(f"TikTok Followers ({song})")

axes[2].plot(d["date"], d["uploads"], linewidth=1, color="C2")
axes[2].set_title(f"TikTok uploads ({song})")
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

#%%

tt_pre12_clean.columns

#%%

import numpy as np

def interpolate_metric(df, value_col, keep_col):
    df = df.sort_values(["song_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = df["date"].astype("int64")
    out_col = value_col + "_interp"
    df[out_col] = np.nan

    for sid, sub in df.groupby("song_id"):
        idx = sub.index
        t = sub["t"].values
        y = sub[value_col].astype(float).values
        keep = sub[keep_col].values.copy()

        keep[0] = True

        valid_mask = np.isfinite(y) & (y != 0)
        if valid_mask.any():
            first_nonzero = np.where(valid_mask)[0][0]
            keep[first_nonzero] = True

        # identify first non-NaN raw value
        first_valid_idx = None
        finite_mask = np.isfinite(y)
        if finite_mask.any():
            first_valid_idx = np.where(finite_mask)[0][0]

        # compute keep mask
        mask = keep & finite_mask

        t_keep = t[mask]
        y_keep = y[mask]

        uniq, uniq_idx = np.unique(t_keep, return_index=True)
        t_keep = t_keep[uniq_idx]
        y_keep = y_keep[uniq_idx]

        # linear interpolation
        base = np.interp(t, t_keep, y_keep)

        # no backfill before first keep
        base[0] = y_keep[0]
        base[t < t_keep[0]] = y_keep[0]

        # no vals before first real data point
        base[:first_valid_idx] = np.nan

        # forward-fill after last keep point
        base[t > t_keep[-1]] = y_keep[-1]

        df.loc[idx, out_col] = base

    return df


tt_pre12_clean = interpolate_metric(tt_pre12_clean, "followers", "followers_keep")
tt_pre12_clean = interpolate_metric(tt_pre12_clean, "likes", "likes_keep")

yt_pre12_clean = interpolate_metric(yt_pre12_clean, "subs", "subs_keep")
yt_pre12_clean = interpolate_metric(yt_pre12_clean, "views", "views_keep")

tt_post2w = interpolate_metric(tt_post2w, "followers", "followers_keep")
tt_post2w = interpolate_metric(tt_post2w, "likes", "likes_keep")

yt_post2w = interpolate_metric(yt_post2w, "subs", "subs_keep")
yt_post2w = interpolate_metric(yt_post2w, "views", "views_keep")

#%%
"""
NEW SECTION
"""

ig_pre12_clean['ig_followers_diff_daily'] = (
    ig_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(1))
)

ig_pre12_clean['ig_media_diff_daily'] = (
    ig_pre12_clean
    .groupby('song_id')['media']
    .transform(lambda s: s - s.shift(1))
)

tt_pre12_clean['tt_followers_diff_daily'] = (
    tt_pre12_clean
    .groupby('song_id')['followers_interp']
    .transform(lambda s: s - s.shift(1))
)

tt_pre12_clean['tt_likes_diff_daily'] = (
    tt_pre12_clean
    .groupby('song_id')['likes_interp']
    .transform(lambda s: s - s.shift(1))
)

tt_pre12_clean['tt_uploads_diff_daily'] = (
    tt_pre12_clean
    .groupby('song_id')['uploads']
    .transform(lambda s: s - s.shift(1))
)

yt_pre12_clean['yt_subs_diff_daily'] = (
    yt_pre12_clean
    .groupby('song_id')['subs_interp']
    .transform(lambda s: s - s.shift(1))
)

yt_pre12_clean['yt_views_diff_daily'] = (
    yt_pre12_clean
    .groupby('song_id')['views_interp']
    .transform(lambda s: s - s.shift(1))
)

#%%
"""
IG FOLLOWERS
"""

import numpy as np
import matplotlib.pyplot as plt

df = ig_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align all artists
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Select top 20 deviations
top10 = (
    df[['song_id','days_since_start','ig_followers_diff_daily']]
    .sort_values('ig_followers_diff_daily', ascending=True)
    .head(10)
)

plt.figure(figsize=(12, 6))

# Plot all lines
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('days_since_start')
    plt.plot(
        group['days_since_start'],
        group['ig_followers_diff_daily'],
        alpha=0.5
    )

# Hardcoded offsets (you can replace these with a circular set)
offsets = [
    (10, -14),(-120,  14),(-14, 10)
]

# Enforce uniqueness constraints
seen_points = set()
seen_songs = set()
label_i = 0   # offset index

for _, row in top10.iterrows():
    song = row['song_id']
    x = row['days_since_start']
    y = row['ig_followers_diff_daily']

    # Skip if already annotated this song_id
    if song in seen_songs:
        continue

    # Skip if this coordinate was already annotated
    if (x, y) in seen_points:
        continue

    # Record uniqueness
    seen_songs.add(song)
    seen_points.add((x, y))

    # Use next offset only when actually annotating
    dx, dy = offsets[label_i]
    label_i += 1

    # Draw annotation
    plt.annotate(
        song,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=9,
        weight='bold',
        arrowprops=dict(arrowstyle='-', lw=0.4)
    )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily Instagram Followers Change")
plt.xlabel("Days since first observation")
plt.ylabel("Daily Follower Change")
plt.tight_layout()
plt.show()


#%%
"""
TT FOLLOWERS
"""


import numpy as np
import matplotlib.pyplot as plt

df = tt_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align all artists by their own start date
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Select top N unique points
N = 8
topN = (
    df[['song_id','days_since_start','tt_followers_diff_daily']]
    .sort_values('tt_followers_diff_daily', ascending=True)
    .head(N)
)

# Hardcoded circular-ish offsets for up to 20 labels
offsets = [
    (14, 0), (11, 8), (4, 13), (-90, 50), (-95, 11),
    (-45, 11), (-11, -8), (-4, -13), (4, -13), (-9, 8),
    (18, 0), (14, 12), (0, 18), (-14, 12), (-18, 0),
    (-14, -12), (0, -18), (14, -12), (20, 0), (-20, 0)
][:N]

plt.figure(figsize=(12, 6))

# Plot all artists
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('days_since_start')
    plt.plot(
        group['days_since_start'],
        group['tt_followers_diff_daily'],
        alpha=0.4
    )

seen_points = set()
seen_songs = set()
label_i = 0   # only increments when a label is used

for _, row in topN.iterrows():
    song = row['song_id']
    x = row['days_since_start']
    y = row['tt_followers_diff_daily']

    # skip if we already used this song_id
    if song in seen_songs:
        continue

    # skip if we already used this exact (x,y)
    if (x, y) in seen_points:
        continue

    # record uniqueness
    seen_songs.add(song)
    seen_points.add((x, y))

    # get next available offset
    dx, dy = offsets[label_i]
    label_i += 1

    plt.annotate(
        song,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=9,
        weight='bold',
        arrowprops=dict(arrowstyle='-', lw=0.4)
    )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily TikTok Followers Change")
plt.xlabel("Days since first observation")
plt.ylabel("Daily Follower Change")
plt.tight_layout()
plt.show()

#%%
"""
TT LIKES
"""

import numpy as np
import matplotlib.pyplot as plt

df = tt_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align all artists by their own start date
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Select top N unique points
N = 5
topN = (
    df[['song_id','days_since_start','tt_likes_diff_daily']]
    .sort_values('tt_likes_diff_daily', ascending=True)
    .head(N)
)

# Hardcoded circular-ish offsets for labels
offsets = [
    (14, 0), (-130, 40), (-150, 13), (4, 13), (-95, 11),
    (-45, 11), (-11, -8), (-4, -13), (4, -13), (-9, 8),
    (18, 0), (14, 12), (0, 18), (-14, 12), (-18, 0),
    (-14, -12), (0, -18), (14, -12), (20, 0), (-20, 0)
][:N]

plt.figure(figsize=(12, 6))

# Plot all artists
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('days_since_start')
    plt.plot(
        group['days_since_start'],
        group['tt_likes_diff_daily'],
        alpha=0.4
    )

seen_points = set()
seen_songs = set()
label_i = 0

# Annotate top-N unique highest increases
for _, row in topN.iterrows():
    song = row['song_id']
    x = row['days_since_start']
    y = row['tt_likes_diff_daily']

    if song in seen_songs:
        continue

    if (x, y) in seen_points:
        continue

    seen_songs.add(song)
    seen_points.add((x, y))

    dx, dy = offsets[label_i]
    label_i += 1

    plt.annotate(
        song,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=9,
        weight='bold',
        arrowprops=dict(arrowstyle='-', lw=0.4)
    )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily TikTok Likes Change")
plt.xlabel("Days since first observation")
plt.ylabel("Daily Likes Change")
plt.tight_layout()
plt.show()

# %%
"""
YT VIEWS
"""

import numpy as np
import matplotlib.pyplot as plt

df = yt_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align each song by its own starting date
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Select top N largest negative dips
N = 15
topN = (
    df[['song_id','days_since_start','yt_views_diff_daily']]
    .sort_values('yt_views_diff_daily', ascending=True)
    .head(N)
)

# Hardcoded offsets for labeling
offsets = [
    (14, 0), (-130, 40), (-150, 13), (4, 13), (-95, 11),
    (-45, 11), (-11, -8), (-4, -13), (4, -13), (-9, 8),
    (18, 0), (14, 12), (0, 18), (-14, 12), (-18, 0),
    (-14, -12), (0, -18), (14, -12), (20, 0), (-20, 0)
][:N]

plt.figure(figsize=(12, 6))

# Plot all artists
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('days_since_start')
    plt.plot(
        group['days_since_start'],
        group['yt_views_diff_daily'],
        alpha=0.4
    )

seen_points = set()
seen_songs = set()
label_i = 0

# Annotate bottom-N dips
for _, row in topN.iterrows():
    song = row['song_id']
    x = row['days_since_start']
    y = row['yt_views_diff_daily']

    if song in seen_songs:
        continue

    if (x, y) in seen_points:
        continue

    seen_songs.add(song)
    seen_points.add((x, y))

    dx, dy = offsets[label_i]
    label_i += 1

    plt.annotate(
        song,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=9,
        weight='bold',
        arrowprops=dict(arrowstyle='-', lw=0.4)
    )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily YouTube Views Change")
plt.xlabel("Days since first observation")
plt.ylabel("Daily Views Change")
plt.tight_layout()
plt.show()

#%%
"""
YT SUBS
"""

import numpy as np
import matplotlib.pyplot as plt

df = yt_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align each song by its own starting date
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Select top N largest negative dips
N = 20
topN = (
    df[['song_id','days_since_start','yt_subs_diff_daily']]
    .sort_values('yt_subs_diff_daily', ascending=True)
    .head(N)
)

# Hardcoded offsets for labeling
offsets = [
    (14, 0), (-130, 40), (-150, 13), (4, 13), (-95, 11),
    (-45, 11), (-11, -8), (-4, -13), (4, -13), (-9, 8),
    (18, 0), (14, 12), (0, 18), (-14, 12), (-18, 0),
    (-14, -12), (0, -18), (14, -12), (20, 0), (-20, 0)
][:N]

plt.figure(figsize=(12, 6))

# Plot all artists
for song_id, group in df.groupby('song_id'):
    group = group.sort_values('days_since_start')
    plt.plot(
        group['days_since_start'],
        group['yt_subs_diff_daily'],
        alpha=0.4
    )

seen_points = set()
seen_songs = set()
label_i = 0

# Annotate bottom-N dips
for _, row in topN.iterrows():
    song = row['song_id']
    x = row['days_since_start']
    y = row['yt_subs_diff_daily']

    if song in seen_songs:
        continue

    if (x, y) in seen_points:
        continue

    seen_songs.add(song)
    seen_points.add((x, y))

    dx, dy = offsets[label_i]
    label_i += 1

    plt.annotate(
        song,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        fontsize=9,
        weight='bold',
        arrowprops=dict(arrowstyle='-', lw=0.4)
    )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily YouTube Subscribers Change")
plt.xlabel("Days since first observation")
plt.ylabel("Daily Subscribers Change")
plt.tight_layout()
plt.show()

#%%
"""
LET"S SEE IF WE CAN PLOT THE DISTRIBUTIONS OF MAX DEVIATIONS WITHIN EACH SONG PLOT
"""

mask = (yt_pre12_clean["subs_interp"] > 100_000) & (yt_pre12_clean["subs_interp"] < 1_000_000)

yt_max_dips = (
    yt_pre12_clean.loc[mask]
    .groupby("song_id")["yt_views_diff_daily"]
    .min()
    .to_frame()
)

x = yt_max_dips["yt_views_diff_daily"].dropna()

stats = {
    "Mean": x.mean(),
    "Median": x.median(),
    "Std": x.std(),
    "5th %ile": np.percentile(x, 5),
    "95th %ile": np.percentile(x, 95),
}

stats_text = "\n".join(
    f"{k}: {v:.0f}" for k, v in stats.items()
)

plt.figure(figsize=(12,6))
sns.histplot(yt_max_dips["yt_views_diff_daily"], bins=70, edgecolor="black")

plt.axvline(stats["Mean"], linestyle="--", linewidth=2, label=f"Mean = {stats['Mean']:.0f}")
plt.axvline(stats["Median"], linestyle="-", linewidth=2, label=f"Median = {stats['Median']:.0f}")

plt.text(
    0.02, 0.98,
    stats_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.9)
)

plt.title("Maximum Dips in YouTube Subs Daily Difference per Song")
plt.xlabel("Daily Difference")
plt.ylabel("Frequency")
plt.show()

#%%

# import numpy as np
# import pandas as pd

# BIN_TABLE = {
#     "youtube_subs": [
#         (0,            1_000,          "0–1K",        0.75),
#         (1_000,        10_000,         "1K–10K",      0.50),
#         (10_000,       100_000,        "10K–100K",    0.50),
#         (100_000,      1_000_000,      "100K–1M",     0.30),
#         (1_000_000,    10_000_000,     "1M–10M",      0.20),
#         (10_000_000,   100_000_000,    "10M–100M",    0.10),
#         (100_000_000,  1_000_000_000,  "100M–1B",     0.05),
#         (1_000_000_000,10_000_000_000, "1B–10B",      0.02),
#     ],
#     "youtube_views": [
#         (0,            1_000,          "0–1K",        0.75),
#         (1_000,        10_000,         "1K–10K",      0.75),
#         (10_000,       100_000,        "10K–100K",    0.75),
#         (100_000,      1_000_000,      "100K–1M",     0.50),
#         (1_000_000,    10_000_000,     "1M–10M",      0.30),
#         (10_000_000,   100_000_000,    "10M–100M",    0.20),
#         (100_000_000,  1_000_000_000,  "100M–1B",     0.10),
#         (1_000_000_000,10_000_000_000, "1B–10B",      0.05),
#     ],
#     "tiktok_followers": [
#         (0,            1_000,          "0–1K",        0.75),
#         (1_000,        10_000,         "1K–10K",      0.50),
#         (10_000,       100_000,        "10K–100K",    0.50),
#         (100_000,      1_000_000,      "100K–1M",     0.30),
#         (1_000_000,    10_000_000,     "1M–10M",      0.30),
#         (10_000_000,   100_000_000,    "10M–100M",    0.10),
#         (100_000_000,  1_000_000_000,  "100M–1B",     0.05),
#         (1_000_000_000,10_000_000_000, "1B–10B",      0.02),
#     ],
#     "tiktok_likes": [
#         (0,            1_000,          "0–1K",        0.75),
#         (1_000,        10_000,         "1K–10K",      0.75),
#         (10_000,       100_000,        "10K–100K",    0.75),
#         (100_000,      1_000_000,      "100K–1M",     0.50),
#         (1_000_000,    10_000_000,     "1M–10M",      0.30),
#         (10_000_000,   100_000_000,    "10M–100M",    0.20),
#         (100_000_000,  1_000_000_000,  "100M–1B",     0.10),
#         (1_000_000_000,10_000_000_000, "1B–10B",      0.05),
#     ],
#     "ig_followers": [
#         (0,            1_000,          "0–1K",        0.75),
#         (1_000,        10_000,         "1K–10K",      0.50),
#         (10_000,       100_000,        "10K–100K",    0.50),
#         (100_000,      1_000_000,      "100K–1M",     0.30),
#         (1_000_000,    10_000_000,     "1M–10M",      0.30),
#         (10_000_000,   100_000_000,    "10M–100M",    0.10),
#         (100_000_000,  1_000_000_000,  "100M–1B",     0.05),
#         (1_000_000_000,10_000_000_000, "1B–10B",      0.02),
#     ],
# }

# def assign_bin_and_threshold(value, metric_key):
#     """
#     Returns (interval_name, allowed_drop_fraction)
#     """
#     for low, high, label, frac in BIN_TABLE[metric_key]:
#         if low <= value < high:
#             return label, frac
#     return np.nan, np.nan

# def flag_drops_with_bins(df, metric_key, level_col, daily_diff_col):
#     df = df.copy()

#     # Assign bin, threshold for each row
#     df[["interval", "allowed_frac_drop"]] = df[level_col].apply(
#         lambda x: pd.Series(assign_bin_and_threshold(x, metric_key))
#     )

#     # Compute actual fractional drop
#     df["actual_frac_drop"] = df.apply(
#         lambda r: (-r[daily_diff_col] / max(r[level_col], 1)) if r[daily_diff_col] < 0 else 0,
#         axis=1
#     )

#     # Flag drops
#     df["is_excessive_drop"] = (
#         (df[daily_diff_col] < 0) & 
#         (df["actual_frac_drop"] > df["allowed_frac_drop"])
#     )

#     # Song_ids with violations
#     flagged_songs = df.loc[df["is_excessive_drop"], "song_id"].unique()

#     return df, flagged_songs

# #%%

# clean_yt, bad_yt = flag_drops_with_bins(
#     yt_pre12_clean,
#     metric_key="youtube_views",
#     level_col="views",
#     daily_diff_col="yt_views_diff_daily"
# )

# bad_yt

# #%%

# for i, song_id in enumerate(reversed(bad_yt)):
#     song_df = yt_pre12_clean[yt_pre12_clean['song_id'] == song_id].copy()
#     song_df = song_df.sort_values('date')

#     ig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#     # Likes
#     axes[0].plot(
#         song_df['date'],
#         song_df['views'],
#         linewidth=1.5
#     )
#     axes[0].set_title(f"View Count: {song_id}")
#     axes[0].set_ylabel("Count")
#     axes[0].grid(alpha=0.3)

#     # Upload
#     axes[1].plot(
#         song_df['date'],
#         song_df['subs'],
#         linewidth=1.5
#     )
#     axes[1].set_title(f"Sub Count: {song_id}")
#     axes[1].set_xlabel("Date")
#     axes[1].set_ylabel("Count")
#     axes[1].grid(alpha=0.3)

#     plt.tight_layout()
#     plt.show()

#     if i >= 20:
#         break

# #%%

# """
# ['Alex Warren',
#  'Zach Top',
#  'BigXthaPlug', (FAKE DATA)
#  'Cris Mj',
#  'd4vd',
#  'Ivan Cornejo',
#  'Ty Myers',
#  'Tommy Richman',
#  'Artemas',
#  'JT',
#  'Addison Rae', (REAL DATA)
#  'Lay Bankz',
#  'Tucker Wetmore',
#  'PARTYNEXTDOOR',
#  'Morgan Wallen', (FAKE DATA)
#  'Yeat', 
#  'Max McNown',
#  'Forrest Frank',
#  'ATEEZ',
#  'Myles Smith',
#  'Parker McCollum', (Parker McCollum)
#  'Gigi Perez',
#  'The Kid LAROI', (Don't Know)
#  'Mitski',
#  'Koe Wetzel',
#  'Bryson Tiller',
#  'The Marias',
#  'Muni Long',
#  'Warren Zeiders',
#  'Benson Boone', (Don't Know)
#  'Odetari']
#  """

# #%%

# clean_sub_yt, bad_sub_yt = flag_drops_with_bins(
#     yt_pre12_clean,
#     metric_key="youtube_subs",
#     level_col="subs",
#     daily_diff_col="yt_subs_diff_daily"
# )

# bad_sub_yt

# #%%

# clean_tt, bad_tt = flag_drops_with_bins(
#     tt_pre12_clean,
#     metric_key="tiktok_followers",
#     level_col="followers",
#     daily_diff_col="tt_followers_diff_daily"
# )

# bad_tt

# #%%

# for song_id in bad_tt:
#     song_df = tt_pre12_clean[tt_pre12_clean['song_id'] == song_id].copy()
#     song_df = song_df.sort_values('date')

#     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#     # Likes
#     axes[0].plot(
#         song_df['date'],
#         song_df['followers'],
#         linewidth=1.5
#     )
#     axes[0].set_title(f"All-Time TikTok Follower Count: {song_id}")
#     axes[0].set_ylabel("Follower Count")
#     axes[0].grid(alpha=0.3)

#     # Upload
#     axes[1].plot(
#         song_df['date'],
#         song_df['uploads'],
#         linewidth=1.5
#     )
#     axes[1].set_title(f"Upload Count: {song_id}")
#     axes[1].set_xlabel("Date")
#     axes[1].set_ylabel("Count")
#     axes[1].grid(alpha=0.3)


#     plt.tight_layout()
#     plt.show()

# #%%

# clean_tt, bad_tt = flag_drops_with_bins(
#     tt_pre12_clean,
#     metric_key="tiktok_likes",
#     level_col="likes",
#     daily_diff_col="tt_likes_diff_daily"
# )

# bad_tt

# # %%

# for song_id in bad_tt:
#     song_df = tt_pre12_clean[tt_pre12_clean['song_id'] == song_id].copy()
#     song_df = song_df.sort_values('date')

#     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#     # Likes
#     axes[0].plot(
#         song_df['date'],
#         song_df['likes'],
#         linewidth=1.5
#     )
#     axes[0].set_title(f"All-Time TikTok Like Count: {song_id}")
#     axes[0].set_ylabel("Like Count")
#     axes[0].grid(alpha=0.3)

#     # Upload
#     axes[1].plot(
#         song_df['date'],
#         song_df['uploads'],
#         linewidth=1.5
#     )
#     axes[1].set_title(f"Upload Count: {song_id}")
#     axes[1].set_xlabel("Date")
#     axes[1].set_ylabel("Count")
#     axes[1].grid(alpha=0.3)

#     plt.tight_layout()
#     plt.show()

# #%%

# clean_ig, bad_ig = flag_drops_with_bins(
#     ig_pre12_clean,
#     metric_key="ig_followers",
#     level_col="followers",
#     daily_diff_col="ig_followers_diff_daily"
# )

# bad_ig

# #%%

# for song_id in bad_ig:
#     song_df = ig_pre12_clean[ig_pre12_clean['song_id'] == song_id].copy()
#     song_df['date'] = pd.to_datetime(song_df['date'])
#     song_df = song_df.sort_values('date')

#     import matplotlib.dates as mdates

#     fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#     # Likes
#     axes[0].plot(
#         song_df['date'],
#         song_df['followers'],
#         linewidth=1.5
#     )
#     axes[0].set_title(f"All-Time Instagram Follower Count: {song_id}")
#     axes[0].set_ylabel("Follower Count")
#     axes[0].grid(alpha=0.3)

#     # Upload
#     axes[1].plot(
#         song_df['date'],
#         song_df['media'],
#         linewidth=1.5
#     )
#     axes[1].set_title(f"Upload Count: {song_id}")
#     axes[1].set_xlabel("Date")
#     axes[1].set_ylabel("Count")
#     axes[1].grid(alpha=0.3)

#     axes[1].xaxis.set_major_locator(mdates.MonthLocator())
#     axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
#     plt.tight_layout()
#     plt.show()

# #%%

# df = tt_pre12_clean[tt_pre12_clean['song_id'] == "Agora Hills — Doja Cat"]
# df['date'] = pd.to_datetime(df['date'])
# df[df['date'].dt.year == 2023].sort_values(by="date")

#%%

ig_pre12_clean.to_csv("data/processed_data/12m_ig_pre_release_imputed.csv", index=False)
tt_pre12_clean.to_csv("data/processed_data/12m_tt_pre_release_imputed.csv", index=False)
yt_pre12_clean.to_csv("data/processed_data/12m_yt_pre_release_imputed.csv", index=False)

ig_post2w.to_csv("data/processed_data/2w_ig_post_release_imputed.csv", index=False)
tt_post2w.to_csv("data/processed_data/2w_tt_post_release_imputed.csv", index=False)
yt_post2w.to_csv("data/processed_data/2w_yt_post_release_imputed.csv", index=False)

#%%
