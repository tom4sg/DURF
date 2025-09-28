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
from utils import expand_to_full_window, plot_two_metrics_song, plot_two_metrics_artist

#%%

emerging_songs = pd.read_csv("data/processed_data/emerging_songs.csv")
solo_songs = emerging_songs[emerging_songs["main_artist"] == emerging_songs["performers"]].copy()
solo_songs = solo_songs.drop_duplicates(subset="song_id")
solo_songs["song_id"].nunique()
solo_songs.groupby('main_artist')['song_id'].count().median()

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

feature_df = pd.DataFrame({
    "song_id": solo_songs["song_id"],
    "title": solo_songs["title"],
    "artist": solo_songs["main_artist"],
    "entry_week_date": solo_songs["chart_week"],
    "entry_week_pos": solo_songs["current_week"],
})

#%%

print(feature_df['artist'].nunique())
print(feature_df['song_id'].nunique())

#%%

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
    feature_df[["artist", "song_id", "release_date"]],
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
df["views"] = df.groupby(grp_interp)["views"].transform(lambda s: s.interpolate())
df["subs"]  = df.groupby(grp_interp)["subs"].transform(lambda s: s.interpolate())

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
    feature_df[["artist", "song_id", "release_date"]],
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
"""
array(['Lil Durk', 'Polo G', 'Moneybagg Yo', 'j-hope'
"""

#%%

"""
There are instances after using expand to full window where we need to beware of NaNs
and impute those, rather than just 0s

Let's impute, but skip the protocol of making 0s NaNs
"""
df.loc[zero_followers_missing, "followers"] = np.nan

grp_interp = ["song_id", "platform"]
df["followers"] = df.groupby(grp_interp)["followers"].transform(lambda s: s.interpolate())

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
    feature_df[["artist", "song_id", "release_date"]],
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
df["followers"] = df.groupby(grp_interp)["followers"].transform(lambda s: s.interpolate())
df["likes"]  = df.groupby(grp_interp)["likes"].transform(lambda s: s.interpolate())

tt_pre12_clean = df

#%%

plot_two_metrics_song(
    tt_pre12_clean,
    metric1="followers",
    metric2="likes",
    outdir="graphs/tt_12_month_imputed",
    filename_pattern="tt_{song}_{platform}.png"
)

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

release_rows = yt_pre12_clean[yt_pre12_clean["date"] == yt_pre12_clean["release_date"]][
    ["song_id", "release_date", "subs", "views"]
].rename(columns={"subs": "yt_subs_release_date", "views": "yt_views_release_date"})

feature_df["release_date"] = pd.to_datetime(feature_df["release_date"])
release_rows["release_date"] = pd.to_datetime(release_rows["release_date"])

feature_df = feature_df.merge(
    release_rows,
    on=["song_id", "release_date"],
    how="left"
)

#%%

yt_pre12_clean.drop(columns=["views_zero_proxy_missing", "subs_zero_proxy_missing"], inplace=True)

#%%

# TT: followers, uploads, likes
release_rows = tt_pre12_clean[tt_pre12_clean["date"] == tt_pre12_clean["release_date"]][
    ["song_id", "release_date", "followers", "uploads", "likes"]
].rename(columns={"followers": "tt_followers_release_date", "uploads": "tt_uploads_release_date", "likes": "tt_likes_release_date"})

feature_df = feature_df.merge(
    release_rows,
    on=["song_id", "release_date"],
    how="left"
)

#%%

tt_pre12_clean.drop(columns=["followers_zero_proxy_missing", "uploads_zero_proxy_missing", "likes_zero_proxy_missing"], inplace=True)

#%%

# IG: followers, media
release_rows = ig_pre12_clean[ig_pre12_clean["date"] == ig_pre12_clean["release_date"]][
    ["song_id", "release_date", "followers", "media"]
].rename(columns={"followers": "ig_followers_release_date", "media": "ig_media_release_date"})

feature_df = feature_df.merge(
    release_rows,
    on=["song_id", "release_date"],
    how="left"
)

#%%

ig_pre12_clean.drop(columns=["followers_zero_proxy_missing", "media_zero_proxy_missing"], inplace=True)
feature_df

#%%
"""
Now, calculate 12 month, 6 month, 3 month, 1 month, absolute growth for 
tiktok likes, tiktok followers
instagram followers
youtube subs, views

And then, calculate mean counts for each, 
and then, do some fourier transform. 

Maybe we can disregard cagr. 

"""


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

social_handles = pd.read_csv('data/processed_data/social_handles.csv')
social_handles["has_ig"] = social_handles["ig_handle"].notnull()
social_handles["has_tt"] = social_handles["tt_handle"].notnull()
social_handles["has_yt"] = social_handles["yt_handle"].notnull()
cols = ["ig_handle", "tt_handle", "yt_handle"]
all_missing = social_handles[cols].isnull().all(axis=1)
social_handles["has_social_media"] = ~all_missing
social_handles.to_csv('data/processed_data/social_handles.csv', index=False)

#%%

feature_df = feature_df.merge(social_handles, on=['artist'], how='left', validate='many_to_one')

#%%
feature_df.to_csv('data/processed_data/feature_df_with_socials.csv', index=False)

#%%

# Things to think about when imputing with 0.0 for social values
# In the situation where an artist doesn't have instagram, we can put in 0.0 for ig_followers
# and put False for has_ig
# Then, in the case where another artist actually has 0.0 ig_followers, has_ig as True will indicate that these fields
# should be viewed differently
# But, what if the artist has_ig, but we weren't able to get data on this artist for any reason?

"""
Artist has no platform (structural zero; you fill with 0.0 but has_* = False).

Artist has platform but your API failed / returned 0 (missing measurement masquerading as 0.0).

How can we distinguish between these two events? I don't want to put 0.0 because
there are instances where somebody has ig, but the api didn't return their data

so if we simply use has_ig dummy, and make everything NaN 0.0, this would make it
seem like artists with ig that had nan were getting no engagement

Apparently we can use LightGBM/XGBoost

as they have branches for missing values
"""

# 54
# %%

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def billions(x, pos):
    return f'{x/1e9:.1f}B'

def millions(x, pos):
    return f'{x/1e6:.1f}M'

#%%

feature_df.columns

#%%

# Instagram

ig_followers_at_release = feature_df.groupby('artist')['ig_followers_release_date'].first()
ig_media_at_release = feature_df.groupby('artist')['ig_media_release_date'].first()

plt.figure(figsize=(8,5))
plt.hist(ig_followers_at_release, bins=50)
plt.xlabel("Followers (Millions)")
plt.ylabel("Artist Count")
plt.title("Distribution of Instagram Followers (release-date)")

ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(millions))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(ig_media_at_release, bins=50)
plt.xlabel("All-time Uploads")
plt.ylabel("Artist Count")
plt.title("Distribution of all-time Instagram uploads (release-date)")

ax = plt.gca()
plt.tight_layout()
plt.show()

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
#%%

feature_df

#%%

mask = feature_df['release_date'] > '2025-01-01'

t2025 = feature_df[mask]

plt.scatter(feature_df["release_date"], feature_df["song_length"])
plt.xlabel("Song Duration")
plt.ylabel("Song Length (Minutes)")
plt.xticks(rotation=45)
plt.show()

#%%

feature_df[feature_df["song_length"] == feature_df["song_length"].min()]

#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

x = feature_df["entry_week_pos"]
y = feature_df['tt_likes_release_date']
z = feature_df["lifespan"]

ax.scatter(x, y, z)

ax.set_xlabel("Entry Week Position (1 = Best)")
ax.set_ylabel("TT likes at Release")
ax.set_zlabel("Lifespan (weeks)")

ax.invert_xaxis()  # flip x-axis if you still want 1=best on right
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
