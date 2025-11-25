"""
Before any feature engineering, let's make sure to split our data, keeping in mind that artists should not
be found in both train and test...

groupkfold with k = 10 might be best option for cross-validation...
"""
#%%

import pandas as pd
from sklearn.model_selection import train_test_split

#%%

songs = pd.read_csv('data/processed_data/songs.csv')

#%%
"""
It is important to note that since many artists have several songs released on same day (album)
The social media features will be the exact same across all those songs, since the data is at the profile-level
Because of this, we can split train and test by ensuring that songs from the same artist are only found in one split
"""

from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(songs, groups=songs['artist']))

train = songs.iloc[train_idx]
test = songs.iloc[test_idx]

#%%

"""
Ok, we finally have our train-test-split!

Now, we can do cross validation on train dataset...

Do more research on groupkfold

"""

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# weeks on chart plot across all songs, solo songs, and collab songs
plt.figure(figsize=(12,6))
sns.histplot(train["lifespan"], bins=30, kde=True, edgecolor="black")
plt.title("How Long Songs Stay on the Billboard Hot 100 (2022–2025)")
plt.xlabel("Weeks on Chart")
plt.ylabel("Frequency")
plt.show()

#%%

train['lifespan'].mean(), train['lifespan'].var()

#%%

"""
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

ig_pre12_clean = ig_pre12_clean.copy()
ig_pre12_clean['date'] = pd.to_datetime(ig_pre12_clean['date'], errors='coerce')
ig_pre12_clean = ig_pre12_clean.sort_values(["song_id", "platform", "date"])

ig_pre12_clean['ig_followers_diff_12m'] = (
    ig_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(365))
)

ig_pre12_clean['ig_followers_diff_6m'] = (
    ig_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(182))
)

ig_pre12_clean['ig_followers_diff_3m'] = (
    ig_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(91))
)

ig_pre12_clean['ig_followers_diff_1m'] = (
    ig_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(30))
)

ig_feat = (
    ig_pre12_clean
      .loc[ig_pre12_clean['date'] == ig_pre12_clean['release_date'],
           ['song_id','release_date',
           'ig_followers_diff_12m','ig_followers_diff_6m', 'ig_followers_diff_3m', 'ig_followers_diff_1m']]
           .drop_duplicates(['song_id','release_date'])
)

#%%

yt_pre12_clean = yt_pre12_clean.copy()
yt_pre12_clean['date'] = pd.to_datetime(yt_pre12_clean['date'], errors='coerce')
yt_pre12_clean = yt_pre12_clean.sort_values(["song_id", "platform", "date"])

yt_pre12_clean['yt_subs_diff_12m'] = (
    yt_pre12_clean
    .groupby('song_id')['subs']
    .transform(lambda s: s - s.shift(365))
)

yt_pre12_clean['yt_subs_diff_6m'] = (
    yt_pre12_clean
    .groupby('song_id')['subs']
    .transform(lambda s: s - s.shift(182))
)

yt_pre12_clean['yt_subs_diff_3m'] = (
    yt_pre12_clean
    .groupby('song_id')['subs']
    .transform(lambda s: s - s.shift(91))
)

yt_pre12_clean['yt_subs_diff_1m'] = (
    yt_pre12_clean
    .groupby('song_id')['subs']
    .transform(lambda s: s - s.shift(30))
)

yt_pre12_clean['yt_views_diff_12m'] = (
    yt_pre12_clean
    .groupby('song_id')['views']
    .transform(lambda s: s - s.shift(365))
)

yt_pre12_clean['yt_views_diff_6m'] = (
    yt_pre12_clean
    .groupby('song_id')['views']
    .transform(lambda s: s - s.shift(182))
)

yt_pre12_clean['yt_views_diff_3m'] = (
    yt_pre12_clean
    .groupby('song_id')['views']
    .transform(lambda s: s - s.shift(91))
)

yt_pre12_clean['yt_views_diff_1m'] = (
    yt_pre12_clean
    .groupby('song_id')['views']
    .transform(lambda s: s - s.shift(30))
)

yt_feat = (
    yt_pre12_clean
      .loc[yt_pre12_clean['date'] == yt_pre12_clean['release_date'],
           ['song_id','release_date',
           'yt_subs_diff_12m','yt_subs_diff_6m', 'yt_subs_diff_3m', 'yt_subs_diff_1m',
           'yt_views_diff_12m','yt_views_diff_6m', 'yt_views_diff_3m', 'yt_views_diff_1m']]
      .drop_duplicates(['song_id','release_date'])
)

#%%

tt_pre12_clean = tt_pre12_clean.copy()
tt_pre12_clean['date'] = pd.to_datetime(tt_pre12_clean['date'], errors='coerce')
tt_pre12_clean = tt_pre12_clean.sort_values(["song_id", "platform", "date"])

tt_pre12_clean['tt_followers_diff_12m'] = (
    tt_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(365))
)

tt_pre12_clean['tt_followers_diff_6m'] = (
    tt_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(182))
)

tt_pre12_clean['tt_followers_diff_3m'] = (
    tt_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(91))
)

tt_pre12_clean['tt_followers_diff_1m'] = (
    tt_pre12_clean
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(30))
)

tt_pre12_clean['tt_likes_diff_12m'] = (
    tt_pre12_clean
    .groupby('song_id')['likes']
    .transform(lambda s: s - s.shift(365))
)

tt_pre12_clean['tt_likes_diff_6m'] = (
    tt_pre12_clean
    .groupby('song_id')['likes']
    .transform(lambda s: s - s.shift(182))
)

tt_pre12_clean['tt_likes_diff_3m'] = (
    tt_pre12_clean
    .groupby('song_id')['likes']
    .transform(lambda s: s - s.shift(91))
)

tt_pre12_clean['tt_likes_diff_1m'] = (
    tt_pre12_clean
    .groupby('song_id')['likes']
    .transform(lambda s: s - s.shift(30))
)

tt_feat = (
    tt_pre12_clean
      .loc[tt_pre12_clean['date'] == tt_pre12_clean['release_date'],
           ['song_id','release_date',
           'tt_followers_diff_12m','tt_followers_diff_6m', 'tt_followers_diff_3m', 'tt_followers_diff_1m',
           'tt_likes_diff_12m','tt_likes_diff_6m', 'tt_likes_diff_3m', 'tt_likes_diff_1m']]
      .drop_duplicates(['song_id','release_date'])
)

#%%

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
    .groupby('song_id')['followers']
    .transform(lambda s: s - s.shift(1))
)

tt_pre12_clean['tt_likes_diff_daily'] = (
    tt_pre12_clean
    .groupby('song_id')['likes']
    .transform(lambda s: s - s.shift(1))
)

tt_pre12_clean['tt_uploads_diff_daily'] = (
    tt_pre12_clean
    .groupby('song_id')['uploads']
    .transform(lambda s: s - s.shift(1))
)

yt_pre12_clean['yt_subs_diff_daily'] = (
    yt_pre12_clean
    .groupby('song_id')['subs']
    .transform(lambda s: s - s.shift(1))
)

yt_pre12_clean['yt_views_diff_daily'] = (
    yt_pre12_clean
    .groupby('song_id')['views']
    .transform(lambda s: s - s.shift(1))
)

#%%

# Now merge on BOTH keys
feature_df = (
    feature_df
      .merge(ig_feat, on=['song_id','release_date'], how='left', validate='many_to_one')
      .merge(tt_feat, on=['song_id','release_date'], how='left', validate='many_to_one')
      .merge(yt_feat, on=['song_id','release_date'], how='left', validate='many_to_one')
)
feature_df

#%%

feature_df.to_csv('data/processed_data/feature_df.csv')

#%%

# Make a working copy
df = tt_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])


df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Z-score filter within each song_id
grouped = df.groupby('song_id')['tt_likes_diff_daily']
z = grouped.transform(lambda x: (x - x.mean()) / x.std())

# Drop / mask outliers
df.loc[z.abs() > 3, 'tt_likes_diff_daily'] = np.nan

# Plot from the filtered df
plt.figure(figsize=(12, 6))
for song_id, group in df.groupby('song_id'):
    if song_id != 'Eternity — Alex Warren':  # skip if you want
        group = group.sort_values('days_since_start')
        plt.plot(
            group['days_since_start'],
            group['tt_likes_diff_daily'],
            alpha=0.5
        )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily TikTok Likes Change")
plt.xlabel('Days since first observation')
plt.ylabel('Daily Like Change')
plt.tight_layout()
plt.show()

#%%

# Make a working copy
df = ig_pre12_clean.copy()
df['date'] = pd.to_datetime(df['date'])

# Align all artists
df['days_since_start'] = (
    df.groupby('song_id')['date']
      .transform(lambda d: (d - d.min()).dt.days)
)

# Z-score filter within each song_id
grouped = df.groupby('song_id')['ig_followers_diff_daily']
z = grouped.transform(lambda x: (x - x.mean()) / x.std())

# Drop / mask outliers
df.loc[z.abs() > 3, 'ig_followers_diff_daily'] = np.nan

# Plot from the filtered df
plt.figure(figsize=(12, 6))
for song_id, group in df.groupby('song_id'):
    if song_id != 'Eternity — Alex Warren':  # skip if you want
        group = group.sort_values('days_since_start')
        plt.plot(
            group['days_since_start'],
            group['ig_followers_diff_daily'],
            alpha=0.5
        )

plt.axhline(0, color='gray', linewidth=1, linestyle='--')
plt.title("Daily Instagram Followers Change")
plt.xlabel('Days since first observation')
plt.ylabel('Daily Follower Change')
plt.tight_layout()
plt.show()

#%%

from matplotlib.ticker import FuncFormatter

def millions(x, pos):
    return f'{x/1e6:.1f}M'

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# column names and titles for each panel
diff_cols = ['ig_followers_diff_1m',
             'ig_followers_diff_3m',
             'ig_followers_diff_6m',
             'ig_followers_diff_12m']
titles = ['Instagram Follower Change (1-month Pre-Release)',
          'Instagram Follower Change (3-month Pre-Release)',
          'Instagram Follower Change (6-month Pre-Release)',
          'Instagram Follower Change (12-month Pre-Release)']

# flatten axes array to 1D so we can loop
axes = axes.ravel()

for ax, col, title in zip(axes, diff_cols, titles):
    ax.scatter(feature_df[col], feature_df['lifespan'], alpha=0.6, color='purple')
    ax.set_title(title)
    ax.set_xlabel('IG Follower Change (Absolute)')
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylabel('Lifespan (weeks)')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# column names and titles for each panel
diff_cols = ['tt_likes_diff_1m',
             'tt_likes_diff_3m',
             'tt_likes_diff_6m',
             'tt_likes_diff_12m']
titles = ['Tiktok Likes Change (1-month Pre-Release)',
          'Tiktok Likes Change (3-month Pre-Release)',
          'Tiktok Likes Change (6-month Pre-Release)',
          'Tiktok Likes Change (12-month Pre-Release)']

# flatten axes array to 1D so we can loop
axes = axes.ravel()

for ax, col, title in zip(axes, diff_cols, titles):
    ax.scatter(feature_df[col], feature_df['lifespan'], alpha=0.6, color='turquoise')
    ax.set_title(title)
    ax.set_xlabel('TT Likes Change (Absolute)')
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylabel('Lifespan (weeks)')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

#%%

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# column names and titles for each panel
diff_cols = ['yt_views_diff_1m',
             'yt_views_diff_3m',
             'yt_views_diff_6m',
             'yt_views_diff_12m']
titles = ['Youtube Views Change (1-month Pre-Release)',
          'Youtube Views Change (3-month Pre-Release)',
          'Youtube Views Change (6-month Pre-Release)',
          'Youtube Views Change (12-month Pre-Release)']

# flatten axes array to 1D so we can loop
axes = axes.ravel()

for ax, col, title in zip(axes, diff_cols, titles):
    ax.scatter(feature_df[col], feature_df['lifespan'], alpha=0.6, color='red')
    ax.set_title(title)
    ax.set_xlabel('YT Views Change (Absolute)')
    ax.xaxis.set_major_formatter(FuncFormatter(millions))
    ax.set_ylabel('Lifespan (weeks)')
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

#%%

def billions(x, pos):
    return f'{x/1e9:.1f}B'

new_df = ig_pre12_clean[ig_pre12_clean['song_id'] == "Don't We — Morgan Wallen"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(new_df['date'], new_df['followers'])

ax.set_xlabel('Date')
ax.set_ylabel('Instagram Followers (All-Time)')
ax.yaxis.set_major_formatter(FuncFormatter(billions))
plt.title('Morgan Wallen Instagram Followers 12-Months prior (Thought You Should Know)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

ig_pre12_clean[ig_pre12_clean['artist'] == 'Morgan Wallen']['song_id'].unique()

#%%

# visualize each metric for every artist to see if there are consistent dips
# in where you think there are socialblade outages

# do same for daily diff plots (accross all artists)
# Look into more hidden data quality issues eg. Alex Warren

# notice, socialblade scrapes the value of likes from profile, which for larger numbers, is rounded off

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

#%%

from utils import plot_ig_daily_change_all
import numpy as np

plot_ig_daily_change_all(ig_pre12_clean)

# %%

df = ig_pre12_clean[ig_pre12_clean['song_id'] == "Burgundy — $uicideboy$"]
mask = (~ig_pre12_clean.groupby('song_id')['ig_followers_diff_daily'].all())
df = ig_pre12_clean.groupby('song_id')['ig_followers_diff_daily'][mask]
df

# %%
ig_pre12_clean.to_csv('data/processed_data/ig_clean.csv', index=False)
# %%

mask = ig_pre12_clean['date'] != ig_pre12_clean['release_date']
ig_pre12_clean.loc[mask, 'ig_followers_diff_daily'].notnull().value_counts()
# %%
ig_pre12_clean[mask]

#%%

ig_pre12_clean.sort_values(["song_id", "platform", "date"], ascending=False)

# %%

ig_pre12_clean['ig_followers_diff_daily'].isna().value_counts()
# %%

mask = ig_pre12_clean['date'] != ig_pre12_clean['release_date'] - pd.Timedelta(days=365)
ig_pre12_clean.loc[mask, 'ig_followers_diff_daily'].isna().value_counts()

# %%

new_df = ig_pre12_clean[ig_pre12_clean['song_id'] == "Vegas — Doja Cat"]
plt.plot(new_df['date'], new_df['ig_followers_diff_daily'])
plt.show()

#%%
import pandas as pd
feature = pd.read_csv('data/processed_data/feature_df.csv')


#%%

# pd.qcut(
#     feature['lifespan'].rank(method='first'),
#     q=4,
#     labels=[1, 2, 3, 4]
# ).value_counts()

feature['lifespan']

#%%

feature['lifespan'].var() > feature['lifespan'].mean()

#%%

feature.to_csv('data/processed_data/feature_df_with_labels.csv', index=False)

#%%


from scipy.signal import detrend
import numpy as np

ig = pd.read_csv('data/processed_data/ig_clean.csv')

song_id = "Get It Sexyy — Sexyy Red"

followers = ig[ig['song_id'] == song_id].copy()
followers['date'] = pd.to_datetime(followers['date'])
followers = followers.dropna(subset=['ig_followers_diff_daily'])

followers['follow_detrend'] = detrend(followers['ig_followers_diff_daily'], type='linear')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Raw and detrended time series
axes[0].plot(followers['date'], followers['ig_followers_diff_daily'], c='blue', label='Raw')
axes[0].plot(followers['date'], followers['follow_detrend'], c='red', label='Linear Detrended')
axes[0].set_title('Input Signal (IG Followers)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Δ Followers')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

fft_vals = np.fft.rfft(followers['follow_detrend'])
fft_freqs = np.fft.rfftfreq(len(followers), d=1)  # d=1 assumes daily samples

# Identify dominant frequency
max_idx = np.argmax(np.abs(fft_vals))
max_freq = fft_freqs[max_idx]

# Plot FFT magnitude
axes[1].plot(fft_freqs, np.abs(fft_vals), color='purple')
axes[1].axvline(max_freq, color='gray', linestyle='--', linewidth=1)
axes[1].set_title(f'Fourier Transform Magnitude (peak={max_freq:.4f})')
axes[1].set_xlabel('Frequency (1/day)')
axes[1].set_ylabel('|Amplitude|')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%

ig[ig['song_id'] == 'Water — Tyla']
#%%

fft_freqs
# %%

print("Daily analysis:")
print(f"  Peak frequency: {max_freq:.4f} cycles/day")
print(f"  Peak period: {1/max_freq:.1f} days")

print("\nWeekly analysis:")
print(f"  Peak frequency: {max_freq:.4f} cycles/week")  
print(f"  Peak period: {1/max_freq:.1f} weeks")
# %%

"""
Notice, .0027 means the cycle is the entire year

This basically mean no cyclical trend was identified for this year, the whole year is a cycle
"""


#%%

import pandas as pd
import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
import seaborn as sns

ig = pd.read_csv('data/processed_data/ig_clean.csv')
ig['date'] = pd.to_datetime(ig['date'])

results = []

for song_id, group in ig.groupby('song_id'):
    g = group.dropna(subset=['ig_followers_diff_daily']).copy()
    if len(g) < 4:
        continue  # skip short or empty series

    detrended = detrend(g['ig_followers_diff_daily'].values, type='linear')
    fft_vals = np.fft.rfft(detrended)
    fft_freqs = np.fft.rfftfreq(len(detrended), d=1)

    # ignore zero frequency (DC component)
    if len(fft_vals) > 1:
        fft_vals[0] = 0  

    max_idx = np.argmax(np.abs(fft_vals))
    max_freq = fft_freqs[max_idx]

    results.append({'song_id': song_id, 'dominant_freq': max_freq})

fft_df = pd.DataFrame(results)

#%%

feature = pd.read_csv('data/processed_data/feature_df.csv')
merged = feature.merge(fft_df, on='song_id', how='left')

# Scatter plot
plt.figure(figsize=(7,5))
plt.scatter(merged['dominant_freq'], merged['lifespan'], alpha=0.6)
plt.xlabel('Dominant Frequency (cycles/day)')
plt.ylabel('Lifespan (weeks on Hot 100)')
plt.title('Song Lifespan vs. Instagram Follower Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%

max_row = merged.loc[merged['dominant_freq'].idxmax()]

print("Song with highest dominant frequency:")
print(f"  Song ID: {max_row['song_id']}")
print(f"  Dominant Frequency: {max_row['dominant_freq']:.6f} cycles/day")
print(f"  Lifespan: {max_row['lifespan']} weeks")

# %%


# Identify song with max frequency
max_row = merged.loc[merged['dominant_freq'].idxmax()]
song_id_max = max_row['song_id']
print(f"Plotting for: {song_id_max}")

# Extract that artist's IG data
song_data = ig[ig['song_id'] == song_id_max].copy()
song_data = song_data.dropna(subset=['ig_followers_diff_daily'])
song_data['date'] = pd.to_datetime(song_data['date'])

# Plot
plt.figure(figsize=(10,4))
plt.plot(song_data['date'], song_data['ig_followers_diff_daily'], c='blue')
plt.title(f'IG Daily Follower Change — {song_id_max}')
plt.xlabel('Date')
plt.ylabel('Δ Followers')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
#%%

plt.figure(figsize=(7,4))
plt.hist(merged['dominant_freq'].dropna(), bins=40, color='purple', alpha=0.7)
plt.xlabel('Dominant Frequency (cycles/day)')
plt.ylabel('Count')
plt.title('Distribution of Dominant IG Follower Frequencies')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# Frequency diagnostics
delta_f = fft_freqs[1] - fft_freqs[0]          # frequency step
nyquist_f = fft_freqs[-1]                      # highest resolvable frequency
max_idx = np.argmax(np.abs(fft_vals))          # index of strongest component
peak_f = fft_freqs[max_idx]                    # dominant frequency
period_days = 1 / peak_f if peak_f > 0 else np.inf

print(f"Δf (frequency step): {delta_f:.6f} cycles/day")
print(f"Nyquist frequency:   {nyquist_f:.6f} cycles/day")
print(f"Peak frequency:      {peak_f:.6f} cycles/day (~{period_days:.1f} days period)")

#%%

bins = [0, 3, 10, 20, feature['lifespan'].max() + 1]
labels = ['Very Short (1-3)', 'Short (4-10)', 'Medium (11-20)', 'Long (21+)']

lifespan_bins = feature['lifespan_bin'] = pd.cut(feature['lifespan'], 
                                  bins=bins, 
                                  labels=labels,
                                  include_lowest=True)

lifespan_bins.value_counts()

#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 10))

# 1. Bar plot of bin counts
plt.subplot(2, 2, 1)
counts = lifespan_bins.value_counts().sort_index()
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
counts.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Distribution of Songs Across Lifespan Bins', fontsize=14, fontweight='bold')
plt.xlabel('Lifespan Category', fontsize=11)
plt.ylabel('Number of Songs', fontsize=11)
plt.xticks(rotation=45, ha='right')
# Add counts on top of bars
for i, v in enumerate(counts):
    plt.text(i, v + 20, str(v), ha='center', fontweight='bold')

# 2. Pie chart with percentages
plt.subplot(2, 2, 2)
percentages = lifespan_bins.value_counts(normalize=True).sort_index() * 100
plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=colors, 
        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
plt.title('Percentage Distribution', fontsize=14, fontweight='bold')

# 3. Histogram of raw lifespan with bin boundaries
plt.subplot(2, 2, 3)
plt.hist(feature['lifespan'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
# Add vertical lines for bin boundaries
for boundary, color in zip([3, 10, 20], ['red', 'orange', 'green']):
    plt.axvline(boundary, color=color, linestyle='--', linewidth=2, 
                label=f'{boundary} weeks')
plt.xlabel('Lifespan (weeks)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Raw Lifespan Distribution with Bin Boundaries', fontsize=14, fontweight='bold')
plt.legend()

# 4. Box plot by bin
plt.subplot(2, 2, 4)
feature.boxplot(column='lifespan', by='lifespan_bin', ax=plt.gca())
plt.suptitle('')  # Remove default title
plt.title('Lifespan Distribution Within Each Bin', fontsize=14, fontweight='bold')
plt.xlabel('Bin Category', fontsize=11)
plt.ylabel('Lifespan (weeks)', fontsize=11)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
