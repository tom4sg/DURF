"""
Ok, we have to consider a few things in the feature engineering process:

Some of our unique artists have several songs, we need to randomly select one per artist

We only want to use data prior to, or on a particular song's release date. 

We must remember that there are one-off null values in Social Data, which we need to account for (Maybe Imputation?).

We want to create features inspired by the following:

- Artist Follower Count (On Release Date)
- Compound Weekly Growth Rate of Artist Followers (the 4 weeks prior to release)
- Relevancy-to-Release Score - How relevant is the content of their posting to the release? (via NLP on comments, captions, hashtags)

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

picked = (solo_songs
          .groupby("main_artist", group_keys=False)
          .apply(lambda g: g.sample(1, random_state=42))
          .reset_index(drop=True))

picked

#%%

feature_df = pd.DataFrame()

#%%

youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
youtube["date"] = pd.to_datetime(youtube["date"])
youtube.drop(columns=["Unnamed: 0"], inplace=True)

tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
tiktok["date"] = pd.to_datetime(tiktok["date"])
tiktok.drop(columns=["Unnamed: 0"], inplace=True)

instagram = pd.read_csv("data/raw_data/social_archives/instagram_archive.csv")
instagram["date"] = pd.to_datetime(instagram["date"])
instagram.drop(columns=["Unnamed: 0"], inplace=True)

#%%
artist_name = "Taylor Swift"

artist_youtube = youtube[youtube["artist_id"] == artist_name].sort_values("date")

normed = artist_youtube[["subs","views"]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
normed["date"] = artist_youtube["date"]
normed

plt.figure(figsize=(12,6))
sns.lineplot(x="date", y="subs", data=normed, label="subscribers")
sns.lineplot(x="date", y="views", data=normed, label="Views")

plt.title(f"{artist_name} TikTok Metrics (Normalized)")
plt.ylabel("Normalized (0–1)")
plt.show()