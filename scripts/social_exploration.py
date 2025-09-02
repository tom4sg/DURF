#%%

import pandas as pd

#%%

instagram = pd.read_csv("data/raw_data/social_archives/instagram_archive.csv")
tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
emerging_hot_100 = pd.read_csv("data/processed_data/emerging_hot_100.csv")

#%%

artist_name = "Taylor Swift"

#%%

artist_instagram = instagram[instagram["artist_id"] == artist_name]
artist_tiktok = tiktok[tiktok["artist_id"] == artist_name]
artist_youtube = youtube[youtube["artist_id"] == artist_name]

#%%

artist_hot_100 = emerging_hot_100[emerging_hot_100["main_artist"] == artist_name]
artist_hot_100

#%%
import matplotlib.pyplot as plt
