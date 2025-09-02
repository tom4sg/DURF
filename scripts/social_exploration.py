#%%

import pandas as pd

#%%

instagram = pd.read_csv("data/raw_data/social_archives/instagram_archive.csv")
tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
emerging_hot_100 = pd.read_csv("data/processed_data/emerging_hot_100.csv")

#%%

artist_name = "Armani White"

#%%

artist_instagram = instagram[instagram["artist_id"] == artist_name]
artist_tiktok = tiktok[tiktok["artist_id"] == artist_name]
artist_youtube = youtube[youtube["artist_id"] == artist_name]

#%%

artist_hot_100 = emerging_hot_100[emerging_hot_100["main_artist"] == artist_name]
artist_hot_100

#%%
import matplotlib.pyplot as plt

# Ensure dates & sort
artist_instagram = artist_instagram.copy()
artist_instagram["date"] = pd.to_datetime(artist_instagram["date"])
artist_instagram = artist_instagram.sort_values("date").reset_index(drop=True)

artist_hot_100 = artist_hot_100.copy()
artist_hot_100["chart_week"] = pd.to_datetime(artist_hot_100["chart_week"])
artist_hot_100 = artist_hot_100.sort_values("chart_week").reset_index(drop=True)

# Basic twin-axis plot
fig, ax1 = plt.subplots(figsize=(11,6))

# Instagram followers (left y-axis)
ax1.plot(
    artist_instagram["date"], 
    artist_instagram["followers"], 
    linewidth=2, 
    label="Instagram followers"
)
ax1.set_xlabel("Date")
ax1.set_ylabel("Instagram followers")
ax1.grid(True, alpha=0.25)

# Billboard rank (right y-axis; invert so #1 is at top)
ax2 = ax1.twinx()
ax2.plot(
    artist_hot_100["chart_week"], 
    artist_hot_100["current_week"], 
    marker="o", 
    linestyle="--", 
    label="Billboard Hot 100 rank"
)
ax2.set_ylabel("Billboard chart position (lower is better)")
ax2.invert_yaxis()

# Shared title & legends
title_artist = artist_instagram["artist_id"].iloc[0] if not artist_instagram.empty else "Artist"
fig.suptitle(f"{title_artist}: Instagram Growth vs. Billboard Rank", y=1.02)

# Two legends (left/right)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper left")

plt.tight_layout()
plt.show()



# %%
