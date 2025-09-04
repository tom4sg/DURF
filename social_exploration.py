#%%

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

#%%

instagram = pd.read_csv("data/raw_data/social_archives/instagram_archive.csv")
tiktok = pd.read_csv("data/raw_data/social_archives/tiktok_archive.csv")
youtube = pd.read_csv("data/raw_data/social_archives/youtube_archive.csv")
emerging_hot_100 = pd.read_csv("data/processed_data/emerging_hot_100.csv")
ugc_data = pd.read_csv("data/raw_data/melodyiq_ugc_sample/folded_kehlani_ugc_and_views_tiktok.csv")
ugc_data

#%%

artist_name = "Kehlani"

#%%

artist_instagram = instagram[instagram["artist_id"] == artist_name]
artist_tiktok = tiktok[tiktok["artist_id"] == artist_name]
artist_youtube = youtube[youtube["artist_id"] == artist_name]

#%%

artist_instagram

#%%

artist_hot_100 = emerging_hot_100[emerging_hot_100["song_id"] == "Folded — Kehlani"]
artist_hot_100

#%%

billboard = artist_hot_100.copy()
billboard["chart_week"] = pd.to_datetime(billboard["chart_week"])
billboard

#%%

ugc = ugc_data.copy()
ugc["Date"] = pd.to_datetime(ugc["Date"], format="%d/%b/%y")
ugc = ugc[["Date", "Total Creates"]]
ugc_weekly = (
    ugc.groupby(pd.Grouper(key="Date", freq="W-SAT"))  # Billboard charts are Saturday weeks
    ["Total Creates"].max()  # cumulative creates → use max per week
    .reset_index()
)

# Align column name
ugc_weekly.rename(columns={"Date": "chart_week"}, inplace=True)

merged = pd.merge(billboard, ugc_weekly, on="chart_week", how="inner")

# %%

fig, ax1 = plt.subplots(figsize=(10,6))

# Billboard rank (lower = better)
ax1.plot(merged["chart_week"], merged["current_week"], color="blue", marker="o", label="Billboard Rank")
ax1.invert_yaxis()
ax1.set_ylabel("Billboard Hot 100 Rank", fontsize=12)

# Second y-axis for TikTok UGC
ax2 = ax1.twinx()
ax2.plot(merged["chart_week"], merged["Total Creates"], color="red", marker="x", label="Total UGCs (TikTok)")
ax2.set_ylabel("Total Creates (TikTok)", fontsize=12)
ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

plt.title("Folded by Kehlani - Billboard Rank vs TikTok UGCs")
plt.show()

# %%

emerging_hot_100_unique = emerging_hot_100.drop_duplicates(subset="song_id")
emerging_hot_100_unique

# %%

emerging_hot_100[emerging_hot_100["song_id"] == "Folded — Kehlani"]

# %%

import ast
import pandas as pd

metadata = pd.read_csv("data/processed_data/metadata.csv").reset_index(drop=True)
metadata["genreNames"] = metadata["genreNames"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

#%%

type(metadata["genreNames"].iloc[1])
metadata

# %%

import matplotlib.pyplot as plt

# Step 1: explode genres into separate rows
genres_exploded = metadata.explode("genreNames")
genres_exploded = genres_exploded[genres_exploded["genreNames"] != "Music"]

#%%

# Step 2: count occurrences
genre_counts = genres_exploded["genreNames"].value_counts()

# Step 3: plot
top_genres = genre_counts.head(30)  # top 15 genres
plt.figure(figsize=(12,6))
top_genres.plot(kind="bar")
plt.title("Distribution of Genre Tags (2022-2025)")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Songs", fontsize=12)
plt.show()

#%%

import numpy as np

print(f"Max weeks on chart: {np.max(emerging_hot_100['wks_on_chart'])}")
print(f"Mean weeks on chart: {np.ceil(np.mean(emerging_hot_100['wks_on_chart']))}")
print(f"Median weeks on chart: {round(np.median(emerging_hot_100['wks_on_chart']), 1)}")
print(f"Standard deviation of weeks on chart: {round(np.std(emerging_hot_100['wks_on_chart']), 1)}")

# Let's do a quick check for the most popular song
longest = emerging_hot_100.sort_values("wks_on_chart", ascending=False).head(40)
longest

# %%

weeks_on_chart_df = pd.read_csv("data/processed_data/emerging_hot_100.csv")
weeks_on_chart_df.sort_values(by="wks_on_chart", ascending=False)
weeks_on_chart_df = weeks_on_chart_df.groupby("song_id")["wks_on_chart"].max().reset_index()
weeks_on_chart_df

#%%

import seaborn as sns

sns.histplot(weeks_on_chart_df["wks_on_chart"], bins=30, kde=True, color="blue", edgecolor="black")
plt.title("How Long Songs Stay on the Billboard Hot 100 (2022–2025)")
plt.xlabel("Weeks on Chart")
plt.ylabel("Frequency")
plt.show()
# %%
# Notice: There is an interesting peak around 20 weeks! Why?
# Let's join metadata with weeks_on_chart_df and see if the release date is related

merged = pd.merge(weeks_on_chart_df, metadata, on="song_id", how="inner").drop(columns=["Unnamed: 0"])
merged

# %%

"""
Look at these rules!

RECURRENT RULES:
Descending songs are removed from the Billboard Hot 100 and Radio Songs simultaneously after 20 weeks on 
the Billboard Hot 100 and if ranking below No. 50, or after 52 weeks if below No. 25. 

"""
# %%

"""
SO if this weren't the case, the gap between week 1 and week 20 would be filled
with a smooth exponential type decay. Records falling from 50 would fall to fill these
spots in the 70s 80s etc.
"""
#%%

!pip uninstall -y pydantic instagrapi
!pip install --no-cache-dir "pydantic<2" "instagrapi<2"


#%%

import os
from instagrapi import Client

#%%

cl = Client()
cl.login(os.getenv("INSTAGRAM_USERNAME"), os.getenv("INSTAGRAM_PASSWORD"))

#%%

target_id = cl.user_id_from_username("taylorswift")
posts = cl.user_medias(target_id, amount=10)
for media in posts:
    # download photos to the current folder
    cl.photo_download(media.pk)

#See [examples/session_login.py](examples/session_login.py) for a standalone script demonstrating these login methods.

# %%
