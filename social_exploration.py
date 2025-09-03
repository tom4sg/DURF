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
