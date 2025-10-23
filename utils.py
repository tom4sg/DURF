import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np


def expand_to_full_window(
    df,
    key_cols=("artist","platform"),
    date_col="date",
    release_col="release_date",
    days_back=28,
):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d[release_col] = pd.to_datetime(d[release_col])

    filled = []
    for keys, g in d.groupby(list(key_cols), dropna=False):
        rd = g[release_col].iloc[0]

        # Taking date range from 38 days back to release date
        window = pd.date_range(rd - pd.Timedelta(days=days_back), rd, freq="D")

        base = pd.DataFrame({date_col: window})
        # date
        # 0 2022-07-17
        # 1 2022-07-18
        # 2 2022-07-19
        # 3 2022-07-20
        for k, v in zip(key_cols, keys):
            base[k] = v
        base[release_col] = rd

        #         date    artist   platform release_date
        # 0 2022-07-17     d4vd  instagram   2022-07-20
        # 1 2022-07-18     d4vd  instagram   2022-07-20
        # 2 2022-07-19     d4vd  instagram   2022-07-20
        # 3 2022-07-20     d4vd  instagram   2022-07-20

        merged = base.merge(g, on=[*key_cols, date_col, release_col], how="left", indicator=True)
        # Above means keep every row from base, match rows from g when all columns match
        merged["was_inserted"] = merged["_merge"].eq("left_only")
        merged = merged.drop(columns="_merge")
        filled.append(merged)

    return pd.concat(filled, ignore_index=True)


def plot_two_metrics_artist(
    df,
    metric1="subs",
    metric2="views",
    artists=None,
    artist_col="artist",
    platform_col="platform",
    date_col="date",
    release_col="release_date",
    outdir="graphs",
    filename_pattern="{metric1}_{metric2}_{artist}_{platform}.png",
    dpi=150,
    show=False,
):
    """
    Plot two metrics over time for each (artist, platform) group.
    """
    # optional filter
    if artists is not None:
        df = df[df[artist_col].isin(artists)]

    for (artist, platform), g in df.groupby([artist_col, platform_col]):
        g = g.sort_values(date_col)

        fig, ax1 = plt.subplots(figsize=(8, 4))
        line1, = ax1.plot(
            g[date_col], g[metric1],
            marker="o", label=metric1
        )
        ax1.set_xlabel(date_col)
        ax1.set_ylabel(metric1)

        ax2 = ax1.twinx()
        line2, = ax2.plot(
            g[date_col], g[metric2],
            marker=".", linestyle="--", label=metric2
        )
        ax2.set_ylabel(metric2)

        rd = g[release_col].iloc[0]
        ax1.axvline(rd, linestyle=":", linewidth=1)

        fig.autofmt_xdate()

        handles = [line1, line2]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="upper left")

        plt.tight_layout()

        if show:
            plt.show()
        else:
            filename = filename_pattern.format(
                metric1=metric1, metric2=metric2,
                artist=artist, platform=platform
            )
            plt.savefig(f"{outdir}/{filename}", dpi=dpi)
            plt.close(fig)

def plot_two_metrics_song(
    df,
    metric1="subs",
    metric2="views",
    artists=None,
    artist_col="artist",
    song_col="song_id",
    platform_col="platform",
    date_col="date",
    release_col="release_date",
    outdir="graphs",
    filename_pattern="{metric1}_{metric2}_{artist}_{song}_{platform}.png",
    dpi=150,
    show=False,
):
    """
    Plot two metrics over time for each (artist, song_id, platform) group.
    """
    # optional filter
    if artists is not None:
        df = df[df[artist_col].isin(artists)]

    # ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # group by artist, song_id, platform
    for (artist, song, platform), g in df.groupby([artist_col, song_col, platform_col]):
        g = g.sort_values(date_col)

        fig, ax1 = plt.subplots(figsize=(8, 4))
        line1, = ax1.plot(
            g[date_col], g[metric1],
            marker="o", label=metric1
        )
        ax1.set_xlabel(date_col)
        ax1.set_ylabel(metric1)

        ax2 = ax1.twinx()
        line2, = ax2.plot(
            g[date_col], g[metric2],
            marker=".", linestyle="--", label=metric2
        )
        ax2.set_ylabel(metric2)

        # vertical line at release date (first row’s release_date)
        rd = g[release_col].iloc[0]
        ax1.axvline(rd, linestyle=":", linewidth=1)

        fig.autofmt_xdate()

        handles = [line1, line2]
        labels = [h.get_label() for h in handles]
        ax1.legend(handles, labels, loc="upper left")

        plt.tight_layout()

        if show:
            plt.show()
        else:
            # sanitize values for filename
            safe_artist = re.sub(r'[\\/*?:"<>|]', "_", str(artist))
            safe_song = re.sub(r'[\\/*?:"<>|]', "_", str(song))
            safe_platform = re.sub(r'[\\/*?:"<>|]', "_", str(platform))

            filename = filename_pattern.format(
                metric1=metric1,
                metric2=metric2,
                artist=safe_artist,
                song=safe_song,
                platform=safe_platform
            )
            plt.savefig(os.path.join(outdir, filename), dpi=dpi)
            plt.close(fig)

def plot_ig_daily_change_all(
    df,
    song_col="song_id",
    artist_col="artist",
    date_col="date",
    metric_col="ig_followers_diff_daily",
    days_back=365,
    outdir="graphs/ig_daily_pre12m",
    dpi=150,
    show=False,
):
    """
    Plots the daily change in Instagram followers for all songs during the
    12 months preceding their release date, saving one plot per song.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns: song_id, artist, date, release_date, ig_followers_diff_daily.
    days_back : int
        Days before release_date to include (default 365).
    outdir : str
        Output directory for saving figures.
    """
    os.makedirs(outdir, exist_ok=True)

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["release_date"] = pd.to_datetime(d["release_date"])

    for (artist, song), g in d.groupby([artist_col, song_col]):
        rd = g["release_date"].iloc[0]
        mask = (g[date_col] >= rd - pd.Timedelta(days=days_back)) & (g[date_col] <= rd)
        g = g.loc[mask].sort_values(date_col)

        if g.empty:
            continue

        # align x-axis as days before release
        g["days_to_release"] = (g[date_col] - rd).dt.days

        # filter extreme z-scores
        z = (g[metric_col] - g[metric_col].mean()) / g[metric_col].std()
        g.loc[z.abs() > 3, metric_col] = np.nan

        plt.figure(figsize=(8, 4))
        plt.plot(
            g["days_to_release"],
            g[metric_col],
            label=f"{artist} — {song}",
        )
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"Daily IG Follower Change — {artist} ({song})")
        plt.xlabel("Days to Release")
        plt.ylabel("Δ Instagram Followers")
        plt.tight_layout()

        if show:
            plt.show()
        else:
            safe_artist = str(artist).replace("/", "_")
            safe_song = str(song).replace("/", "_")
            plt.savefig(f"{outdir}/{safe_artist}_{safe_song}_ig_daily.png", dpi=dpi)
            plt.close()