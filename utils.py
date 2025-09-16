import pandas as pd
import matplotlib.pyplot as plt

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
        window = pd.date_range(rd - pd.Timedelta(days=days_back), rd, freq="D")

        base = pd.DataFrame({date_col: window})
        for k, v in zip(key_cols, keys):
            base[k] = v
        base[release_col] = rd

        merged = base.merge(g, on=[*key_cols, date_col, release_col], how="left", indicator=True)
        merged["was_inserted"] = merged["_merge"].eq("left_only")
        merged = merged.drop(columns="_merge")
        filled.append(merged)

    return pd.concat(filled, ignore_index=True)


def plot_two_metrics(
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