#%%

import requests
import pandas as pd
from datetime import datetime
import s3fs
import os
import pyarrow
import time, random
from data.raw_data.social_handles.social_handles_ig import ig
from data.raw_data.social_handles.social_handles_tt import tt
from data.raw_data.social_handles.social_handles_yt import yt

#%%

for artist, handle in ig.items():
    print(artist, " - ", handle)

print(len(ig.keys()))

#%%

def get_social_data(handles_dict, platform):

    count = 0

    base_url = "https://matrix.sbapis.com/b"

    dfs = []

    for artist, handle in handles_dict.items():
        
        endpoint = f"/{platform}/statistics?query={handle}&history=vault&allow-stale=false"

        headers = {
            "clientid": os.getenv("SOCIAL_BLADE_CLIENT_ID"),
            "token": os.getenv("SOCIAL_BLADE_API_TOKEN")
        }

        response = requests.get(base_url + endpoint, headers=headers)
        print(artist, handle)
        print("--------------------------------")
        print(response.status_code)
        print("--------------------------------")
        print(count, " of ", len(handles_dict))
        count += 1
        time.sleep(1)

        if response.status_code != 200:
            print("Skipping artist: ", artist, " due to status code: ", response.status_code)
            continue


        data = response.json()
        df = pd.DataFrame(data["data"]["daily"])
        df["artist_id"] = artist
        df["platform"] = platform
        df["handle"] = handle
        df["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
        dfs.append(df)

    
    return pd.concat(dfs, ignore_index=True)

#%%

batch = get_social_data(ig, "instagram")
batch.to_csv(f"data/processed_data/instagram_archive.csv")

#%%

batch = get_social_data(tt, "tiktok")
batch.to_csv(f"data/processed_data/tiktok_archive.csv")

#%%

batch = get_social_data(yt, "youtube")
batch.to_csv(f"data/processed_data/youtube_archive.csv")

# %%

"""
For Instagram, we weren't able to get:

404 Error:
Harry Styles
YoungBoy Never Broke Again
Kendrick Lamar
SZA
NengoFlow
Lana Del Rey
Rylo Rodriguez
LISA
Childish Gambino
CHRYSTAL
wifiskeleton
JEONGYEON

400 Error:
Future
Nirvana
4*TOWN
Ryan Gosling
The Citizens of Halloween
Perry Como
HUNTR/X
RUMI



For Tiktok, we weren't able to get:

404 Error:
The Walters
Jake Owen
NengoFlow
Rylo Rodriguez
Mustard
Childish Gambino
Richy Mitch And The Coal Miners
Neton Vega
Gelo

400 Error:
V
Diane Guerrero
XXXTENTACION
Yeat
Lil Shordie Scott
Nirvana
4*TOWN
42 Dugg
Kay Flock
Kendrick Lamar
Hitkidd
Drake
Mac Miller
Jin
RM
Mac Demarco
Junior H
Jimin
Tyler, The Creator
Agust D
Baby Keem
Ryan Gosling
Dominic Fike
Rick Ross
The Citizens of Halloween
Jackson 5
Perry Como
J. Cole
Mustard
Jonathan Bailey
Darlene Love
Saja Boys
HUNTR/X
RUMI
JEONGYEON
Ozzy Osbourne




For Youtube, we weren't able to get:

404 Error:
Diane Guerrero
Dylan Scott
PinkPantheress

400 Error:
munilong
Stephanie Beatriz
Jung Kook
4*TOWN
j-hope
jin
RM
Falling In Reverse
Jimin
Jack Black
Agust D
Ryan Gosling
The Citizens of Halloween
Jonathan Bailey
Aleman
wifiskeleton
Saja Boys
HUNTR/X
Rumi
JEONGYEON

"""
# %%